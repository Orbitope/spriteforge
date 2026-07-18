# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
E1 training loop: palette-index classification (methodology review, Part 3 option A).

Design decision (was an open question in the E1 scaffolding devlog note): the palette
used for both the classification target AND the model's FiLM conditioning is extracted
from the CLEAN ground-truth sprite via extract_palette_kmeans, not from the degraded
input. This gives the most accurate training signal (the GT's true colors). At
inference time there is no GT, so the palette must instead come from either the
degraded input itself or a fixed/user-supplied palette (e.g. one of the bundled presets
in spriteforge/core/palette.py) — this is a real train/inference distribution gap, not
resolved here. Flagging it rather than pretending it's already handled.
"""

from __future__ import annotations

import csv
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from spriteforge.core.palette import extract_palette_kmeans, palette_index_map
from spriteforge.model.palette_unet import PaletteUNet, PaletteUNetConfig
from spriteforge.train.train import get_device


def _build_palette_cache(image_paths: list[Path], cache_dir: Path, target_size: int, num_colors: int) -> None:  # noqa: E501
    """Pre-compute and cache palettes + index maps for all sprites. Run once; skips existing."""
    from spriteforge.core.resize import pad_to_target

    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [p for p in image_paths if not (cache_dir / f"{p.stem}.npz").exists()]
    if not missing:
        return
    print(f"[*] Building palette cache for {len(missing)}/{len(image_paths)} sprites…", flush=True)
    for i, img_path in enumerate(missing):
        try:
            pil_img = Image.open(img_path).convert("RGBA")
            img_rgba = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception:
            img_rgba = np.zeros((target_size, target_size, 4), dtype=np.float32)
        if img_rgba.shape[:2] != (target_size, target_size):
            padded = pad_to_target(img_rgba, target_size=target_size)
            img_rgba = padded if padded is not None else np.zeros(
                (target_size, target_size, 4), dtype=np.float32
            )
        img_rgba[..., :3] = img_rgba[..., :3] * img_rgba[..., 3:4]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            palette_np = extract_palette_kmeans(img_rgba, k=num_colors)
        target_idx_np = palette_index_map(img_rgba, palette_np)
        np.savez_compressed(cache_dir / f"{img_path.stem}.npz", rgba=img_rgba, palette=palette_np, target=target_idx_np)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(missing)} cached", flush=True)
    print(f"[+] Palette cache built: {cache_dir}", flush=True)


class PaletteDataset(Dataset):
    """Loads clean sprites, degrades them on-the-fly, and builds palette-index
    classification targets from the clean ground truth.

    Palettes and index maps are pre-computed once into a cache directory to avoid
    running k-means on every item per epoch (which caused a severe bottleneck at
    scale — ~10 minutes per epoch on 9k sprites before caching)."""

    def __init__(
        self,
        data_dir: str | Path,
        target_size: int = 32,
        num_colors: int = 16,
        apply_degradation: bool = True,
        only_transparent: bool = True,
        cache_dir: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.num_colors = num_colors
        self.apply_degradation = apply_degradation

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self.image_paths = sorted(
            list(self.data_dir.glob("**/*.png"))
            + list(self.data_dir.glob("**/*.bmp"))
            + list(self.data_dir.glob("**/*.webp"))
        )
        if only_transparent:
            self.image_paths = [p for p in self.image_paths if not p.name.startswith("fe_")]
        if not self.image_paths:
            raise ValueError(f"No image files found in {data_dir}")

        # Default cache lives alongside the dataset
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir.parent / f".palette_cache_{num_colors}c"
        _build_palette_cache(self.image_paths, self.cache_dir, target_size, num_colors)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        cache_file = self.cache_dir / f"{img_path.stem}.npz"
        data = np.load(cache_file)
        img_rgba = data["rgba"]
        palette_np = data["palette"]
        target_idx_np = data["target"]

        if self.apply_degradation:
            from spriteforge.core.degrade import degrade, DegradeRanges
            mode = np.random.choice(["standard", "realistic", "color_shift"], p=[0.45, 0.35, 0.2])
            if mode == "realistic":
                ranges = DegradeRanges.realistic_low_noise()
            elif mode == "color_shift":
                ranges = DegradeRanges.color_shift_only()
            else:
                ranges = DegradeRanges.standard()
            degraded_rgba = degrade(img_rgba, ranges=ranges, multiscale=True)
            input_tensor = torch.from_numpy(degraded_rgba).permute(2, 0, 1).contiguous()
        else:
            input_tensor = torch.from_numpy(img_rgba).permute(2, 0, 1).contiguous()

        palette_tensor = torch.from_numpy(palette_np).contiguous()
        target_idx_tensor = torch.from_numpy(target_idx_np).long().contiguous()

        return input_tensor, target_idx_tensor, palette_tensor


def train_palette_model(
    data_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 2e-4,
    num_colors: int = 16,
    hidden_channels: int = 64,
    device: str = "auto",
    output_dir: str | Path = "checkpoints_palette",
    save_interval: int = 10,
    num_workers: int = 0,
    cache_dir: str | Path | None = None,
) -> PaletteUNet:
    """Train the E1 palette-index classification model. No adversarial loss —
    per the methodology review, discrete classification doesn't need a GAN to
    avoid the speckle failure mode; cross-entropy + argmax is discrete by
    construction (see tests/test_palette_unet.py's discreteness guarantee)."""
    device_str = get_device(device)
    print(f"[*] Initializing palette-UNet training on device: {device_str.upper()}")

    dataset = PaletteDataset(data_dir, num_colors=num_colors, cache_dir=cache_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    config = PaletteUNetConfig(num_colors=num_colors, hidden_channels=hidden_channels)
    model = PaletteUNet(config).to(device_str)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "training_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "loss_ce", "pixel_accuracy"])

    print(f"[*] Starting training loop for {epochs} epochs (dataset size: {len(dataset)})...")
    start_time = time.time()

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_pixels = 0

            for inputs, targets, palettes in loader:
                inputs = inputs.to(device_str)
                targets = targets.to(device_str)
                palettes = palettes.to(device_str)

                opt.zero_grad(set_to_none=True)
                logits = model(inputs, palettes)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                opt.step()

                total_loss += loss.item() * inputs.shape[0]
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == targets).sum().item()
                total_pixels += targets.numel()

            scheduler.step()
            epoch_loss = total_loss / len(dataset)
            epoch_acc = total_correct / max(1, total_pixels)
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch:03d}/{epochs}] ({elapsed:.1f}s) | Loss CE: {epoch_loss:.4f} | Pixel Acc: {epoch_acc:.4f}")
            csv_writer.writerow([epoch, round(epoch_loss, 6), round(epoch_acc, 6)])
            csv_file.flush()

            if epoch % save_interval == 0 or epoch == epochs:
                ckpt_path = out_path / f"palette_unet_epoch_{epoch:03d}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "num_colors": num_colors,
                    "hidden_channels": hidden_channels,
                    "epoch": epoch,
                }, ckpt_path)
                print(f"    [+] Saved checkpoint: {ckpt_path}")
    finally:
        csv_file.close()

    return model
