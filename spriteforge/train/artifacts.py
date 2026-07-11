"""
Devlog artifact helpers (Phase 5): periodic training sample grids and before/after slider
assets for the Orbitope content pipeline. Pure helpers — no global state.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from spriteforge.core.io import save_image_float32


def _nn_upscale(arr_rgba: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor integer upscale of a float32 HxWx C array (blocky, no interpolation)."""
    return np.repeat(np.repeat(arr_rgba, scale, axis=0), scale, axis=1)


def save_sample_grid(
    model: torch.nn.Module,
    dataset,
    epoch: int,
    out_dir: str | Path,
    device: str,
    num_samples: int = 16,
    columns: int = 4,
) -> Path:
    """Run inference on evenly-spaced dataset items and save an input/pred/target grid PNG.

    For each row of `columns` samples we stack three sub-rows: degraded input, model
    prediction, ground-truth target (same review-grid style as evaluate.py). Returns the
    written path. Restores the model's train/eval mode on exit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(dataset)
    n = min(num_samples, n_total)
    idxs = np.linspace(0, n_total - 1, n, dtype=int)

    was_training = model.training
    model.eval()
    inputs, preds, targets = [], [], []
    with torch.no_grad():
        for idx in idxs:
            inp, tgt = dataset[int(idx)]
            pred, _, _ = model(inp.unsqueeze(0).to(device))
            inputs.append(inp.numpy())
            preds.append(pred[0].cpu().numpy())
            targets.append(tgt.numpy())
    if was_training:
        model.train()

    blocks = []
    num_rows = (n + columns - 1) // columns
    for r in range(num_rows):
        sub_in = inputs[r * columns:(r + 1) * columns]
        sub_pr = preds[r * columns:(r + 1) * columns]
        sub_tg = targets[r * columns:(r + 1) * columns]
        while len(sub_in) < columns:
            sub_in.append(np.zeros_like(inputs[0]))
            sub_pr.append(np.zeros_like(preds[0]))
            sub_tg.append(np.zeros_like(targets[0]))
        in_row = np.concatenate([sub_in[i].transpose(1, 2, 0) for i in range(columns)], axis=1)
        pr_row = np.concatenate([sub_pr[i].transpose(1, 2, 0) for i in range(columns)], axis=1)
        tg_row = np.concatenate([sub_tg[i].transpose(1, 2, 0) for i in range(columns)], axis=1)
        blocks.append(np.concatenate([in_row, pr_row, tg_row], axis=0))

    grid = np.clip(np.concatenate(blocks, axis=0), 0.0, 1.0).astype(np.float32)
    out_path = out_dir / f"sample_epoch_{epoch:03d}.png"
    save_image_float32(grid, out_path)
    return out_path


def export_before_after_sliders(
    checkpoint_path: str | Path,
    input_paths: list[str | Path],
    out_dir: str | Path,
    device: str = "cpu",
    upscale: int = 8,
) -> list[Path]:
    """Emit paired nearest-neighbor-upscaled before/after PNGs from a checkpoint, ready for
    article embedding. `before` = model-resolution degraded input; `after` = restored output.
    """
    from spriteforge.model.config import get_config
    from spriteforge.model.vqgan import SpriteVQGAN
    from spriteforge.core.io import load_image_float32
    from spriteforge.core.resize import resize_to_target

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = get_config(ckpt.get("config_name", "size_32"))
    model = SpriteVQGAN(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for p in input_paths:
        p = Path(p)
        img = load_image_float32(p)
        resized = resize_to_target(img, target_size=config.target_size)
        with torch.no_grad():
            tensor_in = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device)
            out, _, _ = model(tensor_in)
        restored = out[0].permute(1, 2, 0).cpu().numpy()

        b_path = out_dir / f"{p.stem}_before.png"
        a_path = out_dir / f"{p.stem}_after.png"
        save_image_float32(_nn_upscale(resized, upscale), b_path)
        save_image_float32(_nn_upscale(restored, upscale), a_path)
        written.extend([b_path, a_path])
    return written
