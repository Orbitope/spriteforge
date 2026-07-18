# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
D2 diagnostic (plan: methodology review, Part 4 Stage 1): GAN ablation read.

Compares vqgan_32_best_recon.pt (lowest-recon-loss checkpoint, captured early in training
before the adversarial ramp reaches full weight — see disc_start_epoch/adv_ramp_epochs in
train.py) against the final vqgan_32_epoch_100.pt, on the SAME random samples and the SAME
(degraded-input) task. If best_recon is visibly cleaner, the adversarial term is a primary
speckle driver and taming it (lower adv_weight, adaptive lambda, or dropping the GAN
entirely) is high-value. If both speckle similarly, the codebook/decoder capacity is the
bottleneck regardless of GAN weight.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from spriteforge.model.config import get_config
from spriteforge.model.vqgan import SpriteVQGAN
from spriteforge.train.train import SpriteDataset

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "devlog" / "random_pool_samples"

SOURCE_CONFIGS = {
    "papi": ("checkpoints_bysource_v2/papi", "data_private/train_32_papi"),
    "lpc": ("checkpoints_bysource_v2/lpc", "data_private/train_32_lpc"),
}

SCALE = 8
CELL = 32
LABEL_H = 20
GAP = 4


def load_model(ckpt_path: Path) -> tuple[SpriteVQGAN, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = get_config(ckpt.get("config_name", "size_32"))
    model = SpriteVQGAN(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("epoch", -1)


def up(arr: np.ndarray, scale: int = SCALE) -> Image.Image:
    u8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="RGBA")
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def build_grid(rows: list[list[np.ndarray]], col_labels: list[str]) -> Image.Image:
    cell_px = CELL * SCALE
    n_cols = len(col_labels)
    grid_w = n_cols * cell_px + (n_cols - 1) * GAP
    grid_h = LABEL_H + len(rows) * cell_px + (len(rows) - 1) * GAP
    canvas = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    for c, label in enumerate(col_labels):
        draw.text((c * (cell_px + GAP) + 4, 2), label, fill=(20, 20, 20, 255), font=font)
    for r, row in enumerate(rows):
        y = LABEL_H + r * (cell_px + GAP)
        for c, arr in enumerate(row):
            x = c * (cell_px + GAP)
            tile = up(arr)
            canvas.paste(tile, (x, y), tile)
    return canvas


def run_ablation(source: str, num_samples: int = 10, seed: int = 99) -> Path | None:
    out_dir_rel, pool_rel = SOURCE_CONFIGS[source]
    best_ckpt = ROOT / out_dir_rel / "vqgan_32_best_recon.pt"
    final_ckpt = ROOT / out_dir_rel / "vqgan_32_epoch_100.pt"
    if not (best_ckpt.exists() and final_ckpt.exists()):
        print(f"[!] {source}: missing checkpoint(s)")
        return None

    model_best, epoch_best = load_model(best_ckpt)
    model_final, epoch_final = load_model(final_ckpt)

    # Same seed as build_random_pool_samples.py so this is directly comparable to that doc.
    dataset = SpriteDataset(pool_rel, target_size=32, apply_degradation=True)
    n = len(dataset)
    rng = random.Random(seed)
    idxs = rng.sample(range(n), min(num_samples, n))

    rows = []
    with torch.no_grad():
        for idx in idxs:
            inp, tgt = dataset[idx]
            pred_best, _, _ = model_best(inp.unsqueeze(0))
            pred_final, _, _ = model_final(inp.unsqueeze(0))
            rows.append([
                inp.numpy().transpose(1, 2, 0),
                pred_best[0].numpy().transpose(1, 2, 0),
                pred_final[0].numpy().transpose(1, 2, 0),
                tgt.numpy().transpose(1, 2, 0),
            ])

    grid = build_grid(rows, [
        "Input (degraded)",
        f"best_recon (epoch {epoch_best}, pre-full-GAN)",
        f"epoch_100 (final, full-GAN)",
        "Ground truth",
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{source}_d2_gan_ablation.png"
    grid.save(out_path)
    print(f"[+] {source}: GAN ablation grid (best_recon=epoch {epoch_best} vs final=epoch {epoch_final}) -> {out_path}")
    return out_path


def main():
    for source in SOURCE_CONFIGS:
        run_ablation(source)


if __name__ == "__main__":
    main()
