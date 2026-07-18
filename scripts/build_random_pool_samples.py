# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Builds a collective document showing genuinely random samples from each source's FULL
training pool (not the small curated held-out test set) — an honest, unbiased visual read
on model quality, run per source as each v2 checkpoint becomes available.

Deliberately does NOT filter for "recognizable" content (unlike scripts/build_showcase.py) —
the whole point here is an unbiased sample, warts and all, not a curated best-case demo.
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

# (source_key, checkpoint_path, full_training_pool_dir)
SOURCE_CONFIGS = {
    "papi": ("checkpoints_bysource_v2/papi/vqgan_32_epoch_100.pt", "data_private/train_32_papi"),
    "lpc": ("checkpoints_bysource_v2/lpc/vqgan_32_epoch_100.pt", "data_private/train_32_lpc"),
    "fe": ("checkpoints_bysource_v2/fe/vqgan_32_epoch_100.pt", "data_private/train_32_fe"),
    "oga": ("checkpoints_bysource_v2/oga/vqgan_32_epoch_100.pt", "data_private/train_32_oga"),
}

SCALE = 8
CELL = 32
LABEL_H = 20
GAP = 4


def load_model(ckpt_path: Path) -> SpriteVQGAN:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = get_config(ckpt.get("config_name", "size_32"))
    model = SpriteVQGAN(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


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
        font = ImageFont.load_default(size=14)
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


def build_source_sample(source: str, num_samples: int = 12, seed: int = 99) -> Path | None:
    if source not in SOURCE_CONFIGS:
        raise ValueError(f"unknown source: {source}")
    ckpt_rel, pool_rel = SOURCE_CONFIGS[source]
    ckpt_path = ROOT / ckpt_rel
    if not ckpt_path.exists():
        print(f"[!] {source}: no checkpoint yet at {ckpt_path}, skipping")
        return None

    model = load_model(ckpt_path)
    dataset = SpriteDataset(pool_rel, target_size=32, apply_degradation=True)
    n = len(dataset)
    rng = random.Random(seed)
    idxs = rng.sample(range(n), min(num_samples, n))

    rows = []
    with torch.no_grad():
        for idx in idxs:
            inp, tgt = dataset[idx]
            pred, _, _ = model(inp.unsqueeze(0))
            rows.append([
                inp.numpy().transpose(1, 2, 0),
                pred[0].numpy().transpose(1, 2, 0),
                tgt.numpy().transpose(1, 2, 0),
            ])

    grid = build_grid(rows, ["Input (degraded)", "Restored (unfiltered random sample)", "Ground truth"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{source}_random_pool.png"
    grid.save(out_path)
    print(f"[+] {source}: saved random-pool sample ({len(rows)} images from a {n}-image pool) -> {out_path}")
    return out_path


def main():
    for source in SOURCE_CONFIGS:
        build_source_sample(source)


if __name__ == "__main__":
    main()
