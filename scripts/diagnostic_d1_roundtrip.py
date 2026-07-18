# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
D1 diagnostic (see plan: methodology review, Part 4 Stage 1): clean round-trip test.

Runs CLEAN sprites (no degradation at all — apply_degradation=False) through
encode->quantize->decode. If speckle appears even here, the corruption isn't about inverting
degradation — the autoencoder/decoder itself can't render high-frequency pixel art cleanly,
which means MaskGIT (Method 6) would inherit the same speckle (it only replaces the token
*prior*, not the frozen decoder that renders tokens to pixels).
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
    "papi": ("checkpoints_bysource_v2/papi/vqgan_32_epoch_100.pt", "data_private/train_32_papi"),
    "lpc": ("checkpoints_bysource_v2/lpc/vqgan_32_epoch_100.pt", "data_private/train_32_lpc"),
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


def flat_region_variance(pred: np.ndarray, target: np.ndarray, flat_thresh: float = 0.01) -> float:
    """Quick-and-dirty speckle proxy: for each 3x3 GT neighborhood that's nearly flat (low
    local variance), measure how much the prediction deviates locally. Full D3 metric to
    follow; this is just enough to put a number next to the round-trip visuals."""
    h, w = target.shape[:2]
    gt_rgb = target[..., :3]
    pred_rgb = pred[..., :3]
    devs = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gt_patch = gt_rgb[y-1:y+2, x-1:x+2]
            if gt_patch.var() < flat_thresh:
                pred_patch = pred_rgb[y-1:y+2, x-1:x+2]
                devs.append(pred_patch.var())
    return float(np.mean(devs)) if devs else 0.0


def run_roundtrip(source: str, num_samples: int = 10, seed: int = 42) -> Path | None:
    ckpt_rel, pool_rel = SOURCE_CONFIGS[source]
    ckpt_path = ROOT / ckpt_rel
    if not ckpt_path.exists():
        print(f"[!] {source}: no checkpoint at {ckpt_path}")
        return None

    model = load_model(ckpt_path)
    # apply_degradation=False -> input_tensor is a clone of target_tensor (pure clean sprite)
    dataset = SpriteDataset(pool_rel, target_size=32, apply_degradation=False)
    n = len(dataset)
    rng = random.Random(seed)
    idxs = rng.sample(range(n), min(num_samples, n))

    rows = []
    speckle_scores = []
    with torch.no_grad():
        for idx in idxs:
            clean, target = dataset[idx]
            assert torch.allclose(clean, target), "apply_degradation=False should yield input==target"
            pred, _, _ = model(clean.unsqueeze(0))
            pred_np = pred[0].numpy().transpose(1, 2, 0)
            target_np = target.numpy().transpose(1, 2, 0)
            rows.append([clean.numpy().transpose(1, 2, 0), pred_np, target_np])
            speckle_scores.append(flat_region_variance(pred_np, target_np))

    grid = build_grid(rows, ["Clean input (no degradation)", "Round-trip output", "Ground truth (== input)"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{source}_d1_roundtrip.png"
    grid.save(out_path)
    avg_speckle = float(np.mean(speckle_scores))
    print(f"[+] {source}: round-trip grid -> {out_path}")
    print(f"    flat-region variance proxy (higher = more speckle in flat GT areas): {avg_speckle:.5f}")
    print(f"    per-sample: {[round(s, 5) for s in speckle_scores]}")
    return out_path


def main():
    for source in SOURCE_CONFIGS:
        run_roundtrip(source)


if __name__ == "__main__":
    main()
