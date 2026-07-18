# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Head-to-head on the real eval set for the E1 palette-index model (discrete,
speckle-free by construction) against the deterministic baseline that beat every
VQ-GAN variant.

Pipeline per image (E1 columns): real input -> flood matte -> downscale 32 ->
palette-UNet (argmax over palette classes). No post-hoc snap: the model output is
already discrete.

Two E1 columns exercise the resolved palette-conditioning decision:
  * E1 (kmeans src) — palette extracted from each matted sprite (default mode).
  * E1 (fixed pal)  — one global palette shared across all sprites.

Grid columns: real->32 (bg) | determ matte+snap | E1 (kmeans src) | E1 (fixed pal)

Usage:
    .venv/bin/python3 scripts/eval_e1.py [--ckpt <path>]
    .venv/bin/python3 scripts/eval_e1.py --ckpt checkpoints_palette_e1_sr_k24/palette_unet_epoch_100.pt
"""
from __future__ import annotations
import sys, glob, argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spriteforge.model.config import get_config
from spriteforge.core.resize import resize_to_target
from spriteforge.core.palette import (
    remove_background_flood, extract_palette_kmeans, nearest_neighbor_snap,
)
from spriteforge.model.palette_infer import load_palette_unet, restore_sprite

# Default to K=16 checkpoint; override with --ckpt
DEFAULT_E1_CKPT = ROOT / "checkpoints_palette_e1_sr" / "palette_unet_epoch_100.pt"
REAL_DIR = ROOT / "data_private" / "real_eval_raw"

N = 8          # images to show
SCALE = 7      # nearest-neighbor upscale for viewing
SNAP_K = 12    # deterministic-column palette size


def load_rgba(f: str) -> np.ndarray:
    return np.array(Image.open(f).convert("RGBA"), dtype=np.float32) / 255.0


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def up(rgb: np.ndarray, s: int = SCALE) -> np.ndarray:
    return cv2.resize(to_u8(rgb), (rgb.shape[1] * s, rgb.shape[0] * s), interpolation=cv2.INTER_NEAREST)


def checker(rgba: np.ndarray) -> np.ndarray:
    h, w = rgba.shape[:2]
    c = 8
    yy, xx = np.mgrid[0:h, 0:w]
    pat = (((yy // c + xx // c) % 2) * 0.2 + 0.6).astype(np.float32)[..., None]
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + pat * (1 - a)


def snap(rgba: np.ndarray, k: int = SNAP_K) -> np.ndarray:
    pal = extract_palette_kmeans(rgba, k=k)
    return nearest_neighbor_snap(rgba, pal)


def main() -> int:
    parser = argparse.ArgumentParser(description="E1 palette-index model evaluation")
    parser.add_argument("--ckpt", type=str, default=None, help="E1 checkpoint path (default: K=16)")
    args = parser.parse_args()

    e1_ckpt = Path(args.ckpt) if args.ckpt else DEFAULT_E1_CKPT
    if not e1_ckpt.exists():
        print(f"[!] E1 checkpoint not found: {e1_ckpt}")
        return 1
    model, num_colors = load_palette_unet(e1_ckpt, device="cpu")
    print(f"[+] loaded E1 palette-UNet (K={num_colors}): {e1_ckpt}")

    # Output path based on checkpoint location
    out_dir = e1_ckpt.parent
    out = out_dir / "e1_vs_determ.png"

    ts = get_config("32").target_size
    all_files = sorted(glob.glob(str(REAL_DIR / "*.png")))

    def sharpness(f: str) -> float:
        rgb = resize_to_target(load_rgba(f), target_size=ts)[..., :3]
        g = cv2.cvtColor(to_u8(rgb), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    ranked = sorted(all_files, key=sharpness)
    idx = np.linspace(0, len(ranked) - 1, N).round().astype(int)
    files = [ranked[i] for i in idx]  # evenly spaced mushy -> crisp

    # Matted, target-size inputs for every selected sprite (E1's expected input).
    matted = [resize_to_target(remove_background_flood(load_rgba(f)), target_size=ts) for f in files]

    # Fixed palette: one global K-color palette from all matted sprites' opaque
    # pixels, reused for every image (contrast with per-source extraction).
    opaque_px = np.concatenate([m[..., :3][m[..., 3] >= 0.5] for m in matted], axis=0)
    fixed_pal = extract_palette_kmeans(
        np.concatenate([opaque_px, np.ones((opaque_px.shape[0], 1), np.float32)], axis=1)[None],
        k=num_colors,
    )

    tile = ts * SCALE
    pad = 8
    labelh = 24
    cols = ["real->32 (bg)", "determ. matte+snap", "E1 (kmeans src)", "E1 (fixed pal)"]
    rows = []
    for f, m in zip(files, matted):
        im = load_rgba(f)
        raw32 = resize_to_target(im.copy(), target_size=ts)
        determ = resize_to_target(snap(remove_background_flood(im.copy())), target_size=ts)
        e1_src = restore_sprite(model, num_colors, m, palette=None)        # k-means from source
        e1_fix = restore_sprite(model, num_colors, m, palette=fixed_pal)   # shared fixed palette
        cells = [up(checker(raw32)), up(checker(determ)), up(checker(e1_src)), up(checker(e1_fix))]
        rows.append(np.hstack([np.pad(c, ((0, 0), (0, pad), (0, 0)), constant_values=245) for c in cells]))

    grid = np.vstack([np.pad(r, ((0, pad), (0, 0), (0, 0)), constant_values=245) for r in rows])
    hdr = np.full((labelh, grid.shape[1], 3), 245, np.uint8)
    x = 0
    for c in cols:
        cv2.putText(hdr, c, (x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        x += tile + pad
    grid = np.vstack([hdr, grid])
    cv2.imwrite(str(out), grid[..., ::-1])
    print(f"[+] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
