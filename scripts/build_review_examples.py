# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Generates visual review assets for devlog/TRAINING_REVIEW.md: per-source grids of
randomly selected reconstructions (degraded input / final-checkpoint output /
best-recon-checkpoint output / ground truth), plus a worst-case "failures" grid per
source (highest per-image L1 error against the final checkpoint). Pure image output —
no charts, meant for fast eyeball review.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from spriteforge.core.palette import extract_palette_kmeans, load_palette, nearest_neighbor_snap
from spriteforge.model.config import get_config
from spriteforge.model.vqgan import SpriteVQGAN
from spriteforge.train.train import SpriteDataset

ROOT = Path(__file__).resolve().parent.parent
BYSOURCE = ROOT / "checkpoints_bysource"
DEVLOG = ROOT / "devlog"
SOURCES = ["pmd", "papi", "lpc", "fe"]

TEST_DIRS = {
    "pmd": "data_private/test_pmd_32_clean",
    "papi": "data_private/test_papi_32",
    "lpc": "data_private/test_lpc_32",
    "fe": "data_private/test_fe_32_clean",
}

CELL = 32
SCALE = 6           # nearest-neighbor upscale factor per sprite cell
LABEL_H = 16         # header row height in px
GAP = 3              # gap between cells, in px


def load_model(ckpt_path: Path, device: str = "cpu") -> SpriteVQGAN:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = get_config(ckpt.get("config_name", "size_32"))
    model = SpriteVQGAN(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def nn_upscale_img(arr_rgba: np.ndarray, scale: int) -> Image.Image:
    u8 = to_u8(arr_rgba)
    img = Image.fromarray(u8, mode="RGBA")
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def build_grid(rows: list[list[np.ndarray]], col_labels: list[str], row_labels: list[str] | None = None) -> Image.Image:
    """rows: list of rows, each a list of RGBA (32,32,4) arrays. Renders a labeled grid."""
    n_cols = len(col_labels)
    cell_px = CELL * SCALE
    row_label_w = 90 if row_labels else 0
    grid_w = row_label_w + n_cols * cell_px + (n_cols - 1) * GAP
    grid_h = LABEL_H + len(rows) * cell_px + (len(rows) - 1) * GAP

    canvas = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()

    for c, label in enumerate(col_labels):
        x = row_label_w + c * (cell_px + GAP)
        draw.text((x + 4, 2), label, fill=(20, 20, 20, 255), font=font)

    for r, row in enumerate(rows):
        y = LABEL_H + r * (cell_px + GAP)
        if row_labels:
            draw.text((4, y + cell_px // 2 - 6), row_labels[r], fill=(20, 20, 20, 255), font=font)
        for c, arr in enumerate(row):
            x = row_label_w + c * (cell_px + GAP)
            tile = nn_upscale_img(arr, SCALE)
            canvas.paste(tile, (x, y), tile)
    return canvas


def process_source(
    source: str,
    num_random: int = 5,
    num_failures: int = 3,
    seed: int | None = None,
    fixed_palette: np.ndarray | None = None,
    palette_k: int = 6,
) -> None:
    """
    fixed_palette: if given (K, 3) RGB array, this is the real production case — a
        pre-defined palette (e.g. loaded via spriteforge.core.palette.load_palette) that
        every restored output gets snapped to, no per-image inference. If None, falls back
        to auto-inferring a per-image palette via k-means (exploratory/diagnostic only —
        not how the shipped pipeline would actually run).
    palette_k: cluster count for the auto-inferred comparison columns.
    """
    out_dir = BYSOURCE / source
    final_ckpt = out_dir / "vqgan_32_epoch_100.pt"
    best_ckpt = out_dir / "vqgan_32_best_recon.pt"
    if not final_ckpt.exists():
        print(f"[!] Skipping {source}: no final checkpoint found")
        return

    model_final = load_model(final_ckpt)
    model_best = load_model(best_ckpt) if best_ckpt.exists() else None

    dataset = SpriteDataset(TEST_DIRS[source], target_size=32, apply_degradation=True)
    n = len(dataset)

    rng = random.Random(seed)
    rand_idxs = rng.sample(range(n), min(num_random, n))

    # --- Random review grid ---
    rows = []
    for idx in rand_idxs:
        inp, tgt = dataset[idx]
        with torch.no_grad():
            pred_final, _, _ = model_final(inp.unsqueeze(0))
            row = [inp.numpy().transpose(1, 2, 0), pred_final[0].numpy().transpose(1, 2, 0)]
            if model_best is not None:
                pred_best, _, _ = model_best(inp.unsqueeze(0))
                row.append(pred_best[0].numpy().transpose(1, 2, 0))
            row.append(tgt.numpy().transpose(1, 2, 0))
        rows.append(row)

    col_labels = ["Input (degraded)", "Restored (final)"]
    if model_best is not None:
        col_labels.append("Restored (best-recon)")
    col_labels.append("Ground truth")

    grid = build_grid(rows, col_labels)
    review_dir = out_dir / "review_examples"
    review_dir.mkdir(parents=True, exist_ok=True)
    grid_path = review_dir / "random_examples.png"
    grid.save(grid_path)
    print(f"[+] {source}: saved random review grid -> {grid_path}")

    # --- Palette-snap post-processing grid, on the same random examples ---
    # If a fixed_palette was supplied, that's the actual production case: every restored
    # output snaps to that single pre-defined palette. Auto-inferred (k-means) variants are
    # kept alongside as a diagnostic comparison, not as the intended real usage.
    palette_rows = []
    for idx in rand_idxs:
        inp, tgt = dataset[idx]
        inp_hwc = inp.numpy().transpose(1, 2, 0)
        tgt_hwc = tgt.numpy().transpose(1, 2, 0)
        with torch.no_grad():
            pred_final, _, _ = model_final(inp.unsqueeze(0))
        pred_hwc = pred_final[0].numpy().transpose(1, 2, 0)

        row = [inp_hwc, pred_hwc]
        if fixed_palette is not None:
            row.append(nearest_neighbor_snap(pred_hwc, fixed_palette, alpha_threshold=0.5))

        pal_input = extract_palette_kmeans(inp_hwc, k=palette_k, alpha_threshold=0.5)
        row.append(nearest_neighbor_snap(pred_hwc, pal_input, alpha_threshold=0.5))

        pal_gt = extract_palette_kmeans(tgt_hwc, k=palette_k, alpha_threshold=0.5)
        row.append(nearest_neighbor_snap(pred_hwc, pal_gt, alpha_threshold=0.5))

        row.append(tgt_hwc)
        palette_rows.append(row)

    col_labels = ["Input (degraded)", "Restored (final)"]
    if fixed_palette is not None:
        col_labels.append(f"Snap (fixed palette, {len(fixed_palette)} colors)")
    col_labels += [f"Snap k={palette_k} (auto, from input)", f"Snap k={palette_k} (auto, from GT)", "Ground truth"]

    palette_grid = build_grid(palette_rows, col_labels)
    palette_grid_path = review_dir / "palette_snap_examples.png"
    palette_grid.save(palette_grid_path)
    print(f"[+] {source}: saved palette-snap grid -> {palette_grid_path}")

    # --- Worst-case (failure) grid + palette-snap improvement metrics, over the full eval set ---
    losses = []
    raw_l1_total = 0.0
    auto_snap_l1_total = 0.0
    fixed_snap_l1_total = 0.0
    with torch.no_grad():
        for idx in range(n):
            inp, tgt = dataset[idx]
            pred, _, _ = model_final(inp.unsqueeze(0))
            l1 = F.l1_loss(pred[0], tgt).item()
            losses.append((l1, idx))
            raw_l1_total += l1

            inp_hwc = inp.numpy().transpose(1, 2, 0)
            pred_hwc = pred[0].numpy().transpose(1, 2, 0)
            tgt_hwc = tgt.numpy().transpose(1, 2, 0)

            pal_input = extract_palette_kmeans(inp_hwc, k=palette_k, alpha_threshold=0.5)
            snapped = nearest_neighbor_snap(pred_hwc, pal_input, alpha_threshold=0.5)
            auto_snap_l1_total += float(np.abs(snapped - tgt_hwc).mean())

            if fixed_palette is not None:
                snapped_fixed = nearest_neighbor_snap(pred_hwc, fixed_palette, alpha_threshold=0.5)
                fixed_snap_l1_total += float(np.abs(snapped_fixed - tgt_hwc).mean())

    metrics_path = out_dir / "palette_snap_metrics.json"
    metrics_out = {
        "raw_l1_mean": round(raw_l1_total / n, 4),
        "auto_palette_snap_l1_mean": round(auto_snap_l1_total / n, 4),
        "palette_k": palette_k,
        "num_eval_samples": n,
    }
    if fixed_palette is not None:
        metrics_out["fixed_palette_snap_l1_mean"] = round(fixed_snap_l1_total / n, 4)
        metrics_out["fixed_palette_size"] = len(fixed_palette)
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"[+] {source}: raw L1={raw_l1_total/n:.4f} -> auto-snap L1={auto_snap_l1_total/n:.4f}"
          + (f" -> fixed-snap L1={fixed_snap_l1_total/n:.4f}" if fixed_palette is not None else ""))

    losses.sort(reverse=True)
    worst = losses[:num_failures]

    fail_rows = []
    fail_labels = []
    with torch.no_grad():
        for l1, idx in worst:
            inp, tgt = dataset[idx]
            pred, _, _ = model_final(inp.unsqueeze(0))
            fail_rows.append([
                inp.numpy().transpose(1, 2, 0),
                pred[0].numpy().transpose(1, 2, 0),
                tgt.numpy().transpose(1, 2, 0),
            ])
            fail_labels.append(f"L1={l1:.3f}")

    fail_grid = build_grid(fail_rows, ["Input (degraded)", "Restored (final)", "Ground truth"], row_labels=fail_labels)
    DEVLOG.mkdir(exist_ok=True)
    failures_dir = DEVLOG / "failures"
    failures_dir.mkdir(exist_ok=True)
    fail_path = failures_dir / f"{source}_worst_{num_failures}.png"
    fail_grid.save(fail_path)
    print(f"[+] {source}: saved failure grid ({num_failures} worst) -> {fail_path}")


def build_overview_grid(seed: int | None = None) -> None:
    """One random example per source (input / restored / target), for a fast cross-source scan."""
    rows = []
    row_labels = []
    for source in SOURCES:
        final_ckpt = BYSOURCE / source / "vqgan_32_epoch_100.pt"
        if not final_ckpt.exists():
            continue
        model = load_model(final_ckpt)
        dataset = SpriteDataset(TEST_DIRS[source], target_size=32, apply_degradation=True)
        idx = random.Random(seed).randrange(len(dataset))
        inp, tgt = dataset[idx]
        with torch.no_grad():
            pred, _, _ = model(inp.unsqueeze(0))
        rows.append([
            inp.numpy().transpose(1, 2, 0),
            pred[0].numpy().transpose(1, 2, 0),
            tgt.numpy().transpose(1, 2, 0),
        ])
        row_labels.append(source)

    grid = build_grid(rows, ["Input (degraded)", "Restored (final)", "Ground truth"], row_labels=row_labels)
    DEVLOG.mkdir(exist_ok=True)
    out_path = DEVLOG / "overview_grid.png"
    grid.save(out_path)
    print(f"[+] saved cross-source overview grid -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate visual review assets for devlog/TRAINING_REVIEW.md")
    parser.add_argument(
        "--palette-file", type=str, default=None,
        help="Path to a pre-defined palette (JSON hex list, e.g. spriteforge/data/palettes/pico8.json). "
             "If given, restored outputs snap to this fixed palette (the real production case) "
             "instead of only auto-inferring a palette per image."
    )
    parser.add_argument("--palette-k", type=int, default=6, help="Cluster count for the auto-inferred comparison columns")
    parser.add_argument("--num-random", type=int, default=5, help="Number of random review examples per source")
    args = parser.parse_args()

    fixed_palette = load_palette(args.palette_file) if args.palette_file else None
    if fixed_palette is not None:
        print(f"[*] Using fixed palette from {args.palette_file} ({len(fixed_palette)} colors)")

    for source in SOURCES:
        process_source(source, num_random=args.num_random, fixed_palette=fixed_palette, palette_k=args.palette_k)
    build_overview_grid()


if __name__ == "__main__":
    main()
