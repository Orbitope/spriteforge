"""
Generates SHOWCASE.md — a friendly, non-technical summary with large visual examples,
meant to be shared with friends/reviewers who don't need training curves or loss numbers.
Reuses the same models/eval sets as scripts/build_review_examples.py but renders bigger,
simpler grids with plain-English labels.

Unlike the technical review grids (which sample from SpriteDataset's mixed degradation
strengths, some deliberately mild), this always applies the strongest degradation preset
so "before" clearly looks messed up — and filters out background-only/sliver crops that
don't read as a recognizable sprite, so examples don't look like random noise.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.build_review_examples import SOURCES, load_model, nn_upscale_img
from spriteforge.core.degrade import DegradeRanges, degrade
from spriteforge.core.io import load_image_float32
from spriteforge.core.palette import extract_palette_kmeans, nearest_neighbor_snap
from spriteforge.core.resize import pad_to_target

ROOT = Path(__file__).resolve().parent.parent
BYSOURCE = ROOT / "checkpoints_bysource"
SHOWCASE_DIR = ROOT / "showcase"

SOURCE_BLURBS = {
    "pmd": ("Pokémon-style creature icons", "PMDCollab"),
    "papi": ("Pokémon overworld/battle sprites", "PokéAPI"),
    "lpc": ("Fantasy RPG character gear", "Universal LPC"),
    "fe": ("Tactics-game battle sprites", "Fire Emblem GBA"),
}

# For the visual showcase (unlike the technical review), pull candidates from the much
# larger training pools rather than the 20-image held-out eval sets — more room for the
# content filter below to find genuinely recognizable examples. Not a held-out-eval concern
# here since this is a demo, not a metric.
SHOWCASE_SOURCE_DIRS = {
    "pmd": "data_private/train_32_pmd",
    "papi": "data_private/train_32_papi",
    "lpc": "data_private/train_32_lpc",
    "fe": "data_private/train_32_fe",
}

CELL = 32
SCALE = 11           # bigger than the technical review grids — this is for reading, not auditing
LABEL_H = 24
GAP = 6

# Content filter thresholds for picking recognizable examples (not background-only slivers,
# flat single-color blobs, smoothly-shaded gradient crops, or texture/terrain close-ups).
MIN_ALPHA_COVERAGE = 0.30   # fraction of pixels that must be opaque (>0.5 alpha)
MIN_BBOX_FILL = 0.35        # opaque-pixel bounding box must fill at least this much of itself
MIN_DISTINCT_COLORS = 12    # opaque region must have at least this many distinct quantized colors
MIN_OUTLINE_FRACTION = 0.04  # fraction of opaque pixels that must be near-black (sprite outline)
MAX_BLEEDING_EDGES = 1       # reject if opaque content bleeds off more than this many of the 4 frame edges
MIN_LUMINANCE_STD = 0.09     # reject visually-flat/murky crops with little shape-defining contrast
MIN_DIVERSITY_DIST = 0.20    # min mean-abs-color distance from already-picked examples in a grid


# papi's training data is dominated by "papi_ove_*" (overworld animation-frame crops, which
# are frequently zoomed sub-crops of a larger sheet rather than a whole character) vs. far
# rarer "papi_ico_*" (standalone item/creature icons — whole, centered, showcase-friendly).
# Search icons first so the filter below finds recognizable whole subjects, not action-frame
# fragments.
SHOWCASE_PREFIX_PRIORITY = {
    "papi": ["papi_ico_"],
}


def load_clean_sprites(source: str) -> list[Path]:
    d = Path(SHOWCASE_SOURCE_DIRS[source])
    files = sorted(d.glob("*.png"))
    priority_prefixes = SHOWCASE_PREFIX_PRIORITY.get(source)
    if priority_prefixes:
        preferred = [f for f in files if any(f.name.startswith(p) for p in priority_prefixes)]
        rest = [f for f in files if f not in preferred]
        return preferred + rest
    return files


def is_recognizable(target_rgba: np.ndarray) -> bool:
    """Reject background-only slivers and flat single-color blobs (gem/projectile icons):
    needs real alpha coverage, a filled-out bounding box, and enough color variety to look
    like an actual detailed character/item rather than a flat shape."""
    alpha = target_rgba[..., 3]
    opaque = alpha > 0.5
    coverage = opaque.mean()
    if coverage < MIN_ALPHA_COVERAGE:
        return False

    ys, xs = np.where(opaque)
    if len(ys) == 0:
        return False
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    fill = opaque.sum() / max(1, bbox_area)
    if fill < MIN_BBOX_FILL:
        return False

    opaque_rgb = target_rgba[..., :3][opaque]
    luminance = opaque_rgb.mean(axis=1)
    if luminance.std() < MIN_LUMINANCE_STD:
        return False  # visually flat/murky crop, no real shape-defining contrast

    quantized = (opaque_rgb * 15).astype(np.uint8)  # 16 levels/channel
    distinct = len(np.unique(quantized.reshape(-1, 3), axis=0))
    if distinct < MIN_DISTINCT_COLORS:
        return False

    # Every correctly-cropped sprite in these datasets has a visible black cartoon outline;
    # gradient backgrounds and texture/terrain close-ups don't. Filters out exactly the two
    # failure modes seen above (flat gradients, non-sprite photo-texture crops).
    near_black = opaque_rgb.sum(axis=1) < 0.3
    if near_black.mean() < MIN_OUTLINE_FRACTION:
        return False

    # A whole, self-contained icon has empty/mostly-transparent margins; content that bleeds
    # off multiple frame edges is a zoomed sub-crop of something bigger, not a standalone
    # object — this is what "abstract, unrecognizable" examples turned out to have in common.
    # Skip for sources with no real transparency (e.g. fe's intentionally opaque battle-scene
    # crops, where the whole frame being "opaque" is the format, not a zoomed sub-crop).
    if coverage < 0.95:
        edges = [opaque[0, :], opaque[-1, :], opaque[:, 0], opaque[:, -1]]
        bleeding_edges = sum(1 for e in edges if e.mean() > 0.7)
        if bleeding_edges > MAX_BLEEDING_EDGES:
            return False

    return True


def pick_examples(source: str, num_examples: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Oversamples candidates, filters to recognizable ones, returns (input, target) pairs."""
    files = load_clean_sprites(source)  # priority-tier files (e.g. papi icons) come first
    rng_py = random.Random(seed)
    priority_prefixes = SHOWCASE_PREFIX_PRIORITY.get(source)
    if priority_prefixes:
        split = next(
            (i for i, f in enumerate(files) if not any(f.name.startswith(p) for p in priority_prefixes)),
            len(files),
        )
        priority_tier, rest_tier = files[:split], files[split:]
        rng_py.shuffle(priority_tier)
        rng_py.shuffle(rest_tier)
        shuffled = priority_tier + rest_tier  # exhaust the priority tier before falling back
    else:
        shuffled = files[:]
        rng_py.shuffle(shuffled)

    rng_np = np.random.default_rng(seed)
    picked = []
    picked_signatures = []  # coarse downsampled thumbnails, to avoid near-duplicate picks
    for f in shuffled:
        clean = load_image_float32(f)
        if clean.shape[:2] != (32, 32):
            clean = pad_to_target(clean, target_size=32)
        if not is_recognizable(clean):
            continue

        sig = clean[::4, ::4, :3].reshape(-1)  # 8x8x3 coarse signature
        if any(np.abs(sig - other).mean() < MIN_DIVERSITY_DIST for other in picked_signatures):
            continue  # too similar to something already picked (e.g. same item, different color)

        degraded = degrade(clean, rng=rng_np, ranges=DegradeRanges.standard())
        picked.append((degraded, clean))
        picked_signatures.append(sig)
        if len(picked) >= num_examples:
            break
    return picked


def build_trio_grid(rows: list[list[np.ndarray]], col_labels: list[str]) -> Image.Image:
    n_cols = len(col_labels)
    cell_px = CELL * SCALE
    grid_w = n_cols * cell_px + (n_cols - 1) * GAP
    grid_h = LABEL_H + len(rows) * cell_px + (len(rows) - 1) * GAP

    canvas = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    for c, label in enumerate(col_labels):
        x = c * (cell_px + GAP)
        draw.text((x + 8, 3), label, fill=(30, 30, 30, 255), font=font)

    for r, row in enumerate(rows):
        y = LABEL_H + r * (cell_px + GAP)
        for c, arr in enumerate(row):
            x = c * (cell_px + GAP)
            tile = nn_upscale_img(arr, SCALE)
            canvas.paste(tile, (x, y), tile)
    return canvas


def build_source_showcase(source: str, num_examples: int = 4, seed: int = 123) -> Path:
    final_ckpt = BYSOURCE / source / "vqgan_32_epoch_100.pt"
    model = load_model(final_ckpt)

    examples = pick_examples(source, num_examples, seed)
    if len(examples) < num_examples:
        print(f"[!] {source}: only found {len(examples)}/{num_examples} recognizable examples "
              f"(content filter may be too strict for this source)")

    rows = []
    for degraded, clean in examples:
        inp_t = torch.from_numpy(degraded).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            pred, _, _ = model(inp_t)
        rows.append([degraded, pred[0].numpy().transpose(1, 2, 0), clean])

    grid = build_trio_grid(rows, ["Messy input", "AI restored", "Original sprite"])
    SHOWCASE_DIR.mkdir(exist_ok=True)
    out_path = SHOWCASE_DIR / f"{source}_showcase.png"
    grid.save(out_path)
    print(f"[+] {source}: saved showcase grid -> {out_path}")
    return out_path


def build_palette_trick_demo(source: str = "pmd", seed: int = 7, pool_size: int = 10) -> Path:
    """One clear example of the color-cleanup trick: AI output -> snapped to a clean palette.
    Picks, among recognizable candidates, the one with the most visible color noise in the
    restored output, so the 'trick' is obviously demonstrated rather than a borderline case.
    """
    final_ckpt = BYSOURCE / source / "vqgan_32_epoch_100.pt"
    model = load_model(final_ckpt)

    candidates = pick_examples(source, pool_size, seed)

    best, best_var = None, -1.0
    with torch.no_grad():
        for degraded, clean in candidates:
            inp_t = torch.from_numpy(degraded).permute(2, 0, 1).unsqueeze(0)
            pred, _, _ = model(inp_t)
            pred_hwc = pred[0].numpy().transpose(1, 2, 0)
            var = float(np.var(pred_hwc[..., :3]))
            if var > best_var:
                best_var = var
                best = (degraded, clean, pred_hwc)

    degraded, clean, pred_hwc = best
    pal = extract_palette_kmeans(pred_hwc, k=12, alpha_threshold=0.5)
    snapped = nearest_neighbor_snap(pred_hwc, pal, alpha_threshold=0.5)

    rows = [[degraded, pred_hwc, snapped, clean]]
    grid = build_trio_grid(rows, ["Messy input", "AI restored (noisy)", "+ Color cleanup", "Original sprite"])
    SHOWCASE_DIR.mkdir(exist_ok=True)
    out_path = SHOWCASE_DIR / "palette_trick_demo.png"
    grid.save(out_path)
    print(f"[+] saved palette trick demo -> {out_path}")
    return out_path


def build_overview_grid(seeds: dict[str, int] | None = None) -> Path:
    """One strongly-degraded example per source (input / restored / target), for a fast
    cross-source scan at the top of SHOWCASE.md."""
    seeds = seeds or DEFAULT_SEEDS
    rows = []
    for source in SOURCES:
        final_ckpt = BYSOURCE / source / "vqgan_32_epoch_100.pt"
        if not final_ckpt.exists():
            continue
        model = load_model(final_ckpt)
        degraded, clean = pick_examples(source, 1, seeds.get(source, 123))[0]
        with torch.no_grad():
            pred, _, _ = model(torch.from_numpy(degraded).permute(2, 0, 1).unsqueeze(0))
        rows.append([degraded, pred[0].numpy().transpose(1, 2, 0), clean])

    grid = build_trio_grid(rows, ["Messy input", "AI restored", "Original sprite"])
    SHOWCASE_DIR.mkdir(exist_ok=True)
    out_path = SHOWCASE_DIR / "overview_grid.png"
    grid.save(out_path)
    print(f"[+] saved showcase overview grid -> {out_path}")
    return out_path


# Content-filter heuristics (recognizability, diversity, contrast) narrow the candidate pool
# a lot, and which specific images end up selected is sensitive to shuffle order — these seeds
# were hand-verified to produce a good, diverse, non-repetitive spread per source. Change with
# care; a "random" seed can just as easily land on a bad draw (seen firsthand: default seeds
# picked 3 near-identical wheelchairs for lpc and an all-abstract-crop set for papi).
DEFAULT_SEEDS = {"pmd": 123, "papi": 7, "lpc": 77, "fe": 42}


def main():
    for source in SOURCES:
        build_source_showcase(source, seed=DEFAULT_SEEDS.get(source, 123))
    build_palette_trick_demo()
    build_overview_grid()


if __name__ == "__main__":
    main()
