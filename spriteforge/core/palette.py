# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Palette snapping and extraction in OKLab color space.

Why OKLab?
RGB nearest-neighbor color snapping picks visibly wrong colors because Euclidean distance
in RGB does not correlate with human perceptual difference. OKLab is a perceptual color space
where Euclidean distance maps uniformly to perceived color difference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans


# --------------------------------------------------------------------------- #
# OKLab <-> sRGB Color Conversions (Exact Linear Transformations)
# --------------------------------------------------------------------------- #

_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)

_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float32)

_M2_INV = np.array([
    [1.0000000000, +0.3963377774, +0.2158037573],
    [1.0000000000, -0.1055613458, -0.0638541728],
    [1.0000000000, -0.0894841775, -1.2914855480],
], dtype=np.float32)

_M1_INV = np.array([
    [+4.0767416621, -3.3077115913, +0.2309699292],
    [-1.2684380046, +2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, +1.7076147010],
], dtype=np.float32)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0, 1] to linear RGB [0, 1]."""
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0, 1] to sRGB [0, 1]."""
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055).astype(np.float32)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB float array (..., 3) in [0, 1] to OKLab (..., 3)."""
    lin = srgb_to_linear(rgb)
    lms = np.dot(lin, _M1.T)
    # Cube root preserving sign
    lms_prime = np.sign(lms) * (np.abs(lms) ** (1.0 / 3.0))
    oklab = np.dot(lms_prime, _M2.T)
    return oklab.astype(np.float32)


def oklab_to_rgb(oklab: np.ndarray) -> np.ndarray:
    """Convert OKLab array (..., 3) to sRGB float array (..., 3) in [0, 1]."""
    lms_prime = np.dot(oklab, _M2_INV.T)
    lms = lms_prime ** 3.0
    lin = np.dot(lms, _M1_INV.T)
    return linear_to_srgb(lin)


# --------------------------------------------------------------------------- #
# Palette Snapping (Nearest Neighbor in OKLab Space)
# --------------------------------------------------------------------------- #

def nearest_neighbor_snap(
    img: np.ndarray,
    palette_rgb: np.ndarray,
    alpha_threshold: float = 0.5
) -> np.ndarray:
    """Snap image colors to the nearest color in palette_rgb using Euclidean distance in OKLab.

    Args:
        img: (H, W, 3) or (H, W, 4) float32 in [0, 1].
        palette_rgb: (K, 3) float32 in [0, 1].
        alpha_threshold: Alpha threshold below which pixels are ignored/zeroed.

    Returns:
        snapped image with same shape and alpha channel preserved.
    """
    has_alpha = img.shape[-1] == 4
    if has_alpha:
        rgb = img[..., :3]
        alpha = img[..., 3:4]
    else:
        rgb = img
        alpha = np.ones(img.shape[:2] + (1,), dtype=np.float32)

    # Convert both image and palette to OKLab
    img_oklab = rgb_to_oklab(rgb)  # (H, W, 3)
    pal_oklab = rgb_to_oklab(palette_rgb)  # (K, 3)

    # Compute Euclidean distance in OKLab space: (H, W, K)
    # (img - pal)^2 = img^2 + pal^2 - 2*img*pal
    diff = img_oklab[..., None, :] - pal_oklab[None, None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)  # (H, W, K)

    best_idx = np.argmin(dist_sq, axis=-1)  # (H, W)
    snapped_rgb = palette_rgb[best_idx]  # (H, W, 3)

    if has_alpha:
        # Zero out RGB where alpha is transparent to avoid color edge bleeding
        mask = (alpha >= alpha_threshold).astype(np.float32)
        snapped_rgb = snapped_rgb * mask
        return np.concatenate([snapped_rgb, alpha], axis=-1)
    return snapped_rgb


def palette_index_map(
    img: np.ndarray,
    palette_rgb: np.ndarray,
    alpha_threshold: float = 0.5
) -> np.ndarray:
    """Like nearest_neighbor_snap, but returns per-pixel class indices instead of colors.

    Used to build classification targets for the palette-index model (E1, see
    devlog/2026-07-08-random-pool-samples.md's "Diagnostics" section): index in
    [0, K-1] for the nearest palette color, or K (one past the palette) for pixels
    below alpha_threshold — a dedicated "transparent" class.

    Args:
        img: (H, W, 3) or (H, W, 4) float32 in [0, 1].
        palette_rgb: (K, 3) float32 in [0, 1].

    Returns:
        (H, W) int64 array with values in [0, K] (K = transparent class).
    """
    has_alpha = img.shape[-1] == 4
    rgb = img[..., :3] if has_alpha else img
    k = palette_rgb.shape[0]

    img_oklab = rgb_to_oklab(rgb)
    pal_oklab = rgb_to_oklab(palette_rgb)

    diff = img_oklab[..., None, :] - pal_oklab[None, None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    best_idx = np.argmin(dist_sq, axis=-1).astype(np.int64)

    if has_alpha:
        alpha = img[..., 3]
        best_idx = np.where(alpha >= alpha_threshold, best_idx, k)

    return best_idx


# --------------------------------------------------------------------------- #
# Ordered Dithering (Optional / Default Off)
# --------------------------------------------------------------------------- #

_BAYER_4X4 = (np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
], dtype=np.float32) / 16.0 - 0.5)  # Centered around 0 in [-0.5, 0.4375]


def ordered_dither_snap(
    img: np.ndarray,
    palette_rgb: np.ndarray,
    strength: float = 0.05,
    alpha_threshold: float = 0.5
) -> np.ndarray:
    """Apply 4x4 Bayer ordered dithering before snapping to palette in OKLab.

    Args:
        img: (H, W, 3) or (H, W, 4) float32 in [0, 1].
        palette_rgb: (K, 3) float32 in [0, 1].
        strength: Amount of dither jitter to add in OKLab L-channel.
        alpha_threshold: Transparency cutoff.

    Returns:
        Dithered and palette-snapped image.
    """
    has_alpha = img.shape[-1] == 4
    rgb = img[..., :3] if has_alpha else img
    alpha = img[..., 3:4] if has_alpha else np.ones(img.shape[:2] + (1,), dtype=np.float32)

    img_oklab = rgb_to_oklab(rgb)
    h, w = img.shape[:2]

    # Tile bayer matrix to match image dimensions
    bayer_tiled = np.tile(_BAYER_4X4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    
    # Add dither mainly to lightness (L channel of OKLab)
    dithered_oklab = img_oklab.copy()
    dithered_oklab[..., 0] = np.clip(dithered_oklab[..., 0] + bayer_tiled * strength, 0.0, 1.0)

    dithered_rgb = oklab_to_rgb(dithered_oklab)
    if has_alpha:
        dithered_img = np.concatenate([dithered_rgb, alpha], axis=-1)
    else:
        dithered_img = dithered_rgb

    return nearest_neighbor_snap(dithered_img, palette_rgb, alpha_threshold=alpha_threshold)


# --------------------------------------------------------------------------- #
# Per-Image Palette Extraction (K-Means & Median-Cut in OKLab)
# --------------------------------------------------------------------------- #

def extract_palette_kmeans(
    img: np.ndarray,
    k: int = 16,
    alpha_threshold: float = 0.5,
    random_state: int = 42
) -> np.ndarray:
    """Extract a per-image palette of k colors using K-Means clustering in OKLab space.

    Args:
        img: (H, W, 3) or (H, W, 4) float32 in [0, 1].
        k: Number of palette colors to extract.
        alpha_threshold: Ignore pixels with alpha below this threshold.
        random_state: Random seed for reproducibility.

    Returns:
        (K, 3) float32 RGB palette in [0, 1].
    """
    if img.shape[-1] == 4:
        mask = img[..., 3] >= alpha_threshold
        valid_rgb = img[..., :3][mask]
    else:
        valid_rgb = img.reshape(-1, 3)

    if len(valid_rgb) == 0:
        # Fallback to grayscale palette if image is completely transparent
        return np.stack([np.linspace(0, 1, k)] * 3, axis=-1).astype(np.float32)

    if len(valid_rgb) <= k:
        # Fewer unique pixels than requested k
        unique_rgb = np.unique(valid_rgb, axis=0)
        pad_size = k - len(unique_rgb)
        if pad_size > 0:
            padding = np.zeros((pad_size, 3), dtype=np.float32)
            unique_rgb = np.vstack([unique_rgb, padding])
        return unique_rgb[:k]

    # Convert valid pixels to OKLab for perceptual clustering
    valid_oklab = rgb_to_oklab(valid_rgb)

    kmeans = KMeans(n_clusters=k, n_init=5, random_state=random_state)
    kmeans.fit(valid_oklab)
    centers_oklab = kmeans.cluster_centers_.astype(np.float32)

    return np.clip(oklab_to_rgb(centers_oklab), 0.0, 1.0)


def extract_palette_median_cut(
    img: np.ndarray,
    k: int = 16,
    alpha_threshold: float = 0.5
) -> np.ndarray:
    """Extract a per-image palette of k colors using Median-Cut in OKLab space.

    Args:
        img: (H, W, 3) or (H, W, 4) float32 in [0, 1].
        k: Number of palette colors (should ideally be a power of 2, e.g., 16, 32).
        alpha_threshold: Ignore pixels with alpha below this threshold.

    Returns:
        (K, 3) float32 RGB palette in [0, 1].
    """
    if img.shape[-1] == 4:
        mask = img[..., 3] >= alpha_threshold
        valid_rgb = img[..., :3][mask]
    else:
        valid_rgb = img.reshape(-1, 3)

    if len(valid_rgb) == 0:
        return np.stack([np.linspace(0, 1, k)] * 3, axis=-1).astype(np.float32)

    valid_oklab = rgb_to_oklab(valid_rgb)
    boxes = [valid_oklab]

    while len(boxes) < k:
        # Find box with largest range along any dimension
        best_box_idx = -1
        max_range = -1.0
        split_dim = -1

        for idx, box in enumerate(boxes):
            if len(box) <= 1:
                continue
            ranges = np.ptp(box, axis=0)
            dim = int(np.argmax(ranges))
            if ranges[dim] > max_range:
                max_range = float(ranges[dim])
                best_box_idx = idx
                split_dim = dim

        if best_box_idx == -1:
            break  # Cannot split further

        box_to_split = boxes.pop(best_box_idx)
        # Sort along split_dim and split at median
        sorted_box = box_to_split[np.argsort(box_to_split[:, split_dim])]
        med_idx = len(sorted_box) // 2
        boxes.append(sorted_box[:med_idx])
        boxes.append(sorted_box[med_idx:])

    # Compute average color of each box in OKLab
    centers_oklab = []
    for box in boxes:
        if len(box) > 0:
            centers_oklab.append(np.mean(box, axis=0))
        else:
            centers_oklab.append(np.zeros(3, dtype=np.float32))

    # Pad if we couldn't reach k boxes
    while len(centers_oklab) < k:
        centers_oklab.append(np.zeros(3, dtype=np.float32))

    centers_oklab = np.array(centers_oklab[:k], dtype=np.float32)
    return np.clip(oklab_to_rgb(centers_oklab), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Palette File I/O
# --------------------------------------------------------------------------- #

def load_palette(filepath: str | Path) -> np.ndarray:
    """Load a palette from a JSON file (hex strings or RGB arrays) or image format.

    Returns:
        (K, 3) float32 RGB palette in [0, 1].
    """
    path = Path(filepath)
    if path.suffix.lower() in [".json", ".hex", ".pal"]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith(";")]
        colors = []
        for item in data:
            if isinstance(item, str):
                hex_val = item.lstrip("#")
                if len(hex_val) >= 6:
                    r = int(hex_val[0:2], 16) / 255.0
                    g = int(hex_val[2:4], 16) / 255.0
                    b = int(hex_val[4:6], 16) / 255.0
                    colors.append([r, g, b])
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                val = [float(c) for c in item[:3]]
                if any(c > 1.0 for c in val):
                    val = [c / 255.0 for c in val]
                colors.append(val)
        return np.array(colors, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported palette file format: {path.suffix}")


BUILTIN_PALETTES_DIR = Path(__file__).resolve().parent.parent / "data" / "palettes"


def list_builtin_palettes() -> list[str]:
    """Names of bundled preset palettes (e.g. 'pico8', 'dawnbringer32'), sorted."""
    if not BUILTIN_PALETTES_DIR.is_dir():
        return []
    return sorted(p.stem for p in BUILTIN_PALETTES_DIR.glob("*.json"))


def load_builtin_palette(name: str) -> np.ndarray:
    """Load a bundled preset palette by name (see list_builtin_palettes())."""
    path = BUILTIN_PALETTES_DIR / f"{name}.json"
    if not path.exists():
        available = ", ".join(list_builtin_palettes())
        raise ValueError(f"Unknown builtin palette '{name}'. Available: {available}")
    return load_palette(path)


def save_palette_json(palette_rgb: np.ndarray, filepath: str | Path) -> None:
    """Save a float32 RGB palette [0, 1] as a JSON array of hex strings."""
    hex_colors = []
    for rgb in palette_rgb:
        u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        hex_colors.append(f"#{u8[0]:02x}{u8[1]:02x}{u8[2]:02x}")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(hex_colors, f, indent=2)


# --------------------------------------------------------------------------- #
# Background removal                                                            #
# --------------------------------------------------------------------------- #

def dominant_border_color(rgb_u8: np.ndarray, border_px: int = 4) -> np.ndarray:
    """Return the most-frequent RGB color in the border strip of an image.

    Args:
        rgb_u8: uint8 (H, W, 3) array.
        border_px: width of border strip to sample (default 4 px — suits 32×32 inputs).

    Returns:
        float32 (3,) RGB in [0, 255].
    """
    h, w = rgb_u8.shape[:2]
    border_px = min(border_px, h // 2, w // 2)
    strips = np.concatenate([
        rgb_u8[:border_px, :].reshape(-1, 3),
        rgb_u8[-border_px:, :].reshape(-1, 3),
        rgb_u8[:, :border_px].reshape(-1, 3),
        rgb_u8[:, -border_px:].reshape(-1, 3),
    ])
    quantized = (strips // 8).astype(np.int32)
    keys = quantized[:, 0] * 65536 + quantized[:, 1] * 256 + quantized[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    dominant_key = int(vals[np.argmax(counts)])
    r = (dominant_key >> 16) * 8
    g = ((dominant_key >> 8) & 0xFF) * 8
    b = (dominant_key & 0xFF) * 8
    return np.array([r, g, b], dtype=np.float32)


def remove_background_flood(
    img_rgba: np.ndarray,
    tol: float = 12.0,
    min_foreground_frac: float = 0.03,
    max_background_frac: float = 0.97,
    feather: bool = True,
) -> np.ndarray:
    """Key out a border-connected background using seeded flood fill.

    Unlike :func:`remove_background` (global color-distance to a single sampled
    color), this propagates from the image borders through *gradually changing*
    colors, so soft vignettes and drop-shadows behind a subject get keyed out too,
    while the flood stops at the subject's hard edge (connectivity-aware). This is
    the matte that handles real AI-generated inputs, whose gray backgrounds are
    rarely a single flat color.

    Args:
        img_rgba: float32 (H, W, 4) in [0, 1].
        tol: per-channel floating-range tolerance (0-255 scale) for the flood. The
            flood compares each candidate pixel to its already-filled neighbour, so
            this bounds *local* color change — small values still traverse a smooth
            gradient but won't leap across the subject's edge.
        min_foreground_frac: if fewer than this fraction of pixels survive as
            foreground, treat as a failed keying and return the original.
        max_background_frac: if the flood consumes more than this fraction, it has
            leaked across the subject; return the original.
        feather: soften the 1-px alpha edge so downscale doesn't leave a hard halo.

    Returns:
        float32 (H, W, 4) with background alpha zeroed, or the original array if the
        image already has transparency or the keying is degenerate.
    """
    import cv2  # local import — palette.py keeps cv2 off its top-level deps

    # Already has meaningful transparency — nothing to key.
    if (img_rgba[..., 3] < (20 / 255)).mean() > 0.05:
        return img_rgba

    h, w = img_rgba.shape[:2]
    rgb_u8 = (img_rgba[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    filled = np.zeros((h + 2, w + 2), np.uint8)
    lo = (tol, tol, tol)
    up = (tol, tol, tol)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY  # floating range (no FIXED_RANGE)

    # Seed densely around the whole border so every border-connected background
    # component (including a shadow strip that only touches one edge) is caught.
    step = max(1, min(h, w) // 32)
    seeds = []
    for x in range(0, w, step):
        seeds.append((x, 0))
        seeds.append((x, h - 1))
    for y in range(0, h, step):
        seeds.append((0, y))
        seeds.append((w - 1, y))

    for (sx, sy) in seeds:
        if filled[sy + 1, sx + 1]:
            continue  # already part of a filled component
        cv2.floodFill(bgr, filled, (sx, sy), 0, lo, up, flags)

    bg = filled[1:-1, 1:-1] > 0
    fg = ~bg

    # Clean specks: drop tiny isolated foreground blobs, fill tiny holes in the subject.
    fg_u8 = fg.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, k)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, k)

    frac = float(fg_u8.mean())
    if frac < min_foreground_frac or frac > (1.0 - (1.0 - max_background_frac)):
        return img_rgba  # degenerate — leaked or found nothing

    alpha = fg_u8.astype(np.float32)
    if feather:
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.6)

    result = img_rgba.copy()
    result[..., 3] = np.clip(alpha, 0.0, 1.0)
    return result


def remove_background(
    img_rgba: np.ndarray,
    distance_threshold: float = 30.0,
    border_px: int = 4,
    min_foreground_frac: float = 0.05,
) -> np.ndarray:
    """Key out a uniform background from a float32 RGBA image using border color sampling.

    Samples the dominant color from a `border_px`-wide strip around the image edge,
    then sets any pixel within `distance_threshold` (Euclidean, 0-255 scale) of that
    color to fully transparent.

    Args:
        img_rgba: float32 (H, W, 4) in [0, 1].
        distance_threshold: RGB Euclidean distance (0-255) below which a pixel is
            considered background. 30 works well for solid-color AI-generated backgrounds;
            lower (15-20) for tightly defined game sprite backgrounds.
        border_px: Border strip width to sample for background color detection.
        min_foreground_frac: If fewer than this fraction of pixels survive as foreground
            the image probably has no separable background; returns the original unchanged.

    Returns:
        float32 (H, W, 4) with background pixels zeroed in alpha. Returns the original
        array unchanged if the image already has meaningful transparency or if keying
        yields a degenerate result.
    """
    # Already has meaningful transparency — skip
    if (img_rgba[..., 3] < (20 / 255)).mean() > 0.05:
        return img_rgba

    rgb_255 = img_rgba[..., :3] * 255.0
    rgb_u8 = rgb_255.clip(0, 255).astype(np.uint8)
    bg_color = dominant_border_color(rgb_u8, border_px=border_px)
    dist = np.linalg.norm(rgb_255 - bg_color, axis=-1)
    alpha_mask = (dist >= distance_threshold).astype(np.float32)

    if alpha_mask.mean() < min_foreground_frac:
        return img_rgba  # keying failed — background and content too similar

    result = img_rgba.copy()
    result[..., 3] = alpha_mask
    return result
