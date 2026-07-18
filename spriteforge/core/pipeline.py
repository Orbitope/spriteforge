# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Stage A Pipeline: Deterministic conversion of high-res or arbitrary images to retro pixel-art sprites.
"""
from __future__ import annotations
import numpy as np
from spriteforge.core.resize import resize_to_target
from spriteforge.core.palette import (
    extract_palette_kmeans,
    extract_palette_median_cut,
    load_builtin_palette,
    load_palette,
    nearest_neighbor_snap,
    ordered_dither_snap,
)
from spriteforge.core.alpha import despeckle_alpha, threshold_alpha


def convert_image_to_sprite(
    img: np.ndarray,
    target_size: int = 32,
    palette_mode: str = "per-image-kmeans",
    colors: int = 16,
    palette_preset: str = "pico8",
    palette_file: str | None = None,
    palette: np.ndarray | None = None,
    dither: bool = False,
    dither_strength: float = 0.05,
    despeckle: bool = True,
    despeckle_min_area: int = 2,
    return_palette: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert an arbitrary float32 RGBA/RGB image [0, 1] into a crisp retro pixel-art sprite.

    Args:
        img: Input image array of shape (H, W, 3) or (H, W, 4) in [0, 1].
        target_size: Target square dimension (e.g., 16, 32, 48).
        palette_mode: 'per-image-kmeans', 'per-image-median', 'preset', or 'fixed'.
        colors: Number of colors for palette extraction (kmeans/median modes only).
        palette_preset: Bundled palette name (see palette.list_builtin_palettes()),
            used when palette_mode is 'preset'.
        palette_file: Path to a palette file, used when palette_mode is 'fixed'.
        palette: Explicit (K, 3) float32 RGB palette in [0, 1]. When provided this
            takes precedence over palette_mode/preset/file — the snap uses these
            colors directly. This is the entry point for a user-supplied or
            sub-selected palette (see spriteforge.core.palette_library).
        dither: Whether to apply Bayer ordered dithering during palette snapping.
        dither_strength: Intensity of dithering noise in OKLab space.
        despeckle: Whether to remove isolated transparent/opaque noise speckles.
        despeckle_min_area: Minimum pixel area for speckles to survive.
        return_palette: If True, return (sprite, palette) instead of just sprite —
            lets a caller discover the concrete colors an auto-extracting mode
            (kmeans/median/preset/fixed) picked, e.g. to populate a sub-select UI.

    Returns:
        (target_size, target_size, 4) float32 RGBA array in [0, 1], or
        (sprite, palette) if return_palette is True.
    """
    # 1. Resize down to exact target size using area averaging
    resized = resize_to_target(img, target_size=target_size, method="area")

    # Ensure alpha channel exists
    if resized.shape[-1] == 3:
        alpha = np.ones(resized.shape[:2] + (1,), dtype=np.float32)
        resized = np.concatenate([resized, alpha], axis=-1)

    # 2. Select the palette. An explicit palette array wins over every mode.
    mode = palette_mode.lower()
    if palette is not None:
        palette = np.asarray(palette, dtype=np.float32)
        if palette.ndim != 2 or palette.shape[1] != 3 or palette.shape[0] == 0:
            raise ValueError(f"Explicit palette must be a non-empty (K, 3) array; got {palette.shape}")
    elif "median" in mode:
        palette = extract_palette_median_cut(resized, k=colors)
    elif "preset" in mode:
        palette = load_builtin_palette(palette_preset)
    elif "fixed" in mode:
        if not palette_file:
            raise ValueError("palette_file is required when palette_mode is 'fixed'")
        palette = load_palette(palette_file)
    else:
        palette = extract_palette_kmeans(resized, k=colors)

    # 3. Snap colors to extracted palette in OKLab space
    if dither and dither_strength > 0:
        snapped = ordered_dither_snap(resized, palette, strength=dither_strength)
    else:
        snapped = nearest_neighbor_snap(resized, palette)

    # 4. Threshold alpha to clean up transparent edge boundaries
    snapped = threshold_alpha(snapped, threshold=0.5)

    # 5. Optional despeckling of isolated stray 1-pixel noise
    if despeckle and despeckle_min_area > 1:
        snapped = despeckle_alpha(snapped, min_area=despeckle_min_area)

    return (snapped, palette) if return_palette else snapped
