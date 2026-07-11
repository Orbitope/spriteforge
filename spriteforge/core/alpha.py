"""
Alpha channel handling: hard matte thresholding and stray pixel despeckling.
"""

from __future__ import annotations

import cv2
import numpy as np


def threshold_alpha(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Enforce hard transparency: alpha >= threshold becomes 1.0, otherwise 0.0.

    Where alpha becomes 0.0, RGB channels are zeroed out to prevent color bleeding.

    Args:
        img: (H, W, 4) float32 array in [0, 1].
        threshold: Alpha cutoff value.

    Returns:
        (H, W, 4) float32 array with binary alpha channel.
    """
    if img.shape[-1] != 4:
        return img.copy()

    out = img.copy()
    alpha = out[..., 3:4]
    binary_alpha = (alpha >= threshold).astype(np.float32)

    out[..., :3] = out[..., :3] * binary_alpha
    out[..., 3:4] = binary_alpha
    return out


def despeckle_alpha(img: np.ndarray, min_area: int = 2) -> np.ndarray:
    """Remove stray isolated 1-pixel transparent or opaque speckles.

    Uses connected components analysis on the binary alpha mask to clean up
    noise speckles from downscaling or palette snapping.

    Args:
        img: (H, W, 4) float32 array in [0, 1].
        min_area: Minimum pixel count for an isolated component to survive.

    Returns:
        Despeckled (H, W, 4) float32 array.
    """
    if img.shape[-1] != 4:
        return img.copy()

    out = img.copy()
    alpha_u8 = (out[..., 3] >= 0.5).astype(np.uint8) * 255

    # Find connected components of opaque pixels
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha_u8, connectivity=8)

    clean_mask = np.zeros_like(alpha_u8, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = True

    out[..., 3] = clean_mask.astype(np.float32)
    out[..., :3] = out[..., :3] * out[..., 3:4]
    return out


def apply_matte_background(img: np.ndarray, bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Composite an RGBA image over a solid background color.

    Returns:
        (H, W, 3) float32 RGB array.
    """
    if img.shape[-1] == 3:
        return img.copy()

    rgb = img[..., :3]
    alpha = img[..., 3:4]
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)

    return np.clip(rgb * alpha + bg * (1.0 - alpha), 0.0, 1.0)
