"""
Deterministic downscaling and resize algorithms.

Key features:
- Area-average downscaling (~2-3x target size) for deterministic blur-free reduction.
- Alpha-aware resizing: premultiplies RGB by alpha during interpolation so transparent
  background pixels do not bleed dark or white halos onto sprite outlines.
"""

from __future__ import annotations

import cv2
import numpy as np


def alpha_aware_resize(
    img: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int = cv2.INTER_AREA
) -> np.ndarray:
    """Resize an image while preserving clean color boundaries around alpha cutouts.

    Args:
        img: (H, W, 3) or (H, W, 4) float32 array in [0, 1].
        target_width: Desired width in pixels.
        target_height: Desired height in pixels.
        interpolation: OpenCV interpolation flag (default INTER_AREA for downscaling).

    Returns:
        Resized image with same number of channels as input.
    """
    if img.shape[0] == target_height and img.shape[1] == target_width:
        return img.copy()

    has_alpha = img.shape[-1] == 4
    if not has_alpha:
        return cv2.resize(img, (target_width, target_height), interpolation=interpolation)

    rgb = img[..., :3]
    alpha = img[..., 3:4]

    # Premultiply RGB by Alpha so transparent pixels don't pollute color averages
    premult_rgb = rgb * alpha
    premult_rgba = np.concatenate([premult_rgb, alpha], axis=-1)

    resized_rgba = cv2.resize(premult_rgba, (target_width, target_height), interpolation=interpolation)
    if resized_rgba.ndim == 2:
        resized_rgba = resized_rgba[..., None]

    out_rgb = resized_rgba[..., :3]
    out_alpha = resized_rgba[..., 3:4]

    # Un-premultiply where alpha > 0
    mask = out_alpha > 1e-5
    clean_rgb = np.zeros_like(out_rgb)
    clean_rgb[mask.repeat(3, axis=-1)] = (out_rgb / np.where(out_alpha == 0, 1.0, out_alpha))[mask.repeat(3, axis=-1)]

    return np.clip(np.concatenate([clean_rgb, out_alpha], axis=-1), 0.0, 1.0).astype(np.float32)


def area_average_downscale(img: np.ndarray, factor: float = 2.5) -> np.ndarray:
    """Downscale high-resolution input by a factor using area averaging.

    In the Spriteforge Stage A pipeline, real high-res images are first deterministically
    downscaled to ~2-3x the target sprite resolution before neural restoration or snapping.
    """
    h, w = img.shape[:2]
    new_w = max(1, int(round(w / factor)))
    new_h = max(1, int(round(h / factor)))
    return alpha_aware_resize(img, new_w, new_h, interpolation=cv2.INTER_AREA)


def resize_to_target(
    img: np.ndarray,
    target_size: int,
    method: str = "area"
) -> np.ndarray:
    """Resize an image to exact target_size x target_size (e.g., 16, 32, 48).

    Args:
        img: Input image float32 in [0, 1].
        target_size: Exact square target dimension.
        method: 'area' (INTER_AREA), 'nearest' (INTER_NEAREST), or 'cubic' (INTER_CUBIC).

    Returns:
        (target_size, target_size, C) float32 array in [0, 1].
    """
    interp_map = {
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
        "linear": cv2.INTER_LINEAR,
    }
    interp = interp_map.get(method.lower(), cv2.INTER_AREA)
    return alpha_aware_resize(img, target_size, target_size, interpolation=interp)


def pad_to_target(
    img: np.ndarray,
    target_size: int,
    allow_trim: bool = False
) -> np.ndarray | None:
    """Pad a clean pixel art sprite to exact target_size x target_size WITHOUT resampling.
    
    Why this is critical for training:
    Standard downscaling/resampling destroys crisp pixel edges, creates muddy intermediate colors,
    and ruins the authentic hand-drawn pixel art manifold. For training data ingestion, we MUST NOT
    resample sprites! Instead, we take sprites whose native bounding box fits within target_size
    and center-pad them with transparent pixels (r=0, g=0, b=0, a=0).
    
    Args:
        img: Input sprite float32 array in [0, 1], shape (H, W, 4) or (H, W, 3).
        target_size: Desired square canvas size (e.g., 16, 32, 48).
        allow_trim: If True and sprite is slightly larger than target_size, center-crop it.
            If False (default), returns None if the sprite exceeds target_size.
            
    Returns:
        Exact (target_size, target_size, C) padded float32 array, or None if oversized.
    """
    h, w = img.shape[:2]
    c = img.shape[-1]
    
    if h == target_size and w == target_size:
        return img.copy()
        
    if h > target_size or w > target_size:
        if not allow_trim:
            return None
        # Center-crop if allow_trim is True
        start_y = max(0, (h - target_size) // 2)
        start_x = max(0, (w - target_size) // 2)
        img = img[start_y:start_y+target_size, start_x:start_x+target_size]
        h, w = img.shape[:2]
        
    out = np.zeros((target_size, target_size, c), dtype=np.float32)
    pad_y = (target_size - h) // 2
    pad_x = (target_size - w) // 2
    out[pad_y:pad_y+h, pad_x:pad_x+w] = img
    return out
