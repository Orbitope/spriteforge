# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Image input/output utilities for float32 RGBA/RGB image arrays.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image


def load_image_float32(filepath: str | Path) -> np.ndarray:
    """Load an image from disk as an RGBA float32 array in [0, 1]."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    
    img_pil = Image.open(path).convert("RGBA")
    img_u8 = np.array(img_pil, dtype=np.uint8)
    return img_u8.astype(np.float32) / 255.0


def save_image_float32(img: np.ndarray, filepath: str | Path) -> None:
    """Save an RGBA/RGB float32 array in [0, 1] to disk as a PNG image."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    img_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if img_u8.shape[-1] == 4:
        img_pil = Image.fromarray(img_u8, mode="RGBA")
    else:
        img_pil = Image.fromarray(img_u8, mode="RGB")
    img_pil.save(path, format="PNG")
