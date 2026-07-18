# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
CC0 sprite sheet ingestion and normalization.
Supports both Kenney structured sheets and automatic bounding-box alpha contour slicing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from spriteforge.core.resize import pad_to_target, resize_to_target
from spriteforge.data.provenance import log_provenance


def compute_phash(img_u8: np.ndarray) -> str:
    """Compute a simple 8x8 perceptual hash of an RGBA image for deduplication."""
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGBA2GRAY) if img_u8.shape[-1] == 4 else img_u8
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean_val = np.mean(resized)
    bits = (resized > mean_val).flatten()
    return "".join("1" if b else "0" for b in bits)


def slice_by_alpha_contours(
    sheet_rgba: np.ndarray,
    min_area: int = 100,
    min_w: int = 10,
    min_h: int = 10,
    max_aspect_ratio: float = 3.5
) -> list[np.ndarray]:
    """Automatically slice an unannotated sprite sheet into isolated sprites using alpha contours.
    
    Includes smart filtering to eliminate UI junk, TM symbols, punctuation, and health bars:
    - Rejects crops smaller than min_w x min_h (default 10x10, filtering out 6x6 TM symbols/dots).
    - Rejects extreme aspect ratios (like thin 30x3 UI lines or health bars).
    """
    if sheet_rgba.shape[-1] != 4:
        return [sheet_rgba]

    # Threshold alpha at > 0.5 (128 u8) to ignore semi-transparent noise dots or anti-aliasing artifacts
    alpha_u8 = np.where(sheet_rgba[..., 3] > 0.5, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(alpha_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sprites = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Smart Junk Filtering: reject tiny symbols (TM, dots) and thin UI lines
        if w * h < min_area or w < min_w or h < min_h:
            continue
        if (w / h) > max_aspect_ratio or (h / w) > max_aspect_ratio:
            continue
        
        crop = sheet_rgba[y:y+h, x:x+w].copy()
        # Ignore crops that are almost completely transparent
        if np.max(crop[..., 3]) < 0.5:
            continue

        sprites.append(crop)
    return sprites


def slice_vertical_strip(sheet_rgba: np.ndarray, frame_size: int | None = None) -> list[np.ndarray]:
    """Split a vertically-stacked animation strip (single-column grid of square frames) into
    individual frames. For opaque-background strips (e.g. FE-Repo Map Sprites), alpha is
    uniform across the whole strip, so `slice_by_alpha_contours` finds zero components —
    this does simple fixed-size grid division instead.

    frame_size defaults to the strip's width (frames are assumed square, matching the strip's
    width — confirmed convention for FE-Repo Map Sprites: e.g. 32-wide strips contain 32x32
    frames, 16-wide strips contain 16x16 frames). Height not evenly divisible by frame_size is
    truncated (the remainder is dropped, not padded).
    """
    h, w = sheet_rgba.shape[:2]
    fs = frame_size or w
    if fs <= 0 or w != fs:
        return [sheet_rgba]  # not a single-column strip in the expected convention

    num_frames = h // fs
    return [sheet_rgba[i * fs:(i + 1) * fs, :, :].copy() for i in range(num_frames)]


def ingest_sprite_pack(
    input_dir: str | Path,
    output_dir: str | Path,
    target_size: int = 32,
    source_name: str = "Unknown CC0 Source"
) -> int:
    """Ingest images from input_dir, slice/normalize to target_size, deduplicate, and save."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input directory not found: {in_path}")

    log_provenance(source_name=source_name, source_url=str(in_path), author="Ingested Pack", license_status="CC0 1.0 Universal")

    seen_hashes = set()
    count = 0

    for img_file in in_path.glob("**/*.*"):
        if img_file.suffix.lower() not in [".png", ".bmp", ".tga", ".webp"]:
            continue

        try:
            pil_img = Image.open(img_file).convert("RGBA")
            img_rgba = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"[!] Skipping {img_file.name}: {e}")
            continue

        # If image is larger than 4x target_size and aspect ratio isn't 1:1, assume sprite sheet
        h, w = img_rgba.shape[:2]
        if (h > target_size * 2 or w > target_size * 2) and (h != w or h > 128):
            slices = slice_by_alpha_contours(img_rgba)
        else:
            slices = [img_rgba]

        for idx, sprite in enumerate(slices):
            padded = pad_to_target(sprite, target_size=target_size)
            if padded is None:
                continue
            u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)
            
            phash = compute_phash(u8)
            if phash in seen_hashes:
                continue
            seen_hashes.add(phash)

            out_name = f"{img_file.stem}_s{idx:03d}_{count:05d}.png"
            Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
            count += 1

    return count
