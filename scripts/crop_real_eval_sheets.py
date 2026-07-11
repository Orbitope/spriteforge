"""
One-off script: crop individual figures out of real AI-generated character
sheets (for the Stage 2 "real-input eval set", item 1d — see
devlog/2026-07-08-stage2-degradation-fixes.md). These are genuine, uncurated
AI outputs supplied by the user, not synthetic degrade() output — the point is
to break the circularity of evaluating only on our own degradation distribution.

Approach: detect the sheet's flat background color, build a foreground mask,
merge nearby components with a small dilation (joins a character's own limbs/
wings/weapon without joining separate letters of caption text), then discard
small/thin blobs (captions, titles) and keep the rest as individual crops.
Not used for training data, so rough crops with some background/caption
bleed-through are acceptable — this mirrors what a real user would upload.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC_DIR = Path(
    "/private/tmp/claude-501/-Users-mwburke-projects-spriteforge/"
    "1e2768ff-507c-4c57-b888-34471b197ad1/scratchpad/eval_source_images"
)
OUT_DIR = Path("/Users/mwburke/projects/spriteforge/data_private/real_eval_raw")

MIN_AREA = 1500
MIN_HEIGHT = 60
PADDING = 10


def background_color(img: np.ndarray) -> np.ndarray:
    """Mode color over a sample of border pixels (background dominates edges)."""
    h, w = img.shape[:2]
    border = np.concatenate([
        img[0:10, :].reshape(-1, 3),
        img[-10:, :].reshape(-1, 3),
        img[:, 0:10].reshape(-1, 3),
        img[:, -10:].reshape(-1, 3),
    ])
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return colors[np.argmax(counts)]


def crop_sheet(path: Path, out_prefix: str) -> int:
    img = np.array(Image.open(path).convert("RGB"))
    bg = background_color(img)

    dist = np.linalg.norm(img.astype(np.float32) - bg.astype(np.float32), axis=-1)
    fg_mask = (dist > 20).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    merged = cv2.dilate(fg_mask, kernel, iterations=1)
    merged = cv2.erode(merged, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    h, w = img.shape[:2]
    saved = 0
    for i in range(1, num_labels):  # skip background label 0
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA or bh < MIN_HEIGHT:
            continue
        # Use the ORIGINAL (undilated) mask's bbox within this component's region
        # so the crop hugs the real figure, not the dilated blob.
        region_mask = (labels == i).astype(np.uint8)
        orig_in_region = fg_mask & (region_mask * 255)
        ys, xs = np.where(orig_in_region > 0)
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        x0 = max(0, x0 - PADDING)
        y0 = max(0, y0 - PADDING)
        x1 = min(w, x1 + PADDING)
        y1 = min(h, y1 + PADDING)

        crop = img[y0:y1, x0:x1]
        if crop.shape[0] < MIN_HEIGHT or crop.shape[1] < 20:
            continue

        saved += 1
        out_path = OUT_DIR / f"{out_prefix}_{saved:02d}.png"
        Image.fromarray(crop).save(out_path)

    return saved


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("pasted_02.png", "sheet1_mastercast"),
        ("pasted_04.png", "sheet3_professions"),
        ("pasted_05.png", "sheet4_hooded"),
        ("pasted_06.png", "sheet5_overworld"),
    ]
    total = 0
    for fname, prefix in targets:
        path = SRC_DIR / fname
        if not path.exists():
            print(f"[!] missing {path}", file=sys.stderr)
            continue
        n = crop_sheet(path, prefix)
        print(f"{fname} -> {n} crops")
        total += n
    print(f"total: {total} crops saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
