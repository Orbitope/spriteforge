"""
Stage 2, item 1d: run the real (user-supplied, uncurated) AI-generated character
crops through the actual model pipeline — this is the whole point of assembling
a real-input eval set, not just having the crops sit on disk. No ground truth
exists for these, so there's no PSNR; the point is a qualitative/rubric read
that our own synthetic degrade() distribution cannot provide (see the
methodology review's Part 1d critique).

Pipeline per image: crude background-color matte -> alpha-aware area-downscale
to 32x32 -> VQGAN forward pass -> OKLab palette snap. Runs on CPU explicitly to
avoid contending with the in-flight v2 training marathon on MPS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spriteforge.core.resize import resize_to_target
from spriteforge.core.palette import extract_palette_kmeans, nearest_neighbor_snap
from spriteforge.model.config import get_config
from spriteforge.model.vqgan import SpriteVQGAN

RAW_DIR = Path("/Users/mwburke/projects/spriteforge/data_private/real_eval_raw")
OUT_PATH = Path("/Users/mwburke/projects/spriteforge/devlog/random_pool_samples/real_input_eval.png")
CHECKPOINT = Path("/Users/mwburke/projects/spriteforge/checkpoints_bysource_v2/lpc/vqgan_32_epoch_100.pt")


def background_color(img: np.ndarray) -> np.ndarray:
    border = np.concatenate([
        img[0:10, :].reshape(-1, 3), img[-10:, :].reshape(-1, 3),
        img[:, 0:10].reshape(-1, 3), img[:, -10:].reshape(-1, 3),
    ])
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return colors[np.argmax(counts)]


def crude_matte(rgb_u8: np.ndarray) -> np.ndarray:
    """Rough alpha matte from background-color distance. Not a real matting model —
    this is a stand-in for a preprocessing step the product pipeline doesn't have yet
    (see devlog note): real user images arrive with opaque backgrounds, not alpha."""
    bg = background_color(rgb_u8)
    dist = np.linalg.norm(rgb_u8.astype(np.float32) - bg.astype(np.float32), axis=-1)
    alpha = np.clip((dist - 15) / 25.0, 0.0, 1.0).astype(np.float32)  # soft threshold
    rgb = rgb_u8.astype(np.float32) / 255.0
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def load_model(checkpoint_path: Path) -> SpriteVQGAN:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = get_config(ckpt.get("config_name", "size_32"))
    model = SpriteVQGAN(config).to("cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    model = load_model(CHECKPOINT)
    paths = sorted(RAW_DIR.glob("*.png"))
    print(f"[*] {len(paths)} real input images found")

    rows = []
    for p in paths:
        rgb_u8 = np.array(Image.open(p).convert("RGB"))
        rgba = crude_matte(rgb_u8)
        rgba[..., :3] = rgba[..., :3] * rgba[..., 3:4]  # premultiply, matches training convention
        input_32 = resize_to_target(rgba, target_size=32, method="area")

        x = torch.from_numpy(input_32).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            pred, _, _ = model(x)
        pred_np = pred[0].permute(1, 2, 0).numpy()

        # Palette extracted from the model's OWN (already-hallucinated) output —
        # this only cleans up speckle, it cannot fix a wrong hue.
        own_palette = extract_palette_kmeans(pred_np, k=16)
        snapped_own = nearest_neighbor_snap(pred_np, own_palette)

        # Palette extracted from the SOURCE input instead — tests whether the
        # hallucination is only a continuous-color-space artifact that a
        # source-conditioned snap could recover, or something deeper.
        source_palette = extract_palette_kmeans(rgba, k=16)
        snapped_source = nearest_neighbor_snap(pred_np, source_palette)

        def to_u8(img):
            return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)

        row = np.concatenate(
            [to_u8(input_32), to_u8(pred_np), to_u8(snapped_own), to_u8(snapped_source)], axis=1
        )
        rows.append(row)

    # Pad to same width, stack vertically with separators
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 4), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
        padded.append(np.full((2, max_w, 4), 128, dtype=np.uint8))

    grid = np.concatenate(padded, axis=0)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, mode="RGBA").save(OUT_PATH)
    print(f"[+] saved {len(paths)}-row comparison grid to {OUT_PATH}")
    print("columns: input(32x32 matted) | raw model output | snap-to-own-palette | snap-to-source-palette")


if __name__ == "__main__":
    main()
