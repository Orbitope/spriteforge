"""
Head-to-head on the real eval set: legacy-degradation sr checkpoint vs the new
ai_style-degradation sr checkpoint. Full deterministic-first pipeline per image:

    real input -> flood matte -> downscale 32 -> VQ-GAN -> source-palette snap

Renders a grid: [real@32 | legacy output | ai_style output] for N images.

Usage:
    .venv/bin/python3 scripts/eval_aistyle_vs_legacy.py
"""
from __future__ import annotations
import sys, glob
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spriteforge.model.config import get_config
from spriteforge.model.vqgan import SpriteVQGAN
from spriteforge.core.resize import resize_to_target
from spriteforge.core.palette import (
    remove_background_flood, extract_palette_kmeans, nearest_neighbor_snap,
)

# (label, checkpoint) — any that don't exist yet are skipped, so this runs as soon
# as at least the legacy model is present and fills in more columns over time.
MODEL_SPECS = [
    ("legacy sr", ROOT / "checkpoints_sr_variants" / "sr" / "vqgan_32_epoch_100.pt"),
    ("ai_style (span)", ROOT / "checkpoints_sr_aistyle" / "vqgan_32_epoch_100.pt"),
    ("ai_style (crisp)", ROOT / "checkpoints_sr_aistyle_crisp" / "vqgan_32_epoch_100.pt"),
]
REAL_DIR = ROOT / "data_private" / "real_eval_raw"
OUT = ROOT / "checkpoints_sr_aistyle" / "aistyle_vs_legacy.png"

N = 8          # images to show
SCALE = 7      # nearest-neighbor upscale for viewing
SNAP_K = 12


def load_model(ckpt_path: Path) -> SpriteVQGAN:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = get_config(ckpt.get("config_name", "32"))
    m = SpriteVQGAN(config)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def load_rgba(f: str) -> np.ndarray:
    return np.array(Image.open(f).convert("RGBA"), dtype=np.float32) / 255.0


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def up(rgb: np.ndarray, s: int = SCALE) -> np.ndarray:
    return cv2.resize(to_u8(rgb), (rgb.shape[1] * s, rgb.shape[0] * s), interpolation=cv2.INTER_NEAREST)


def checker(rgba: np.ndarray) -> np.ndarray:
    h, w = rgba.shape[:2]
    c = 8
    yy, xx = np.mgrid[0:h, 0:w]
    pat = (((yy // c + xx // c) % 2) * 0.2 + 0.6).astype(np.float32)[..., None]
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + pat * (1 - a)


def run_vqgan(model: SpriteVQGAN, resized_in: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(resized_in).permute(2, 0, 1).unsqueeze(0).contiguous()
    with torch.no_grad():
        out, _, _ = model(t)
    return out.squeeze(0).permute(1, 2, 0).cpu().numpy()


def snap(rgba: np.ndarray, k: int = SNAP_K) -> np.ndarray:
    pal = extract_palette_kmeans(rgba, k=k)
    return nearest_neighbor_snap(rgba, pal)


def main() -> int:
    models = []
    for label, path in MODEL_SPECS:
        if path.exists():
            models.append((label, load_model(path)))
            print(f"[+] loaded {label}: {path}")
        else:
            print(f"[.] skip {label} (not trained yet): {path}")
    if not models:
        print("[!] no checkpoints available")
        return 1
    ts = get_config("32").target_size

    # Pick a subset that SPANS the sharpness spectrum (mushy complex-downscales ->
    # crisp near-sprites), so the head-to-head exposes each model's behavior at both
    # ends of the real-input distribution rather than only the easy cases.
    all_files = sorted(glob.glob(str(REAL_DIR / "*.png")))

    def sharpness(f: str) -> float:
        rgb = resize_to_target(load_rgba(f), target_size=ts)[..., :3]
        g = cv2.cvtColor(to_u8(rgb), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    ranked = sorted(all_files, key=sharpness)
    idx = np.linspace(0, len(ranked) - 1, N).round().astype(int)
    files = [ranked[i] for i in idx]  # evenly spaced mushy -> crisp

    tile = ts * SCALE
    pad = 8
    labelh = 24
    # Column layout:
    #   real->32 (raw, WITH background — this is exactly what the models were trained
    #             to receive, since ai_style injects a background),
    #   deterministic (flood matte -> palette snap, no model — the baseline to beat),
    #   then one column per model: raw-with-bg -> VQGAN -> palette snap.
    cols = ["real->32 (bg)", "determ. matte+snap"] + [label for label, _ in models]
    rows = []
    for f in files:
        im = load_rgba(f)
        raw32 = resize_to_target(im.copy(), target_size=ts)            # WITH background
        determ = snap(remove_background_flood(im.copy()))              # matte then snap, full-res
        determ = resize_to_target(determ, target_size=ts)
        cells = [up(checker(raw32)), up(checker(determ))]
        for _, model in models:
            cells.append(up(checker(snap(run_vqgan(model, raw32)))))   # feed raw-with-bg
        rows.append(np.hstack([np.pad(c, ((0, 0), (0, pad), (0, 0)), constant_values=245) for c in cells]))

    grid = np.vstack([np.pad(r, ((0, pad), (0, 0), (0, 0)), constant_values=245) for r in rows])
    hdr = np.full((labelh, grid.shape[1], 3), 245, np.uint8)
    x = 0
    for c in cols:
        cv2.putText(hdr, c, (x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        x += tile + pad
    grid = np.vstack([hdr, grid])
    cv2.imwrite(str(OUT), grid[..., ::-1])
    print(f"[+] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
