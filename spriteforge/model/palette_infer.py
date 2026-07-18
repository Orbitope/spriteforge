# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
E1 inference: run the palette-index classification UNet and reconstruct an RGBA
sprite from predicted indices.

Resolves the palette-conditioning gap flagged in spriteforge/train/train_palette.py.
At training time the FiLM palette comes from the clean ground truth; at inference
there is no GT, so the palette is sourced one of two ways:

  * "source" (default) — k-means over the (matted) input sprite itself. Adapts to
    each sprite's own colors; the natural analogue of the deterministic
    source-palette snap that already wins the VQ-GAN comparison.
  * fixed — a caller-supplied (K, 3) palette reused for every sprite (e.g. a
    curated preset or a per-project palette). Pass `palette=` to select this.

Either way the palette must have exactly `num_colors` (=K) entries, since the
FiLM conditioner was trained with that width.

Discreteness guarantee: the output only ever contains colors that are literally
in the conditioning palette (argmax over classes, then index -> palette[idx]),
plus a dedicated transparent class. Off-palette speckle is impossible by
construction — the whole point of E1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from spriteforge.core.palette import extract_palette_kmeans
from spriteforge.model.palette_unet import PaletteUNet, PaletteUNetConfig


def load_palette_unet(ckpt_path: str | Path, device: str = "cpu") -> tuple[PaletteUNet, int]:
    """Load an E1 checkpoint. Returns (model, num_colors)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_colors = int(ckpt["num_colors"])
    config = PaletteUNetConfig(
        num_colors=num_colors,
        hidden_channels=int(ckpt.get("hidden_channels", 64)),
    )
    model = PaletteUNet(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, num_colors


def _fit_palette_width(palette: np.ndarray, k: int) -> np.ndarray:
    """Coerce an arbitrary palette to exactly k rows (FiLM was trained on width k).
    Fewer -> pad by repeating the last color; more -> k-means down to k."""
    palette = np.asarray(palette, dtype=np.float32)
    if palette.shape[0] == k:
        return palette
    if palette.shape[0] < k:
        pad = np.repeat(palette[-1:], k - palette.shape[0], axis=0)
        return np.concatenate([palette, pad], axis=0)
    # too many colors: re-cluster to k in the palette's own RGB space
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=42, n_init=4).fit(palette)
    return km.cluster_centers_.astype(np.float32)


def restore_sprite(
    model: PaletteUNet,
    num_colors: int,
    input_rgba: np.ndarray,
    palette: np.ndarray | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Restore a single (H, W, 4) float32 sprite in [0, 1] with the E1 model.

    Args:
        input_rgba: matted, target-size RGBA input (transparent background, as the
            model was trained — feed a flood-matted sprite, not a raw screenshot).
        palette: (K, 3) fixed palette to condition on. If None, the palette is
            extracted k-means from `input_rgba` (the "source sprite" mode).

    Returns:
        (H, W, 4) float32 RGBA whose colors are drawn only from the palette, with a
        hard transparent class for background pixels.
    """
    if palette is None:
        palette = extract_palette_kmeans(input_rgba, k=num_colors)
    palette = _fit_palette_width(palette, num_colors)

    inp = torch.from_numpy(input_rgba).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    pal = torch.from_numpy(palette).unsqueeze(0).contiguous().to(device)
    with torch.no_grad():
        logits = model(inp, pal)               # (1, K+1, H, W)
        idx = torch.argmax(logits, dim=1)[0].cpu().numpy()  # (H, W) in [0, K]

    h, w = idx.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    opaque = idx < num_colors                  # class K == transparent
    safe_idx = np.clip(idx, 0, num_colors - 1)
    out[..., :3] = palette[safe_idx]
    out[..., 3] = opaque.astype(np.float32)
    out[..., :3] *= out[..., 3:4]              # zero color under transparent pixels
    return out
