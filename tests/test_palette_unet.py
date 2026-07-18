# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

import numpy as np
import torch

from spriteforge.core.palette import extract_palette_kmeans, palette_index_map
from spriteforge.model.palette_unet import PaletteUNet, PaletteUNetConfig


def _dummy_sprite(size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.zeros((size, size, 4), dtype=np.float32)
    # A few flat regions plus a transparent border, roughly sprite-shaped.
    img[4:-4, 4:-4, 0] = 0.8
    img[4:-4, 4:-4, 1] = 0.3
    img[4:-4, 4:-4, 2] = 0.2
    img[4:-4, 4:-4, 3] = 1.0
    img[12:20, 12:20, 0] = 0.1
    img[12:20, 12:20, 1] = 0.1
    img[12:20, 12:20, 2] = 0.9
    img[12:20, 12:20, 3] = 1.0
    img += rng.normal(0, 0.01, size=img.shape).astype(np.float32)
    img[..., 3] = np.clip(img[..., 3], 0.0, 1.0)
    return np.clip(img, 0.0, 1.0)


def test_palette_index_map_shapes_and_range():
    img = _dummy_sprite()
    palette = extract_palette_kmeans(img, k=16)
    idx_map = palette_index_map(img, palette)
    assert idx_map.shape == (32, 32)
    assert idx_map.min() >= 0
    assert idx_map.max() <= 16  # 16 = transparent class
    # Border should be transparent-classed.
    assert idx_map[0, 0] == 16


def test_palette_unet_forward_shape():
    config = PaletteUNetConfig(num_colors=16, hidden_channels=16)
    model = PaletteUNet(config)
    x = torch.rand(2, 4, 32, 32)
    palette = torch.rand(2, 16, 3)
    logits = model(x, palette)
    assert logits.shape == (2, 17, 32, 32)


def test_palette_unet_backward_smoke():
    """Full loop: image -> palette -> target indices -> forward -> CE loss -> backward."""
    config = PaletteUNetConfig(num_colors=16, hidden_channels=16)
    model = PaletteUNet(config)

    img = _dummy_sprite()
    palette_np = extract_palette_kmeans(img, k=16)
    target_idx_np = palette_index_map(img, palette_np)

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    palette = torch.from_numpy(palette_np).unsqueeze(0)
    target = torch.from_numpy(target_idx_np).unsqueeze(0)

    logits = model(x, palette)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert has_grad


def test_decode_to_rgba_is_discrete():
    """Every output pixel must be exactly a palette color (or transparent) — no bleeding."""
    config = PaletteUNetConfig(num_colors=8, hidden_channels=8)
    model = PaletteUNet(config)
    x = torch.rand(1, 4, 32, 32)
    palette = torch.rand(1, 8, 3)
    logits = model(x, palette)
    rgba = model.decode_to_rgba(logits, palette)
    assert rgba.shape == (1, 4, 32, 32)

    rgb = rgba[0, :3].permute(1, 2, 0).reshape(-1, 3)
    alpha = rgba[0, 3].reshape(-1)
    opaque_rgb = rgb[alpha > 0]
    if opaque_rgb.shape[0] > 0:
        pal = palette[0]  # (8, 3)
        dists = torch.cdist(opaque_rgb, pal)
        min_dists = dists.min(dim=-1).values
        assert torch.all(min_dists < 1e-5), "every opaque output pixel must exactly match a palette entry"
