"""Tests for E1 inference (restore_sprite): discreteness, sub-select containment,
and palette-width adaptation. The underlying UNet is tested in test_palette_unet;
here we cover the inference wrapper that the GUI/CLI reach.
"""
from __future__ import annotations

import numpy as np
import pytest

from spriteforge.model.palette_infer import restore_sprite
from spriteforge.model.palette_unet import PaletteUNet, PaletteUNetConfig

K = 8


@pytest.fixture
def model():
    # Small untrained model — restore_sprite's guarantees are structural (argmax
    # over classes + gather from palette), independent of trained weights.
    return PaletteUNet(PaletteUNetConfig(num_colors=K, hidden_channels=8))


def _sprite():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (32, 32, 4)).astype(np.float32)
    img[..., 3] = (img[..., 3] > 0.3).astype(np.float32)  # some transparent pixels
    return img


def _opaque_colors(out):
    return np.unique(np.round(out[out[..., 3] >= 0.5][:, :3], 4), axis=0)


def _each_in(colors, palette, tol=1e-4):
    for c in colors:
        assert np.min(np.sum((palette - c) ** 2, axis=1)) < tol, c


def test_restore_shape_and_binary_alpha(model):
    out = restore_sprite(model, K, _sprite(), palette=None)
    assert out.shape == (32, 32, 4)
    assert np.all(np.isin(out[..., 3], [0.0, 1.0]))


def test_restore_discrete_from_default_kmeans(model):
    """With palette=None the output is drawn only from the k-means source palette."""
    img = _sprite()
    from spriteforge.core.palette import extract_palette_kmeans

    src_pal = extract_palette_kmeans(img, k=K)
    out = restore_sprite(model, K, img, palette=None)
    _each_in(_opaque_colors(out), src_pal)


def test_restore_subset_smaller_than_k(model):
    """A sub-selected palette (fewer than K) restricts output to those colors."""
    subset = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)  # 3 < K
    out = restore_sprite(model, K, _sprite(), palette=subset)
    _each_in(_opaque_colors(out), subset)


def test_restore_palette_larger_than_k(model):
    """A palette larger than K is reclustered down to K; output stays discrete."""
    big = np.random.default_rng(1).uniform(0, 1, (20, 3)).astype(np.float32)  # 20 > K
    out = restore_sprite(model, K, _sprite(), palette=big)
    assert out.shape == (32, 32, 4)
    # At most K distinct opaque colors survive the width adaptation.
    assert _opaque_colors(out).shape[0] <= K
