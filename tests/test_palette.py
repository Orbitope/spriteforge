from __future__ import annotations

import numpy as np
import pytest

from spriteforge.core.palette import (
    extract_palette_kmeans,
    extract_palette_median_cut,
    list_builtin_palettes,
    load_builtin_palette,
    nearest_neighbor_snap,
    oklab_to_rgb,
    rgb_to_oklab,
    srgb_to_linear,
    linear_to_srgb,
)


def test_srgb_linear_roundtrip():
    """Verify sRGB <-> Linear RGB conversion roundtrip."""
    rgb = np.random.uniform(0, 1, (10, 10, 3)).astype(np.float32)
    lin = srgb_to_linear(rgb)
    rec = linear_to_srgb(lin)
    np.testing.assert_allclose(rgb, rec, rtol=1e-5, atol=1e-5)


def test_oklab_rgb_roundtrip():
    """Verify exact sRGB -> OKLab -> sRGB conversion roundtrip."""
    rgb = np.random.uniform(0, 1, (20, 20, 3)).astype(np.float32)
    oklab = rgb_to_oklab(rgb)
    rec = oklab_to_rgb(oklab)
    np.testing.assert_allclose(rgb, rec, rtol=1e-4, atol=1e-4)


def test_oklab_perceptual_difference():
    """
    Verify OKLab distance differs perceptually from RGB nearest neighbor.
    In RGB space, bright yellow and bright cyan might be equidistant to white/green,
    but in OKLab perceptual lightness and chroma separate them accurately.
    """
    # Create an image of pure blue (0, 0, 1)
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[..., 2] = 1.0  # Pure blue

    # Palette: dark blue (0, 0, 0.4) vs bright cyan (0, 1, 1)
    pal = np.array([
        [0.0, 0.0, 0.4],
        [0.0, 1.0, 1.0]
    ], dtype=np.float32)

    snapped = nearest_neighbor_snap(img, pal)
    assert snapped.shape == (4, 4, 3)
    # All pixels should snap identically to one of the palette colors
    assert np.all(snapped == snapped[0, 0])


def test_kmeans_palette_extraction():
    """Verify K-Means extracts requested number of colors."""
    img = np.random.uniform(0, 1, (32, 32, 4)).astype(np.float32)
    img[..., 3] = 1.0  # Opaque
    pal = extract_palette_kmeans(img, k=8)

    assert pal.shape == (8, 3)
    assert np.all(pal >= 0.0) and np.all(pal <= 1.0)


def test_median_cut_palette_extraction():
    """Verify Median-Cut extracts requested number of colors."""
    img = np.random.uniform(0, 1, (32, 32, 4)).astype(np.float32)
    img[..., 3] = 1.0  # Opaque
    pal = extract_palette_median_cut(img, k=16)

    assert pal.shape == (16, 3)
    assert np.all(pal >= 0.0) and np.all(pal <= 1.0)


def test_list_builtin_palettes_includes_known_presets():
    names = list_builtin_palettes()
    for expected in [
        "pico8", "dawnbringer32", "sweetie16", "endesga32", "resurrect64", "gameboy",
        "palette31", "lospec500", "cc29", "mulfok32",
    ]:
        assert expected in names


@pytest.mark.parametrize("name,expected_k", [
    ("pico8", 16),
    ("dawnbringer32", 32),
    ("sweetie16", 16),
    ("endesga32", 32),
    ("resurrect64", 64),
    ("gameboy", 4),
    ("palette31", 31),
    ("lospec500", 42),
    ("cc29", 29),
    ("mulfok32", 32),
])
def test_load_builtin_palette_shapes(name, expected_k):
    pal = load_builtin_palette(name)
    assert pal.shape == (expected_k, 3)
    assert pal.dtype == np.float32
    assert np.all(pal >= 0.0) and np.all(pal <= 1.0)


def test_load_builtin_palette_unknown_name_raises():
    with pytest.raises(ValueError):
        load_builtin_palette("not_a_real_palette")
