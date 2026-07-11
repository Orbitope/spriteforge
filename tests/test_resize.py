from __future__ import annotations

import numpy as np
import pytest

from spriteforge.core.resize import alpha_aware_resize, area_average_downscale, pad_to_target, resize_to_target


def test_resize_to_target_square():
    """Verify resize_to_target outputs exact N x N dimensions."""
    img = np.random.uniform(0, 1, (100, 150, 4)).astype(np.float32)
    resized_32 = resize_to_target(img, target_size=32, method="area")
    resized_16 = resize_to_target(img, target_size=16, method="nearest")

    assert resized_32.shape == (32, 32, 4)
    assert resized_16.shape == (16, 16, 4)


def test_alpha_aware_resize_prevents_background_bleed():
    """
    Verify alpha-aware resizing does not bleed dark background colors into opaque sprite edges.
    """
    # Create a 4x4 image where center 2x2 is pure white opaque (1, 1, 1, 1)
    # and outer border is pure black transparent (0, 0, 0, 0).
    img = np.zeros((4, 4, 4), dtype=np.float32)
    img[1:3, 1:3] = 1.0

    # Resize down to 2x2 with area average
    resized = alpha_aware_resize(img, 2, 2)

    # Where alpha > 0, the RGB values should remain pure white (1.0), NOT diluted by black border!
    mask = resized[..., 3] > 0
    assert np.allclose(resized[..., :3][mask], 1.0, atol=1e-4)


def test_pad_to_target_no_resampling():
    """Verify pad_to_target center-pads without modifying original pixel values."""
    sprite = np.ones((20, 16, 4), dtype=np.float32)
    padded = pad_to_target(sprite, target_size=32)
    
    assert padded is not None
    assert padded.shape == (32, 32, 4)
    # The center 20x16 area should be exact 1.0s
    pad_y = (32 - 20) // 2
    pad_x = (32 - 16) // 2
    assert np.allclose(padded[pad_y:pad_y+20, pad_x:pad_x+16], 1.0)
    # Outer border should be pure zero
    assert np.all(padded[:pad_y, :, :] == 0.0)
    assert np.all(padded[:, :pad_x, :] == 0.0)


def test_pad_to_target_skips_oversized():
    """Verify pad_to_target returns None when sprite exceeds target_size without allow_trim."""
    sprite = np.ones((40, 40, 4), dtype=np.float32)
    assert pad_to_target(sprite, target_size=32, allow_trim=False) is None

