from __future__ import annotations

import numpy as np
import pytest

from spriteforge.core.degrade import degrade, DegradeRanges, _u


def test_degrade_output_shape_and_range():
    """Verify degrade returns float32 RGBA array in [0, 1] with correct shape."""
    clean = np.ones((32, 32, 4), dtype=np.float32)
    clean[..., :3] = 0.5  # Mid-gray opaque sprite

    rng = np.random.default_rng(seed=42)
    degraded = degrade(clean, rng=rng)

    assert degraded.shape == (32, 32, 4)
    assert degraded.dtype == np.float32
    assert np.all(degraded >= 0.0) and np.all(degraded <= 1.0)


def test_degrade_reproducibility():
    """Verify explicit RNG generator produces identical degraded outputs for same seed."""
    clean = np.random.uniform(0, 1, (32, 32, 4)).astype(np.float32)

    rng1 = np.random.default_rng(seed=12345)
    out1, log1 = degrade(clean, rng=rng1, return_log=True)

    rng2 = np.random.default_rng(seed=12345)
    out2, log2 = degrade(clean, rng=rng2, return_log=True)

    assert log1 == log2
    np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)


def test_degrade_coercion_from_rgb_uint8():
    """Verify degrade handles uint8 RGB (3-channel) inputs by adding opaque alpha."""
    clean_u8 = np.full((16, 16, 3), 128, dtype=np.uint8)
    degraded = degrade(clean_u8)

    assert degraded.shape == (16, 16, 4)
    assert degraded.dtype == np.float32
    assert np.all(degraded >= 0.0) and np.all(degraded <= 1.0)
