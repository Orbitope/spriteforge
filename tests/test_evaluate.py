# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

import numpy as np

from spriteforge.train.evaluate import compute_speckle_score


def _flat_rgba(color: tuple[float, float, float], size: int = 32) -> np.ndarray:
    img = np.zeros((size, size, 4), dtype=np.float32)
    img[..., 0] = color[0]
    img[..., 1] = color[1]
    img[..., 2] = color[2]
    img[..., 3] = 1.0
    return img


def test_speckle_score_zero_for_identical_flat_images():
    target = _flat_rgba((0.5, 0.3, 0.8))
    pred = target.copy()
    assert compute_speckle_score(pred, target) == 0.0


def test_speckle_score_zero_when_prediction_also_flat():
    """Flat GT + flat (but differently-colored) prediction: no local noise, even though the
    colors don't match — speckle measures noise, not color accuracy."""
    target = _flat_rgba((0.5, 0.3, 0.8))
    pred = _flat_rgba((0.1, 0.9, 0.2))
    assert compute_speckle_score(pred, target) == 0.0


def test_speckle_score_high_for_salt_and_pepper_noise():
    target = _flat_rgba((0.5, 0.5, 0.5))
    rng = np.random.default_rng(0)
    pred = target.copy()
    noise = rng.uniform(0.0, 1.0, size=pred[..., :3].shape).astype(np.float32)
    pred[..., :3] = noise
    score = compute_speckle_score(pred, target)
    assert score > 0.01, f"expected high speckle score for pure noise, got {score}"


def test_speckle_score_ignores_non_flat_regions():
    """A GT with real high-frequency detail (checkerboard) should not be penalized for the
    prediction failing to match pixel-for-pixel there — only flat regions are scored."""
    size = 32
    target = np.zeros((size, size, 4), dtype=np.float32)
    target[..., 3] = 1.0
    checker = (np.indices((size, size)).sum(axis=0) % 2).astype(np.float32)
    for c in range(3):
        target[..., c] = checker
    # prediction is a totally different checkerboard phase (maximally "wrong" in the
    # detailed region) but the metric should still read as low/zero since GT has no flat area
    pred = target.copy()
    pred[..., :3] = 1.0 - pred[..., :3]
    assert compute_speckle_score(pred, target) == 0.0


def test_speckle_score_higher_for_more_speckle():
    """Monotonicity sanity check: more noise -> higher score."""
    target = _flat_rgba((0.4, 0.4, 0.4))
    rng = np.random.default_rng(1)

    mild = target.copy()
    mild[..., :3] += rng.normal(0, 0.02, size=mild[..., :3].shape).astype(np.float32)
    mild = np.clip(mild, 0.0, 1.0)

    heavy = target.copy()
    heavy[..., :3] += rng.normal(0, 0.3, size=heavy[..., :3].shape).astype(np.float32)
    heavy = np.clip(heavy, 0.0, 1.0)

    assert compute_speckle_score(mild, target) < compute_speckle_score(heavy, target)
