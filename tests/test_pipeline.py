from __future__ import annotations

import numpy as np
import pytest

from spriteforge.core.alpha import despeckle_alpha, threshold_alpha
from spriteforge.core.palette import BUILTIN_PALETTES_DIR, extract_palette_kmeans, nearest_neighbor_snap
from spriteforge.core.pipeline import convert_image_to_sprite
from spriteforge.core.resize import resize_to_target


def test_stage_a_pipeline_end_to_end():
    """Verify deterministic Stage A conversion pipeline runs cleanly end-to-end without errors."""
    # Simulate high-res RGBA input (e.g. 256x256 photo)
    high_res = np.random.uniform(0, 1, (256, 256, 4)).astype(np.float32)
    # Give it a circle alpha cutout
    y, x = np.ogrid[:256, :256]
    mask = ((x - 128)**2 + (y - 128)**2) < 100**2
    high_res[..., 3] = mask.astype(np.float32)

    # 1. Resize to target 32x32
    sprite_32 = resize_to_target(high_res, target_size=32, method="area")
    assert sprite_32.shape == (32, 32, 4)

    # 2. Extract 16-color palette
    pal = extract_palette_kmeans(sprite_32, k=16)
    assert pal.shape == (16, 3)

    # 3. Snap to OKLab palette
    snapped = nearest_neighbor_snap(sprite_32, pal)
    assert snapped.shape == (32, 32, 4)

    # 4. Hard alpha & despeckle
    out = threshold_alpha(snapped, threshold=0.5)
    out = despeckle_alpha(out, min_area=2)

    assert out.shape == (32, 32, 4)
    # Alpha should be strictly binary 0.0 or 1.0
    assert np.all(np.logical_or(out[..., 3] == 0.0, out[..., 3] == 1.0))


@pytest.mark.parametrize("palette_mode", ["per-image-kmeans", "per-image-median", "preset"])
def test_convert_image_to_sprite_palette_modes(palette_mode):
    img = np.random.uniform(0, 1, (64, 64, 4)).astype(np.float32)
    out = convert_image_to_sprite(img, target_size=32, palette_mode=palette_mode, colors=16)
    assert out.shape == (32, 32, 4)


def test_convert_image_to_sprite_fixed_palette_mode():
    img = np.random.uniform(0, 1, (64, 64, 4)).astype(np.float32)
    out = convert_image_to_sprite(
        img, target_size=32, palette_mode="fixed",
        palette_file=str(BUILTIN_PALETTES_DIR / "sweetie16.json"),
    )
    assert out.shape == (32, 32, 4)


def test_convert_image_to_sprite_fixed_without_file_raises():
    img = np.random.uniform(0, 1, (64, 64, 4)).astype(np.float32)
    with pytest.raises(ValueError):
        convert_image_to_sprite(img, target_size=32, palette_mode="fixed")
