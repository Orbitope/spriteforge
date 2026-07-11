"""Smoke tests for `spriteforge convert` — the --palette-mode choices and cmd_convert's
handling of them previously did not match at all (choices included "k-means"/"gameboy"/"nes"
while the function only checked for "fixed"/"per-image-kmeans"/"per-image-median", and
--palette-file was never registered as an argument), so every documented mode failed at
runtime. These tests run the actual CLI end-to-end to catch that class of drift."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spriteforge.cli import main


@pytest.fixture
def sample_input(tmp_path):
    path = tmp_path / "input.png"
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 255, (48, 48, 4))).astype(np.uint8)
    img[..., 3] = 255
    Image.fromarray(img).save(path)
    return path


@pytest.mark.parametrize("mode", ["kmeans", "median-cut", "preset"])
def test_convert_runs_for_each_palette_mode(sample_input, tmp_path, mode):
    out_path = tmp_path / f"out_{mode}.png"
    ret = main([
        "convert", "-i", str(sample_input), "-o", str(out_path), "--palette-mode", mode,
    ])
    assert ret == 0
    assert out_path.exists()


def test_convert_fixed_palette_mode(sample_input, tmp_path):
    from spriteforge.core.palette import BUILTIN_PALETTES_DIR

    out_path = tmp_path / "out_fixed.png"
    ret = main([
        "convert", "-i", str(sample_input), "-o", str(out_path),
        "--palette-mode", "fixed", "--palette-file", str(BUILTIN_PALETTES_DIR / "sweetie16.json"),
    ])
    assert ret == 0
    assert out_path.exists()


def test_convert_fixed_without_palette_file_errors(sample_input, tmp_path):
    out_path = tmp_path / "out.png"
    ret = main([
        "convert", "-i", str(sample_input), "-o", str(out_path), "--palette-mode", "fixed",
    ])
    assert ret == 1
