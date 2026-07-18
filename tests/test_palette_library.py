# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""Tests for the palette library (user-dir vs bundled, save/load/import/delete).

The user palette dir is monkeypatched to a tmp path so tests never touch the
real ~/.spriteforge.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spriteforge.core import palette_library as pl


@pytest.fixture(autouse=True)
def _tmp_user_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "USER_PALETTE_DIR", tmp_path / "palettes")
    yield


def test_list_palettes_includes_builtins():
    names = {p.name for p in pl.list_palettes()}
    assert "pico8" in names
    assert all(p.source == "builtin" for p in pl.list_palettes())  # no user palettes yet


def test_save_load_roundtrip():
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    path = pl.save_user_palette("My Pal", colors)
    assert path.exists()
    got = pl.load_named("My Pal")
    assert got.source == "user"
    assert got.colors.shape == (3, 3)
    np.testing.assert_allclose(got.colors, colors, atol=1 / 255)


def test_list_includes_user_and_delete():
    pl.save_user_palette("tmp_pal", np.array([[0.5, 0.5, 0.5]], dtype=np.float32))
    assert "tmp_pal" in pl.list_user_palettes()
    assert any(p.name == "tmp_pal" and p.source == "user" for p in pl.list_palettes())
    pl.delete_user_palette("tmp_pal")
    assert "tmp_pal" not in pl.list_user_palettes()


def test_user_shadows_builtin():
    """A user palette named like a builtin resolves to the user's version."""
    custom = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    pl.save_user_palette("pico8", custom)
    got = pl.load_named("pico8")
    assert got.source == "user"
    np.testing.assert_allclose(got.colors, custom, atol=1 / 255)
    # and it appears only once in the merged list (the builtin is shadowed)
    pico_entries = [p for p in pl.list_palettes() if p.name == "pico8"]
    assert len(pico_entries) == 1 and pico_entries[0].source == "user"


def test_save_rejects_bad_shape():
    with pytest.raises(ValueError):
        pl.save_user_palette("bad", np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        pl.save_user_palette("bad", np.zeros((4, 4), dtype=np.float32))


def test_delete_unknown_raises():
    with pytest.raises(ValueError):
        pl.delete_user_palette("does_not_exist")


def test_import_hex(tmp_path):
    f = tmp_path / "p.hex"
    f.write_text("ff0000\n00ff00\n0000ff\n")
    pal = pl.import_palette(f)
    assert pal.shape == (3, 3)
    np.testing.assert_allclose(pal[0], [1, 0, 0], atol=1 / 255)


def test_import_gpl(tmp_path):
    f = tmp_path / "p.gpl"
    f.write_text("GIMP Palette\nName: Test\n# comment\n255 0 0 red\n0 128 0\n")
    pal = pl.import_palette(f)
    assert pal.shape == (2, 3)
    np.testing.assert_allclose(pal[0], [1, 0, 0], atol=1 / 255)
    np.testing.assert_allclose(pal[1], [0, 128 / 255, 0], atol=1 / 255)


def test_import_png(tmp_path):
    f = tmp_path / "strip.png"
    arr = np.zeros((2, 2, 4), np.uint8)
    arr[..., 3] = 255
    arr[0, 0, :3] = [255, 0, 0]
    arr[0, 1, :3] = [0, 255, 0]
    arr[1, 0, :3] = [0, 0, 255]
    arr[1, 1, :3] = [255, 255, 0]
    Image.fromarray(arr, "RGBA").save(f)
    pal = pl.import_palette(f)
    assert pal.shape[0] == 4  # four distinct opaque colors


def test_import_unsupported_raises(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("nope")
    with pytest.raises(ValueError):
        pl.import_palette(f)


def test_import_and_save(tmp_path):
    f = tmp_path / "cool.hex"
    f.write_text("#123456\n#abcdef\n")
    name, path = pl.import_and_save(f)
    assert name == "cool"
    assert path.exists()
    assert "cool" in pl.list_user_palettes()
