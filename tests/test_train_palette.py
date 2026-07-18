# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spriteforge.train.train_palette import PaletteDataset, train_palette_model


@pytest.fixture
def tiny_dataset_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(4):
        img = np.zeros((32, 32, 4), dtype=np.uint8)
        img[4:-4, 4:-4, 0] = 200
        img[4:-4, 4:-4, 1] = 100 + i * 10
        img[4:-4, 4:-4, 2] = 50
        img[4:-4, 4:-4, 3] = 255
        Image.fromarray(img).save(tmp_path / f"sprite_{i}.png")
    return tmp_path


def test_palette_dataset_getitem_shapes(tiny_dataset_dir):
    ds = PaletteDataset(tiny_dataset_dir, target_size=32, num_colors=8, apply_degradation=False)
    assert len(ds) == 4
    inp, target_idx, palette = ds[0]
    assert inp.shape == (4, 32, 32)
    assert target_idx.shape == (32, 32)
    assert palette.shape == (8, 3)
    assert target_idx.dtype == __import__("torch").long


def test_train_palette_model_smoke_runs_and_saves_checkpoint(tiny_dataset_dir, tmp_path):
    out_dir = tmp_path / "ckpt"
    model = train_palette_model(
        data_dir=tiny_dataset_dir,
        epochs=2,
        batch_size=2,
        num_colors=8,
        hidden_channels=8,
        device="cpu",
        output_dir=out_dir,
        save_interval=2,
    )
    assert (out_dir / "palette_unet_epoch_002.pt").exists()
    assert (out_dir / "training_log.csv").exists()
