"""
Unit tests for SpriteDataset, VQ-GAN training loop, and evaluation metrics (Phase 5).
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from spriteforge.train.train import SpriteDataset, train_model, get_device
from spriteforge.train.evaluate import compute_psnr, evaluate_checkpoint


@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """Create a temporary directory with 4 dummy RGBA PNG sprites."""
    data_dir = tmp_path / "dummy_sprites"
    data_dir.mkdir()
    for i in range(4):
        img_data = np.random.randint(0, 256, (32, 32, 4), dtype=np.uint8)
        # Ensure alpha > 0.5
        img_data[..., 3] = 255
        Image.fromarray(img_data, mode="RGBA").save(data_dir / f"sprite_{i:03d}.png")
    return data_dir


def test_get_device():
    dev = get_device("cpu")
    assert dev == "cpu"
    dev_auto = get_device("auto")
    assert dev_auto in ["cpu", "cuda", "mps"]


def test_sprite_dataset(dummy_dataset_dir):
    ds = SpriteDataset(dummy_dataset_dir, target_size=32, apply_degradation=True)
    assert len(ds) == 4

    input_tensor, target_tensor = ds[0]
    assert input_tensor.shape == (4, 32, 32)
    assert target_tensor.shape == (4, 32, 32)
    assert input_tensor.dtype == torch.float32
    assert target_tensor.dtype == torch.float32


def test_train_model_and_evaluate(dummy_dataset_dir, tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ds = SpriteDataset(dummy_dataset_dir, target_size=32, apply_degradation=True)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    # Train for 1 epoch on CPU
    model = train_model(
        config_name="32",
        train_loader=loader,
        epochs=1,
        lr=1e-3,
        device="cpu",
        output_dir=ckpt_dir,
        save_interval=1
    )
    assert model is not None

    # Check that checkpoint was saved
    ckpt_path = ckpt_dir / "vqgan_32_epoch_001.pt"
    assert ckpt_path.exists()

    # Evaluate checkpoint
    eval_grid_path = tmp_path / "eval_grid.png"
    metrics = evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        test_loader=loader,
        device="cpu",
        output_grid_path=eval_grid_path
    )

    assert "psnr" in metrics
    assert "l1_loss" in metrics
    assert "codebook_usage" in metrics
    assert eval_grid_path.exists()


def test_compute_psnr():
    t1 = torch.ones(4, 32, 32)
    t2 = torch.ones(4, 32, 32)
    assert compute_psnr(t1, t2) == 100.0

    t3 = torch.zeros(4, 32, 32)
    psnr_diff = compute_psnr(t1, t3)
    assert 0.0 <= psnr_diff < 100.0
