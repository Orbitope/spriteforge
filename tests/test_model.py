# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Unit tests for VQ-GAN model architecture, Vector Quantizer, Discriminator, and Losses (Phase 4).
"""

import pytest
import torch
from spriteforge.model.config import get_config, CONFIG_16, CONFIG_32, CONFIG_48
from spriteforge.model.vqgan import SpriteVQGAN, VectorQuantizerEMA
from spriteforge.model.discriminator import PatchDiscriminator
from spriteforge.model.losses import SobelEdgeLoss, SpriteReconstructionLoss


def test_config_retrieval():
    assert get_config("32").target_size == 32
    assert get_config(16).target_size == 16
    assert get_config("48").target_size == 48
    with pytest.raises(ValueError):
        get_config("64")


def test_vqgan_forward_32():
    config = CONFIG_32
    model = SpriteVQGAN(config)
    model.eval()

    x = torch.rand(2, 4, 32, 32)
    with torch.no_grad():
        out, vq_loss, indices = model(x)

    assert out.shape == (2, 4, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert isinstance(vq_loss, torch.Tensor)
    # CONFIG_32 uses num_downsamples=1 (32x32 -> 16x16 latent grid), per its own
    # comment in config.py — deliberately shallow to preserve 1-pixel outlines.
    assert indices.shape == (2, 16, 16)


def test_vqgan_forward_16_and_48():
    for conf in [CONFIG_16, CONFIG_48]:
        model = SpriteVQGAN(conf)
        x = torch.rand(1, 4, conf.target_size, conf.target_size)
        out, vq_loss, indices = model(x)
        assert out.shape == (1, 4, conf.target_size, conf.target_size)


def test_patch_discriminator():
    critic = PatchDiscriminator(in_channels=4, hidden_channels=16)
    x = torch.rand(2, 4, 32, 32)
    logits = critic(x)
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1  # 1 channel output map

    features = critic.extract_features(x)
    assert len(features) > 0
    for feat in features:
        assert feat.shape[0] == 2


def test_sobel_edge_loss():
    loss_fn = SobelEdgeLoss()
    pred = torch.rand(2, 4, 32, 32)
    target = torch.rand(2, 4, 32, 32)
    loss = loss_fn(pred, target)
    assert loss.item() >= 0.0

    # Identical tensors should have 0 edge loss
    zero_loss = loss_fn(pred, pred)
    assert zero_loss.item() == pytest.approx(0.0, abs=1e-5)


def test_sprite_reconstruction_loss():
    loss_fn = SpriteReconstructionLoss(edge_weight=0.5, alpha_weight=1.0)
    pred = torch.rand(2, 4, 32, 32)
    target = torch.rand(2, 4, 32, 32)
    loss = loss_fn(pred, target)
    assert loss.item() >= 0.0
