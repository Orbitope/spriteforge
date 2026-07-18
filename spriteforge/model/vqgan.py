# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
VQ-GAN reconstruction model: Encoder -> Vector Quantizer -> Decoder.
Outputs continuous RGBA in [0, 1]. Palette snapping is performed post-inference in OKLab.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spriteforge.model.config import ModelConfig


class VectorQuantizerEMA(nn.Module):
    """Vector Quantization layer with Exponential Moving Average (EMA) codebook updates."""
    def __init__(self, num_codes: int, code_dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps

        embedding = torch.randn(num_codes, code_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embedding_avg", embedding.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z shape: (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_permuted.view(-1, self.code_dim)

        # Distances: (z - e)^2 = z^2 + e^2 - 2ze
        dist = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.t())
        )

        encoding_indices = torch.argmin(dist, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_codes).float()

        if self.training:
            # Update EMA cluster size
            cluster_size = torch.sum(encodings, dim=0)
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            # Laplace smoothing of cluster size
            n = torch.sum(self.cluster_size.data)
            self.cluster_size.data = (
                (self.cluster_size.data + self.eps) / (n + self.num_codes * self.eps) * n
            )

            # Update EMA embedding average
            dw = torch.matmul(encodings.t(), flat_z)
            self.embedding_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Normalize embedding
            self.embedding.data = self.embedding_avg.data / self.cluster_size.data.unsqueeze(1)

            # Dead codebook restarting: re-initialize codes whose EMA usage dropped below 1.0
            dead_mask = self.cluster_size < 1.0
            if torch.any(dead_mask):
                num_dead = int(torch.sum(dead_mask).item())
                if flat_z.shape[0] >= num_dead:
                    rand_idx = torch.randperm(flat_z.shape[0], device=flat_z.device)[:num_dead]
                    self.embedding.data[dead_mask] = flat_z[rand_idx].detach()
                    self.embedding_avg.data[dead_mask] = flat_z[rand_idx].detach() * 1.0
                    self.cluster_size.data[dead_mask] = 1.0

        quantized = torch.matmul(encodings, self.embedding).view_as(z_permuted)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Commitment loss
        commitment_loss = F.mse_loss(quantized.detach(), z)
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, commitment_loss, encoding_indices.view(z.shape[0], z.shape[2], z.shape[3])


class SpriteVQGAN(nn.Module):
    """Shallow VQ-GAN tailored for low-pixel retro sprite restoration."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Simple encoder
        enc_layers = [
            nn.Conv2d(config.in_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        curr_channels = config.hidden_channels
        for _ in range(config.num_downsamples):
            enc_layers.extend([
                nn.Conv2d(curr_channels, curr_channels * 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            curr_channels *= 2
        enc_layers.append(nn.Conv2d(curr_channels, config.codebook_dim, kernel_size=3, padding=1))
        self.encoder = nn.Sequential(*enc_layers)

        # Quantizer
        self.quantizer = VectorQuantizerEMA(
            num_codes=config.codebook_size,
            code_dim=config.codebook_dim,
            decay=config.decay
        )

        # Simple decoder
        dec_layers = [
            nn.Conv2d(config.codebook_dim, curr_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(config.num_downsamples):
            dec_layers.extend([
                nn.ConvTranspose2d(curr_channels, curr_channels // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            curr_channels //= 2
        dec_layers.extend([
            nn.Conv2d(curr_channels, config.out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Continuous RGBA in [0, 1]
        ])
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        quantized, vq_loss, indices = self.quantizer(z)
        out = self.decoder(quantized)
        return out, vq_loss, indices
