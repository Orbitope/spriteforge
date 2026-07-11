"""
E1: palette-index classification model (methodology review, Part 3 option A).

Reframes restoration as dense per-pixel classification over a fixed-size palette
instead of continuous RGB regression. Argmax at inference is *truly* discrete —
no possible off-palette bleeding, no post-hoc snap step, no GAN required. See
devlog/2026-07-08-random-pool-samples.md's "Diagnostics" section for why: the
VQ-GAN's continuous decoder speckles even on a clean, undegraded round-trip
(D1), which this architecture cannot do by construction.

Conditioning: the target palette (K colors in [0, 1] RGB, extracted via
spriteforge.core.palette.extract_palette_kmeans or a fixed per-source palette)
is injected via FiLM (feature-wise linear modulation) at the bottleneck, so one
network serves any palette rather than baking in a fixed color set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PaletteUNetConfig:
    in_channels: int = 4
    hidden_channels: int = 64
    num_colors: int = 32  # classification classes = num_colors + 1 (transparent)


class FiLM(nn.Module):
    """Maps a flattened (K, 3) palette to per-channel (gamma, beta) for a feature map."""

    def __init__(self, palette_colors: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(palette_colors * 3, out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * 2, out_channels * 2),
        )
        self.out_channels = out_channels

    def forward(self, feat: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
        # palette: (B, K, 3) -> (B, K*3)
        gamma_beta = self.net(palette.flatten(1))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x


class PaletteUNet(nn.Module):
    """Small single-downsample UNet: 32x32 -> 16x16 bottleneck -> 32x32.

    Shallow by design, matching the project's existing shallow-encoder rationale
    (spriteforge/model/config.py's docstring) — at 32x32 a deep encoder destroys
    the spatial detail needed for 1-pixel outlines.
    """

    def __init__(self, config: PaletteUNetConfig | None = None):
        super().__init__()
        self.config = config or PaletteUNetConfig()
        h = self.config.hidden_channels
        num_classes = self.config.num_colors + 1  # + transparent class

        self.enc1 = _ConvBlock(self.config.in_channels, h)
        self.down = nn.Conv2d(h, h * 2, kernel_size=4, stride=2, padding=1)  # 32 -> 16
        self.bottleneck = _ConvBlock(h * 2, h * 2)
        self.film = FiLM(self.config.num_colors, h * 2)
        self.up = nn.ConvTranspose2d(h * 2, h, kernel_size=4, stride=2, padding=1)  # 16 -> 32
        self.dec1 = _ConvBlock(h * 2, h)  # concat skip: h (up) + h (enc1 skip)
        self.head = nn.Conv2d(h, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, H, W) degraded input RGBA in [0, 1].
            palette: (B, K, 3) per-sample palette colors in [0, 1], K == config.num_colors.

        Returns:
            logits: (B, num_colors + 1, H, W) — argmax over dim 1 gives the
            predicted class per pixel (index num_colors == transparent).
        """
        skip = self.enc1(x)
        z = self.down(skip)
        z = self.bottleneck(z)
        z = self.film(z, palette)
        z = self.up(z)
        z = torch.cat([z, skip], dim=1)
        z = self.dec1(z)
        return self.head(z)

    def decode_to_rgba(self, logits: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
        """Argmax classes -> RGBA image using the same palette used for conditioning.
        Truly discrete: every output pixel is exactly a palette color or fully transparent.
        """
        b, _, h, w = logits.shape
        classes = torch.argmax(logits, dim=1)  # (B, H, W)
        transparent_idx = self.config.num_colors
        is_transparent = classes == transparent_idx
        clamped = torch.clamp(classes, max=self.config.num_colors - 1)  # index into palette

        # Gather per-pixel colors: palette (B, K, 3), clamped (B, H, W) -> (B, H, W, 3)
        flat_idx = clamped.view(b, -1, 1).expand(-1, -1, 3)
        colors = torch.gather(palette, 1, flat_idx).view(b, h, w, 3)
        colors = colors.permute(0, 3, 1, 2)  # (B, 3, H, W)

        alpha = (~is_transparent).float().unsqueeze(1)  # (B, 1, H, W)
        colors = colors * alpha  # zero RGB where transparent, matches training convention
        return torch.cat([colors, alpha], dim=1)
