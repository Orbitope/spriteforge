"""
Small patch discriminator (the critic) for realistic sprite texture restoration.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """Lightweight 7x7 PatchDiscriminator for tiny resolutions."""
    def __init__(self, in_channels: int = 4, hidden_channels: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate feature maps for Feature Matching loss."""
        features = []
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        return features
