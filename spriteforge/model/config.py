"""
Dataclass configurations for VQ-GAN models across target sizes (16, 32, 48).

Architectural note:
At tiny resolutions (16x16), standard deep encoders destroy spatial and positional detail.
We use shallow downsampling (e.g., 1 downsample layer for 16x16 -> 8x8 latent grid) to preserve
1-pixel outlines and character identity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    target_size: int
    in_channels: int = 4
    out_channels: int = 4
    hidden_channels: int = 64
    codebook_size: int = 512
    codebook_dim: int = 64
    num_downsamples: int = 2
    use_ema: bool = True
    decay: float = 0.99


CONFIG_16 = ModelConfig(
    target_size=16,
    hidden_channels=48,
    codebook_size=256,
    codebook_dim=32,
    num_downsamples=1  # 16x16 -> 8x8 latent grid
)

CONFIG_32 = ModelConfig(
    target_size=32,
    hidden_channels=64,
    codebook_size=512,
    codebook_dim=64,
    num_downsamples=1  # 32x32 -> 16x16 latent grid
)

CONFIG_48 = ModelConfig(
    target_size=48,
    hidden_channels=64,
    codebook_size=512,
    codebook_dim=64,
    num_downsamples=2  # 48x48 -> 12x12 latent grid
)


def get_config(size_or_name: int | str) -> ModelConfig:
    if str(size_or_name) in ["16", "size_16"]:
        return CONFIG_16
    elif str(size_or_name) in ["32", "size_32"]:
        return CONFIG_32
    elif str(size_or_name) in ["48", "size_48"]:
        return CONFIG_48
    raise ValueError(f"Unknown config: {size_or_name}")
