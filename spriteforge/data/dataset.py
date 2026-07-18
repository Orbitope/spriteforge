# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
PyTorch Dataset yielding (degraded, clean) sprite pairs on the fly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from spriteforge.core.degrade import degrade, DegradeRanges


class SpriteDataset(Dataset):
    """Dataset that loads clean sprites and applies synthetic degradation on the fly.

    Yields:
        tuple[torch.Tensor, torch.Tensor]: (degraded_tensor, clean_tensor) in shape (C, H, W).
    """
    def __init__(
        self,
        root_dir: str | Path,
        ranges: DegradeRanges | None = None,
        transform: Callable | None = None,
        seed: int = 42
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.ranges = ranges or DegradeRanges()
        self.transform = transform
        self.base_seed = seed
        
        self.image_paths = sorted([
            p for p in self.root_dir.glob("**/*.*")
            if p.suffix.lower() in [".png", ".bmp", ".tga", ".webp"]
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        pil_img = Image.open(path).convert("RGBA")
        clean_np = np.array(pil_img, dtype=np.float32) / 255.0

        # Deterministic seeding per sample and epoch
        rng = np.random.default_rng(seed=self.base_seed + idx * 10007)
        degraded_np = degrade(clean_np, rng=rng, ranges=self.ranges)

        # Convert (H, W, C) float32 in [0, 1] to PyTorch (C, H, W) tensor
        clean_tensor = torch.from_numpy(clean_np).permute(2, 0, 1).float()
        degraded_tensor = torch.from_numpy(degraded_np).permute(2, 0, 1).float()

        if self.transform:
            degraded_tensor, clean_tensor = self.transform(degraded_tensor, clean_tensor)

        return degraded_tensor, clean_tensor
