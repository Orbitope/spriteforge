# Spriteforge

> High-resolution image → constrained low-pixel 2D sprite (16×16, 32×32, 48×48).

Spriteforge converts arbitrary high-resolution images or AI-generated art into clean, retro 2D sprites at target sizes (16×16, 32×32, 48×48). It enforces hard transparency and deterministic palette constraints in OKLab color space, powered by a VQ-GAN trained exclusively on CC0 sprites via synthetic degradation inversion.

## Features (v1)
- **Deterministic Core (Phase 1):** Area-average downscaling, OKLab nearest-neighbor/k-means/median-cut palette snapping, and hard matte thresholding.
- **Synthetic Degradation (Phase 3):** 14 physically-motivated and AI-source degradation primitives to bridge real-world photo/AI art inputs to clean sprite manifolds.
- **Learned Restoration (Phase 4-5):** Sprites-only VQ-GAN architecture designed for tiny resolutions without codebook collapse or spatial detail destruction.
- **100% Python & Self-Contained:** Desktop app built with PySide6 and packaged via PyInstaller.

## Installation

```bash
# Clone and install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## CLI Usage

```bash
# Run deterministic conversion (Stage A baseline / no model required)
spriteforge convert --input photo.png --output sprite_32.png --size 32 --palette-mode per-image --colors 16

# Run degradation calibration grid generator
spriteforge calibrate --input photo.png --output calibration_grid.png --size 32

# Train VQ-GAN model
spriteforge train --config size_32 --data-dir ./data_processed

# Evaluate trained model
spriteforge eval --checkpoint models/spriteforge_32.pt --input test.png --output eval_grid.png
```

## License

**Dual Licensed:**

- **Open Source (AGPL v3):** Free for non-commercial use, research, and for commercial use if you open-source your modifications.
- **Commercial License:** Required for proprietary/closed-source commercial use, SaaS, and redistribution. Contact matthew.wesley.burke@gmail.com.

See `LICENSE` for full terms and `ASSETS_PROVENANCE.md` for training data provenance (CC0 sprites only).
