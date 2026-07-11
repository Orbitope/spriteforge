# Synthetic Degradation Rationale & Calibration Protocol

> The degradation pipeline (`core/degrade.py`) is the crux of Spriteforge: it is the **only bridge** between "trained on sprites only" and "works on real inputs."

## 1. Rationale for Primitives
Real-world high-resolution images, especially AI-generated art (Midjourney, DALL-E, Stable Diffusion) and downscaled photos, exhibit distinct artifacts when reduced toward retro sprite grids ($16\times16$, $32\times32$, $48\times48$):

1. **AI-Source Local Inconsistency:**
   - `local_color_drift`: AI images often lack flat color fills; regions that should be solid color wander smoothly in hue and brightness.
   - `palette_inflation`: High-frequency per-pixel jitter that causes near-identical colors to diverge into hundreds of unique RGB values.
   - `region_texture`: Faint hallucinations or texture in flat background/fill areas.
   - `palette_bleed`: Color mixing across adjacent flat boundary regions.
2. **Optics & Sampling Artifacts:**
   - `antialias_edges` & `blur`: Downscaling real images softens hard 1-pixel boundaries into multi-pixel anti-aliased gradients.
   - `subpixel_shift`: Real subjects rarely align perfectly to the target integer sprite grid.
   - `motion_blur`: Directional smearing from source motion or anisotropic rendering.
3. **Matte & Quantization:**
   - `alpha_soften` & `alpha_morph`: Haloed, soft, or misaligned transparency cutouts.
   - `posterize`, `jpeg`, & `noise`: Bit-depth banding, compression ringing, and sensor noise.

## 2. Phase 3 Calibration Protocol (Human Checkpoint)
Before training any VQ-GAN model, we must verify that `degrade(clean_sprite)` visually matches real downscaled inputs:

1. **Generate Side-by-Side Grid:**
   Run `spriteforge calibrate --input <real_ai_image.png> --size 32 --output grid.png`.
   This renders:
   - Left: Real high-res image downscaled deterministically to $32\times32$.
   - Right: Clean reference sprites passed through `degrade()`.
2. **Visual Inspection:**
   - Do the degraded sprites show the same degree of edge softness as the real downscaled image?
   - Is `local_color_drift` strong enough to mimic AI color wandering without destroying character identity?
   - Are alpha halos comparable?
3. **Iterative Tuning:**
   Adjust `DegradeRanges` defaults in `core/degrade.py` until the distribution of synthetic degraded sprites visually overlaps with real downscaled inputs.
4. **Commit & Lock:**
   Once verified by human inspection, log the locked range parameters and calibration notes here before proceeding to Phase 4 (VQ-GAN training).
