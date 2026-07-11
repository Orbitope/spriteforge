# Spriteforge — Project Plan & Build Spec

> High-resolution image → constrained low-pixel 2D sprite (16×16, 32×32, 48×48).
> A pretrained reconstruction model (VQ-GAN) trained **only on sprites**, bridged to
> real inputs via synthetic degradation, with a deterministic palette-snap constraint.

## 0. Product summary
**What it does:** takes an arbitrary high-resolution image and produces a clean
sprite at a target size (16/32/48), constrained to either a fixed palette or a
per-image palette, with hard transparency.

**Core insight:** the hard part is not downscaling (solved) — it is the final
snap onto the "real sprite" manifold. We learn that manifold from sprites alone
via a reconstruction model, and teach it to accept real (non-sprite) inputs by
training it to invert a synthetic degradation pipeline.

## 1. Pipeline overview (end to end)
```
INFERENCE (shipped, Python binary):
  real image
    → downscale to ~2–3× target (area average)          [deterministic]
    → model inference (torch): denoise/restore to sprite [learned]
    → resize to exact target N×N                         [deterministic]
    → palette snap in OKLab (fixed or per-image)         [deterministic]
    → alpha threshold → hard transparency                [deterministic]
    → N×N indexed sprite (PNG)

TRAINING (same codebase, offline):
  clean CC0 sprite (target = ground truth)
    → degrade() → synthetic "downscaled real" input
    → VQ-GAN learns degraded → clean reconstruction
    → checkpoint saved; optionally export to ONNX later
```
