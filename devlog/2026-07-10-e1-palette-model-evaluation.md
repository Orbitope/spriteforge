# E1 Palette-Index Model: Discrete Restoration Evaluation (2026-07-10)

## Summary

Implemented and evaluated the E1 palette-index classification model (discrete per-pixel UNet) as an alternative to VQ-GAN. E1 is **speckle-free by construction** — the architecture's argmax over a fixed palette guarantees zero off-palette color noise. K=24 variant visibly reduces posterization vs K=16, though both still lose to deterministic on overall fidelity.

**Verdict:** E1 is the only viable *learned* restoration path if speckle-free output is required. Deterministic remains the default; E1 is a secondary "learned polish" option.

---

## Context

Prior sessions established that VQ-GAN speckles on every variant tested — even at K=24 classes in the degradation profile. The root cause: the continuous decoder in VQ-GAN can emit any RGB value, and no training objective ties pixels to a discrete palette or their neighbors. Three diagnostic runs (legacy sr, ai_style span, ai_style crisp, K=16–24 degradation) all showed the same speckle defect.

The E1 model was scaffolded but never trained (see `spriteforge/model/palette_unet.py`). This session: train, infer, and evaluate.

---

## Implementation

### New Files

**`spriteforge/model/palette_infer.py`** — E1 inference module

Resolves the train/inference palette-conditioning gap:
- **Training:** FiLM palette comes from clean ground-truth sprite (accurate supervision signal).
- **Inference:** Palette sourced one of two ways:
  - `palette=None` (default) → k-means extraction from the matted input sprite. Adapts per-sprite, most natural analogue to deterministic source-palette snap.
  - `palette=<array>` → fixed (K, 3) palette reused for every sprite. Alternative mode, less flexible.

Key functions:
- `load_palette_unet(ckpt_path, device)` — Load a checkpoint, return (model, num_colors).
- `restore_sprite(model, num_colors, input_rgba, palette=None, device)` — Restore a single sprite. Output is guaranteed discrete: colors are literal entries from the palette, plus a transparent class.
- `_fit_palette_width(palette, k)` — Coerce arbitrary palette to exact K width (pad or re-cluster).

Discreteness guarantee: Output only contains palette colors (argmax → palette[idx]) + transparent. **Off-palette speckle is impossible by construction.**

**`scripts/eval_e1.py`** — Head-to-head evaluation on real eval set

Parametrized to accept `--ckpt <path>` for any E1 checkpoint. Renders a 4-column grid:
1. Real input → 32×32 (with background, as models receive it).
2. Deterministic (matte + palette snap, no model) — the baseline to beat.
3. E1 (kmeans src) — model output with per-sprite k-means palette.
4. E1 (fixed pal) — model output with a global shared palette.

Samples 8 images spanning mushy ↔ crisp from the real eval set, rendered at 7× zoom.

---

## Experiments

### K=16 Baseline

**Setup:**
- Model trained on sr_combined (9.5k sprites after filtering transparent-bg exclusions).
- 100 epochs, batch 32, lr 2e-4, CosineAnnealingLR.
- Cross-entropy loss (no adversarial; discrete by construction).
- Checkpoint: `checkpoints_palette_e1_sr/palette_unet_epoch_100.pt`.

**Results:**
- Final loss CE: 0.896
- Pixel accuracy: 0.697 (K=16 classes means harder target than simpler tasks).
- **Visual:** Speckle-free ✓. But heavily posterized: color banding, loss of shading gradients, interior detail muddied. Silhouettes read well; smooth shading collapses to flat color blocks. Farmer sprite (mushy) skin tones become yellow-tan blobs instead of gradient.

**Finding:** Discrete architecture solves the speckle defect, but K=16 is too coarse for natural shading.

---

### K=24 Improved

**Setup:**
- Same pipeline, K=24 colors (increased classification classes).
- Palette cache built once on ~26k sprites (broader glob finds more variants).
- 100 epochs.
- Checkpoint: `checkpoints_palette_e1_sr_k24/palette_unet_epoch_100.pt`.

**Results:**
- Final loss CE: 1.053 (higher loss expected; 24 classes is a harder target than 16).
- Pixel accuracy: 0.686 (slightly lower, consistent with harder target).
- **Visual:** Speckle still absent ✓. Posterization **visibly reduced**. Smoother gradients, more natural shading transitions. Yellows no longer blocky. Farmer sprite skin tones have better variation. Blacks cleaner, less fringing.

**Finding:** Extra palette colors directly address the posterization problem. K=24 is closer to viable, though still behind deterministic on overall fidelity.

---

## Visual Comparison: K=16 vs K=24

Grid columns for both:
```
[Real→32 (bg)] [Determ matte+snap] [E1 kmeans-src] [E1 fixed-pal]
```

**Key observations (K=16 → K=24):**

| Sprite | K=16 Issue | K=24 Fix |
|--------|-----------|---------|
| Mushy black creature | Blocky yellows, no shading | Smoother color transitions, more natural palette usage |
| Gold humanoid | Posterized skin tones | Better gradation in mid-tones |
| Yellow/brown armored character | Muddy interior, hard edges | Cleaner structure, smoother shading |
| Farmer (crisp) | Skin collapsed to yellow blob | Flesh tones more varied |

**Per-source vs fixed palette:**
- E1 (kmeans src) is clearly better; fixed palette adds stray dots and color errors (e.g., unexpected bright yellows along edges).
- Confirms design choice: per-sprite k-means is the right conditioning strategy.

---

## Key Findings

### 1. Discrete architecture is the fix for speckle
VQ-GAN's continuous decoder + no neighbor coherence loss = inevitable speckle. E1's argmax over a fixed palette makes speckle **impossible by construction**. Thesis confirmed on real inputs.

### 2. K=24 reduces posterization, but doesn't solve fidelity gap
More colors help (smoother shading, better detail), but the learned output still lags deterministic on:
- Mid-tone shading smoothness (deterministic palette snap is perceptually better).
- Edge crispness (deterministic matte has cleaner boundaries).

The discrete constraint is the fundamental limiting factor. E1 can't learn continuous blending; it outputs indexed palette colors.

### 3. Per-source k-means >> fixed palette
Adapting the palette to each sprite (source mode, default) is vastly better than forcing one global palette. Fixed palette adds errors and artifacts. The design choice was correct.

### 4. Deterministic still wins on fidelity
Deterministic pipeline (flood matte + downscale + k-means snap, no learned model) continues to beat every learned variant on real inputs. This is the production default.

### 5. E1 is the only learned path still in play
If users want a learned restoration that's speckle-free, E1 is it. The architecture guarantee is powerful. But it's a trade (discrete → limited fidelity). Communicate this as "speckle-free learned option" not "better than deterministic."

---

## Code Changes

### Added
- `spriteforge/model/palette_infer.py` — E1 inference (load checkpoint, restore sprite with palette conditioning).
- `scripts/eval_e1.py` — Parametrized eval script (`--ckpt <path>`). Runs K=16 and K=24 in the same loop.

### Modified
- `scripts/eval_e1.py` — Changed from hardcoded `E1_CKPT` to argparse `--ckpt` (reusable for any K).

### Training Outputs
- `checkpoints_palette_e1_sr/` — K=16 baseline (100 epochs, loss 0.896, acc 0.697).
- `checkpoints_palette_e1_sr_k24/` — K=24 improved (100 epochs, loss 1.053, acc 0.686).
- Both include `e1_vs_determ.png` (4-col grid, 8 real eval images, 7× zoom).

---

## Recommendations

### Immediate (Ship)
- **Deterministic as default.** Proven winner on real inputs; no learned model needed.
- **E1-K24 as optional secondary.** For users who prioritize speckle-free aesthetics. Caveat: less smooth shading than deterministic, but zero color noise.

### Next Experiments (If Time)
1. **K=32** — Push color palette further. May reach deterministic's smoothness without increasing model size significantly.
2. **Class-weighted CE loss** — Address yellow bias (dominant colors in sprites). Current loss treats all misclassifications equally; weighting rare classes higher might fix the color-balance blotches.
3. **[[idea-luminance-decouple]]** — Decouple structure (luminance) from color. Model predicts structure only, post-hoc deterministic palette snap. Combines learned structure with guaranteed discreteness. Architectural pivot, but worth exploring if K=32 doesn't close the gap.

### Long-term
- If learned polish is critical and continuous output is acceptable, reconsider VQ-GAN with **neighbor coherence loss** (total-variation or mode-consistency penalty). One loss term, might suppress speckle enough for acceptable output.
- **Autoregressive palette indices** (PixelCNN-style) — highest ceiling for learned discrete output. Models pixel-to-pixel coherence explicitly. Slower training and inference, but theoretically cleanest learned restoration.

---

## Files Touched

```
spriteforge/model/palette_infer.py       [NEW] Inference module for E1
scripts/eval_e1.py                       [MODIFIED] Parametrized for any checkpoint
checkpoints_palette_e1_sr/               Training output (K=16)
checkpoints_palette_e1_sr_k24/           Training output (K=24)
```

---

## Related Prior Work

- `devlog/2026-07-08-random-pool-samples.md` — D1 diagnostic showing VQ-GAN speckles on clean round-trip; E1's motivation.
- `spriteforge/model/palette_unet.py` — Existing E1 architecture scaffold.
- `spriteforge/train/train_palette.py` — Training loop (unchanged; works as-is).
- Memory: `[[idea-luminance-decouple]]` — Logged alternative direction if E1 needs more work.

---

## Conclusion

E1 palette-index model is **production-ready as a speckle-free option**, though it trades some fidelity for that guarantee. K=24 is the sweet spot between color palette size and posterization. Default recommendation: ship deterministic; offer E1 as a user option for discrete-only workflows.

The experiment closed the loop on learned restoration: VQ-GAN speckles by design, E1 is discrete by design. Next moves are refinements (more colors, loss tuning, architectural variants) or a pivot to structure-only learning ([[idea-luminance-decouple]]).
