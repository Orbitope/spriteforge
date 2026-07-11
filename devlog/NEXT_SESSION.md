# Next Session State

## What's Running

**E1 palette classification training** (PID 28023, started 2026-07-08 ~20:10 PDT)
- Data: `data_private/train_32_sr_combined` (9,562 sprites)
- Output: `checkpoints_palette_e1_sr/`
- Log: `checkpoints_palette_e1_sr/train_stdout.log` (stdout is buffered — check `training_log.csv` for progress)
- Status at session end: epoch 3/100 complete (loss 1.57, acc 52.9%)
- Estimated completion: ~21:30 PDT (65 more minutes at ~40s/epoch on MPS)

## What Was Completed This Session

1. **Spriters Resource full 6-game scrape**: 9,682 raw sprites → 9,562 after cleanup
   - Games: FF6 (264), Chrono Trigger (3,559), Zelda ALTTP (1,117), MMX (576), CV:AoS (1,214), CV:DoS (2,832)
   - Cleanup: 8 green-background blobs, 112 mono-color degenerate slices removed
   - Test split: `data_private/test_sr_32/` (20 symlinks)

2. **Training pipeline updated**: `scripts/train_v2_sources.sh` now includes `sr` source (replacing `oga`)

3. **E1 training performance fix**: Added `_build_palette_cache()` to pre-compute k-means palettes+index-maps
   - Cache dir: `data_private/.palette_cache_16c/` (9,562 .npz files, ~35MB total)
   - Speed: ~40s/epoch vs ~10min/epoch before caching

4. **All 56 tests pass**

## Next Steps After E1 Training Completes

### 1. Evaluate E1 checkpoint
```bash
# Quick quality check: run palette-snap in E1 mode on a test image
cat checkpoints_palette_e1_sr/training_log.csv  # check final loss/acc
```
Target: loss < 1.0, pixel accuracy > 70% by epoch 100 (based on trajectory)

### 2. Wire E1 into the GUI

Currently, `gui.py` Stage B has palette snap modes: "None", "Source image (k-means)", and preset names.
Add a "Palette UNet (E1)" option that:
1. Loads `PaletteUNet` from the best checkpoint
2. Runs it on the degraded input with the source palette as FiLM conditioning
3. Argmax-decodes logits → palette indices → mapped back to RGB

The inference gap (GT palette at training vs source-derived palette at inference) may limit quality —
test this before shipping to the GUI.

### 3. Consider v3 VQ-GAN training on sr data

The v2 marathon trained on papi/lpc/fe separately. A v3 run on sr (humanoid-only, ~10k sprites)
would let us benchmark sr-specific quality. Could run alongside a v3-combined (papi+lpc+fe+sr).

### 4. Potential additional games

If the sr dataset needs more diversity, good candidate SNES games on Spriters Resource:
- `snes/secretofmana` — 3 playable characters, varied enemy types
- `snes/supermetroid` — Samus + all enemies
- `snes/finalfantasy5` — FF5 job system → high costume variety

Add to `SPRITERS_RESOURCE_GAME_SLUGS` in `spriteforge/data/scrapers.py`.

## Architecture Notes

- VQ-GAN: continuous decoder → speckle artifacts (known, quantified via D3 metric)
- E1 UNet: 32→16→32 skip-connection arch, FiLM-modulated by palette, cross-entropy loss
- Palette snap modes: source-image k-means (implemented), preset (implemented), E1 (training)
- Best v2 source so far: fe (mean D3 speckle 0.00025), then papi (0.00097), lpc (0.00196)
- sr quality unknown until trained; expected similar to fe given consistent art style

## File Locations

- SR dataset: `data_private/train_32_sr_combined/`
- Palette cache: `data_private/.palette_cache_16c/`
- E1 checkpoints: `checkpoints_palette_e1_sr/`
- v2 checkpoints: `checkpoints_bysource_v2/{papi,lpc,fe}/`
- Test splits: `data_private/test_{papi,lpc,fe,sr}_32{_v2,}/`
