# Spriters Resource dataset: completion + E1 training launch

## Final dataset stats

9,562 sprites after cleanup across 6 games:

| Game | Platform | Sprites |
|------|----------|---------|
| Chrono Trigger | SNES | 3,559 |
| Castlevania: Dawn of Sorrow | DS | 2,832 |
| Castlevania: Aria of Sorrow | GBA | 1,214 |
| Zelda: A Link to the Past | SNES | 1,117 |
| Mega Man X | SNES | 576 |
| Final Fantasy VI | SNES | 264 |
| **Total** | | **9,562** |

Location: `data_private/train_32_sr_combined/`
Test split (20 sprites, symlinks): `data_private/test_sr_32/`

## Cleanup passes run

Two post-scrape cleanup passes removed 120 garbage sprites:

1. **Green-background remnants** (>80% saturated green opaque pixels): 8 removed — primarily `Pengator_Gibdo_Zora` sheets in Zelda where the border-keying failed on a green chroma-key background.
2. **Degenerate mono-color sprites** (<2 quantized color groups): 112 removed — mostly solid-fill shadow blobs that slipped through the alpha-contour slicer's minimum-area filter.

These represent ~1.2% of the raw output (120/9,682). Remaining contamination rate estimated <2%.

## Visual spot-check (10 samples × 6 games)

- **FF6**: Mostly clean. A few decorative sheet elements (scale textures, horizontal color bars) made it through — not harmful at training scale.
- **Chrono Trigger**: Excellent. Clean recognizable characters and enemies, good pose variety.
- **Zelda ALTTP**: Good after green cleanup. Pink sprites (Arrghus boss, Kholdstare eye) are intentional character colors, not artifacts.
- **Mega Man X**: Excellent. Best alpha-quality source — all sheets were originally transparent.
- **CV: Aria of Sorrow**: Good. GBA sprites are lower-res but consistent. Lightning-bolt effect sprites are edge cases that won't affect training.
- **CV: Dawn of Sorrow**: Good. DS resolution gives the most detail. A few effect/projectile sprites included (acceptable).

## Training pipeline updates

- `scripts/train_v2_sources.sh`: added `sr` to the default source loop (replacing oga), added `train_dir_for`/`test_dir_for` cases for `sr`.

## E1 palette classification training

Launched on `data_private/train_32_sr_combined`, 100 epochs, batch_size=32, lr=2e-4, 16 colors, 64 hidden channels, MPS device.

**Bottleneck discovered**: `extract_palette_kmeans` called per-item per-epoch = ~10 min/epoch at 9k sprites. Fixed by adding `_build_palette_cache()` which pre-computes palettes + index maps to `.npz` files in `data_private/.palette_cache_16c/` on first run. Subsequent epochs load from cache → O(disk read) per item.

Cache building in progress (PID 28023), ~25 min estimated. Output: `checkpoints_palette_e1_sr/`.

## Next

- Monitor E1 training progress (checkpoints_palette_e1_sr/training_log.csv)
- After training: evaluate palette-index accuracy vs. the CPU smoke-test baseline (2.80→2.73 loss, 2%→55% accuracy at 3 epochs on 24 samples)
- If E1 quality is good: wire it into the GUI as "Palette UNet" snap mode
- Consider v3 VQ-GAN training on sr combined to benchmark humanoid-specific quality
