# Spriters Resource scraper (humanoid character sources, oga replacement)

## Context

Dropped oga from the training pipeline after v2 review — its 932-image dataset was 20-50x smaller than papi/lpc/fe and produced mean speckle 68x worse than fe on the random-pool methodology (see devlog/2026-07-08-random-pool-samples.md). Prioritized sourcing larger, more consistent humanoid-specific datasets instead.

Found two candidates:
- **Battle for Wesnoth** (github.com/wesnoth/wesnoth): 7,118 GPL/CC-BY-SA unit sprites, 72x72, consistent fantasy pixel art, but has a team-color (magenta placeholder) system requiring a remap step before use — fixable but adds complexity.
- **The Spriters Resource** (spriters-resource.com): 6 major games (FF6, Chrono Trigger, Zelda ALTTP, Mega Man X, Castlevania AoS, Castlevania DoS) with ~800 individually-labeled character sheets total, organized by category ("Playable Characters", "Enemies & Bosses" vs. "Tilesets", "Backgrounds", "Cutscenes"), directly scrapable, same legal category as fe ("private research only, commercial IP").

Chose Spriters Resource for the larger volume and cleaner per-game structure (no remap step).

## Scraper implementation

**File**: `spriteforge/data/scrapers.py`, section 10 (new), function `download_spriters_resource_games()`.

**Features**:
- Live scraping by game slug (e.g. `"snes/mmx"` → `spriters-resource.com/snes/mmx/`)
- Per-asset page fetch to extract title and image URL (title format: "PLATFORM - Game - Category - Name.png")
- Category filtering: keep `["playable character", "non-playable", "npc", "guest character", "enem", "boss"]`, exclude `["tileset", "background", "map", "cutscene", "effect", "misc", "unused", "title", "font", "ui", "world of"]`
- Local cache in `data_private/.cache/spriters_resource/` to avoid re-fetching (rate-polite, important given potentially hundreds of HTTP requests)
- Rate limiting: configurable `request_delay` (default 0.3s per request)
- Alpha-contour slicing (`slice_by_alpha_contours`) with phash deduplication
- Polite User-Agent headers on all requests

**Configured game slugs** (in `SPRITERS_RESOURCE_GAME_SLUGS`):
1. `snes/ff6` — 181 assets
2. `snes/chronotrigger` — 260 assets
3. `snes/legendofzeldaalinktothepast` — 73 assets
4. `snes/mmx` — 51 assets
5. `game_boy_advance/cvaos` — 99 assets
6. `ds_dsi/cstlevniadawnofsorrow` — 136 assets
**Total**: 800 raw asset sheets

## Critical bug: opaque composite reference sheets

**Found during smoke-testing**: many "Playable Characters" sheets (confirmed for MMX's X, Vile, Sigma, all 6 Mavericks, plus LPC's full character source lists) are **not floating-transparent multi-frame sheets** — they're **fully opaque composite reference images** with multiple poses laid out on a uniform background, text labels, color swatches, and data tables baked directly into the pixels.

`slice_by_alpha_contours` assumes a floating-transparent background (e.g. 74% transparent, per the Bosspider/Enemies 1 sheets that work correctly). On a 0% transparent opaque sheet, it silently area-downscales the **entire composite image** into one 32×32 blob of noise instead of failing loudly or recovering individual frames.

**Test results** (MMX sheet inspection):
- Vile (1140×1115 px): 0% transparent, but after background-keying at distance<15 in RGB: 77% of pixels match background, leaving ~140 individual frames (pose variants, equipment variants, team-color swaps) cleanly separated ✓
- X (1420×3294 px): 0% transparent, background-keying recovered 2 blob slices (not actually separate poses, just upper/lower halves of one giant layout — needs a better slicing strategy for this specific sheet structure, or just skip it) ✗
- Playable Characters sheets across MMX: most are 0% transparent; only 2 of 23 had real transparency (Bosspider 74%, Enemies 1 74%)

**Implication**: without a fix, ~70% of MMX's "character" sheets would silently corrupt into noise instead of yielding usable frames. This would happen invisibly at scale, not during smoke-test.

## Fix: background-color-keying with safety filter

**User suggestion**: instead of discarding opaque sheets, extract their background color (sample borders), convert matching pixels to transparent, then slice.

**Implementation plan** (not yet wired):
1. Detect transparent_frac < 5% (currently just skips; will instead trigger fallback path)
2. Sample the most-frequent color from the border pixels (top 10px, bottom 10px, left 10px, right 10px rows/cols)
3. Create a distance mask: `alpha = (distance_to_border_color >= threshold).astype(np.float32)`
4. Re-slice with `slice_by_alpha_contours(rgba_with_keyed_alpha, min_area=36)`
5. **Safety filter**: if result is <5 slices or any slice is >50% of sheet dimensions (indicating a degenerate whole-sheet downscale), skip this sheet entirely rather than risk polluting the dataset with a single huge noise blob

**Tested manually on Vile**:
- Border color: `[112, 136, 168]` (blueish-gray)
- Threshold: distance < 15 in RGB
- Result: 143 individual slices after contour-finding
- Visual inspection: clean, recognizable robot poses (Vile across ~8 different armor configurations + action frames)

**Not yet tested on X** (needs the full implementation to see if the safety filter correctly rejects it as degenerate).

## Remaining work

1. **Wire background-keying into scraper**: update the `transparent_frac < 0.05` branch to call a new helper `_sr_keyed_alpha_from_png_bytes(png_bytes)` that returns either transparent-keyed RGBA or None (skip).
2. **Add safety filter**: after slicing, check if `len(slices) < 5` or any slice is oversized (> 50% sheet dims), skip if so.
3. **Full-scale test**: run on all 6 game slugs, sample a few outputs from each to verify category filtering + slicing + safety filter work together.
4. **Optional**: add a logging flag to report how many sheets were skipped due to safety filter (useful for tuning threshold if needed).

## Quota approaching

Current session approaching context limit. Next session should:
1. Add the background-keying helper and wire it into the scraper
2. Re-run smoke test on MMX to verify it recovers ~140 Vile frames now instead of 15
3. Run full scrape on all 6 games with real output directory
4. Spot-check 10-20 random slices from different games visually
5. If good: save to `data_private/train_32_sr_combined/` and update `scripts/train_v2_sources.sh` to replace oga with sr
6. Plan next training run (depends on whether we still want to retrain v2 or skip to E1 palette-classification)
