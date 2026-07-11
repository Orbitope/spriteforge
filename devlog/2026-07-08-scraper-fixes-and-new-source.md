# Devlog — 2026-07-08: Fixing the scraper bugs, and a new environment/furniture source

Follow-up to
[2026-07-08-data-pipeline-investigation.md](2026-07-08-data-pipeline-investigation.md), which
identified but did not yet fix three scraper bugs. This entry covers applying those fixes,
re-ingesting, and adding a new source for environment/prop content (furniture, plants,
flooring) to pair with the existing humanoid character sources — per a specific ask: sprites
in the 16-48px range with both humanoid characters *and* background elements available, for
use in an actual game.

## The three fixes, applied

1. **`papi` (PokéAPI)** — two bugs, not one. `slice_by_alpha_contours` was removed entirely
   (every source file is already a single whole creature, confirmed last entry). But fixing
   that alone wasn't enough — a second bug surfaced on the first re-test: `pad_to_target(...,
   allow_trim=True)` **center-crops** oversized input rather than resizing it. For a 512×512
   "official artwork" render, that grabs a random 32×32 fragment of the creature, not the whole
   thing — visually identical symptom to the slicing bug, different root cause. Fixed by
   resizing (`resize_to_target`, area-average) when the source exceeds `target_size`, and only
   using the padding path for already-small native pixel art (e.g. the 40×40 gen-i sprites,
   where padding preserves crisp original pixels that resizing would blur).

2. **`lpc`** — restricted to paths under `spritesheets/`, dropping `readme-images/` and the
   GitHub logo file that were previously getting swept in and sliced.

3. **`fe` `Map Sprites`** — a third bug, found live while testing the fix for the first two:
   `slice_by_alpha_contours` returned **zero** slices on these files (confirmed empirically —
   alpha is uniformly 1.0 across the whole opaque-background strip, so there's no contour to
   find). Added `slice_vertical_strip` (`spriteforge/data/ingest.py`) — simple grid division,
   frame size = strip width, frame count = height ÷ width, a convention confirmed by sampling
   dimensions across ~200 real files (`-stand.png` files: mostly 16×48 = 3 frames; `-walk.png`
   files: mostly 32×480 = 15 frames). Also added category-path filtering (previously *none* —
   the scraper sliced every PNG in the entire ~340k-file repo, including 25k spell-effect
   animations and 16k portrait headshots that aren't battle/map sprites at all) and excluded
   the "Advance Wars Animation Ports" subfolder (vehicle sprites, not humanoid).

## Re-ingestion results (from the already-cached zips, no network needed)

| Source | Before (buggy) | After (fixed) |
|---|---|---|
| papi | 30,493 | 20,624 |
| lpc | 30,493 | 23,030 |
| fe | 20,132 | 49,145 |

`fe` grew substantially — `Map Sprites`' grid-slicing produces many frames per source file,
and this was previously entirely untapped content sitting in an archive already on disk.

New train/test splits built (`scripts/build_v2_splits.py`, random 20-image held-out set,
seed 42, zero train/test overlap confirmed) and visually spot-checked: `papi` and `fe` came
back excellent (whole, clean, correctly-scaled creatures and characters). `lpc`'s sample
included some partial-body animation frames (a sword mid-swing, a cloak fragment) — legitimate
content, not a bug, same as the "climb" arm-reach frame noted in the first devlog entry;
individual frames of a fast attack/motion animation don't always show the whole character.

**Note:** hit one process error mid-session — a `find ... -delete` command intended to clear
old *symlink* directories instead matched and deleted the freshly re-ingested `_v2` data.
Caught immediately (the delete's own output looked wrong), and fully recoverable since it only
affected already-cached, already-reproducible data — re-ran the ~7-minute ingestion again with
no real loss. Worth naming plainly here rather than glossing over it.

## New source: OpenGameArt environment/prop packs

Added `download_opengameart_packs()` (`spriteforge/data/scrapers.py`) — the first scraper in
this codebase that pulls from *individual* OpenGameArt content pages rather than one big
GitHub repo archive. Two things learned building it, both from testing against the real site
rather than assuming:

- **OpenGameArt "collections"** (e.g. `opengameart.org/content/nearly-all-the-lpc-assets-in-one-place`)
  are curated link lists, not bulk downloads — "Download pending. Collection being built,"
  permanently, is the actual page state. The real pattern is: **individual content pages**
  each have their own attachment, either a `.zip` or (on two of the eight pages tried) a
  standalone `.png` sheet with no zip wrapper at all. Built both paths.
- Verified the license explicitly: CC-BY-SA 3.0/4.0 + GPL 3.0, same family as the existing
  `lpc` character source — **not CC0**, despite `docs/DATA_SOURCES.md` explicitly listing "no
  LPC / copyleft" as something to avoid. That doc reflects an earlier, stricter intended
  policy than what this codebase actually does in practice (every non-`pmd`-CC0 source here,
  `lpc`/`fe`/`papi`, is already logged `private_research` tier, non-CC0, in
  `ASSETS_PROVENANCE.md`) — worth reconciling that doc at some point, not done here.

Harvested from 7 of 8 curated pack pages (one, `lpc-interiors`, has no downloadable attachment
of any kind — dropped from the list):

| Pack | Sprites |
|---|---|
| Crops | 249 |
| Flowers/Plants/Fungi/Wood | 206 |
| Wooden Furniture | 202 |
| Fruit Trees | 112 |
| Upholstery | 106 |
| Floors | 53 |
| House Interior & Decorations | 24 |
| **Total** | **952** |

Visually spot-checked (16-image random sample): strong majority clean and correctly cropped —
beds, dressers, armchairs, side tables, palm trees, berry bushes, flower bouquets, a pumpkin,
floor-plank textures. A small minority (roughly 1 in 16 in the sample) were ambiguous partial
crops. Good enough to use, not perfect — matches the pattern from every other source this
session: automated ingestion gets most of the way, a visual pass catches the rest.

This directly answers the "humanoid + background (furniture, plants, ground)" ask — the
humanoid side was already covered by `pmd`/`papi`/`lpc`/`fe`; this adds the missing
environment side, same 32×32 scale, compatible license tier.

## Status / not yet done

- No retraining on any of this new/fixed data yet — that's a separate multi-hour undertaking,
  intentionally not started without a separate go-ahead given how long this session already
  is.
- `lpc-interiors` still has no automated path (page has no attachment). Low priority.
- `docs/DATA_SOURCES.md`'s "avoid LPC/Spriters Resource" guidance is now clearly stale
  relative to actual practice — flagged, not fixed.
