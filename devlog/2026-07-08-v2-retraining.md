# Devlog — 2026-07-08: Retraining on the fixed data (in progress)

Follow-up to
[2026-07-08-scraper-fixes-and-new-source.md](2026-07-08-scraper-fixes-and-new-source.md).
The three scraper bugs are fixed, the data is re-ingested and verified, and a new
furniture/plant source exists — none of it has actually been trained on yet. This entry
tracks that run. Will be updated with final results once it completes; started as a live
entry rather than written after the fact, per a specific ask to keep the devlog current
*during* the work, not just after.

## What's training

Four sources, 100 epochs each, sequential, same validated hyperparameters as the first
successful round (`--disc-start 10 --adv-ramp-epochs 20` — the ramped-adversarial-warmup fix
from earlier in the day):

| Source | Train images | Test images (held out) | Note |
|---|---|---|---|
| `papi` | 20,604 | 20 | Bug-fixed (no more over-slicing/center-crop) |
| `lpc` | 23,010 | 20 | Bug-fixed (`spritesheets/`-only path filter) |
| `fe` | 49,125 | 20 | Bug-fixed + Map Sprites now included |
| `oga` | 932 | 20 | **New** — OpenGameArt furniture/plants/environment, never trained before |

`pmd` is *not* being retrained — no bug was found in its ingestion, and its existing
checkpoint (PSNR 21.44 dB from the first round) is still valid. Output goes to
`checkpoints_bysource_v2/` specifically so it doesn't overwrite or get confused with the
original `checkpoints_bysource/` results.

Driver: `scripts/train_v2_sources.sh`, mirrors the structure of the original
`scripts/train_all_sources.sh` but scoped to just these four sources and the new data paths.

## Pre-flight check on `oga`

Since this is a genuinely new content domain (furniture/plants/flooring, not characters) that
has never been through this training pipeline before, ran a 2-epoch smoke test before
committing to the full 100-epoch run (same due-diligence habit as everything else today):
trained without crashing, recon loss started around 3.0 (notably higher than the ~1.5-2.2
range seen on character sources) — plausibly because `oga` mixes several visually distinct
sub-styles (furniture, crops, floor textures) in one small 932-image dataset, which is a
harder reconstruction target for a single small codebook than a single coherent character
style. Also noticed `active_codes` dropped 512 → 204 between epoch 1 and 2 in just the smoke
test — worth watching over the full run for actual codebook collapse, though 2 epochs isn't
enough signal to call it a problem yet.

## Estimated timeline

`papi`/`lpc`/`fe` are similar in scale to the first round (~1.5-2 hours each based on that
run's timing). `oga` is much smaller (932 vs 20-49k images) and should take only a few
minutes. Rough total: 4-6 hours for all four sequentially.

---

## Progress log

**~50 min in:** `papi` at 79/100 epochs. Recon loss stable in the 0.81-0.89 range since epoch
25 (plateaued, not still falling — matches the healthy dip-then-plateau pattern seen on every
source in the first successful round), `active_codes` steady at 512/512 (no codebook
collapse). No failures in `run_status.log`. `lpc`, `fe`, `oga` haven't started yet — driver
script runs them sequentially.

**~58 min in — `papi` complete:** trained cleanly, evaluated cleanly. **PSNR 21.31 dB**, up
from the original buggy run's 18.82 dB, despite training on ~10k *fewer* images (20,624 vs.
30,493 — the fix removed fragmented/corrupted duplicates, not just added clean ones).
Codebook usage 89.65% (459/512). The small 20-image held-out eval grid looked clean at a
glance — every example a whole, correctly-scaled, recognizable creature, no trace of the old
zoomed-fragment problem.

**Correction, after a closer look:** the above was too generous. A held-out set of 20 images
isn't enough to trust, and a first-glance read of a compressed grid isn't real scrutiny —
both mistakes this session already learned not to make with eval *data*, repeated here with
eval *judgment*. Pulling a larger, genuinely random, unfiltered sample directly from the full
20,604-image training pool (not the small holdout) tells a different story: see
[2026-07-08-random-pool-samples.md](2026-07-08-random-pool-samples.md). Roughly half of a
12-image random draw shows heavy rainbow/multicolor speckle noise on detailed or
high-frequency content — not a minor cosmetic issue. **The scraper fixes improved the training
data (confirmed, real, measurable via the PSNR delta) but did not fix the model's speckle
problem, which remains the dominant failure mode.** These are two separate axes and shouldn't
have been conflated. `lpc` now training.

**`lpc` complete:** PSNR 22.27 dB, codebook usage 87.89% (450/512). Same honesty standard
applied immediately this time (see
[2026-07-08-random-pool-samples.md](2026-07-08-random-pool-samples.md)) — random-pool sample
shows the identical speckle pattern seen on `papi`: heavy corruption concentrated on
detailed/textured content (a dragon wing, scale armor), holding up much better on flat-color
items (boots). This is now confirmed across two independent sources with different content
styles — strong evidence the speckle is a general architecture/objective problem (see the
methodology review in the plan file), not something specific to one dataset. `fe` now
training; `oga` queued after.

**Methodology review requested and delivered separately** (plan file at
`/Users/mwburke/.claude/plans/actually-chekc-out-this-unified-key.md`, approved) — critiques
the degradation pipeline's scale mismatch, the disabled AI-artifact primitives, and the
circularity of eval-on-synthetic-degradation; assesses that Phase 6 (MaskGIT) as originally
specified would not fix the speckle since the frozen decoder still renders it; and proposes a
gated diagnostic-first experiment plan (round-trip test, GAN ablation, a real speckle metric)
before committing to a specific model alternative. Full document in the plan file; will be
copied into a proper devlog entry once the Stage 1 diagnostics land.
