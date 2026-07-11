# Devlog — 2026-07-08: Training diagnostics, a broken run, two corrupted eval sets, and palette-snap experiments

## Goal

Implement the Master Plan v4 diagnostics (codebook-orthogonality metric, per-epoch CSV
logging, periodic sample grids, before/after slider export), then retrain the VQ-GAN on all
four available sprite sources (`pmd`, `papi`, `lpc`, `fe`) to get a real read on model quality
per source.

This turned into three rounds, not one — the first two surfaced real bugs that would have
gone unnoticed if we'd only looked at aggregate numbers.

## Round 0 — diagnostics implementation

Added to `spriteforge/model/losses.py`, `spriteforge/train/train.py`, `spriteforge/train/artifacts.py`,
and `spriteforge/cli.py`:

- `OrthogonalRegularization` — **metric only**, not a training loss. The codebook
  (`VectorQuantizerEMA.embedding`) is an EMA-updated buffer with `requires_grad=False`, so it
  has no gradient path; adding it to the loss would be a silent no-op. Logged every epoch
  instead, under `torch.no_grad()`.
- Per-epoch CSV log (`training_log.csv`): loss_g, loss_d, recon, vq, ortho, active_codes.
- Sample grids every `--sample-interval` epochs, written to `<output_dir>/samples/`.
- `vqgan_32_best_recon.pt` — a second checkpoint tracking the lowest-recon-loss epoch seen,
  independent of whatever the final epoch looks like.
- `spriteforge slider` CLI command for before/after article-embed assets.
- Fixed a pre-existing bug while in this code: checkpoints save `critic_state_dict`, but
  `--resume` was reading `ckpt["disc_state_dict"]` — a key mismatch that silently no-op'd
  critic restoration on resume.

## Round 1 — first full run (50 epochs × 4 sources), and it was broken

Trained all four sources for 50 epochs each with default hyperparameters
(`disc_start_epoch=3`). Two problems surfaced from actually looking at the results instead of
trusting the summary numbers:

**1. Training instability.** `recon_loss` dropped for the first 2 epochs, then the moment the
discriminator turned on at epoch 3 — at full weight, instantly, against a critic that had
seen a single training step — reconstruction quality visibly regressed and never recovered
(1.08 at epoch 2 → 1.61 by epoch 50, worse than where it started).

**2. `test_pmd_32` (the pre-existing held-out eval set) was corrupted.** PSNR came back at
7.41 dB — catastrophically bad. Pulling up the actual eval grid image (not just the number)
showed why: the 20 "sprites" in that test set weren't sprites at all — they were PMDCollab
*portrait/mood-caption sheets* with text baked in ("CRYING", "ANGRY", "HAPPY", "DIZZY"...),
full-frame, no alpha transparency. Completely different distribution from the actual training
data. The scraper already excludes filenames containing `"portrait"`, so this wasn't from the
same pipeline that built the 50k-image training set — it's stale/mismatched data.

Lesson: **a low PSNR is ambiguous.** It can mean the model is bad, or it can mean the eval
data is bad. The only way to tell the difference is to look at the images.

## Round 2 — fixes, then relaunch at 100 epochs

- **Ramped adversarial warmup**: `total_loss_g = recon + vq + ramp * adv_weight * loss_adv + ramp * fm_weight * loss_fm`,
  where `ramp` climbs linearly from 0 to 1 over `--adv-ramp-epochs` starting at
  `--disc-start`. Validated on a 6-epoch smoke test before committing to a full run: recon
  loss now *improves* monotonically through the discriminator warmup instead of regressing.
- **Rebuilt `test_pmd_32_clean`**: random 20-image held-out sample carved directly out of the
  real `train_32_pmd` distribution (49,980 remaining for training, no leakage). Visually
  verified clean.
- **Rebuilt `test_papi_32`**: the original was "last 20 files alphabetically," which turned
  out to be dominated by one sprite variant (`papi_ove_substitute_*`) purely because of sort
  order. Replaced with a proper random sample.
- Relaunched: **100 epochs per source**, `--disc-start 10 --adv-ramp-epochs 20`.

(Bash 3.2 on macOS doesn't support associative arrays — the driver script
`scripts/train_all_sources.sh` uses `case` statements instead. Also hit a classic `while read`
gotcha: list files without a trailing newline silently drop their last line in a `while read`
loop — cost two dropped entries in the `papi`/`pmd` split rebuilds before switching to
`while read -r f || [ -n "$f" ]`.)

## Round 3 — a second corrupted eval set, found while writing the review doc

After the 100-epoch run finished, `fe` (Fire Emblem) came back with PSNR 9.38 dB — again
anomalously low relative to its training loss (which was the *best* of all four sources,
0.52). Same diagnostic instinct: pulled up the actual eval grid instead of trusting the
number. Found bold isolated letters — "M", "T", and a "TM" glyph — sitting among the real
sprites. 3 of the 20 `test_fe_32` images were fragments of a mis-sliced trademark watermark,
not sprites at all.

Rebuilt `test_fe_32_clean` the same way as `pmd` (random sample from the real
`train_32_fe` distribution, contact-sheet visually verified — this time confirming the
sample was legitimately diverse battle-sprite content, not contamination). Reran eval only
(no retraining needed, checkpoint was already trained): **PSNR 9.38 → 25.1 dB**, now the best
of the four sources.

**Also fixed:** the review-doc generator's `parse_eval_metrics` was silently returning "no
eval metrics found" for every source. NumPy 2.x reprs scalars as `np.float64(21.44)` instead
of a bare number, which `ast.literal_eval` can't parse — it was throwing and getting caught
by a bare `except Exception: return None`. Added a regex unwrap before parsing.

## Final results (100 epochs, `test_*_clean` eval sets)

| Source | PSNR | Codebook usage | Note |
|---|---|---|---|
| pmd | 21.44 dB | 94% (481/512) | |
| papi | 18.82 dB | 97% (496/512) | |
| lpc | 24.96 dB | 69% (351/512) | lower codebook use — more visually uniform domain |
| fe | 25.10 dB | 83% (427/512) | best score, after the eval-set fix |

Full per-source training curves, sample grids, and eval grids in `checkpoints_bysource/<source>/`.
The stale 50-epoch v1 run is archived at `checkpoints_bysource_v1_50ep_stale/`.

## Palette-snap post-processing experiment

Tested whether Stage A's deterministic palette-snap (`spriteforge/core/palette.py`,
`extract_palette_kmeans` + `nearest_neighbor_snap`, OKLab space) meaningfully cleans up
Stage B's (VQ-GAN's) continuous output. Built `scripts/build_review_examples.py` and
`scripts/build_showcase.py` to generate the comparison grids.

**Visually**, snapping consistently removes the speckled/muddy color noise that's the raw
model's most visible flaw — flat regions that should be one color collapse from a noisy
multi-color speckle field into a clean fill.

**Numerically, it's not a clean win**, and the reason matters:

| Palette source | k=6 | k=12 | k=16 |
|---|---|---|---|
| From degraded input (realistic) | 0.0327 | 0.0305 | 0.0296 |
| From model's own output (self) | 0.0275 | 0.0299 | 0.0280 |
| From ground truth (ceiling) | 0.0249 | 0.0292 | 0.0235 |

(raw model L1 on this run ≈ 0.026-0.030, `pmd`)

Only a *ground-truth-derived* palette reliably beats raw output on L1. Snapping using the
degraded input's own colors never wins — the input's colors are already corrupted by
blur/noise/color-jitter, so a palette built from it inherits and compounds that corruption.
Snapping from the model's own (partially-denoised) output does better than snapping from raw
input, sitting close to break-even.

**Also tested**: snapping to a fixed, pre-defined palette (`spriteforge/data/palettes/pico8.json`,
16 colors, used as a stand-in since no real production palette exists yet) instead of an
auto-inferred one. This was *worse* across all four sources (e.g. `fe`: raw L1 0.030 →
fixed-snap L1 0.091) — an unrelated palette doesn't collapse the noise, it just remaps it onto
different, still-wrong discrete colors. **A fixed palette only helps if it's actually
representative of the domain's true colors.**

The palette-source and target-k are now both configurable (`--palette-file`, `--palette-k` on
`scripts/build_review_examples.py`) — swap in the real production palette once one exists and
rerun before drawing conclusions from the PICO-8 numbers.

## Takeaways for next time

1. **A metric that looks anomalous relative to other signals (training loss, visual
   inspection) is a data-quality red flag before it's a model-quality verdict.** Both
   corrupted eval sets this round were caught exactly this way — PSNR didn't match training
   loss, so we looked at the actual pixels instead of trusting the number.
2. **L1/PSNR and perceptual quality are not the same axis.** Palette-snapping visibly cleans
   up the output while sometimes making L1 slightly worse — forcing pixels onto a small
   discrete palette can move them numerically further from a continuous ground truth even as
   it looks more correct to a human eye.
3. Smoke-test hyperparameter/architecture changes on a short run before committing to a
   multi-hour retrain — the ramped-adversarial-warmup fix was validated on 6 epochs before
   the full 100-epoch × 4-source relaunch.

## Next steps

- Attack the color-speckle problem at its source (larger codebook and/or a loss term that
  penalizes per-pixel color noise directly), rather than relying on palette-snap as a
  band-aid.
- Once a real production palette exists, rerun the fixed-palette comparison against it.
- If codebook utilization data (now logged every epoch) shows collapse on a longer run,
  revisit the Phase 6 MaskGIT fallback from the original master plan.
