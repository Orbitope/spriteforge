# Devlog — 2026-07-06 (backfill): earlier mixed-source training run

**Note: this entry is reconstructed on 2026-07-08 from leftover artifacts only — no
conversation history, notes, or commit log exists for this session.** No rationale, decisions,
or hyperparameters beyond what's directly recoverable from the files themselves. Treat
everything below as inferred, not reported.

## What's recoverable

Four sets of leftover artifacts in the repo root and `checkpoints*/`, all timestamped
2026-07-06, two days before the session documented in
[2026-07-08-training-pipeline-v2.md](2026-07-08-training-pipeline-v2.md):

- `checkpoints/vqgan_32_epoch_{005,010,015,020}.pt` (21:31–22:23) — a base training run,
  config `size_32`, standard checkpoint format (model/critic/optimizer state dicts + epoch +
  config_name — same schema this session's `train.py` still uses).
- `eval_epoch5_bg_inject.png` (21:08), `eval_epoch10_v2.png` (21:55), `eval_epoch15_v2.png`
  (22:23), `eval_epoch20_v2.png` (22:27) — eval grids at increasing epochs. The
  `bg_inject` name suggests this run specifically tested the `inject_background` degradation
  primitive (`spriteforge/core/degrade.py`) — simulating AI generators placing sprites on
  solid/noisy backgrounds instead of transparency. The `_v2` suffix on the later ones implies
  this superseded an earlier (unrecovered) v1 attempt.
- `checkpoints_finetune/vqgan_32_epoch_{002,004,006,008,010}.pt` (22:44–23:09) — a subsequent
  fine-tuning run, same schema, presumably resumed from one of the `checkpoints/` epochs.
- `eval_finetune_ep10.png` (23:11), `eval_sample_diverse.png` / `eval_sample_random1.png` /
  `eval_sample_random2.png` (23:17) — eval grids from the fine-tuned model, including
  diverse/random sampling modes matching `evaluate_checkpoint`'s existing `sample_mode`
  parameter.

## What the images show

Unlike this session's per-source split (`pmd`/`papi`/`lpc`/`fe` trained separately), this run
appears to have trained on a **mixed pool** — `eval_sample_diverse.png` shows creature sprites,
clothing/armor pieces, and weapon icons side by side in the same eval grid, suggesting a single
combined dataset rather than four separate ones.

Quality between epoch 10 and epoch 20 (`eval_epoch10_v2.png` vs `eval_epoch20_v2.png`) looks
roughly comparable by eye — no obvious dramatic improvement or degradation visible across that
window, though without the per-epoch loss log this session's diagnostics would have produced
(`training_log.csv` didn't exist yet — that was added on 2026-07-08), there's no way to confirm
this quantitatively.

## Why this matters going forward

This session (2026-07-08) rediscovered, the hard way, several things a written record would
have caught immediately: two corrupted eval sets, an adversarial-training instability. It's
plausible some of this was already known from 2026-07-06 and simply never written down. The
lesson isn't about this specific run — it's that **checkpoints and eval images without a
written log are close to worthless for future reference.** The diagnostics added on
2026-07-08 (`training_log.csv`, sample grids, this devlog convention) exist specifically to
prevent this gap from recurring.
