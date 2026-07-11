# Stage 2 — degradation pipeline fixes (methodology review, Part 1)

Context: a full methodology review (see the plan doc referenced from
[2026-07-08-random-pool-samples.md](2026-07-08-random-pool-samples.md)'s Diagnostics section)
concluded the speckle problem is mostly architecture/objective, not data — but flagged four
real, independent problems with the degradation pipeline itself (`spriteforge/core/degrade.py`)
that are worth fixing regardless, since they're cheap and orthogonal to whatever happens with
the model. This entry tracks that work as it lands, live, not retroactively.

## 1a. Multi-scale wrapper — done

Every primitive used to run directly on the 32×32 grid. For codec/optics primitives (JPEG,
blur, motion blur, noise) this is a real scale mismatch: a real photo is compressed/blurred
at source resolution, *then* downscaled — JPEG's 8×8 block grid at native resolution becomes
a barely-visible ~1px artifact after area-downscale to 32×32, not a block spanning a quarter
of the sprite like it does when applied directly at 32×32.

Added an opt-in `multiscale: bool = False` param to `degrade()`. When `True`, the
codec/optics primitives (`motion_blur`, `blur`, `noise`, `jpeg`) are deferred out of the main
per-primitive loop, run instead at a random 4-8x nearest-upscale, then area-downscaled back
via `cv2.INTER_AREA`. Verified two ways:
- Determinism test: same seed → identical output, both with and without multiscale.
- Direct visual check: forced `p_jpeg=1.0` (quality 15-25) with every other primitive
  disabled, compared old direct-32×32 JPEG vs new multiscale JPEG on the same real papi
  sprite (`/tmp/jpeg_scale_comparison.png`). Old version produced an obvious blotchy
  discolored quarter-sprite artifact; new version was barely distinguishable from clean,
  showing only fine correlated chroma noise — the actual signature of real JPEG-then-shrink.

**One bug caught during this**: my first implementation accidentally split the *entire*
per-primitive loop into two sequential loops (not just the scale-group), which silently
changed RNG draw order and primitive interleaving even for the `multiscale=False` default
path. That would have broken the controlled data comparison for whatever training run used
it next. Caught by re-reading my own diff before running anything; fixed by making
`multiscale=False` a byte-identical copy of the original single loop, with `multiscale=True`
as a fully separate branch that only fires for the four scale-sensitive primitives.

`spriteforge/train/train.py`'s `SpriteDataset.__getitem__` now passes `multiscale=True` on
every degrade() call — this is the actual production wiring, not just available-but-unused.

## 1c. Re-enabling the AI-artifact primitives — done

`local_color_drift` ("the signature AI-source artifact" per its own docstring) and
`palette_inflation` were at `p=0.0` in every single preset — disabled at some earlier point
"to keep hues faithful." Given the product explicitly targets AI-generated source images,
training on zero examples of the one artifact family that motivated the project was a real
gap, not a minor omission.

Re-enabled both at low probability so the *majority* of samples are still faithful-hued, but
the model now sees the pattern some of the time:
- `standard` (base `DegradeRanges`): `p_local_color_drift=0.15`, `p_palette_inflation=0.15`
- `realistic_low_noise`: `0.08` / `0.08` (already the "mild" preset, kept proportionally low)
- `color_shift_only`: `0.12` / `0.12`

## 1b. Rebalancing the training preset mixture — done

`SpriteDataset.__getitem__` sampled presets at `p=[0.2 standard, 0.5 realistic_low_noise,
0.3 color_shift_only]` — 80% mild, and `color_shift_only` applies zero spatial degradation
at all. That buries exactly the hard, full-severity cases the model fails on in the
random-pool sampling (see the "detailed content corrupts, flat content is fine" pattern
documented across papi and lpc in the random-pool doc).

Changed to `p=[0.45 standard, 0.35 realistic_low_noise, 0.2 color_shift_only]` — standard
(full randomized severity) is now the plurality case instead of a fifth of samples.

## 1d. Real-input eval set — done, and it surfaced a real new failure mode

The user supplied 4 genuine AI-generated character sheets (bee/insect-warrior designs, several
different art styles/generations — "Drone: Functionalist Master Cast," a bee-professions
grid, a hooded-figure grid, and an "Overworld Bee Roster" grid). A 5th sheet (an
animation-frame/multi-pose lineup) was explicitly excluded per user instruction as too
complex for this pass.

**Cropping** (`scripts/crop_real_eval_sheets.py`): background-color-distance masking +
morphological merge (7x7 ellipse dilate/erode, to join a character's own limbs/wings/weapon
without joining separate caption letters) + connected-component filtering (area, height) to
drop title/caption text, keeping only figure-sized blobs. Two sheets needed a manual
follow-up split (row/column gap detection) where characters were stacked closely enough that
the automated pass merged 2-3 into one crop — reasonable given these are genuinely messy,
non-gridded-for-slicing images, not clean sprite sheets. Final count: **44 individual crops**
in `data_private/real_eval_raw/` (one crop still contains 2 characters where no clean
separator existed even after retrying — left as-is rather than force a bad split, since a
real user upload could just as easily contain multiple figures anyway).

**Running them through the pipeline** (`scripts/eval_real_inputs.py`): resized each crop to
32x32 (alpha-aware area downscale) and ran them through the finished `lpc` v2 checkpoint
(closest existing domain: humanoid character sprites), then OKLab palette-snapped the output.
No ground truth exists for these, so this is a qualitative read, not a metric — exactly the
point of 1d. Grid saved to `devlog/random_pool_samples/real_input_eval.png` (44 rows: input |
raw output | palette-snapped).

**Finding — consistent across all 44 samples, not cherry-picked:** yellow/gold-colored
subjects are heavily hallucinated toward reddish-brown/orange/rust tones in the output,
regardless of what color they actually are in the input. Darker-toned subjects (black,
dark green, charcoal-gray) keep roughly correct hue, though still heavily speckled/noisy.
This is a **new failure mode the random-pool sampling on synthetic data never surfaced**,
for a specific, identifiable reason: `degrade()`'s hue jitter is disabled in every preset
(`hue_deg=(0.0, 0.0)` everywhere — see Part 1c above), so held-out eval and random-pool
sampling both only ever tested "does the model preserve a color that's already close to
what it saw in training," never "does it preserve an out-of-training-distribution color
under real domain shift." The lpc training data skews toward earth-tone/armor palettes;
this model appears to have learned a color prior tied to that, not general hue fidelity.

**Caveat, stated honestly rather than glossed over:** these real crops don't have an alpha
channel (flat gray sheet background, not transparency), and the product itself has no
background-removal preprocessing step yet — so `eval_real_inputs.py`'s `crude_matte()` is a
rough color-distance stand-in, not a real matting model. Some of the visible noise/mottling
at edges could be an artifact of this crude alpha estimate rather than the VQGAN itself.
The color-hallucination pattern, though, is consistent on large flat interior regions far
from any edge (e.g. a solid yellow bee thorax turning solid red-brown in the center of the
shape, nowhere near a matte boundary), so that specific finding is not explained by matting
noise. This is a genuine, new, checkpoint-specific (`lpc`) result — it has not yet been
checked against `papi`, `fe`, or `oga` checkpoints, which may behave differently given their
different training-palette distributions.

**Implication for the plan:** this reinforces Part 1c's decision to re-enable
`local_color_drift`/`palette_inflation`, but suggests hue-preserving training coverage (not
just those two primitives) is also needed — the model needs to see genuinely
out-of-training-palette colors during training and be supervised to preserve them, not just
tolerate mild local drift. Worth adding as an explicit consideration when Stage 3 (E1/E2)
training actually starts.

### Follow-up: does snapping to a source-extracted palette recover the hallucinated color?

Prompted by the question "how does it do after applying a palette shift taken from the
overall image?" — i.e., instead of the usual post-process (extract a palette from the
model's *own* output, then OKLab-snap to that, which only cleans up speckle and cannot
touch a wrong hue), extract the palette from the **source input image** instead and snap
the raw model output to that. Added as a 4th column in `eval_real_inputs.py`
(`snap-to-own-palette` vs `snap-to-source-palette`), re-run over all 44 real crops.

**Result: consistent and substantial recovery.** Across essentially every sample, snapping
to the source-extracted palette pulls color identity back to roughly correct — the yellow/
gold bees that were rendered red-brown by the raw model output become gold/tan again; green
armor stays green instead of drifting brown; gray/silver armor stays gray instead of warming
up. This works because the model's output *structure* (region boundaries, where the armor
vs. skin vs. background regions are) is still roughly right even when the *color* it painted
those regions is wrong — nearest-neighbor-in-OKLab assignment per pixel is enough to
re-color those regions correctly once given the right palette to choose from. Shape/edge
noise (speckle) is unaffected either way, since palette snapping only ever touches color, not
structure — this is a color fix, not a speckle fix, and the two remain separate problems.

**Practical implication:** source-palette-conditioned snapping (as opposed to snapping to a
palette derived from the model's own, already-corrupted output) is a real, deployable
mitigation for the color-hallucination failure mode identified above — it doesn't require
retraining and is compatible with the palette-mapping post-process infrastructure that
already exists (`spriteforge/core/palette.py`). It's not a full fix (speckle persists, and a
source palette isn't always available/well-defined for a real messy photo — an AI-generated
sheet like these has a fairly clean intended palette to extract, but a photo of a real toy or
a heavily-shaded render might not).

Checked the actual pipeline before recommending anything further (rather than assuming):
Stage B in the GUI (`spriteforge/app/gui.py`'s `StageBStudioTab.run_restoration`) runs the
VQ-GAN and exports the **raw continuous output directly — no palette snap at all**. The only
palette-snap step in the codebase lives in Stage A's `convert` CLI command
(`spriteforge/cli.py:cmd_convert`), which is a separate, purely deterministic pipeline
(downscale -> snap -> hard alpha) with no neural model involved. So there's no existing
"Stage B output, palette-snapped" path today at all — a user would have to manually feed the
Stage B `.png` export back through `convert`, at which point `img_resized` (the thing the
palette gets extracted from by default) *is* the already-hallucinated Stage B output, not the
true source. That matches the "own-palette" column above, not the better "source-palette"
one. Concretely worth doing: either (a) add a palette-snap step directly to Stage B with an
option to extract from the *original pre-restoration* input rather than the restored output,
or (b) let Stage A's `convert` accept a separate `--palette-source-image` distinct from the
image being converted. Not implemented yet — flagging as a real, concrete follow-up rather
than doing a speculative fix blind.

## Status / what's next

Stage 2 code changes (1a, 1b, 1c) are complete and unit-tested (`tests/test_degrade.py`,
`tests/test_evaluate.py` both pass, 8/8). These changes are **live in `train.py`** as of this
commit, so they'll apply automatically to any future training run — they were not applied
retroactively to the in-flight v2 marathon (`fe`/`oga`), which was already running with the
old degrade() call when this landed; that's intentional, so the v2 marathon stays a clean,
controlled comparison against v1. The next run to actually use multiscale + rebalanced
presets + re-enabled AI-artifact primitives will be Stage 3 (E1 palette-index classification
UNet, or any VQ-GAN retrain), once the v2 marathon finishes.

Item 1d (real-input eval set) remains open, and Stage 3 *training* is still gated on the v2
marathon completing (to avoid MPS contention), per the plan's sequencing note.

## E1 scaffolding (Stage 3, code only — no training yet)

While the marathon runs (CPU-only work, no MPS contention), started on E1 — the palette-index
classification UNet from the methodology review's Part 3 option A, the primary recommendation
for actually fixing speckle (not just the input distribution).

- `spriteforge/core/palette.py`: added `palette_index_map()` — same nearest-neighbor-in-OKLab
  logic as the existing `nearest_neighbor_snap`, but returns per-pixel class indices (0..K-1
  for palette colors, K for a dedicated "transparent" class) instead of colors. This is the
  classification target builder.
- `spriteforge/model/palette_unet.py` (new): `PaletteUNet` — a shallow single-downsample UNet
  (32x32 -> 16x16 bottleneck -> 32x32, ~200K params at hidden_channels=32) that takes degraded
  RGBA input plus a per-sample palette (K colors, extracted via the existing
  `extract_palette_kmeans` or a fixed source palette), FiLM-modulates the bottleneck on the
  palette, and outputs per-pixel logits over K+1 classes. `decode_to_rgba()` argmaxes and
  gathers palette colors directly — output is *provably* discrete: verified in
  `tests/test_palette_unet.py` that every opaque output pixel exactly matches a palette entry
  (distance < 1e-5), which a continuous decoder can never guarantee.
- 4 tests (`tests/test_palette_unet.py`): target-map shape/range, forward-pass shape,
  full forward+CE-loss+backward smoke test (confirms gradients flow), and the discreteness
  guarantee above. All pass. Also sanity-checked directly against a real papi sprite — full
  16/17 classes used, expected shapes throughout.
- Still open before this can train: decide whether to keep the existing Sobel edge loss as
  an auxiliary term on the decoded RGBA. Actual full training run still deferred until the
  v2 marathon finishes (MPS contention).

### Training loop — written and CPU-smoke-tested (still no real training run)

Added `spriteforge/train/train_palette.py` (`PaletteDataset` + `train_palette_model`) and
`scripts/train_palette_unet.py` (CLI entry). Resolved the palette-conditioning design
question noted above: **training uses a palette extracted from the clean ground-truth
sprite** (most accurate signal available during training) for both the classification
target and the model's FiLM conditioning. This is explicitly flagged in the module
docstring as a real train/inference gap, not swept under the rug — at inference there's no
GT, so the palette has to come from somewhere else (the degraded input itself, or a
fixed/preset palette — the same bundled presets added in
[2026-07-08-palette-presets-and-source-snap.md](2026-07-08-palette-presets-and-source-snap.md)
are a natural fit here too). No adversarial loss, no critic — per the plan, discrete
classification doesn't need a GAN to avoid speckle; cross-entropy + argmax is discrete by
construction, already verified in `tests/test_palette_unet.py`.

Smoke-tested on CPU only (the v2 marathon was still running on MPS, so a real training run
was deliberately not launched here — this was CPU-only verification, no contention): a
24-image subset of papi's training data, 3 epochs, batch size 8. Loss dropped 2.80 -> 2.73
and pixel accuracy climbed 2% -> 55% in 3 epochs on this tiny overfit-able set, confirming
gradients flow correctly end-to-end (dataset -> model -> loss -> backward -> checkpoint).
Added `tests/test_train_palette.py` (dataset shape test + a 2-epoch training smoke test) —
56/56 tests pass across the whole suite now.

A real E1 training run (full dataset, full epoch count) is still pending, gated on the v2
marathon finishing per the plan's sequencing note.
