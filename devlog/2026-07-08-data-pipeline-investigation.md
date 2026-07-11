# Devlog — 2026-07-08: Why some sprites looked wrong (a data pipeline investigation)

*Written with an eye toward reuse in an article/video — this is the story of a bug hunt,
not just a bug report.*

## The symptom

While curating example images for a friendly showcase document, two of the four sprite
sources — `papi` (PokéAPI) and `lpc` (Universal LPC) — kept producing "recognizable" example
grids full of odd, abstract, zoomed-in content: a marble-like blue orb, a striped pattern that
turned out to be a woodgrain texture, three near-identical wheelchairs in a row. Nothing was
crashing. The numbers looked fine (`papi` trained to 18.82 dB PSNR, `lpc` to 24.96 dB — both
respectable). But the *pictures* looked wrong in a way the metrics didn't explain.

The instinct that mattered here — same one that caught the two corrupted eval sets earlier
this session — was: **don't rationalize a weird picture as "that's just what the data looks
like." Go find out why.**

## First theory (wrong): "the source data is just abstract"

Initial hypothesis: maybe `papi`'s PokéAPI content is genuinely composed of tight item/detail
crops rather than whole creatures — a legitimate stylistic property of the source, not a bug.
This felt plausible and was nearly where the investigation stopped. It would have been the
wrong conclusion.

## Second theory (right): the ingestion pipeline is over-slicing whole sprites

The tell was a single statistic: one source file, `papi_ove_110`, had produced **91 separate
"sprite" slices** during ingestion. No sprite sheet reasonably contains 91 distinct sub-images
worth training on individually — that's the signature of something being sliced far too
aggressively.

`spriteforge/data/scrapers.py`'s `download_pokeapi_repo` runs every harvested PNG through
`slice_by_alpha_contours` (`spriteforge/data/ingest.py`) — a function built for genuine sprite
*sheets*: a single image file containing a grid of many separate sprites, which needs
splitting into individual training examples (this is exactly what's needed for `pmd`
(PMDCollab) and `lpc` (Universal LPC), whose source files really are multi-frame grids).

The cached source archive was still on disk (`data_private/.cache/pokeapi_master.zip`,
~1.5GB, downloaded 2026-07-06 — a fortunate leftover from the earlier undocumented session),
so this was directly checkable without re-downloading anything. Every single matched path
turned out to be **one creature per file**, never a sheet:

- `sprites/pokemon/versions/generation-i/red-blue/1.png` — 40×40, one Pokémon (35,472 files)
- `sprites/pokemon/other/home/0.png` — 512×512, one Pokémon, official "HOME" render (part of
  5,926 files under `other/`)
- `sprites/pokemon/other/official-artwork/*.png` — same pattern

There is no multi-sprite sheet anywhere in the ~41,000 files this scraper's `mode="overworld"`
filter matches. (One more thing the investigation turned up along the way: the filter's
intended target — paths actually containing the substring `"overworld"` — matched **zero**
files. The entire 29,494-file `papi_ove_*` dataset came from the much broader `"versions"` and
`"other"` substring fallbacks instead, which is a second bug, distinct from the slicing issue.)

Running contour-based slicing on a single, whole, already-correctly-cropped 512×512 creature
render does the wrong thing: `cv2.findContours` finds every disconnected patch of opaque
pixels — antialiasing gaps, a separated tail or ear, a floating accessory — and treats each
one as its own "sprite." That's the 91-slices-from-one-file result, and it's what showed up
downstream as abstract zoomed fragments in the training data: not a stylistic property of
PokéAPI's art, but a whole sprite getting shattered into meaningless pieces by a slicing step
it should never have gone through.

**Fix identified (not yet applied):** for sources whose files are already one-sprite-per-file
(this PokéAPI dump, and likely similar "official art dump" style sources generally), skip
`slice_by_alpha_contours` entirely and pad/resize each file directly.

## A second, smaller bug: non-sprite files in `lpc`

Using the same locally-cached-archive trick (`data_private/.cache/lpc_master.zip`), a related
but distinct problem turned up in `download_lpc_humanoids`. The path filter only excludes
filenames containing `"shadow"` or `"preview"` — nothing else. That's not enough:

- `readme-images/credits-sheet.png`, `readme-images/example.png` — documentation images, not
  sprites at all — pass the filter and get contour-sliced into the training set.
- `sources/github-mark.png` — the GitHub logo. Also passes. Also gets sliced.
- `spritesheets/body/wheelchair/adult/...` — this one isn't a bug exactly: the Universal LPC
  Spritesheet Generator supports wheelchair-using character bodies as a first-class option,
  same tier as other body types. It's legitimate content, just not what "modular character
  armor" (the function's docstring) led anyone to expect. Worth knowing about, not worth
  filtering out reflexively.

**Fix identified (not yet applied):** restrict the path filter to `spritesheets/` specifically
(excludes the logo/readme content for free), and make a deliberate call on whether to keep
non-armor body-type categories like `wheelchair` in scope.

## The methodological point (the part worth keeping for later)

Both bugs were invisible from the metrics. `papi` and `lpc` trained to perfectly reasonable
PSNR numbers *despite* a meaningful fraction of their training data being algorithmically
shattered garbage or off-topic non-sprite images — the model just learned to shrug off that
fraction as noise, the same way it learns to ignore any minority-class junk. **A healthy
aggregate metric does not mean healthy training data.** The only way either bug surfaced was
generating actual images and looking at them with the specific question "does this look like
what I think it should look like" — the same discipline that caught the corrupted `pmd` and
`fe` eval sets earlier in this session.

Also notable: both root causes were fully diagnosable **without any new network access** —
the original scrape archives were still sitting in `data_private/.cache/`. Caching the raw
source archive, not just the processed output, turned out to matter for exactly this kind of
after-the-fact forensic check.

## Status

Root causes identified and written up; fixes are **not yet implemented**. Next step (separate
piece of work): apply both fixes, and evaluate new tactics/SRPG-style sprite sources with this
lesson in hand — check whether a source's files are whole-sprite-per-file or genuine sheets
*before* deciding whether to run them through contour slicing, rather than assuming.
