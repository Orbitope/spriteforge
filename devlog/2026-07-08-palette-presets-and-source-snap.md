# Palette presets + source-image palette snap (application feature)

Follow-up to the real-input eval set finding (see
[2026-07-08-stage2-degradation-fixes.md](2026-07-08-stage2-degradation-fixes.md)): the model
consistently hallucinates yellow/gold subjects toward red-brown, and snapping the raw model
output to a palette extracted from the *original source image* (not the model's own output)
substantially recovers correct color identity. This entry covers making that the actual
default behavior in the app, plus adding a handful of well-regarded bundled preset palettes.

## Bundled presets

Pulled 4 well-known, well-regarded palettes' actual hex codes from their lospec.com pages
(the palette-list index page only has links, not the color data itself — had to fetch each
palette's own detail page): DawnBringer 32, Sweetie 16, Endesga 32, Resurrect 64. Added a
standard 4-shade Game Boy DMG green palette from memory (well-established, no fetch needed).
Combined with the existing PICO-8 palette, `spriteforge/data/palettes/` had 6 presets.

**Update:** user supplied a specific list of 6 palettes they like by lospec URL: 👌31,
Resurrect 64 (already had it), Lospec500, CC-29, Sweetie 16 (already had it), mulfok32. Fetched
each palette's own detail page for the actual hex codes (same reason as above — the listing
page only has links) and added the 4 new ones (`palette31.json` — named without the emoji
prefix for a clean filename/identifier, `lospec500.json`, `cc29.json`, `mulfok32.json`). Kept
the original 4 rather than replacing them, since the ask was "here's a list I like," not
"replace what's there" — 10 presets total now. Both GUI tabs and the CLI read
`list_builtin_palettes()` dynamically, so the new presets appeared in both dropdowns and
`--palette-preset`'s choices with no additional wiring. Verified: all 10 load with correct
color counts (`cc29`→29, `dawnbringer32`→32, `endesga32`→32, `gameboy`→4, `lospec500`→42,
`mulfok32`→32, `palette31`→31, `pico8`→16, `resurrect64`→64, `sweetie16`→16), both GUI combo
boxes list all 10 (headless-verified), and the CLI's `convert --palette-mode preset` runs
end-to-end against two of the new ones.

Added `list_builtin_palettes()` and `load_builtin_palette(name)` to
`spriteforge/core/palette.py` so both the CLI and GUI can discover and load these by name
without hardcoding paths.

## Fixed a real, pre-existing bug in `convert`

While wiring presets into the CLI, found that `spriteforge convert`'s `--palette-mode` was
completely broken — not a new bug, a pre-existing one. The argparse `choices` were
`["k-means", "median-cut", "gameboy", "nes"]`, but `cmd_convert`'s actual if/elif chain
checked for `"fixed"`, `"per-image-kmeans"`, `"per-image-median"` — no overlap at all, so
every documented mode hit the `else: unknown palette mode` branch and exited 1. Additionally,
`--palette-file` was referenced in the function body but was never registered as an argparse
argument, so the `"fixed"` branch (had it been reachable) would have raised an
`AttributeError` instead of the intended error message. Verified before writing anything:

```
$ spriteforge convert -i in.png -o out.png --palette-mode k-means
[!] Error: unknown palette mode k-means
```

Fixed by making the choices and the handling agree: `["kmeans", "median-cut", "preset",
"fixed"]`, registering `--palette-file` and a new `--palette-preset` (choices = the bundled
preset names), and adding a `"preset"` branch that calls `load_builtin_palette`. Verified all
four modes actually produce output now. Added `tests/test_cli_convert.py` (previously zero
CLI tests existed) so this class of "choices vs. handler" drift gets caught in CI going
forward, not just by manual poking.

`spriteforge/core/pipeline.py`'s `convert_image_to_sprite()` (used by the GUI's Stage A tab,
a separate implementation from the CLI's `cmd_convert` — pre-existing duplication, not
touched further here) got the same `preset`/`fixed` support added, with tests.

## Source-image palette snap, wired as the Stage B default

`spriteforge/app/gui.py`'s Stage B tab (`StageBStudioTab`) previously ran the VQ-GAN and
exported the raw continuous output directly — no palette-snap step existed there at all, so
there was no "default behavior" to fix, just a gap to fill. Added:

- A "Palette Snap" combo box: `None (raw output)`, `Source image (k-means)` (**default** —
  this is the fix the eval-set finding called for), `Restored output (k-means)`, one
  `Preset: <name>` entry per bundled palette, and `Custom file...` (opens a file picker,
  falls back to the default if cancelled rather than leaving a dangling selection).
- A "Max Colors" spinner for the two k-means modes.
- `run_restoration()` now keeps the raw model output (`self.restored_sprite_raw`) separate
  from the palette-snapped result (`self.restored_sprite`), and `_apply_palette_snap()`
  dispatches on the combo box selection — `"Source image (k-means)"` extracts the palette
  from `resized_in` (the true pre-restoration input at the model's target resolution),
  matching exactly what `scripts/eval_real_inputs.py` found to work well.
- `export_sprite()` now saves the palette-snapped result by default (comment updated so this
  isn't a silent behavior change future-me forgets about).
- Stage A's combo box also got the bundled presets added (`preset:<name>` entries), wired
  through `convert_image_to_sprite`'s new `palette_preset` parameter, for consistency across
  both tabs — Stage A didn't have the "own output vs. source" ambiguity Stage B had (it only
  ever operates on one image, no model in the loop), so this was just about exposing the
  same presets, not a default-behavior fix.

**Verification**: no live PySide6 display in this environment, so verified headlessly instead
of skipping it — `QT_QPA_PLATFORM=offscreen`, instantiated both `StageBStudioTab` and
`StageAStudioTab` directly, confirmed the combo box defaults and full option lists, and ran
`_apply_palette_snap()` / `convert_sprite()` through every mode (`None`, source-kmeans,
output-kmeans, two different presets) against synthetic arrays — all produced correctly
shaped output. This confirms the wiring is correct; it does not confirm the visual/UX
polish (label alignment, dialog behavior) since no on-screen rendering was observed.

Full test suite: 49/50 pass (the one failure, `test_vqgan_forward_32` expecting an 8x8 latent
grid instead of the actual 16x16, is pre-existing and unrelated — `vqgan.py`/`config.py`
weren't touched by this work).
