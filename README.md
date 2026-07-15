# Spriteforge

> High-resolution image → constrained low-pixel 2D sprite (16×16, 32×32, 48×48).

Spriteforge turns arbitrary high-resolution images or AI-generated art into clean, retro 2D sprites at target sizes (16×16, 32×32, 48×48). It enforces hard transparency and deterministic palette constraints in OKLab color space, with an optional learned restoration path for degraded inputs.

## Features

- **Deterministic conversion:** area-average downscaling, OKLab nearest-neighbor / k-means / median-cut palette snapping, hard alpha matting, and despeckling — no model required, and the strongest path for most inputs.
- **Palette library:** define, import, save, and reuse named palettes across sessions. Sub-select individual colors and watch the sprite update live.
- **Learned restoration:** two neural engines for cleaning up degraded or AI-generated sprites — a VQ-GAN and the discrete E1 Palette UNet (speckle-free by construction).
- **Desktop app + CLI:** a PySide6 GUI for interactive work, and a `spriteforge` CLI for scripted/batch conversion.

## Installation

```bash
# Clone, then install in editable mode with dev dependencies
pip install -e ".[dev]"
```

This installs the `spriteforge` command.

---

# User Guide

## Launching the app

```bash
spriteforge gui
```

The window has three tabs: **🎨 Convert** (deterministic pixelation + palette reduction), **🧠 Restore** (neural cleanup of degraded inputs), and **🔬 Evaluate** (degradation calibration grids).

Most people want the **Convert** tab.

## Convert tab — image → sprite

1. **Load Input Image** — pick any PNG/JPG/BMP/WebP.
2. **Target Size** — 16×16, 32×32, or 48×48.
3. **Palette Mode** — how colors are chosen (see below).
4. **Generate Retro Sprite** — the sharp, nearest-neighbor preview appears on the right. A preview also regenerates automatically as you change settings.
5. **Export Sprite PNG** — saves the sprite at exact target resolution (transparency preserved).

Other controls: **Max Colors** (palette size for the auto-extraction modes), **Bayer dithering**, **despeckling** (removes stray 1-pixel noise), and **Remove background before converting** (flood-matte the background to transparent first).

### Palette Mode options

- `per-image-kmeans` — extract a palette from this image with k-means (default, good general choice).
- `per-image-median` — extract with median-cut.
- `palette:<name>` — snap to a saved or built-in named palette (e.g. `palette:pico8`).

To create or choose named palettes, use the **🎨 Define / manage palettes…** button (see [Palettes](#palettes)).

### Sub-selecting colors (live)

Under the palette controls is an always-visible checklist: **"Uncheck to exclude a color (updates live)."** After a conversion it fills with the colors actually in use. **Uncheck any color and the sprite re-converts immediately** using only the checked colors — no dialog, no save step. This works for every palette mode.

## Restore tab — clean up a degraded sprite

For inputs that are noisy, blurry, or AI-generated and need restoring rather than just downscaling.

1. **Engine** — `VQ-GAN` or `Palette UNet (E1)`.
2. **Load Checkpoint (.pt)** — a trained model file. (The app auto-detects which engine a checkpoint belongs to.)
3. **Load Degraded Input** — the image to restore.
4. **Run Neural Restoration**.
5. **Export Restored Sprite**.

The **Palette** control drives both engines: for VQ-GAN it snaps the model's continuous output to your palette; for E1 it *conditions* the model (the output is drawn only from that palette, adapted to the model's trained color count). Options include `None (raw output)`, `Source image (k-means)`, `Restored output (k-means)`, any `palette:<name>`, or `Custom file…`. The same live sub-select checklist applies here too.

> E1 is discrete by construction, so its output never speckles. VQ-GAN can produce richer shading but may show color noise.

## Palettes

Click **🎨 Define / manage palettes…** (on either tab) to open the Palette Manager.

- **Build a palette:** type hex values (`#rrggbb`, `#rgb`, or `rgba(...)`), or use **🎨 Pick…** for a color wheel.
- **📚 Load from library:** start from a built-in preset or one you saved.
- **📥 Import file:** load a palette from `.json`, `.hex`, `.pal`, GIMP `.gpl`, or a `.png` color strip (e.g. a [Lospec](https://lospec.com/palette-list) export).
- **Sub-select:** each swatch has a checkbox — checked colors are the ones used. **Save checked only** persists just that subset as a new palette.
- **Save to your library:** type a name in the **Save as:** field, then **💾 Save all colors** (or **Save checked only**). Saved palettes appear in the dropdowns immediately — no restart.
- **Delete from library:** removes one of your saved palettes (built-in presets can't be deleted).

**Where palettes live:**

- Your saved/imported palettes: `~/.spriteforge/palettes/*.json`
- Bundled presets (read-only): shipped inside the package — `cc29`, `dawnbringer32`, `endesga32`, `gameboy`, `lospec500`, `mulfok32`, `palette31`, `pico8`, `resurrect64`, `sweetie16`.

A saved palette that shares a name with a built-in takes precedence.

---

## CLI reference

The deterministic converter runs headless — handy for batching.

```bash
# Extract a palette from the image itself (k-means) and snap to it
spriteforge convert -i photo.png -o sprite_32.png --size 32 --palette-mode kmeans --colors 16

# Snap to a bundled preset
spriteforge convert -i photo.png -o sprite.png --palette-mode preset --palette-preset pico8

# Snap to a palette file (.json/.hex/.pal)
spriteforge convert -i photo.png -o sprite.png --palette-mode fixed --palette-file my_palette.json

# Optional flags: --dither, --despeckle, --alpha-thresh 0.5, --resize-method area
```

`--palette-mode` accepts `kmeans`, `median-cut`, `preset`, or `fixed`. Run `spriteforge convert --help` for the full list.

Other subcommands (developer-oriented): `train` (train a VQ-GAN), `eval` (evaluate a checkpoint), `calibrate` (degradation grids), `ingest`/`scrape` (dataset prep), `gui` (launch the app). Each supports `--help`.

---

## License

**Dual Licensed:**

- **Open Source (AGPL v3):** Free for non-commercial use, research, and for commercial use if you open-source your modifications.
- **Commercial License:** Required for proprietary/closed-source commercial use, SaaS, and redistribution. Contact matthew.wesley.burke@gmail.com.

See `LICENSE` for full terms and `ASSETS_PROVENANCE.md` for training data provenance (CC0 sprites only).
