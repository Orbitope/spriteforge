# Spriteforge — Turning Messy Images Into Clean Retro Game Sprites

*A quick, non-technical look at what this project does and how well it's working so far.*

## What is this?

Spriteforge takes a messy, blurry, or AI-generated image of a game character and turns it
into a **clean, tiny, retro-style sprite** — the kind you'd see in an old Pokémon or SNES-era
game. Think 32×32 pixels, hard flat colors, a crisp black outline, and a transparent
background so it drops straight into a game engine.

The hard part isn't shrinking the image down — any image editor can do that. The hard part
is that a naive shrink just gives you a **blurry mess of mushy colors**. What you actually
want is for the small image to look like a *real, hand-drawn sprite* — flat color regions,
sharp edges, no noise. That's the part this project trains an AI model to do.

## How it works, in four steps

1. **Shrink** the input image down close to the target size.
2. **AI restoration** — a small neural network (trained only on real game sprites) tries to
   "clean up" the shrunken image into something that looks like a genuine sprite.
3. **Color cleanup** — snap every pixel to a small, clean set of colors (like a retro game's
   limited color palette), which kills off any leftover color noise from step 2.
4. **Transparency** — cut a hard, clean transparent background instead of a blurry halo.

The AI model (step 2) never sees a single real-world photo during training. Instead, it's
trained exclusively on clean sprites that we deliberately *degrade* first (blur, noise, color
drift, fake JPEG artifacts, stray backgrounds) — then it learns to undo exactly that kind of
damage. That's the trick that lets it generalize to real messy input later.

## Does it actually work?

Short answer: **mostly yes, with a known rough edge.** We trained four separate versions of
the model, one for each art style below, and evaluated each against sprites it never saw
during training. Here's one random example from each:

![Cross-source overview](showcase/overview_grid.png)

Left = the messy input the model has to work with. Middle = what the AI produces. Right = the
real, original sprite it's trying to reconstruct.

## The color cleanup trick, illustrated

Step 2 above (the AI) tends to leave a bit of "speckle" — small flecks of the wrong color
scattered around, especially in flat regions that should be one solid color. Step 3 (color
cleanup) fixes almost all of it:

![Color cleanup demo](showcase/palette_trick_demo.png)

That messy middle panel with the color speckling is what the raw AI produces. One deterministic
cleanup pass later, it's a clean, flat-colored sprite much closer to the original.

## Random examples, by art style

### Pokémon-style creatures (PMDCollab)
Small, colorful monster icons with transparent backgrounds.

![pmd showcase](showcase/pmd_showcase.png)

### Pokémon overworld sprites (PokéAPI)
Close-up detail crops from walking/battle sprite sheets — more zoomed-in and abstract
than the other styles, since this source's crops often catch a detail (a marking, a gem,
an emblem) rather than a whole creature.

![papi showcase](showcase/papi_showcase.png)

### Fantasy RPG gear (Universal LPC)
Modular character armor, weapons, clothing, and equipment/furniture pieces — this source
turned out to include more than just "armor," including some unexpected everyday items.

![lpc showcase](showcase/lpc_showcase.png)

### Tactics-game battle sprites (Fire Emblem GBA)
Characters and weapon-attack animations over in-game backgrounds (these are the one style
that's meant to have an opaque background, not transparency).

![fe showcase](showcase/fe_showcase.png)

*(These are randomly sampled from each style's training pool, filtered to skip
background-only/off-frame crops so every example is actually recognizable — run
`python scripts/build_showcase.py` for a fresh set, or edit `DEFAULT_SEEDS` in that script
to reshuffle which examples get picked.)*

## Where it's at right now

- The AI restoration step gets the **overall shape and rough color scheme** right most of the
  time, which is the hard part.
- Its biggest weakness is **fine color speckle/noise** in areas that should be flat — visible
  in the middle column above. The color-cleanup step (step 3) removes most, but not all, of it.
- Quality varies a bit by art style — some styles have simpler, more repetitive color schemes
  and reconstruct more cleanly than others (see the four grids above for a sense of it).
- This is a from-scratch model trained on a laptop, not a giant pretrained foundation model —
  the target was "usable starting point for a retro sprite pipeline," not photorealism.

## What's next

- Reduce the color-speckle problem at the source (bigger color vocabulary for the AI, and/or
  a training signal that penalizes per-pixel color noise more directly), rather than relying
  entirely on the cleanup step to paper over it.
- Once we lock in the actual game/project's color palette, swap it in for the cleanup step
  instead of guessing one from each image — that should sharpen results further.

*For the full technical write-up (training curves, metrics, what broke and how it got fixed),
see [devlog/](devlog/).*
