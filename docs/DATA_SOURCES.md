# CC0 Data Sources & Fetch Guide

> To protect future licensing and commercial distribution, Spriteforge models must be trained **exclusively on verified CC0 (Public Domain) sprite assets**.

When Kenney is insufficient (e.g., you need higher stylistic diversity, richer RPG characters, isometric sprites, or modern animations), use the following verified CC0 and Public Domain sources.

---

## 1. Top CC0 & Public Domain Repositories

### A. OpenGameArt.org (CC0 Filtered)
OpenGameArt is the largest open game asset repository. Many prolific pixel artists release their work exclusively under CC0 here.
- **Search Filter URL:** [CC0 2D Art on OGA](https://opengameart.org/art-search-advanced?keys=&field_art_type_tid%5B%5D=9&field_art_license_tid%5B%5D=4)
- **Top Prolific CC0 Creators to Search:**
  - **Buch / Stoddard:** Famous for high-quality RPG characters, monsters, and dungeon environments.
  - **Calciumtrice:** Animated fantasy character sprites and enemies.
  - **0x72:** Famous for *DungeonTileset II*, industrial, and sci-fi sprite packs.
  - **surt:** Thousands of retro arcade and console-style pixel sprites.
  - **Ansimuz:** High-quality backgrounds, characters, and tilesets (check individual tags for CC0 vs CC-BY).
- **⚠️ WARNING:** Always manually verify on the asset page that the license says **CC0 / Public Domain** and does not include NC (Non-Commercial) or BY (Attribution).

### B. Itch.io (CC0 / Free Tagged Assets)
Itch.io has a thriving pixel art community releasing entire commercial-ready packs under CC0.
- **Search URL:** [itch.io Free CC0 Game Assets](https://itch.io/game-assets/free/tag-cc0)
- **How to Search:** Filter by `Free`, `Game Assets`, and tags: `cc0`, `public-domain`, `pixel-art`, `sprites`.
- **Notable CC0 Packs on Itch:** Look for uploads by **0x72**, **Penusbmic**, **Szadi art**, and **Analog Studios** (always verify the license block on the individual download page).

### C. The Glitch Game Assets Archive (10,000+ CC0 Sprites)
When the MMO *Glitch* shut down, the developers (Tiny Speck / Slack creators) released over **10,000+ original game sprites, character animations, items, and UI elements into the public domain (CC0)**.
- **Why it's great:** Highly detailed, professionally illustrated, unique aesthetic, and massive scale.
- **Where to find:**
  - [OpenGameArt Glitch Repository](https://opengameart.org/content/glitch-svgs)
  - GitHub community mirrors (search `Glitch Game Assets CC0`).
- **Note:** The artwork and animations are 100% CC0; just avoid using the trademarked "Glitch" logo or game title in your commercial branding.

---

## 2. The "Synthetic CC0 Flywheel" (3D → 2D Sprites)

If you need thousands of consistent, multi-angle character sprites at exact $16\times16$, $32\times32$, or $48\times48$ resolutions and human-drawn CC0 packs are too scarce, **bootstrap from CC0 3D models!**

### The Quaternius 3D → 2D Pipeline
- **Source:** [Quaternius.com](https://quaternius.com) releases massive modular 3D character, mech, monster, and animal packs under **CC0 1.0 Universal**.
- **How to Bootstrap:**
  1. Download Quaternius rigged/animated CC0 3D models (e.g., *Animated RPG Characters*, *Cyberpunk*, *Mechs*).
  2. Write a lightweight Blender Python script to render each animation frame from 8 isometric or orthographic camera angles against a transparent background.
  3. Pass the rendered high-res PNGs through Spriteforge's Stage A deterministic pipeline (`spriteforge convert --size 32 --palette-mode per-image-kmeans`).
  4. **Result:** An infinite, legally pristine, perfectly aligned CC0 sprite training dataset with ground-truth transparency!

---

## 3. What to AVOID (The Commercial Trap)

To protect your future SaaS, Unity Asset Store package, or closed-source licensing:
1. **NO Spriters Resource / Ripped Console Sprites:** Do not scrape `spriters-resource.com` or Pokémon/Nintendo/Capcom sprite databases. Ripped commercial IP will permanently poison your model weights and expose you to copyright litigation.
2. **NO Liberated Pixel Cup (LPC) / GPL / CC-BY-SA:** The Liberated Pixel Cup (LPC) has thousands of great sprites, but they are licensed under **CC-BY-SA 3.0** and **GPL 3.0**. Copyleft and ShareAlike licenses can create legal ambiguity or contagion for proprietary model weights and commercial tools. Stick strictly to **CC0 / Public Domain**.
