# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Scrape Spriters Resource with the expanded game slug list and merge into
the existing train_32_sr_combined dataset.

New slugs in this run:
  snes/secretofmana, snes/supermetroid, snes/finalfantasy5,
  snes/dragonquest6, snes/terranigma, snes/megamanx2, snes/megamanx3,
  game_boy_advance/finalfantasytactics, game_boy_advance/mothereginnebeginn,
  ds_dsi/finalfantasy4

Previously scraped games are still present in the output dir; duplicate
filenames are skipped automatically by the downloader.

Usage:
    python scripts/scrape_sr_expanded.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spriteforge.data.scrapers import (
    download_spriters_resource_games,
    SPRITERS_RESOURCE_GAME_SLUGS,
)

OUTPUT_DIR = ROOT / "data_private" / "train_32_sr_combined"

if __name__ == "__main__":
    print(f"Scraping {len(SPRITERS_RESOURCE_GAME_SLUGS)} games → {OUTPUT_DIR}")
    for slug in SPRITERS_RESOURCE_GAME_SLUGS:
        print(f"  {slug}")
    print()
    n = download_spriters_resource_games(
        output_dir=OUTPUT_DIR,
        game_slugs=SPRITERS_RESOURCE_GAME_SLUGS,
        target_size=32,
        request_delay=0.5,
    )
    print(f"\nDone — {n} new sprites written to {OUTPUT_DIR}")
    total = sum(1 for _ in OUTPUT_DIR.glob("*.png"))
    print(f"Total sprites in dataset: {total}")
