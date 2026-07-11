"""
Re-runs ingestion for papi/lpc/fe with the three scraper bugs fixed (see
devlog/2026-07-08-data-pipeline-investigation.md), using the already-cached source archives
(no network access needed). Writes to fresh *_v2 directories so the old (buggy) data stays
available for comparison.
"""
from __future__ import annotations

import time
from pathlib import Path

from spriteforge.data.scrapers import download_pokeapi_repo, download_lpc_humanoids, download_fire_emblem_repo

ROOT = Path(__file__).resolve().parent.parent


def main():
    t0 = time.time()
    print("=== Re-ingesting papi (skip over-slicing + resize-not-crop fix) ===")
    papi_count = download_pokeapi_repo(
        "data_private/train_32_papi_v2", mode="overworld", target_size=32
    )
    print(f"[+] papi: {papi_count} sprites ({time.time()-t0:.0f}s elapsed)\n")

    t1 = time.time()
    print("=== Re-ingesting lpc (spritesheets/-only path filter) ===")
    lpc_count = download_lpc_humanoids("data_private/train_32_lpc_v2", target_size=32)
    print(f"[+] lpc: {lpc_count} sprites ({time.time()-t1:.0f}s elapsed)\n")

    t2 = time.time()
    print("=== Re-ingesting fe (category filter + Map Sprites grid-slicing) ===")
    fe_count = download_fire_emblem_repo("data_private/train_32_fe_v2", target_size=32)
    print(f"[+] fe: {fe_count} sprites ({time.time()-t2:.0f}s elapsed)\n")

    print(f"=== Done in {time.time()-t0:.0f}s total ===")
    print(f"papi: {papi_count}, lpc: {lpc_count}, fe: {fe_count}")


if __name__ == "__main__":
    main()
