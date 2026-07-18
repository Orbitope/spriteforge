# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Builds train/test splits from the freshly re-ingested (bug-fixed) papi/lpc/fe data:
data_private/train_32_{source}_v2 -> data_private/train_32_{source} (training symlinks) +
data_private/test_{source}_32_v2 (held-out random 20-image test symlinks).

Read-only with respect to the _v2 source directories — only ever symlinks FROM them, never
deletes or modifies their contents.
"""
from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["papi", "lpc", "fe"]
HOLDOUT_SIZE = 20
SEED = 42


def build_split(source: str) -> None:
    v2_dir = ROOT / f"data_private/train_32_{source}_v2"
    train_dir = ROOT / f"data_private/train_32_{source}"
    test_dir = ROOT / f"data_private/test_{source}_32_v2"

    files = sorted(v2_dir.glob("*.png"))
    if len(files) < HOLDOUT_SIZE + 100:
        raise RuntimeError(f"{source}: only {len(files)} files in {v2_dir}, refusing to split")

    rng = random.Random(SEED)
    shuffled = files[:]
    rng.shuffle(shuffled)
    test_files = shuffled[:HOLDOUT_SIZE]
    train_files = shuffled[HOLDOUT_SIZE:]

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    # clear only the destination symlink dirs (never touch v2_dir, the real data)
    for p in train_dir.glob("*.png"):
        p.unlink()
    for p in test_dir.glob("*.png"):
        p.unlink()

    # fe filenames start with "fe_", which SpriteDataset's only_transparent filter excludes
    # by default (fe sprites are intentionally opaque, not transparent) — rename symlinks
    # with a "src_" prefix to bypass that filter, same workaround used earlier this session.
    prefix = "src_" if source == "fe" else ""

    for f in train_files:
        (train_dir / f"{prefix}{f.name}").symlink_to(f.resolve())
    for f in test_files:
        (test_dir / f"{prefix}{f.name}").symlink_to(f.resolve())

    print(f"[+] {source}: {len(train_files)} train, {len(test_files)} test -> {train_dir}, {test_dir}")


def main():
    for source in SOURCES:
        build_split(source)


if __name__ == "__main__":
    main()
