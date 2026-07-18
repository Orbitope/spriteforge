# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Automated downloading and scraping tools for collecting high-quality game sprites
for private/personal model training.

Supports:
1. PMDCollab (Pokémon Mystery Dungeon Sprite Collab) — 100,000+ standardized character sprites.
2. Hugging Face Pixel Art Datasets — Direct downloading of pre-compiled pixel art datasets.
3. Spriters Resource Sheet Processing — Automated contour slicing and padding for downloaded sheets.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from spriteforge.core.resize import pad_to_target, resize_to_target
from spriteforge.data.ingest import compute_phash, slice_by_alpha_contours, slice_vertical_strip
from spriteforge.data.provenance import log_provenance


def should_skip_path(
    path_str: str,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> bool:
    """Check if a file path should be skipped based on filter/exclude keyword rules."""
    path_lower = path_str.lower()
    if exclude_keywords:
        for kw in exclude_keywords:
            if kw.lower() in path_lower:
                return True
    if filter_keywords:
        if not any(kw.lower() in path_lower for kw in filter_keywords):
            return True
    return False


# --------------------------------------------------------------------------- #
# 1. PMDCollab Downloader (The Holy Grail of Standardized Pixel Characters)
# --------------------------------------------------------------------------- #

PMD_COLLAB_ZIP_URL = "https://github.com/PMDCollab/SpriteCollab/archive/refs/heads/master.zip"


def download_pmdcollab(
    output_dir: str | Path,
    target_size: int = 32,
    max_sprites: int | None = None,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Download and process the PMDCollab sprite archive.

    Why PMDCollab?
    It contains tens of thousands of professionally standardized, animated 2D pixel art
    characters with clean hard alpha cutouts and consistent multi-angle views.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log_provenance(
        source_name="PMDCollab (SpriteCollab)",
        source_url="https://github.com/PMDCollab/SpriteCollab",
        author="PMDCollab Community Artists",
        license_status="Private Research / Community License (Non-CC0)",
        license_tier="private_research",
        notes="Harvested for private model training."
    )

    cache_dir = Path("data_private/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_zip = cache_dir / "pmdcollab_master.zip"

    if cache_zip.exists():
        print(f"[*] Using cached PMDCollab archive from {cache_zip}...")
        try:
            with open(cache_zip, "rb") as f:
                zip_bytes = f.read()
        except Exception as e:
            print(f"[!] Cached archive corrupted ({e}). Re-downloading...")
            cache_zip.unlink(missing_ok=True)
            zip_bytes = None
    else:
        zip_bytes = None

    if zip_bytes is None:
        print("[*] Downloading PMDCollab master archive (this may take a minute)...")
        try:
            with urllib.request.urlopen(PMD_COLLAB_ZIP_URL) as resp:
                zip_bytes = resp.read()
            with open(cache_zip, "wb") as f:
                f.write(zip_bytes)
        except Exception as e:
            print(f"[!] Failed to download PMDCollab archive: {e}")
            return 0

    print("[*] Extracting and processing sprites...")
    seen_hashes = set()
    count = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for file_info in zf.infolist():
            if file_info.is_dir() or not file_info.filename.lower().endswith(".png"):
                continue

            # Focus on character sheets and animation frames
            if "shadow" in file_info.filename.lower() or "portrait" in file_info.filename.lower():
                continue
            if should_skip_path(file_info.filename, filter_keywords, exclude_keywords):
                continue

            with zf.open(file_info) as f:
                try:
                    pil_img = Image.open(f).convert("RGBA")
                    img_rgba = np.array(pil_img, dtype=np.float32) / 255.0
                except Exception:
                    continue

            # PMD sheets are grids of animation frames. Slice by alpha contours!
            slices = slice_by_alpha_contours(img_rgba, min_area=36)
            
            for idx, sprite in enumerate(slices):
                if max_sprites and count >= max_sprites:
                    print(f"[+] Reached requested limit of {max_sprites} sprites.")
                    return count

                padded = pad_to_target(sprite, target_size=target_size)
                if padded is None:
                    continue
                u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)

                phash = compute_phash(u8)
                if phash in seen_hashes:
                    continue
                seen_hashes.add(phash)

                clean_stem = Path(file_info.filename).stem.replace(" ", "_")
                out_name = f"pmd_{clean_stem}_s{idx:03d}_{count:05d}.png"
                Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
                count += 1

                if count % 500 == 0:
                    print(f"    -> Processed {count} sprites...")

    print(f"[+] Processed {count} PMDCollab sprites into {out_path}")
    return count


# --------------------------------------------------------------------------- #
# 2. Hugging Face Dataset Downloader
# --------------------------------------------------------------------------- #

def download_huggingface_dataset(
    dataset_name: str,
    output_dir: str | Path,
    target_size: int = 32,
    max_sprites: int | None = None,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Download sprites from a Hugging Face pixel art dataset using `datasets` library."""
    try:
        import datasets
    except ImportError:
        print("[!] Error: `datasets` package not installed. Run `pip install datasets`.", file=sys.stderr)
        return 0

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log_provenance(
        source_name=f"Hugging Face: {dataset_name}",
        source_url=f"https://huggingface.co/datasets/{dataset_name}",
        author="Various HF Contributors",
        license_status="Private Research / Unknown",
        license_tier="private_research",
        notes="Harvested for private model training."
    )

    print(f"[*] Loading Hugging Face dataset: {dataset_name} (streaming mode)...")
    try:
        ds = datasets.load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"[!] Failed to load HF dataset {dataset_name}: {e}")
        return 0

    seen_hashes = set()
    count = 0

    ds_iter = iter(ds)
    while True:
        if max_sprites and count >= max_sprites:
            break
        try:
            item = next(ds_iter)
        except StopIteration:
            break
        except Exception as e:
            print(f"[!] Warning: skipped sample due to HF download/decode error ({e})")
            continue

        img_col = None
        for col_name in ["image", "img", "sprite", "pixel_art"]:
            if col_name in item and item[col_name] is not None:
                img_col = col_name
                break
        
        if img_col is None:
            continue

        try:
            val = item[img_col]
            if isinstance(val, dict):
                if "bytes" in val and val["bytes"] is not None:
                    pil_img = Image.open(io.BytesIO(val["bytes"]))
                elif "path" in val and val["path"] is not None:
                    pil_img = Image.open(val["path"])
                else:
                    continue
            else:
                pil_img = val
            pil_img = pil_img.convert("RGBA")
            img_rgba = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception:
            continue

        h, w = img_rgba.shape[:2]
        if (h > target_size * 2 or w > target_size * 2) and (h != w or h > 128):
            slices = slice_by_alpha_contours(img_rgba)
        else:
            slices = [img_rgba]

        for sprite in slices:
            if max_sprites and count >= max_sprites:
                break
            
            padded = pad_to_target(sprite, target_size=target_size, allow_trim=True)
            if padded is None:
                continue
            u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)

            phash = compute_phash(u8)
            if phash in seen_hashes:
                continue
            seen_hashes.add(phash)

            safe_ds_name = dataset_name.replace("/", "_")
            out_name = f"hf_{safe_ds_name}_{count:06d}.png"
            Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
            count += 1

            if count > 0 and count % 500 == 0:
                print(f"    -> Harvested {count} HF sprites...")

    print(f"[+] Successfully harvested {count} sprites from Hugging Face.")
    return count


# --------------------------------------------------------------------------- #
# 3. Spriters Resource Sheet Processing (Local Slicing)
# --------------------------------------------------------------------------- #

def process_spriters_resource_dir(
    raw_dir: str | Path,
    output_dir: str | Path,
    target_size: int = 32,
    max_sprites: int | None = None,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Process locally downloaded sprite sheets from The Spriters Resource."""
    in_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Directory not found: {in_path}")

    log_provenance(
        source_name="The Spriters Resource (Raw Sheets)",
        source_url=str(in_path),
        author="Various Game Developers / Rippers",
        license_status="Private Research Only (Commercial IP / Non-CC0)",
        license_tier="private_research",
        notes="STRICTLY FOR PRIVATE MODEL TRAINING. Do not redistribute weights or images."
    )

    seen_hashes = set()
    count = 0

    print(f"[*] Slicing raw sheets from {in_path}...")
    for sheet_file in in_path.glob("**/*.*"):
        if max_sprites and count >= max_sprites:
            break
        if sheet_file.suffix.lower() not in [".png", ".bmp", ".gif", ".webp"]:
            continue
        if should_skip_path(str(sheet_file), filter_keywords, exclude_keywords):
            continue

        try:
            pil_img = Image.open(sheet_file).convert("RGBA")
            sheet_rgba = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception as e:
            continue

        # Slice floating sprites via alpha contours
        slices = slice_by_alpha_contours(sheet_rgba, min_area=64)

        for idx, sprite in enumerate(slices):
            padded = pad_to_target(sprite, target_size=target_size, allow_trim=True)
            if padded is None:
                continue
            u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)

            phash = compute_phash(u8)
            if phash in seen_hashes:
                continue
            seen_hashes.add(phash)

            out_name = f"sr_{sheet_file.stem}_s{idx:03d}_{count:05d}.png"
            Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
            count += 1

        if count > 0 and count % 500 == 0:
            print(f"    -> Processed {count} sprites...")

    print(f"[+] Processed {count} Spriters Resource sprites into {out_path}")
    return count


# --------------------------------------------------------------------------- #
# 4. Universal LPC Humanoid Generator Downloader (50,000+ Modular Heroes)
# --------------------------------------------------------------------------- #

LPC_ZIP_URL = "https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator/archive/refs/heads/master.zip"


def download_lpc_humanoids(
    output_dir: str | Path,
    target_size: int = 32,
    max_sprites: int | None = None,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Download and harvest modular humanoid sprites from the Liberated Pixel Cup (LPC).

    Why LPC?
    It contains tens of thousands of modular humanoid character animations (knights, mages,
    archers, rogues) across 4 directional walking/attacking views.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log_provenance(
        source_name="Universal LPC Spritesheet Generator",
        source_url="https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator",
        author="Liberated Pixel Cup Artists",
        license_status="Private Research (CC-BY-SA 3.0 / GPL 3.0)",
        license_tier="private_research",
        notes="Harvested for private model training. Do not share weights trained on copyleft data."
    )

    cache_dir = Path("data_private/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_zip = cache_dir / "lpc_master.zip"

    if cache_zip.exists():
        print(f"[*] Using cached LPC archive from {cache_zip}...")
        try:
            with open(cache_zip, "rb") as f:
                zip_bytes = f.read()
        except Exception as e:
            print(f"[!] Cached archive corrupted ({e}). Re-downloading...")
            cache_zip.unlink(missing_ok=True)
            zip_bytes = None
    else:
        zip_bytes = None

    if zip_bytes is None:
        print("[*] Downloading Universal LPC Spritesheet archive...")
        try:
            with urllib.request.urlopen(LPC_ZIP_URL) as resp:
                zip_bytes = resp.read()
            with open(cache_zip, "wb") as f:
                f.write(zip_bytes)
        except Exception as e:
            print(f"[!] Failed to download LPC archive: {e}")
            return 0

    print("[*] Extracting and harvesting LPC humanoid frames...")
    seen_hashes = set()
    count = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for file_info in zf.infolist():
            if file_info.is_dir() or not file_info.filename.lower().endswith(".png"):
                continue
            
            # Restrict to actual character sheets under spritesheets/ — without this, the
            # harvest also sweeps in non-sprite content that happens to be PNGs elsewhere in
            # the repo (readme-images/credits-sheet.png, readme-images/example.png, even
            # sources/github-mark.png — the GitHub logo), all of which pass the previous
            # shadow/preview filter and get contour-sliced into the training set. Confirmed by
            # inspecting the cached archive directly.
            if "/spritesheets/" not in file_info.filename.lower():
                continue
            if "shadow" in file_info.filename.lower() or "preview" in file_info.filename.lower():
                continue
            if should_skip_path(file_info.filename, filter_keywords, exclude_keywords):
                continue

            with zf.open(file_info) as f:
                try:
                    pil_img = Image.open(f).convert("RGBA")
                    img_rgba = np.array(pil_img, dtype=np.float32) / 255.0
                except Exception:
                    continue

            # LPC sheets are grids of 64x64 animation frames. Slice by contours!
            slices = slice_by_alpha_contours(img_rgba, min_area=100)
            
            for idx, sprite in enumerate(slices):
                if max_sprites and count >= max_sprites:
                    print(f"[+] Reached requested limit of {max_sprites} LPC sprites.")
                    return count

                # For LPC, allow center-trimming if a 64x64 frame has empty margins around a 32x32 character
                padded = pad_to_target(sprite, target_size=target_size, allow_trim=True)
                if padded is None:
                    continue
                u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)

                phash = compute_phash(u8)
                if phash in seen_hashes:
                    continue
                seen_hashes.add(phash)

                clean_stem = Path(file_info.filename).stem.replace(" ", "_")
                out_name = f"lpc_{clean_stem}_s{idx:03d}_{count:05d}.png"
                Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
                count += 1

                if count % 500 == 0:
                    print(f"    -> Harvested {count} LPC sprites...")

    print(f"[+] Successfully harvested {count} LPC humanoid sprites to {out_path}")
    return count


# --------------------------------------------------------------------------- #
# 5. Fire Emblem Asset Repository Downloader (GBA Chibi RPG Knights & Mages)
# --------------------------------------------------------------------------- #

FE_REPO_ZIP_URL = "https://github.com/Klokinator/FE-Repo/archive/refs/heads/main.zip"


def download_fire_emblem_repo(
    output_dir: str | Path,
    target_size: int = 32,
    max_sprites: int | None = None,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Download and harvest chibi RPG humanoid sprites from the Fire Emblem Asset Repo."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log_provenance(
        source_name="Fire Emblem Asset Repository (GBA/DS)",
        source_url="https://github.com/Klokinator/FE-Repo",
        author="FE Modding Community / Intelligent Systems",
        license_status="Private Research Only (Fan IP / Non-CC0)",
        license_tier="private_research",
        notes="Harvested for private model training."
    )

    cache_dir = Path("data_private/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_zip = cache_dir / "fe_repo_main.zip"

    if cache_zip.exists():
        print(f"[*] Using cached Fire Emblem archive from {cache_zip}...")
        try:
            with open(cache_zip, "rb") as f:
                zip_bytes = f.read()
        except Exception as e:
            print(f"[!] Cached archive corrupted ({e}). Re-downloading...")
            cache_zip.unlink(missing_ok=True)
            zip_bytes = None
    else:
        zip_bytes = None

    if zip_bytes is None:
        print("[*] Downloading Fire Emblem Asset Repository archive (this is a large ~500MB archive)...")
        try:
            with urllib.request.urlopen(FE_REPO_ZIP_URL) as resp:
                zip_bytes = resp.read()
            with open(cache_zip, "wb") as f:
                f.write(zip_bytes)
        except Exception as e:
            print(f"[!] Failed to download FE Repo (trying fallback master.zip): {e}")
            try:
                fallback_url = "https://github.com/Klokinator/FE-Repo/archive/refs/heads/master.zip"
                with urllib.request.urlopen(fallback_url) as resp:
                    zip_bytes = resp.read()
                with open(cache_zip, "wb") as f:
                    f.write(zip_bytes)
            except Exception as e2:
                print(f"[!] Failed to download FE Repo: {e2}")
                return 0

    print("[*] Extracting and harvesting Fire Emblem humanoid sprites...")
    seen_hashes = set()
    count = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for file_info in zf.infolist():
            if file_info.is_dir() or not file_info.filename.lower().endswith(".png"):
                continue

            fn_lower = file_info.filename.lower()
            # The repo root also contains non-humanoid-sprite categories (Spells n Skills,
            # Portrait Repository headshots, Tilesets, Maps, Class Cards, Item Icons, BGs,
            # unsorted/archival junk) that previously got swept in with no filtering at all.
            # Restrict to the two categories that are actually humanoid battle/overworld
            # sprites.
            in_battle_anims = "/battle animations/" in fn_lower
            in_map_sprites = "/map sprites/" in fn_lower
            if not (in_battle_anims or in_map_sprites):
                continue
            # Map Sprites includes an "Advance Wars Animation Ports" subfolder — vehicle/tank
            # sprites, not humanoid characters. Everything else in Map Sprites (confirmed by
            # sampling): Lords, Infantry by weapon type, Mounted units, Magi, Bards, Monsters.
            if in_map_sprites and "/advance wars animation ports/" in fn_lower:
                continue

            if should_skip_path(file_info.filename, filter_keywords, exclude_keywords):
                continue

            with zf.open(file_info) as f:
                try:
                    pil_img = Image.open(f).convert("RGBA")
                    img_rgba = np.array(pil_img, dtype=np.float32) / 255.0
                except Exception:
                    continue

            if in_map_sprites:
                # Map Sprites are opaque-background vertical animation strips (frame width ==
                # strip width, frame count == height // width) — alpha is uniform across the
                # whole strip, so slice_by_alpha_contours finds zero components. Confirmed
                # empirically. Use grid-based frame splitting instead.
                slices = slice_vertical_strip(img_rgba)
            else:
                slices = slice_by_alpha_contours(img_rgba, min_area=100)

            for idx, sprite in enumerate(slices):
                if max_sprites and count >= max_sprites:
                    print(f"[+] Reached requested limit of {max_sprites} FE sprites.")
                    return count

                padded = pad_to_target(sprite, target_size=target_size, allow_trim=True)
                if padded is None:
                    continue
                u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)

                phash = compute_phash(u8)
                if phash in seen_hashes:
                    continue
                seen_hashes.add(phash)

                clean_stem = Path(file_info.filename).stem.replace(" ", "_")
                out_name = f"fe_{clean_stem}_s{idx:03d}_{count:05d}.png"
                Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
                count += 1

                if count % 500 == 0:
                    print(f"    -> Harvested {count} FE sprites...")

    print(f"[+] Successfully harvested {count} Fire Emblem sprites to {out_path}")
    return count


# --------------------------------------------------------------------------- #
# 6. Ragnarok Online / MapleStory Chibi Batch Processor
# --------------------------------------------------------------------------- #

def process_ragnarok_maplestory_dir(
    raw_dir: str | Path,
    output_dir: str | Path,
    target_size: int = 32,
    max_sprites: int | None = None,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Process locally downloaded Ragnarok Online or MapleStory chibi humanoid archives."""
    in_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Directory not found: {in_path}")

    log_provenance(
        source_name="Ragnarok Online / MapleStory Chibi Archive",
        source_url=str(in_path),
        author="Gravity / Nexon / Community Dumps",
        license_status="Private Research Only (Commercial IP / Non-CC0)",
        license_tier="private_research",
        notes="STRICTLY FOR PRIVATE MODEL TRAINING."
    )

    seen_hashes = set()
    count = 0

    print(f"[*] Harvesting Ragnarok/MapleStory chibi humanoids from {in_path}...")
    for sheet_file in in_path.glob("**/*.*"):
        if max_sprites and count >= max_sprites:
            break
        if sheet_file.suffix.lower() not in [".png", ".bmp", ".gif", ".webp"]:
            continue
        if should_skip_path(str(sheet_file), filter_keywords, exclude_keywords):
            continue

        try:
            pil_img = Image.open(sheet_file).convert("RGBA")
            sheet_rgba = np.array(pil_img, dtype=np.float32) / 255.0
        except Exception:
            continue

        slices = slice_by_alpha_contours(sheet_rgba, min_area=64)

        for idx, sprite in enumerate(slices):
            if max_sprites and count >= max_sprites:
                break
            padded = pad_to_target(sprite, target_size=target_size, allow_trim=True)
            if padded is None:
                continue
            u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)

            phash = compute_phash(u8)
            if phash in seen_hashes:
                continue
            seen_hashes.add(phash)

            out_name = f"ro_ms_{sheet_file.stem}_s{idx:03d}_{count:05d}.png"
            Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
            count += 1

        if count > 0 and count % 500 == 0:
            print(f"    -> Harvested {count} chibi sprites...")

    print(f"[+] Processed {count} Ragnarok/MapleStory sprites into {out_path}")
    return count


# --------------------------------------------------------------------------- #
# 7. PokeAPI Sprite Archive (Canonical Items, Icons & Overworlds - Items 1 & 4)
# --------------------------------------------------------------------------- #

POKEAPI_ZIP_URL = "https://github.com/PokeAPI/sprites/archive/refs/heads/master.zip"


def download_pokeapi_repo(
    output_dir: str | Path,
    mode: str = "all",
    target_size: int = 32,
    max_sprites: int | None = None,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Download and harvest canonical items, icons, and overworld sprites from PokeAPI.

    mode:
      - 'icons': Focuses on inanimate objects, held items, berries, and Pokéball icons (Item 1).
      - 'overworld': Focuses on Gen 3-5 overworld walking NPCs, trainers, and battle sprites (Item 4).
      - 'all': Harvests both items and overworld sprites.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log_provenance(
        source_name=f"PokeAPI Sprites ({mode.upper()})",
        source_url="https://github.com/PokeAPI/sprites",
        author="Nintendo / Game Freak / PokeAPI Community",
        license_status="Private Research Only (Commercial IP / Non-CC0)",
        license_tier="private_research",
        notes="Harvested for private model training. Do not redistribute weights or images."
    )

    cache_dir = Path("data_private/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_zip = cache_dir / "pokeapi_master.zip"

    if cache_zip.exists():
        print(f"[*] Using cached PokeAPI archive from {cache_zip}...")
        try:
            with open(cache_zip, "rb") as f:
                zip_bytes = f.read()
        except Exception as e:
            print(f"[!] Cached archive corrupted ({e}). Re-downloading...")
            cache_zip.unlink(missing_ok=True)
            zip_bytes = None
    else:
        zip_bytes = None

    if zip_bytes is None:
        print("[*] Downloading PokeAPI sprite archive (~100MB)...")
        try:
            with urllib.request.urlopen(POKEAPI_ZIP_URL) as resp:
                zip_bytes = resp.read()
            with open(cache_zip, "wb") as f:
                f.write(zip_bytes)
        except Exception as e:
            print(f"[!] Failed to download PokeAPI archive: {e}")
            return 0

    print(f"[*] Extracting and harvesting PokeAPI sprites (mode: {mode})...")
    seen_hashes = set()
    count = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for file_info in zf.infolist():
            if file_info.is_dir() or not file_info.filename.lower().endswith(".png"):
                continue

            fn_lower = file_info.filename.lower()
            if mode == "icons":
                # Only grab item icons, held items, berries, balls, and UI badges (Item 1)
                if "items" not in fn_lower and "badges" not in fn_lower and "berries" not in fn_lower:
                    continue
            elif mode == "overworld":
                # Only grab overworld, generation-iii, generation-iv, generation-v (Item 4)
                if "versions" not in fn_lower and "overworld" not in fn_lower and "other" not in fn_lower:
                    continue

            if should_skip_path(file_info.filename, filter_keywords, exclude_keywords):
                continue

            with zf.open(file_info) as f:
                try:
                    pil_img = Image.open(f).convert("RGBA")
                    img_rgba = np.array(pil_img, dtype=np.float32) / 255.0
                except Exception:
                    continue

            # Every file harvested here (versions/, other/home/, other/official-artwork/) is
            # already a single, whole, correctly-cropped creature render — never a multi-sprite
            # sheet. Running slice_by_alpha_contours on these shatters them into disconnected
            # fragments (antialiasing gaps, separated tails/ears/accessories become their own
            # "sprite"): confirmed empirically, one file produced 91 fragment slices. Treat
            # each file as one sprite directly.
            slices = [img_rgba]

            for idx, sprite in enumerate(slices):
                if max_sprites and count >= max_sprites:
                    print(f"[+] Reached requested limit of {max_sprites} PokeAPI sprites.")
                    return count

                # pad_to_target(allow_trim=True) center-CROPS oversized input rather than
                # resizing it — fine for a sprite that's oversized by a few stray pixels, wrong
                # for a 512x512 whole-creature render (crops to a random 32x32 fragment of it,
                # confirmed empirically). Since these are always single whole images now (no
                # slicing), resize down when oversized instead of cropping; pad (no resample)
                # when already small, to preserve crisp native pixel art like the 40x40 gen-i
                # sprites.
                h, w = sprite.shape[:2]
                if h > target_size or w > target_size:
                    padded = resize_to_target(sprite, target_size=target_size, method="area")
                else:
                    padded = pad_to_target(sprite, target_size=target_size, allow_trim=True)
                if padded is None:
                    continue
                u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)

                phash = compute_phash(u8)
                if phash in seen_hashes:
                    continue
                seen_hashes.add(phash)

                clean_stem = Path(file_info.filename).stem.replace(" ", "_")
                out_name = f"papi_{mode[:3]}_{clean_stem}_s{idx:03d}_{count:05d}.png"
                Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
                count += 1

                if count > 0 and count % 500 == 0:
                    print(f"    -> Harvested {count} PokeAPI sprites...")

    print(f"[+] Successfully harvested {count} PokeAPI sprites to {out_path}")
    return count


# --------------------------------------------------------------------------- #
# 8. Retro RPG & Gothic Character Archives (FFRK, Castlevania, Mother 2 - Items 2 & 3)
# --------------------------------------------------------------------------- #

def download_retro_rpg_sprites(
    output_dir: str | Path,
    target_size: int = 32,
    max_sprites: int | None = None,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Download and harvest canonical SNES/16-bit RPG & Gothic character sprites (FFRK style, EarthBound/Mother 2, Castlevania)."""
    count = download_huggingface_dataset(
        "botmaster/mother-2-battle-sprites",
        output_dir=output_dir,
        target_size=target_size,
        max_sprites=max_sprites,
        filter_keywords=filter_keywords,
        exclude_keywords=exclude_keywords
    )
    count += download_huggingface_dataset(
        "andrewburns/anime_sprites_v1",
        output_dir=output_dir,
        target_size=target_size,
        max_sprites=max_sprites,
        filter_keywords=filter_keywords,
        exclude_keywords=exclude_keywords
    )
    return count


# --------------------------------------------------------------------------- #
# 9. OpenGameArt Individual Content Pages (environment/prop packs — furniture,
#    plants, terrain — to pair with existing humanoid character sources)
# --------------------------------------------------------------------------- #

# Curated individual OpenGameArt content-page URLs, not "collection" bookmark pages —
# collections (e.g. opengameart.org/content/nearly-all-the-lpc-assets-in-one-place) are just
# link lists with no bulk download; each URL below is a real page with its own attached zip.
# License is CC-BY-SA 3.0/4.0 + GPL 3.0 (same family as the existing LPC character source, NOT
# CC0 — docs/DATA_SOURCES.md's "avoid LPC/copyleft" guidance is stale relative to actual
# practice in this codebase; every non-CC0 source here is logged as private_research tier).
LPC_ADJACENT_PACK_URLS = [
    "https://opengameart.org/content/lpc-wooden-furniture",
    "https://opengameart.org/content/lpc-upholstery",
    "https://opengameart.org/content/lpc-interiors",
    "https://opengameart.org/content/lpc-house-interior-and-decorations",
    "https://opengameart.org/content/lpc-floors",
    "https://opengameart.org/content/lpc-fruit-trees",
    "https://opengameart.org/content/lpc-crops",
    "https://opengameart.org/content/lpc-flowers-plants-fungi-wood",
]


def _find_opengameart_zip_urls(page_html: str) -> list[str]:
    """OpenGameArt attachment links live under sites/default/files/<name>.zip. Confirmed by
    inspecting a real page: they appear as full absolute URLs
    (https://opengameart.org/sites/default/files/layers.zip), not root-relative paths —
    match both forms defensively. Also excludes credits .txt files that get zipped
    (e.g. credits-furniture.txt_.zip) by requiring the stem not start with "credits".
    """
    matches = re.findall(r'(?:https://opengameart\.org)?(/sites/default/files/[^"\'<>\s]+\.zip)', page_html)
    base = "https://opengameart.org"
    urls = [base + m for m in dict.fromkeys(matches)]
    return [u for u in urls if not Path(u).stem.lower().startswith("credits")]


def _find_opengameart_png_urls(page_html: str) -> list[str]:
    """Fallback for pages with a single attached PNG sheet instead of a zip (confirmed on
    lpc-upholstery, lpc-interiors). Excludes thumbnail/style/license/logo images that also
    live under sites/default/files/ but aren't the actual asset (styles/thumbnail/,
    license_images/, archive/sara-logo.png).
    """
    matches = re.findall(r'(?:https://opengameart\.org)?(/sites/default/files/[^"\'<>\s]+\.png)', page_html)
    base = "https://opengameart.org"
    urls = [base + m for m in dict.fromkeys(matches)]
    return [
        u for u in urls
        if "/styles/" not in u and "/license_images/" not in u and "sara-logo" not in u
    ]


def download_opengameart_packs(
    output_dir: str | Path,
    page_urls: list[str] | None = None,
    target_size: int = 32,
    max_sprites: int | None = None,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Download and harvest environment/prop packs from individual OpenGameArt content pages
    (not collections — see LPC_ADJACENT_PACK_URLS). Each page's zip is fetched fresh (not
    cached locally like the other scrapers' single master archives, since this pulls from
    several small independent pages rather than one large repo).

    Due-diligence note (see devlog/2026-07-08-data-pipeline-investigation.md): verify a visual
    sample of the output before large-scale training use. These packs are assumed to be
    multi-object sheets (appropriate for slice_by_alpha_contours), which is typical for
    OpenGameArt furniture/prop packs, but — as pmd/papi/lpc/fe all separately demonstrated
    this session — that assumption can be wrong for a specific source and silently produces
    bad crops. Check before trusting at scale.
    """
    page_urls = page_urls or LPC_ADJACENT_PACK_URLS
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    count = 0

    for page_url in page_urls:
        page_slug = page_url.rstrip("/").split("/")[-1]
        log_provenance(
            source_name=f"OpenGameArt: {page_slug}",
            source_url=page_url,
            author="OpenGameArt Community",
            license_status="Private Research (CC-BY-SA 3.0/4.0 / GPL 3.0)",
            license_tier="private_research",
            notes="Harvested for private model training. Do not share weights trained on copyleft data."
        )

        try:
            req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                page_html = resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[!] Failed to fetch {page_url}: {e}")
            continue

        # Gather (source_name, png_bytes) pairs from either a zip attachment or, failing
        # that, directly-attached PNG sheet(s) — confirmed both patterns exist across
        # different pages.
        sources: list[tuple[str, bytes]] = []
        zip_urls = _find_opengameart_zip_urls(page_html)
        for zip_url in zip_urls:
            print(f"[*] Downloading {zip_url}...")
            try:
                req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp:
                    zip_bytes = resp.read()
                zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
            except Exception as e:
                print(f"[!] Failed to download/open {zip_url}: {e}")
                continue
            with zf:
                for file_info in zf.infolist():
                    if file_info.is_dir() or not file_info.filename.lower().endswith(".png"):
                        continue
                    if should_skip_path(file_info.filename, filter_keywords, exclude_keywords):
                        continue
                    with zf.open(file_info) as f:
                        sources.append((file_info.filename, f.read()))

        if not zip_urls:
            png_urls = _find_opengameart_png_urls(page_html)
            for png_url in png_urls:
                print(f"[*] Downloading {png_url}...")
                try:
                    req = urllib.request.Request(png_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req) as resp:
                        sources.append((png_url.split("/")[-1], resp.read()))
                except Exception as e:
                    print(f"[!] Failed to download {png_url}: {e}")

        if not sources:
            print(f"[!] No zip or PNG attachment found on {page_url}, skipping.")
            continue

        for src_name, png_bytes in sources:
            try:
                pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                img_rgba = np.array(pil_img, dtype=np.float32) / 255.0
            except Exception:
                continue

            slices = slice_by_alpha_contours(img_rgba, min_area=36)

            for idx, sprite in enumerate(slices):
                if max_sprites and count >= max_sprites:
                    print(f"[+] Reached requested limit of {max_sprites} OpenGameArt sprites.")
                    return count

                h, w = sprite.shape[:2]
                if h > target_size or w > target_size:
                    padded = resize_to_target(sprite, target_size=target_size, method="area")
                else:
                    padded = pad_to_target(sprite, target_size=target_size, allow_trim=True)
                if padded is None:
                    continue
                u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)

                phash = compute_phash(u8)
                if phash in seen_hashes:
                    continue
                seen_hashes.add(phash)

                clean_stem = Path(src_name).stem.replace(" ", "_")
                out_name = f"oga_{page_slug}_{clean_stem}_s{idx:03d}_{count:05d}.png"
                Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
                count += 1

    print(f"[+] Successfully harvested {count} OpenGameArt sprites to {out_path}")
    return count


# --------------------------------------------------------------------------- #
# 10. The Spriters Resource — live scraping by game slug (large, consistent,
#     single-game humanoid sources: FF6, Chrono Trigger, Zelda ALTTP, Mega Man X,
#     Castlevania AoS/DoS). Chosen to replace oga (OpenGameArt furniture/plants),
#     dropped from default training per 2026-07-08 review — its 932-image
#     dataset was too small and produced far worse speckle than the other
#     sources. These games each ship hundreds of individually-labeled sprite
#     sheets, closer to fe's scale, which was the best-performing v2 source.
# --------------------------------------------------------------------------- #

SPRITERS_RESOURCE_BASE = "https://www.spriters-resource.com"

# Category keep/exclude lists derived from the actual section headers observed on FF6,
# Chrono Trigger, Zelda ALTTP, Mega Man X, and the two Castlevania pages — categories vary
# slightly in naming per game (e.g. "Non-Playable Characters" vs "Non-Playable & Guest
# Characters") so matching is substring-based, not exact.
SPRITERS_RESOURCE_CATEGORY_KEEP = [
    "playable character",
    "non-playable",
    "npc",
    "guest character",
    "enem",
    "boss",
]
SPRITERS_RESOURCE_CATEGORY_EXCLUDE = [
    "tileset",
    "background",
    "map",
    "cutscene",
    "effect",
    "misc",
    "unused",
    "title",
    "font",
    "ui",
    "world of",
]

SPRITERS_RESOURCE_GAME_SLUGS = [
    # SNES — original 6
    "snes/ff6",
    "snes/chronotrigger",
    "snes/legendofzeldaalinktothepast",
    "snes/mmx",
    # SNES — round 2
    "snes/secretofmana",
    "snes/supermetroid",
    "snes/finalfantasy5",
    "snes/dragonquest6",
    "snes/terranigma",
    "snes/megamanx2",
    "snes/megamanx3",
    # SNES — round 3: JRPGs + high character density
    "snes/earthbound",
    "snes/tacticsogre",
    "snes/bahamutlagoon",
    "snes/livealive",
    "snes/dragonquest5",
    "snes/romancingsaga2",
    "snes/romancingsaga3",
    "snes/megaman7",
    # GBA — round 1
    "game_boy_advance/cvaos",
    "game_boy_advance/finalfantasytactics",
    "game_boy_advance/mothereginnebeginn",
    # GBA — round 3: Fire Emblem + action + RPG
    "game_boy_advance/fireemblem",
    "game_boy_advance/fireemblem7",
    "game_boy_advance/fireemblem8",
    "game_boy_advance/goldensun",
    "game_boy_advance/goldensunthelostage",
    "game_boy_advance/megamanzero",
    "game_boy_advance/megamanzero2",
    "game_boy_advance/megamanzero3",
    "game_boy_advance/megamanzero4",
    # DS/DSi — round 1
    "ds_dsi/cstlevniadawnofsorrow",
    "ds_dsi/finalfantasy4",
    # DS/DSi — round 3
    "ds_dsi/radianthistoria",
    "ds_dsi/finalfantasy3",
    # Genesis — tactical RPGs with large humanoid rosters
    "genesis/shiningforce",
    "genesis/shiningforce2",
]


def _sr_dominant_border_color(rgb: np.ndarray, border_px: int = 10) -> np.ndarray:
    """Return the most-frequent RGB color found in the border strip of a sheet."""
    h, w = rgb.shape[:2]
    strips = np.concatenate([
        rgb[:border_px, :].reshape(-1, 3),
        rgb[-border_px:, :].reshape(-1, 3),
        rgb[:, :border_px].reshape(-1, 3),
        rgb[:, -border_px:].reshape(-1, 3),
    ])
    # Quantize to nearest-8 to group near-identical shades together
    quantized = (strips // 8).astype(np.int32)
    keys = quantized[:, 0] * 65536 + quantized[:, 1] * 256 + quantized[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    dominant_key = vals[np.argmax(counts)]
    r = (dominant_key >> 16) * 8
    g = ((dominant_key >> 8) & 0xFF) * 8
    b = (dominant_key & 0xFF) * 8
    return np.array([r, g, b], dtype=np.float32)


def _sr_key_background(sheet_rgba: np.ndarray, distance_threshold: float = 20.0) -> np.ndarray | None:
    """Attempt background-color-keying on a fully-opaque sheet.

    Samples the most frequent color in the border pixels and sets any pixel within
    `distance_threshold` (Euclidean RGB, 0-255 scale) of that color to transparent.
    Returns a new RGBA array (float32, 0-1), or None if the result looks degenerate
    (fewer than 5% of pixels became foreground — suggests a sheet where the "background"
    and the actual content are too close in color to separate cleanly, or where the sheet
    has no uniform background at all and the dominant border color is content).

    A safety check after keying: if the result yields too few foreground pixels (< 5% of
    sheet) the keying failed and we return None to tell the caller to skip this sheet.
    This prevents the crop step from producing a handful of huge noise blobs instead of
    individual sprites.
    """
    rgb_255 = (sheet_rgba[..., :3] * 255.0)
    bg_color = _sr_dominant_border_color(rgb_255.astype(np.uint8))
    dist = np.linalg.norm(rgb_255 - bg_color.astype(np.float32), axis=-1)
    alpha_mask = (dist >= distance_threshold).astype(np.float32)
    if alpha_mask.mean() < 0.05:
        return None
    result = sheet_rgba.copy()
    result[..., 3] = alpha_mask
    return result


def _sr_fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _sr_fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _sr_category_allowed(category: str) -> bool:
    cat_lower = category.lower()
    if any(kw in cat_lower for kw in SPRITERS_RESOURCE_CATEGORY_EXCLUDE):
        return False
    return any(kw in cat_lower for kw in SPRITERS_RESOURCE_CATEGORY_KEEP)


def download_spriters_resource_games(
    output_dir: str | Path,
    game_slugs: list[str] | None = None,
    target_size: int = 32,
    max_sprites: int | None = None,
    request_delay: float = 0.3,
    filter_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None
) -> int:
    """Scrape individually-labeled character sheets from The Spriters Resource for a list
    of game page slugs (e.g. "snes/ff6" -> spriters-resource.com/snes/ff6/).

    Each game's listing page groups assets under category headers ("Playable Characters",
    "Enemies & Bosses", vs. "Tilesets", "Backgrounds", "Cutscenes", etc.) — this only keeps
    categories that look like actual characters (see SPRITERS_RESOURCE_CATEGORY_KEEP/EXCLUDE),
    read per-asset from each asset page's own labeled title (format
    "PLATFORM - Game - Category - Name.png"), not from the listing page's HTML structure
    (more robust — doesn't depend on correctly associating asset IDs with section blocks).

    Downloaded PNGs are cached to data_private/.cache/spriters_resource/ so re-runs (or a
    later run with more game_slugs) don't re-fetch already-downloaded assets — this hits
    up to ~2 HTTP requests per asset (listing page's asset-page fetch + image download)
    across potentially hundreds of assets per game, so caching matters both for politeness
    and for iteration speed while testing category filters.

    Due-diligence note (see devlog/2026-07-08-data-pipeline-investigation.md and the oga
    finding in devlog/2026-07-08-random-pool-samples.md): verify a visual sample of sliced
    output before large-scale training use — assumed to be alpha-transparent multi-frame
    sheets (slice_by_alpha_contours), typical for this site's community-cleaned rips, but
    that assumption has been wrong for specific sources before (papi, fe) and silently
    produced bad crops. Check before trusting at scale.
    """
    game_slugs = game_slugs or SPRITERS_RESOURCE_GAME_SLUGS
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("data_private/.cache/spriters_resource")
    cache_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    count = 0

    for slug in game_slugs:
        if max_sprites and count >= max_sprites:
            break

        slug_flat = slug.replace("/", "_")
        listing_url = f"{SPRITERS_RESOURCE_BASE}/{slug}/"

        log_provenance(
            source_name=f"The Spriters Resource: {slug}",
            source_url=listing_url,
            author="Various Game Developers / Rippers",
            license_status="Private Research Only (Commercial IP / Non-CC0)",
            license_tier="private_research",
            notes="STRICTLY FOR PRIVATE MODEL TRAINING. Do not redistribute weights or images."
        )

        try:
            listing_html = _sr_fetch(listing_url)
        except Exception as e:
            print(f"[!] Failed to fetch listing page {listing_url}: {e}")
            continue

        asset_ids = sorted(set(re.findall(rf'href="/{re.escape(slug)}/asset/(\d+)/"', listing_html)), key=int)
        print(f"[*] {slug}: found {len(asset_ids)} candidate assets")

        game_cache_dir = cache_dir / slug_flat
        game_cache_dir.mkdir(parents=True, exist_ok=True)
        kept = 0

        for asset_id in asset_ids:
            if max_sprites and count >= max_sprites:
                break

            cached_png = game_cache_dir / f"{asset_id}.png"
            cached_title = game_cache_dir / f"{asset_id}.title.txt"

            if cached_png.exists() and cached_title.exists():
                title = cached_title.read_text(encoding="utf-8").strip()
                png_bytes = cached_png.read_bytes()
            else:
                asset_url = f"{SPRITERS_RESOURCE_BASE}/{slug}/asset/{asset_id}/"
                try:
                    asset_html = _sr_fetch(asset_url)
                    time.sleep(request_delay)
                except Exception as e:
                    print(f"[!] Failed to fetch asset page {asset_url}: {e}")
                    continue

                title_match = re.search(r'data-download="([^"]*)"', asset_html)
                file_match = re.search(r'data-file="([^"]*)"', asset_html)
                if not title_match or not file_match:
                    continue
                title = title_match.group(1)
                file_path = file_match.group(1)

                try:
                    png_bytes = _sr_fetch_bytes(f"{SPRITERS_RESOURCE_BASE}{file_path}")
                    time.sleep(request_delay)
                except Exception as e:
                    print(f"[!] Failed to download image for asset {asset_id}: {e}")
                    continue

                cached_png.write_bytes(png_bytes)
                cached_title.write_text(title, encoding="utf-8")

            # Title format: "PLATFORM - Game - Category - Name.png"
            parts = [p.strip() for p in title.split(" - ")]
            if len(parts) < 2:
                continue
            category = parts[-2]
            if not _sr_category_allowed(category):
                continue
            if should_skip_path(title, filter_keywords, exclude_keywords):
                continue

            try:
                pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                sheet_rgba = np.array(pil_img, dtype=np.float32) / 255.0
            except Exception:
                continue

            # Many Spriters Resource sheets are fully opaque composites (multiple poses
            # on a uniform background, text labels baked in) — confirmed for MMX's X,
            # Vile, Sigma, all 6 Mavericks. slice_by_alpha_contours assumes floating-
            # transparent frames; on a 0% transparent sheet it silently area-downscales
            # the entire composite into a single 32x32 noise blob.
            # Fix: if the sheet is opaque (< 5% transparent pixels), attempt background-
            # color-keying by sampling the border color and masking it out. If keying
            # also fails to yield enough foreground pixels (> 5%), skip this sheet.
            transparent_frac = float((sheet_rgba[..., 3] < (20 / 255)).mean())
            if transparent_frac < 0.05:
                keyed = _sr_key_background(sheet_rgba)
                if keyed is None:
                    continue
                sheet_rgba = keyed

            slices = slice_by_alpha_contours(sheet_rgba, min_area=36)

            # Safety check: if slicing an opaque-keyed sheet yielded fewer than 3 results
            # or any single slice spans more than half the original sheet in both dimensions
            # (a sign the keying separated large text blocks rather than individual sprites),
            # skip entirely rather than save degenerate blobs.
            sheet_h, sheet_w = sheet_rgba.shape[:2]
            valid_slices = []
            for s in slices:
                sh, sw = s.shape[:2]
                if sh > sheet_h * 0.5 and sw > sheet_w * 0.5:
                    continue
                valid_slices.append(s)
            if transparent_frac < 0.05 and len(valid_slices) < 3:
                continue
            slices = valid_slices

            for idx, sprite in enumerate(slices):
                if max_sprites and count >= max_sprites:
                    break

                h, w = sprite.shape[:2]
                if h > target_size or w > target_size:
                    padded = resize_to_target(sprite, target_size=target_size, method="area")
                else:
                    padded = pad_to_target(sprite, target_size=target_size, allow_trim=True)
                if padded is None:
                    continue
                u8 = np.clip(padded * 255.0 + 0.5, 0, 255).astype(np.uint8)

                phash = compute_phash(u8)
                if phash in seen_hashes:
                    continue
                seen_hashes.add(phash)

                clean_stem = re.sub(r"[^a-zA-Z0-9]+", "_", parts[-1].replace(".png", "")).strip("_")
                out_name = f"sr_{slug_flat}_{clean_stem}_s{idx:03d}_{count:05d}.png"
                Image.fromarray(u8, mode="RGBA").save(out_path / out_name)
                count += 1
                kept += 1

        print(f"[+] {slug}: kept {kept} sprites after category filtering + slicing")

    print(f"[+] Successfully harvested {count} Spriters Resource sprites to {out_path}")
    return count

