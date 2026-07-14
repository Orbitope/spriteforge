"""
Palette library: a unified, reusable store of named palettes for sprite imputation.

Two sources are merged into one namespace:

  * BUILTIN — the bundled presets in spriteforge/data/palettes/ (read-only; pico8,
    dawnbringer32, ...). Shipped with the package.
  * USER — palettes the user defines, imports, or saves, kept in a per-user data
    dir (~/.spriteforge/palettes/) so they persist across sessions and survive
    reinstalls. A user palette shadows a builtin of the same name.

This module is deliberately Qt-free so it stays in `core/` and is unit-testable;
the GUI (spriteforge/app/gui.py) is the only consumer that adds widgets on top.

Palettes follow the codebase convention everywhere: (K, 3) float32 sRGB in [0, 1].
All file I/O and extraction reuse the primitives in spriteforge.core.palette.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from spriteforge.core.palette import (
    BUILTIN_PALETTES_DIR,
    extract_palette_kmeans,
    list_builtin_palettes,
    load_builtin_palette,
    load_palette,
    save_palette_json,
)

# Per-user palette store. Kept out of the package dir so user palettes are never
# mixed with bundled presets and are not wiped on reinstall/upgrade.
USER_PALETTE_DIR = Path.home() / ".spriteforge" / "palettes"

# Cap for palettes extracted from images (a .png strip can contain thousands of
# unique pixels; beyond this we k-means down to a representative set).
_IMPORT_COLOR_CAP = 64

_SUPPORTED_IMPORT_SUFFIXES = (".json", ".hex", ".pal", ".gpl", ".png")


@dataclass
class NamedPalette:
    """A palette with provenance. `path` is None only for transient palettes."""

    name: str
    colors: np.ndarray  # (K, 3) float32 sRGB in [0, 1]
    source: str  # "builtin" | "user"
    path: Path | None = None


def _sanitize_name(name: str) -> str:
    """Reduce an arbitrary label to a safe filename stem (no extension)."""
    stem = Path(str(name)).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not stem:
        raise ValueError(f"Palette name is empty after sanitizing: {name!r}")
    return stem


def _ensure_user_dir() -> Path:
    USER_PALETTE_DIR.mkdir(parents=True, exist_ok=True)
    return USER_PALETTE_DIR


def list_user_palettes() -> list[str]:
    """Names of user-saved palettes (sorted). Empty if the dir doesn't exist yet."""
    if not USER_PALETTE_DIR.is_dir():
        return []
    return sorted(p.stem for p in USER_PALETTE_DIR.glob("*.json"))


def list_palettes() -> list[NamedPalette]:
    """All available palettes, builtin first then user, sorted within each source.

    A user palette with the same name as a builtin shadows it (the builtin is
    dropped so the name resolves to the user's version, matching load_named()).
    """
    user_names = set(list_user_palettes())
    out: list[NamedPalette] = []
    for name in list_builtin_palettes():
        if name in user_names:
            continue  # shadowed by a user palette of the same name
        out.append(
            NamedPalette(
                name=name,
                colors=load_builtin_palette(name),
                source="builtin",
                path=BUILTIN_PALETTES_DIR / f"{name}.json",
            )
        )
    for name in sorted(user_names):
        out.append(
            NamedPalette(
                name=name,
                colors=load_palette(USER_PALETTE_DIR / f"{name}.json"),
                source="user",
                path=USER_PALETTE_DIR / f"{name}.json",
            )
        )
    return out


def load_named(name: str) -> NamedPalette:
    """Resolve a palette by name, preferring a user palette over a builtin.

    The user-dir lookup applies the same name sanitizing as save_user_palette so
    a label round-trips (save "My Pal" -> load "My Pal").
    """
    user_stem = _sanitize_name(name) if name.strip() else name
    user_path = USER_PALETTE_DIR / f"{user_stem}.json"
    if user_path.exists():
        return NamedPalette(name=user_stem, colors=load_palette(user_path), source="user", path=user_path)
    if name in list_builtin_palettes():
        return NamedPalette(
            name=name,
            colors=load_builtin_palette(name),
            source="builtin",
            path=BUILTIN_PALETTES_DIR / f"{name}.json",
        )
    raise ValueError(f"Unknown palette '{name}'.")


def save_user_palette(name: str, colors: np.ndarray) -> Path:
    """Save (or overwrite) a user palette. Returns the written path.

    Reuses spriteforge.core.palette.save_palette_json (hex-array JSON).
    """
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Palette must be (K, 3); got {colors.shape}")
    if colors.shape[0] == 0:
        raise ValueError("Cannot save an empty palette.")
    _ensure_user_dir()
    path = USER_PALETTE_DIR / f"{_sanitize_name(name)}.json"
    save_palette_json(colors, path)
    return path


def delete_user_palette(name: str) -> None:
    """Delete a user palette. Never touches bundled presets."""
    path = USER_PALETTE_DIR / f"{_sanitize_name(name)}.json"
    if not path.exists():
        raise ValueError(f"No user palette named '{name}'.")
    path.unlink()


def _load_gpl(path: Path) -> np.ndarray:
    """Parse a GIMP .gpl palette: 'R G B  name' rows, skipping header/comments."""
    colors: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(("GIMP", "Name:", "Columns:")):
                continue
            parts = line.split()
            if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
                r, g, b = (int(parts[i]) / 255.0 for i in range(3))
                colors.append([r, g, b])
    if not colors:
        raise ValueError(f"No colors parsed from GIMP palette: {path}")
    return np.array(colors, dtype=np.float32)


def _load_png(path: Path) -> np.ndarray:
    """Extract distinct colors from an image (Lospec/Aseprite PNG-strip export).

    Uses opaque unique pixels; if there are more than the cap, k-means down to a
    representative set so the result stays a usable palette.
    """
    img = np.array(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    opaque = img[img[..., 3] >= 0.5][:, :3]
    if opaque.shape[0] == 0:
        raise ValueError(f"Image has no opaque pixels to extract a palette from: {path}")
    unique = np.unique(np.round(opaque, 6), axis=0)
    if unique.shape[0] > _IMPORT_COLOR_CAP:
        rgba = np.concatenate([opaque, np.ones((opaque.shape[0], 1), np.float32)], axis=1)
        return extract_palette_kmeans(rgba[None], k=_IMPORT_COLOR_CAP)
    return unique.astype(np.float32)


def import_palette(filepath: str | Path) -> np.ndarray:
    """Load a palette from an external file, extending load_palette with .gpl/.png.

    Returns the (K, 3) float32 palette. Does NOT save it — the caller decides
    whether to persist it via save_user_palette (the GUI does).
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_IMPORT_SUFFIXES:
        raise ValueError(
            f"Unsupported palette format '{suffix}'. Supported: {', '.join(_SUPPORTED_IMPORT_SUFFIXES)}"
        )
    if suffix == ".gpl":
        return _load_gpl(path)
    if suffix == ".png":
        return _load_png(path)
    return load_palette(path)  # .json / .hex / .pal


def import_and_save(filepath: str | Path, name: str | None = None) -> tuple[str, Path]:
    """Import an external palette and persist it into the user library.

    Returns (saved_name, saved_path). The name defaults to the file's stem.
    """
    colors = import_palette(filepath)
    stem = _sanitize_name(name if name is not None else Path(filepath).stem)
    path = save_user_palette(stem, colors)
    return stem, path
