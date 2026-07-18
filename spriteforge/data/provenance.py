# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-Commercial
# Copyright (C) 2026 Matthew Burke <matthew.wesley.burke@gmail.com>

"""
Enforce and record CC0 and Private Research provenance for all ingested training datasets.
"""

from __future__ import annotations

import datetime
from pathlib import Path


def log_provenance(
    source_name: str,
    source_url: str,
    author: str,
    license_status: str = "CC0 1.0 Universal",
    license_tier: str = "cc0_public",
    notes: str = "",
    log_file: str | Path = "ASSETS_PROVENANCE.md"
) -> None:
    """Append an asset or pack record to the provenance log.

    Args:
        license_tier: 'cc0_public' (safe to share/sell weights), 'private_research' (strictly
            for personal/private model weights), or 'custom_user'.
    """
    path = Path(log_file)
    if not path.exists():
        path = Path(__file__).resolve().parent.parent.parent / "ASSETS_PROVENANCE.md"

    if not path.exists():
        return  # Silently skip if log doesn't exist

    date_str = datetime.date.today().isoformat()
    tier_badge = f"`[{license_tier.upper()}]`"
    row = f"| {tier_badge} {source_name} | {source_url} | {author} | **{license_status}** | {date_str} | {notes} |\n"

    with open(path, "a", encoding="utf-8") as f:
        f.write(row)
