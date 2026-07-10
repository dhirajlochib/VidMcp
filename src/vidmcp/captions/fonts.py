"""Resolve fonts for burn-in captions."""

from __future__ import annotations

import os
from pathlib import Path

_CANDIDATES = [
    Path(__file__).resolve().parents[3] / "scripts" / "fonts" / "Outfit-Medium.ttf",
    Path(__file__).resolve().parents[3] / "scripts" / "fonts" / "Outfit.ttf",
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def resolve_font(prefer: str | None = None) -> Path | None:
    env = os.environ.get("VIDMCP_FONT_DIR")
    if env:
        d = Path(env)
        for name in ("Outfit-Medium.ttf", "Outfit.ttf", "DejaVuSans.ttf", "Arial.ttf"):
            p = d / name
            if p.exists():
                return p
    if prefer and Path(prefer).exists():
        return Path(prefer)
    for p in _CANDIDATES:
        if p.exists():
            return p
    return None
