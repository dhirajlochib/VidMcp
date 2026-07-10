"""Brand kit — persistent fonts/colors/logo/style auto-injected across all surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import orjson

from vidmcp.config import get_settings
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.brand")

# default kit = the visual language from the shipped demo edits
DEFAULT_KIT: dict[str, Any] = {
    "name": "default",
    "colors": {
        "primary": "#d4ff2a",    # lime
        "secondary": "#2affd1",  # cyan
        "accent": "#b18cff",     # violet
        "paper": "#efefeb",
        "muted": "#969ba5",
        "dark": "#06080e",
    },
    "font": None,               # path to .ttf; None → captions/fonts resolver
    "logo": None,               # path to PNG with alpha
    "logo_position": "top_right",
    "watermark_opacity": 0.55,
    "safe_margin_pct": 5.0,
    "caption_style": "brand",
    "lut": "filmic_soft",
    "bgm_style": "cinematic",
    "lower_third_layout": "bar_left",
}


def _brand_dir() -> Path:
    d = get_settings().workspace_root / "brand"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_brand_kit(name: str = "default") -> dict[str, Any]:
    p = _brand_dir() / f"{name}.json"
    if p.exists():
        try:
            kit = orjson.loads(p.read_bytes())
            return {**DEFAULT_KIT, **kit}
        except Exception as e:  # noqa: BLE001
            log.warning("brand_kit_corrupt", path=str(p), error=str(e))
    return dict(DEFAULT_KIT)


def set_brand_kit(kit: dict[str, Any], name: str = "default") -> dict[str, Any]:
    merged = {**DEFAULT_KIT, **(kit or {}), "name": name}
    p = _brand_dir() / f"{name}.json"
    p.write_bytes(orjson.dumps(merged, option=orjson.OPT_INDENT_2))
    return {"ok": True, "name": name, "path": str(p), "kit": merged}


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def color(kit: dict[str, Any], key: str) -> tuple[int, int, int]:
    return hex_to_rgb(kit.get("colors", {}).get(key, DEFAULT_KIT["colors"].get(key, "#ffffff")))


def extract_brand_from_video(video_path: str, name: str = "extracted", save: bool = True) -> dict[str, Any]:
    """Dominant palette from an existing branded video → candidate kit."""
    from vidmcp.utils.video_io import sample_frames

    frames = sample_frames(Path(video_path), max_frames=12, max_side=320)
    if not frames:
        return {"ok": False, "message": "Cannot sample video"}
    pixels = np.concatenate([f[2].reshape(-1, 3) for f in frames]).astype(np.float32)
    # k-means for 5 dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels[:: max(1, len(pixels) // 20000)], 5, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=5)
    order = np.argsort(counts)[::-1]
    cols = [centers[i] for i in order]  # BGR

    def to_hex(bgr) -> str:
        return f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"

    # most-saturated non-dominant color = primary accent
    def sat(bgr) -> float:
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        return float(hsv[1])

    accents = sorted(cols[1:], key=sat, reverse=True)
    kit = {
        "name": name,
        "colors": {
            "dark": to_hex(min(cols, key=lambda c: sum(c))),
            "paper": to_hex(max(cols, key=lambda c: sum(c))),
            "primary": to_hex(accents[0]) if accents else DEFAULT_KIT["colors"]["primary"],
            "secondary": to_hex(accents[1]) if len(accents) > 1 else DEFAULT_KIT["colors"]["secondary"],
            "accent": to_hex(accents[2]) if len(accents) > 2 else DEFAULT_KIT["colors"]["accent"],
            "muted": DEFAULT_KIT["colors"]["muted"],
        },
    }
    if save:
        return set_brand_kit(kit, name=name)
    return {"ok": True, "kit": {**DEFAULT_KIT, **kit}}
