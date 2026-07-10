"""Speech-locked infographic overlays (brand cards)."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vidmcp.captions.fonts import resolve_font
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames, probe_video

log = get_logger("vidmcp.infographics")

LIME = (212, 255, 42)
CYAN = (42, 255, 209)
PAPER = (239, 239, 235)
MUTED = (150, 155, 165)


def _font(size: int) -> ImageFont.ImageFont:
    p = resolve_font()
    if p:
        try:
            return ImageFont.truetype(str(p), size)
        except Exception:
            pass
    return ImageFont.load_default()


def derive_beats_from_transcript(text: str, words: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Heuristic keyword → timed beats."""
    beats: list[dict[str, Any]] = []
    text_l = (text or "").lower()
    # number highlights from words
    if words:
        for w in words:
            tok = str(w.get("word") or "")
            if re.search(r"\d", tok):
                beats.append(
                    {
                        "start": float(w["start"]),
                        "end": float(w["end"]) + 1.5,
                        "kind": "number",
                        "value": re.sub(r"[^\d.%]", "", tok) or tok,
                        "label": "KEY NUMBER",
                    }
                )
    keywords = [
        ("gps", "GPS", "Satellite clocks need relativity corrections"),
        ("einstein", "EINSTEIN", "Time is relative"),
        ("dilat", "TIME DILATION", "Speed warps the clock"),
        ("dimension", "DIMENSIONS", "0D → 1D → 2D → 3D"),
        ("light", "LIGHT SPEED", "Near-c travel slows aging"),
        ("twin", "TWIN PARADOX", "Traveler vs Earth clocks"),
    ]
    for key, title, sub in keywords:
        if key in text_l:
            # place mid-video if no word timing
            start = 2.0
            if words:
                for w in words:
                    if key in str(w.get("word") or "").lower() or key in text_l:
                        start = float(w.get("start") or 2.0)
                        break
            beats.append({"start": start, "end": start + 4.0, "kind": "card", "title": title, "sub": sub})
    if not beats:
        beats.append({"start": 0.5, "end": 4.0, "kind": "card", "title": "KEY IDEA", "sub": (text or "Lesson")[:60]})
    # dedupe close numbers
    return beats[:12]


def _draw_card(draw: ImageDraw.ImageDraw, w: int, h: int, beat: dict, alpha: float) -> None:
    A = int(255 * alpha)
    font = _font(max(22, h // 28))
    font_sm = _font(max(16, h // 40))
    if beat.get("kind") == "number":
        panel_w = int(w * 0.32)
        px, py = w - panel_w - 36, int(h * 0.18)
        draw.rounded_rectangle((px, py, px + panel_w, py + 200), 16, fill=(8, 10, 16, int(210 * alpha)), outline=(*LIME, A), width=2)
        draw.text((px + 24, py + 24), beat.get("label", "NUMBER"), font=font_sm, fill=(*MUTED, A))
        draw.text((px + 24, py + 60), str(beat.get("value", "")), font=_font(max(48, h // 14)), fill=(*LIME, A))
    else:
        panel_w = int(w * 0.36)
        px, py = w - panel_w - 36, int(h * 0.2)
        draw.rounded_rectangle((px, py, px + panel_w, py + 160), 16, fill=(8, 10, 16, int(210 * alpha)), outline=(*CYAN, A), width=2)
        draw.text((px + 24, py + 28), str(beat.get("title", "")), font=font, fill=(*PAPER, A))
        draw.text((px + 24, py + 90), str(beat.get("sub", ""))[:48], font=font_sm, fill=(*CYAN, A))


def burn_infographics(
    video: Path | str,
    out: Path | str,
    beats: list[dict[str, Any]],
) -> dict[str, Any]:
    video = Path(video)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, meta.fps, (meta.width, meta.height))
    for idx, frame in iter_frames(video):
        t = idx / max(meta.fps, 1e-6)
        overlay = Image.new("RGBA", (meta.width, meta.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for b in beats:
            if b["start"] <= t <= b["end"]:
                fade = 0.35
                al = 1.0
                if t < b["start"] + fade:
                    al = (t - b["start"]) / fade
                if t > b["end"] - fade:
                    al = min(al, (b["end"] - t) / fade)
                _draw_card(draw, meta.width, meta.height, b, max(0.0, min(1.0, al)))
        ov = np.array(overlay)
        a = ov[:, :, 3:4].astype(np.float32) / 255.0
        rgb = ov[:, :, :3][:, :, ::-1].astype(np.float32)
        base = frame.astype(np.float32)
        out_f = base * (1 - a) + rgb * a
        writer.write(np.clip(out_f, 0, 255).astype(np.uint8))
    writer.release()
    return {"ok": True, "path": str(out), "n_beats": len(beats)}
