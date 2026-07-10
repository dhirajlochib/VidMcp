"""Caption cues, ASS writer, ffmpeg burn-in."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from vidmcp.captions.fonts import resolve_font
from vidmcp.captions.styles import STYLES
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.captions")


def words_to_cues(
    words: list[dict[str, Any]],
    *,
    max_chars: int = 42,
    max_duration: float = 4.0,
) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    if not words:
        return cues
    buf: list[dict[str, Any]] = []
    char_count = 0
    for w in words:
        token = str(w.get("word") or "").strip()
        if not token:
            continue
        add = len(token) + (1 if buf else 0)
        dur = (buf[-1]["end"] - buf[0]["start"]) if buf else 0.0
        if buf and (char_count + add > max_chars or dur > max_duration):
            cues.append(
                {
                    "start": float(buf[0]["start"]),
                    "end": float(buf[-1]["end"]),
                    "text": " ".join(x["word"].strip() for x in buf),
                }
            )
            buf = []
            char_count = 0
        buf.append({"word": token, "start": float(w["start"]), "end": float(w["end"])})
        char_count += add
    if buf:
        cues.append(
            {
                "start": float(buf[0]["start"]),
                "end": float(buf[-1]["end"]),
                "text": " ".join(x["word"].strip() for x in buf),
            }
        )
    return cues


def _ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        cs = 0
        s += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_ass(
    cues: list[dict[str, Any]],
    path: Path | str,
    *,
    style: str = "brand",
    play_res_x: int = 1920,
    play_res_y: int = 1080,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    st = STYLES.get(style, STYLES["brand"])
    font = resolve_font()
    font_name = font.stem if font else "Arial"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{st['fontsize']},{st['primary']},{st['accent']},{st['outline']},{st['back']},{st['bold']},0,0,0,100,100,0,0,1,2,0,2,40,40,{st['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for c in cues:
        text = str(c["text"]).replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{_ts(c['start'])},{_ts(c['end'])},Default,,0,0,0,,{text}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def burn_captions(
    video: Path | str,
    out: Path | str,
    *,
    cues: list[dict[str, Any]] | None = None,
    ass_path: Path | str | None = None,
    style: str = "brand",
) -> dict[str, Any]:
    video = Path(video)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if ass_path is None:
        if not cues:
            raise ValueError("cues or ass_path required")
        ass_path = out.with_suffix(".ass")
        write_ass(cues, ass_path, style=style)
    ass_path = Path(ass_path)
    # escape for ffmpeg subtitles filter
    ass_esc = str(ass_path).replace("\\", "/").replace(":", "\\:")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"ass={ass_esc}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # fallback: re-encode audio too
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"subtitles={ass_esc}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    return {"ok": True, "path": str(out), "ass_path": str(ass_path), "style": style, "n_cues": len(cues or [])}
