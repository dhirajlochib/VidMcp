"""Per-platform loudness targeting — two-pass loudnorm with true-peak ceilings."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from vidmcp.compositor.ffmpeg_ops import measure_loudness
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.loudness")

# target integrated LUFS, true peak dBTP, LRA hint
TARGETS: dict[str, dict[str, float]] = {
    "youtube": {"lufs": -14.0, "tp": -1.0, "lra": 11.0},
    "reels": {"lufs": -14.0, "tp": -1.0, "lra": 9.0},
    "tiktok": {"lufs": -14.0, "tp": -1.0, "lra": 9.0},
    "podcast": {"lufs": -16.0, "tp": -1.5, "lra": 11.0},
    "broadcast": {"lufs": -23.0, "tp": -1.0, "lra": 15.0},
    "square": {"lufs": -14.0, "tp": -1.0, "lra": 9.0},
}


def target_for(name: str) -> dict[str, float]:
    key = (name or "youtube").lower()
    for k in TARGETS:
        if k in key:
            return TARGETS[k]
    return TARGETS["youtube"]


def normalize_loudness(
    src: Path | str,
    out: Path | str,
    *,
    target: str | dict[str, float] = "youtube",
    audio_only: bool = False,
) -> dict[str, Any]:
    """Two-pass loudnorm (measure → linear apply). Works on video (copies video) or wav."""
    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = target_for(target) if isinstance(target, str) else target
    measured = measure_loudness(src)
    ln = (
        f"loudnorm=I={spec['lufs']}:TP={spec['tp']}:LRA={spec['lra']}"
    )
    if measured.get("input_i") is not None:
        ln += (
            f":measured_I={measured['input_i']}:measured_TP={measured.get('input_tp', -1)}"
            f":measured_LRA={measured.get('input_lra', 7)}:measured_thresh={measured.get('input_thresh', -24)}"
            ":linear=true"
        )
    cmd = ["ffmpeg", "-y", "-i", str(src), "-af", ln]
    if audio_only or src.suffix.lower() in {".wav", ".mp3", ".m4a", ".aac", ".flac"}:
        cmd += ["-ar", "48000", str(out)]
    else:
        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    achieved = measure_loudness(out)
    return {
        "ok": True,
        "path": str(out),
        "target_lufs": spec["lufs"],
        "target_tp": spec["tp"],
        "lufs_in": measured.get("input_i"),
        "lufs_out": achieved.get("input_i"),
        "tp_out": achieved.get("input_tp"),
    }
