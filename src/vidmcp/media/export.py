"""Export aspect-ratio presets without stretch."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video

log = get_logger("vidmcp.export")

PRESETS: dict[str, tuple[int, int] | None] = {
    "youtube_16x9": (1920, 1080),
    "reels_9x16": (1080, 1920),
    "square_1x1": (1080, 1080),
    "source": None,
}


def export_render(
    video: Path | str,
    out: Path | str,
    *,
    preset: str = "youtube_16x9",
    audio: Path | str | None = None,
    crf: int = 18,
    loudnorm: bool = False,
    target_lufs: float = -14.0,
) -> dict[str, Any]:
    video = Path(video)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset {preset}. Available: {sorted(PRESETS)}")
    size = PRESETS[preset]
    meta = probe_video(video)

    vf = "null"
    tw, th = meta.width, meta.height
    if size is not None:
        tw, th = size
        # scale to fit, pad center — never stretch
        vf = (
            f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=0x050507,"
            f"setsar=1"
        )

    cmd = ["ffmpeg", "-y", "-i", str(video)]
    audio_path = Path(audio) if audio else None
    if audio_path and audio_path.exists():
        cmd.extend(["-i", str(audio_path)])

    cmd.extend(["-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf)])

    if audio_path and audio_path.exists():
        af = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11" if loudnorm else "anull"
        cmd.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-af",
                af,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )
    else:
        cmd.extend(["-map", "0:v:0", "-map", "0:a:0?"])
        if loudnorm:
            cmd.extend(["-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    cmd.extend(["-shortest", "-movflags", "+faststart", str(out)])
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"export_render failed: {err}") from e

    out_meta = probe_video(out)
    return {
        "ok": True,
        "path": str(out),
        "preset": preset,
        "width": out_meta.width,
        "height": out_meta.height,
        "duration_sec": out_meta.duration_sec,
        "has_audio": out_meta.has_audio,
    }
