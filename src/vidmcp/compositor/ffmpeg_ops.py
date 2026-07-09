"""FFmpeg helpers for final encode / audio mux / mask videos."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.ffmpeg")


def run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    cmd = ["ffmpeg", "-y", *args]
    log.debug("ffmpeg_cmd", cmd=" ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True)


def mux_audio(video_no_audio: Path, source_with_audio: Path, out_path: Path) -> Path:
    """Copy audio from source onto rendered video."""
    out_path = Path(out_path)
    try:
        run_ffmpeg(
            [
                "-i",
                str(video_no_audio),
                "-i",
                str(source_with_audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(out_path),
            ]
        )
        return out_path
    except subprocess.CalledProcessError as e:
        log.warning("audio_mux_failed_copy_video", error=e.stderr[-500:] if e.stderr else str(e))
        # fallback copy
        out_path.write_bytes(Path(video_no_audio).read_bytes())
        return out_path


def encode_frames_dir(
    frames_pattern: str,
    out_path: Path,
    *,
    fps: float,
    crf: int = 18,
) -> Path:
    out_path = Path(out_path)
    run_ffmpeg(
        [
            "-framerate",
            str(fps),
            "-i",
            frames_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            str(out_path),
        ]
    )
    return out_path


def extract_preview_gif(video: Path, out_gif: Path, *, fps: float = 8, scale: int = 480) -> Path:
    out_gif = Path(out_gif)
    try:
        run_ffmpeg(
            [
                "-i",
                str(video),
                "-vf",
                f"fps={fps},scale={scale}:-1:flags=lanczos",
                "-frames:v",
                "48",
                str(out_gif),
            ]
        )
    except subprocess.CalledProcessError:
        log.warning("preview_gif_failed")
    return out_gif
