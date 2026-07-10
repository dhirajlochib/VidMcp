"""FFmpeg helpers for final encode / audio mux / mask videos."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.ffmpeg")


def run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    cmd = ["ffmpeg", "-y", *args]
    log.debug("ffmpeg_cmd", cmd=" ".join(cmd[:20]))
    return subprocess.run(cmd, check=check, capture_output=True)


def run_ffmpeg_af(
    input_path: Path | str,
    output_path: Path | str,
    *,
    af: str | None = None,
    filter_complex: str | None = None,
    extra_inputs: list[Path | str] | None = None,
    map_audio: str = "0:a:0",
    ar: int = 48000,
    ac: int = 2,
) -> Path:
    """Apply audio filter(s) and write wav/aac output."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = ["-i", str(input_path)]
    for p in extra_inputs or []:
        cmd.extend(["-i", str(p)])
    if filter_complex:
        cmd.extend(["-filter_complex", filter_complex, "-map", map_audio if map_audio.startswith("[") else f"[{map_audio}]" if False else map_audio])
        # if map is like [a] use as-is
        if map_audio.startswith("["):
            # replace last map
            # rebuild carefully
            cmd = ["-i", str(input_path)]
            for p in extra_inputs or []:
                cmd.extend(["-i", str(p)])
            cmd.extend(["-filter_complex", filter_complex, "-map", map_audio])
        cmd.extend(["-ar", str(ar), "-ac", str(ac), str(output_path)])
    else:
        if af:
            cmd.extend(["-af", af])
        cmd.extend(["-ar", str(ar), "-ac", str(ac), str(output_path)])
    try:
        run_ffmpeg(cmd, check=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-1000:]
        log.error("ffmpeg_af_failed", error=err)
        raise RuntimeError(f"ffmpeg audio filter failed: {err}") from e
    return output_path


def measure_loudness(path: Path | str) -> dict[str, Any]:
    """EBU R128 / loudnorm print. Best-effort metrics."""
    path = Path(path)
    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        text = (proc.stderr or "") + (proc.stdout or "")
        # find last JSON object
        m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.DOTALL)
        if not m:
            m = re.search(r"\{[\s\S]*?\"input_i\"[\s\S]*?\}", text)
        if m:
            data = json.loads(m.group(0))
            return {
                "ok": True,
                "input_i": float(data.get("input_i", 0) or 0),
                "input_tp": float(data.get("input_tp", 0) or 0),
                "input_lra": float(data.get("input_lra", 0) or 0),
                "input_thresh": float(data.get("input_thresh", 0) or 0),
                "raw": data,
            }
    except Exception as e:  # noqa: BLE001
        log.warning("measure_loudness_failed", error=str(e)[:200])
    return {"ok": False, "input_i": None, "input_tp": None}


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
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
        )
        return out_path
    except subprocess.CalledProcessError as e:
        log.warning("audio_mux_failed_copy_video", error=e.stderr[-500:] if e.stderr else str(e))
        out_path.write_bytes(Path(video_no_audio).read_bytes())
        return out_path


def mux_audio_replace(
    video: Path | str,
    audio: Path | str,
    out_path: Path | str,
    *,
    reencode_video: bool = False,
    crf: int = 18,
) -> Path:
    """Replace/attach audio track onto video."""
    video = Path(video)
    audio = Path(audio)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vcodec = ["-c:v", "copy"] if not reencode_video else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf)]
    run_ffmpeg(
        [
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            *vcodec,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    )
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
