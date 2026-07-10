"""Vocal denoise + enhance + loudnorm (production path)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from vidmcp.audio.media import extract_wav, has_audio_stream
from vidmcp.compositor.ffmpeg_ops import measure_loudness
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.audio.process")


def process_audio(
    video_or_wav: Path | str,
    out_wav: Path | str,
    *,
    strength: float = 0.7,
    highpass_hz: float = 100.0,
    lowpass_hz: float = 10000.0,
    target_lufs: float = -14.0,
    agate: bool = True,
) -> dict[str, Any]:
    """
    Denoise + vocal enhance + loudnorm.
    strength 0..1 scales afftdn / gate aggressiveness.
    """
    src = Path(video_or_wav)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    strength = max(0.0, min(1.0, float(strength)))

    work = out_wav.parent / "_process_work"
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "raw.wav"

    if src.suffix.lower() in {".wav", ".aiff", ".flac", ".mp3", ".m4a", ".aac"}:
        # normalize to wav
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "2", "-ar", "48000", str(raw)],
            check=True,
            capture_output=True,
        )
    else:
        if not has_audio_stream(src):
            return {
                "ok": False,
                "path": None,
                "warnings": ["No audio stream on video"],
            }
        extract_wav(src, raw, sr=48000)
        # re-extract stereo 48k
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "48000", str(raw)],
            check=True,
            capture_output=True,
        )

    pre = measure_loudness(raw)
    nr = int(10 + 12 * strength)  # 10..22
    nf = int(-35 + 8 * (1 - strength))  # quieter floor when stronger
    gate = "agate=threshold=0.02:ratio=2:attack=5:release=80," if agate else ""

    af = (
        f"highpass=f={highpass_hz},"
        f"lowpass=f={lowpass_hz},"
        f"afftdn=nr={nr}:nf={nf}:tn=1,"
        f"anlmdn=s=0.00025:p=0.002:r=0.002:m=15,"
        f"acompressor=threshold=-22dB:ratio=3.5:attack=12:release=160:makeup=5,"
        f"equalizer=f=200:t=q:w=1.0:g=-2.5,"
        f"equalizer=f=3000:t=q:w=1.1:g={2.5 + 2.5 * strength},"
        f"equalizer=f=5500:t=q:w=1.2:g={1.5 + 1.5 * strength},"
        f"{gate}"
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
    )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw),
                "-af",
                af,
                "-ar",
                "48000",
                "-ac",
                "2",
                str(out_wav),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-800:]
        log.error("process_audio_failed", error=err)
        raise RuntimeError(f"process_audio failed: {err}") from e

    post = measure_loudness(out_wav)
    # duration
    try:
        dur = float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(out_wav),
                ],
                text=True,
            ).strip()
        )
    except Exception:
        dur = 0.0

    return {
        "ok": True,
        "path": str(out_wav),
        "duration_sec": dur,
        "strength": strength,
        "target_lufs": target_lufs,
        "lufs_in": pre.get("input_i"),
        "lufs_out": post.get("input_i"),
        "true_peak_out": post.get("input_tp"),
        "warnings": [],
    }
