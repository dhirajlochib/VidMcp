"""Audio media helpers: extract, ensure track, mux narration."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.audio.media")


def has_audio_stream(video_path: Path) -> bool:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            text=True,
        )
        return "audio" in out
    except Exception:
        return False


def extract_wav(video_path: Path, wav_path: Path, sr: int = 16000) -> Path:
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-vn",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return wav_path


def synthesize_narration_wav(
    text: str,
    wav_path: Path,
    *,
    voice: str | None = None,
) -> Path:
    """macOS `say` or espeak → wav. Fallback: generated tone bed (not speech)."""
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    aiff = wav_path.with_suffix(".aiff")

    if shutil.which("say"):
        cmd = ["say", "-o", str(aiff)]
        if voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav_path)],
            check=True,
            capture_output=True,
        )
        aiff.unlink(missing_ok=True)
        return wav_path

    if shutil.which("espeak"):
        wav_tmp = wav_path.with_suffix(".espeak.wav")
        subprocess.run(["espeak", text, "-w", str(wav_tmp)], check=True, capture_output=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_tmp), "-ac", "1", "-ar", "16000", str(wav_path)],
            check=True,
            capture_output=True,
        )
        wav_tmp.unlink(missing_ok=True)
        return wav_path

    # sine bed matching approx duration from word count
    dur = max(3.0, 0.35 * len(text.split()))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={dur}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    log.warning("tts_unavailable_used_tone_bed")
    return wav_path


def mux_audio_onto_video(video_path: Path, audio_wav: Path, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path


def ensure_video_with_narration(
    video_path: Path,
    *,
    narration: str,
    out_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """If video has no audio (or force), mux TTS narration."""
    video_path = Path(video_path)
    out_path = Path(out_path or video_path.with_name(video_path.stem + "_narrated.mp4"))
    if has_audio_stream(video_path) and not force:
        return {"ok": True, "path": str(video_path), "had_audio": True, "muxed": False}
    work = out_path.parent / "_narration"
    work.mkdir(parents=True, exist_ok=True)
    wav = work / "narration.wav"
    synthesize_narration_wav(narration, wav)
    mux_audio_onto_video(video_path, wav, out_path)
    return {
        "ok": True,
        "path": str(out_path),
        "had_audio": False,
        "muxed": True,
        "narration": narration[:200],
        "wav": str(wav),
    }
