"""process_audio denoise pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from vidmcp.audio.process import process_audio


def _make_noisy_wav(path: Path, sr: int = 48000, dur: float = 1.5):
    # ffmpeg sine + noise
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={dur}",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=d={dur}:c=pink:a=0.15",
            "-filter_complex",
            "[0][1]amix=inputs=2:duration=first",
            "-ar",
            str(sr),
            "-ac",
            "2",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_process_audio_writes_and_keeps_duration(tmp_path: Path):
    raw = tmp_path / "noisy.wav"
    out = tmp_path / "clean.wav"
    _make_noisy_wav(raw)
    result = process_audio(raw, out, strength=0.8)
    assert result["ok"]
    assert Path(result["path"]).exists()
    assert result["duration_sec"] > 1.0
    assert abs(result["duration_sec"] - 1.5) < 0.25
