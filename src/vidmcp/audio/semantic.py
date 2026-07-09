"""Audio–semantic coupling: energy/onset timeline + keyword cues for scene beats / FX.

Uses librosa when available; otherwise ffmpeg+numpy RMS envelope (always works).
"""

from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.audio")


def _extract_wav(video: Path, wav: Path, sr: int = 16000) -> Path:
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", str(sr), "-vn", str(wav)],
        check=True,
        capture_output=True,
    )
    return wav


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        frames = w.readframes(n)
        if w.getsampwidth() == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
    return audio, sr


def extract_audio_timeline(video_path: Path, *, hop_ms: float = 50.0, work_dir: Path | None = None) -> dict[str, Any]:
    video_path = Path(video_path)
    work_dir = Path(work_dir or video_path.parent / "audio_work")
    wav = work_dir / "audio.wav"
    try:
        _extract_wav(video_path, wav)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"audio extract failed: {e}", "has_audio": False}

    audio, sr = _load_wav_mono(wav)
    hop = max(1, int(sr * hop_ms / 1000.0))
    # RMS envelope
    n = len(audio) // hop
    rms = []
    for i in range(n):
        seg = audio[i * hop : (i + 1) * hop]
        rms.append(float(np.sqrt(np.mean(seg**2) + 1e-12)))
    rms_a = np.array(rms, dtype=np.float32)
    if rms_a.max() > 0:
        rms_n = rms_a / rms_a.max()
    else:
        rms_n = rms_a

    # onset-like peaks
    diff = np.diff(rms_n, prepend=rms_n[0])
    thr = float(np.percentile(diff, 90)) if len(diff) else 0.1
    onsets = [i for i, d in enumerate(diff) if d > thr and rms_n[i] > 0.25]
    # debounce
    debounced = []
    last = -999
    for i in onsets:
        if i - last >= 3:
            debounced.append(i)
            last = i
    times = [i * hop_ms / 1000.0 for i in debounced]

    # optional librosa refine
    backend = "rms"
    try:
        import librosa

        y, _ = librosa.load(str(wav), sr=sr)
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr)
        times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
        backend = "librosa"
    except Exception:  # noqa: BLE001
        pass

    return {
        "ok": True,
        "has_audio": True,
        "sample_rate": sr,
        "duration_sec": float(len(audio) / max(sr, 1)),
        "rms": rms_n.tolist()[:: max(1, len(rms_n) // 200)],
        "onset_times": times[:200],
        "mean_energy": float(rms_n.mean()) if len(rms_n) else 0.0,
        "peak_energy": float(rms_n.max()) if len(rms_n) else 0.0,
        "backend": backend,
        "wav_path": str(wav),
    }


def sync_audio_semantics(
    audio_timeline: dict[str, Any],
    *,
    transcript: str | None = None,
    keywords: list[str] | None = None,
    scene_steps: int = 5,
) -> dict[str, Any]:
    """Map onsets + optional keyword cues to FX/scene beat events."""
    onsets = list(audio_timeline.get("onset_times") or [])
    keywords = keywords or []
    events: list[dict[str, Any]] = []

    # scene step markers evenly on strong onsets
    if onsets and scene_steps > 0:
        picks = np.linspace(0, len(onsets) - 1, num=min(scene_steps, len(onsets)), dtype=int)
        for si, idx in enumerate(picks):
            events.append(
                {
                    "type": "scene_step",
                    "t": float(onsets[int(idx)]),
                    "step": int(si),
                    "action": "advance_manim_or_procedural_step",
                }
            )

    # energy-driven particle density curve
    rms = audio_timeline.get("rms") or []
    if rms:
        events.append(
            {
                "type": "particle_density_curve",
                "samples": [float(x) for x in rms],
                "action": "modulate_particle_intensity",
            }
        )

    # transcript keyword hits (uniform proxy if no word timestamps)
    if transcript and keywords:
        words = re.findall(r"[A-Za-z']+", transcript.lower())
        dur = float(audio_timeline.get("duration_sec") or 1.0)
        for kw in keywords:
            kw_l = kw.lower()
            hits = [i for i, w in enumerate(words) if kw_l in w]
            for hi in hits:
                t = dur * (hi / max(len(words), 1))
                events.append(
                    {
                        "type": "keyword",
                        "keyword": kw,
                        "t": float(t),
                        "action": "emphasize_formula_or_fx",
                    }
                )

    events.sort(key=lambda e: float(e.get("t") or 0))
    return {
        "ok": True,
        "events": events,
        "n_events": len(events),
        "onset_count": len(onsets),
    }
