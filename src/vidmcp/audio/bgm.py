"""Copyright-safe ambient BGM generation + speech-ducked mix."""

from __future__ import annotations

import struct
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.audio.bgm")


def _soft_sin(freq: float, t: np.ndarray, amp: float) -> np.ndarray:
    ph = 2 * np.pi * freq * t
    return amp * (0.65 * np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph))


def generate_ambient_bgm(
    out_wav: Path | str,
    duration_sec: float,
    *,
    seed: int = 0,
    style: str = "cinematic",
    sr: int = 48000,
) -> dict[str, Any]:
    """Original ambient pad (not copyrighted material). Mid-heavy for laptop speakers."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = max(1.0, float(duration_sec))
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    rng = np.random.default_rng(seed)

    styles = {
        "cinematic": dict(pad_scale=1.0, piano_scale=1.0, swell=28.0),
        "soft": dict(pad_scale=0.75, piano_scale=0.5, swell=40.0),
        "pulse": dict(pad_scale=0.9, piano_scale=0.35, swell=12.0),
    }
    st = styles.get(style, styles["cinematic"])

    pad = np.zeros_like(t)
    for f, a in [
        (55, 0.10),
        (82.4, 0.09),
        (110, 0.08),
        (164.8, 0.07),
        (220, 0.06),
        (329.6, 0.05),
        (440, 0.035),
    ]:
        lfo = 1 + 0.05 * np.sin(2 * np.pi * (0.04 + f * 1e-5) * t)
        pad += _soft_sin(f, t, a * st["pad_scale"]) * lfo
    swell = 0.55 + 0.45 * np.sin(2 * np.pi * t / st["swell"])
    pad *= swell

    piano = np.zeros_like(t)
    motif = [
        (1.5, 261.6, 0.12, 3),
        (5, 311.1, 0.1, 3),
        (9, 392, 0.11, 4),
        (14, 349, 0.09, 3),
        (18, 261.6, 0.1, 4),
        (24, 196, 0.11, 5),
        (30, 329.6, 0.12, 5),
        (36, 392, 0.1, 4),
    ]
    for start, f, amp, length in motif:
        i0 = int(start * sr)
        n = int(length * sr)
        if i0 >= len(t):
            continue
        n = min(n, len(t) - i0)
        tt = np.arange(n) / sr
        tone = _soft_sin(f, tt, amp * st["piano_scale"]) + 0.35 * _soft_sin(f * 2, tt, amp * 0.35 * st["piano_scale"])
        tone *= np.exp(-tt * 0.5)
        piano[i0 : i0 + n] += tone

    dust = rng.normal(0, 1, len(t)).astype(np.float64)
    for _ in range(4):
        dust = np.convolve(dust, np.ones(80) / 80, mode="same")
    dust *= 0.012

    mix = pad + piano + dust
    fade = int(2 * sr)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)
    peak = float(np.max(np.abs(mix)) + 1e-9)
    mix = mix / peak * 0.55
    left = mix * 0.95 + np.roll(mix, 100) * 0.05
    right = mix * 0.95 + np.roll(mix, -80) * 0.05
    stereo = np.stack([left, right], axis=1)
    pcm = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return {"ok": True, "path": str(out_wav), "duration_sec": duration_sec, "style": style, "sr": sr}


def _speech_envelope(wav_path: Path, hop_ms: float = 50.0) -> tuple[np.ndarray, int]:
    """Return mono float envelope sampled every hop_ms and sample rate."""
    # load via ffmpeg to 16k mono
    import tempfile

    tmp = wav_path.with_suffix(".env16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-ac", "1", "-ar", "16000", str(tmp)],
        check=True,
        capture_output=True,
    )
    with wave.open(str(tmp), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(1, int(sr * hop_ms / 1000))
    env = []
    for i in range(0, len(data), hop):
        chunk = data[i : i + hop]
        env.append(float(np.sqrt(np.mean(chunk**2))) if len(chunk) else 0.0)
    try:
        tmp.unlink()
    except OSError:
        pass
    arr = np.array(env, dtype=np.float32)
    if arr.max() > 1e-6:
        arr = arr / arr.max()
    return arr, sr


def mix_bgm(
    vocals_wav: Path | str,
    out_wav: Path | str,
    *,
    bgm_wav: Path | str | None = None,
    bgm_volume: float = 0.35,
    duck: bool = True,
    fade_in: float = 1.2,
    fade_out: float = 2.5,
    style: str = "cinematic",
    seed: int = 0,
) -> dict[str, Any]:
    """Mix BGM under vocals. Uses normalize=0 so BGM stays audible."""
    vocals_wav = Path(vocals_wav)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # duration of vocals
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
                str(vocals_wav),
            ],
            text=True,
        ).strip()
        or "0"
    )

    if bgm_wav is None:
        bgm_path = out_wav.parent / "bgm_generated.wav"
        generate_ambient_bgm(bgm_path, max(dur + 1.0, 3.0), seed=seed, style=style)
    else:
        bgm_path = Path(bgm_wav)

    # optional duck: pre-scale bgm with envelope via volume filter expression is hard;
    # use sidechaincompress if available, else static volume.
    vol = max(0.05, min(1.0, float(bgm_volume)))
    fade_out_start = max(0.0, dur - fade_out)

    if duck:
        # sidechaincompress: bg ducked by vocals
        fc = (
            f"[1:a]atrim=0:{dur},asetpts=PTS-STARTPTS,volume={vol},"
            f"afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_out_start}:d={fade_out}[bg];"
            f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.05[vox];"
            f"[bg][vox]sidechaincompress=threshold=0.02:ratio=6:attack=50:release=300:level_sc=1[bgd];"
            f"[vox][bgd]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
            f"alimiter=limit=0.94:level=false[a]"
        )
        map_a = "[a]"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(vocals_wav),
                    "-i",
                    str(bgm_path),
                    "-filter_complex",
                    fc,
                    "-map",
                    map_a,
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(out_wav),
                ],
                check=True,
                capture_output=True,
            )
            method = "sidechaincompress"
        except subprocess.CalledProcessError:
            duck = False  # fall through
            method = "fallback_static"

    if not duck or not out_wav.exists():
        fc = (
            f"[1:a]atrim=0:{dur},asetpts=PTS-STARTPTS,volume={vol},"
            f"afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_out_start}:d={fade_out}[bg];"
            f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.05[vox];"
            f"[vox][bg]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
            f"alimiter=limit=0.94:level=false[a]"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(vocals_wav),
                "-i",
                str(bgm_path),
                "-filter_complex",
                fc,
                "-map",
                "[a]",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(out_wav),
            ],
            check=True,
            capture_output=True,
        )
        method = "static_mix"

    return {
        "ok": True,
        "path": str(out_wav),
        "bgm_path": str(bgm_path),
        "bgm_volume": vol,
        "duck": duck,
        "method": method,
        "duration_sec": dur,
        "style": style,
    }
