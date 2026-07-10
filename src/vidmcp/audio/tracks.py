"""Multi-track audio model — named tracks (voice/bgm/sfx/ambience), VAD ducking v2, mixdown.

Tracks live in manifest.source_meta['audio_tracks']; mixdown is numpy-based
(load via ffmpeg → float32 48k stereo), then loudness-targeted per platform.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from vidmcp.audio.loudness import normalize_loudness
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.tracks")

SR = 48000


def _load_stereo(path: Path, sr: int = SR) -> np.ndarray:
    """Decode any audio/video to float32 stereo [n,2]."""
    tmp = path.parent / f"_dec_{path.stem}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-ar", str(sr), "-ac", "2",
         "-c:a", "pcm_s16le", str(tmp)],
        check=True, capture_output=True,
    )
    with wave.open(str(tmp), "rb") as wf:
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    tmp.unlink(missing_ok=True)
    return (data.reshape(-1, 2).astype(np.float32)) / 32768.0


def _write_stereo(samples: np.ndarray, path: Path, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((clipped * 32767).astype(np.int16).tobytes())


def _db(db: float) -> float:
    return float(10 ** (db / 20.0))


def speech_envelope(voice: np.ndarray, sr: int = SR, lookahead_s: float = 0.2) -> np.ndarray:
    """Smoothed speech-presence envelope [0,1] with lookahead (per sample, mono)."""
    mono = np.abs(voice).mean(axis=1)
    win = int(sr * 0.05)
    kernel = np.ones(win, np.float32) / win
    env = np.convolve(mono, kernel, mode="same")
    thresh = max(float(np.percentile(env, 30)) * 2.5, 0.004)
    presence = np.clip(env / thresh, 0, 1) ** 0.5
    # attack/release smoothing (fast attack 60ms, slow release 400ms)
    out = np.empty_like(presence)
    a_att = 1.0 - np.exp(-1.0 / (sr * 0.06))
    a_rel = 1.0 - np.exp(-1.0 / (sr * 0.4))
    prev = 0.0
    for i, p in enumerate(presence):
        coeff = a_att if p > prev else a_rel
        prev = prev + coeff * (p - prev)
        out[i] = prev
    # lookahead: shift envelope earlier so ducking starts before speech onset
    la = int(sr * lookahead_s)
    if la > 0:
        out = np.concatenate([out[la:], np.full(la, out[-1], np.float32)])
    return out


def duck_music(bgm: np.ndarray, voice: np.ndarray, *, floor_db: float = -13.0, sr: int = SR) -> np.ndarray:
    """VAD-driven sidechain: BGM dips to floor_db under speech, smooth, no pumping."""
    n = min(len(bgm), len(voice))
    env = speech_envelope(voice[:n], sr)
    gain = 1.0 - env * (1.0 - _db(floor_db))
    out = bgm.copy()
    out[:n] *= gain[:, None]
    return out


def stereo_widen(x: np.ndarray, amount: float = 0.25) -> np.ndarray:
    """Mid/side widening, mono-compatible (side gain limited)."""
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2
    side = side * (1.0 + float(np.clip(amount, 0, 0.6)))
    return np.stack([mid + side, mid - side], axis=1)


DEFAULT_TRACK_GAINS = {"voice": 0.0, "bgm": -14.0, "sfx": -16.0, "ambience": -30.0}


def get_tracks(project: Any) -> dict[str, Any]:
    return (project.manifest.source_meta or {}).setdefault("audio_tracks", {})


def set_track(project: Any, name: str, src: str, *, gain_db: float | None = None, meta: dict | None = None) -> dict[str, Any]:
    tracks = get_tracks(project)
    tracks[name] = {
        "src": src,
        "gain_db": gain_db if gain_db is not None else DEFAULT_TRACK_GAINS.get(name, -12.0),
        **(meta or {}),
    }
    project.manifest.append_history("set_audio_track", {"track": name, "src": src})
    project.save()
    return {"ok": True, "track": name, "tracks": sorted(tracks)}


def audio_tracks_project(project: Any) -> dict[str, Any]:
    tracks = get_tracks(project)
    # bootstrap from creator pipeline artifacts when empty
    pipe = (project.manifest.source_meta or {}).get("audio_pipeline") or {}
    if not tracks:
        if pipe.get("vocals"):
            tracks["voice"] = {"src": pipe["vocals"], "gain_db": 0.0}
        if pipe.get("bgm"):
            tracks["bgm"] = {"src": pipe["bgm"], "gain_db": -14.0, "duck": True}
    return {"ok": True, "tracks": tracks, "sr": SR}


def mixdown_project(
    project: Any,
    target: str = "youtube",
    widen_bgm: float = 0.25,
    duck_floor_db: float = -13.0,
) -> dict[str, Any]:
    m = project.manifest
    info = audio_tracks_project(project)
    tracks = info["tracks"]
    if not tracks:
        # fall back to source audio as single voice track
        if not m.source_video:
            return {"ok": False, "message": "No audio tracks and no source video"}
        tracks = {"voice": {"src": m.source_video, "gain_db": 0.0}}

    loaded: dict[str, np.ndarray] = {}
    for name, spec in tracks.items():
        p = project.abs(spec["src"])
        if not p.exists():
            log.warning("track_missing", track=name, src=spec["src"])
            continue
        buf = _load_stereo(p)
        buf *= _db(float(spec.get("gain_db", 0.0)))
        offset = float(spec.get("offset_sec", 0.0))
        if offset > 0:
            buf = np.concatenate([np.zeros((int(offset * SR), 2), np.float32), buf])
        loaded[name] = buf
    if not loaded:
        return {"ok": False, "message": "No decodable tracks"}

    n = max(len(b) for b in loaded.values())
    voice = loaded.get("voice")
    mix = np.zeros((n, 2), np.float32)
    for name, buf in loaded.items():
        b = buf
        if name in ("bgm", "ambience", "music"):
            if voice is not None and tracks.get(name, {}).get("duck", True):
                b = duck_music(b, voice, floor_db=duck_floor_db)
            if widen_bgm > 0:
                b = stereo_widen(b, widen_bgm)
        pad = np.zeros((n - len(b), 2), np.float32)
        mix += np.concatenate([b, pad]) if len(b) < n else b[:n]

    # soft-knee limiter before loudnorm
    peak = float(np.abs(mix).max())
    if peak > 0.98:
        mix *= 0.98 / peak

    audio_dir = project.root / "audio"
    raw = audio_dir / "mixdown_raw.wav"
    final = audio_dir / f"mixdown_{target}.wav"
    _write_stereo(mix, raw)
    norm = normalize_loudness(raw, final, target=target, audio_only=True)
    pipe = m.source_meta.setdefault("audio_pipeline", {})
    pipe["mix"] = project.rel(final)
    m.append_history("mixdown_audio", {"target": target, "lufs_out": norm.get("lufs_out")})
    project.save()
    return {
        "ok": True,
        "wav": project.rel(final),
        "target": target,
        "n_tracks": len(loaded),
        "tracks": sorted(loaded),
        "lufs_out": norm.get("lufs_out"),
        "tp_out": norm.get("tp_out"),
        "duration_sec": round(n / SR, 2),
    }


def edit_audio_track_project(project: Any, track: str, ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply simple ops to a track spec: set_gain, set_src, set_offset, enable_duck."""
    tracks = get_tracks(project)
    spec = tracks.setdefault(track, {"src": None, "gain_db": DEFAULT_TRACK_GAINS.get(track, -12.0)})
    for op in ops:
        kind = op.get("op")
        if kind == "set_gain":
            spec["gain_db"] = float(op.get("db", 0.0))
        elif kind == "set_src":
            spec["src"] = str(op.get("src"))
        elif kind == "set_offset":
            spec["offset_sec"] = float(op.get("sec", 0.0))
        elif kind == "enable_duck":
            spec["duck"] = bool(op.get("value", True))
        else:
            return {"ok": False, "message": f"Unknown track op '{kind}'"}
    project.manifest.append_history("edit_audio_track", {"track": track, "ops": [o.get("op") for o in ops]})
    project.save()
    return {"ok": True, "track": track, "spec": spec}
