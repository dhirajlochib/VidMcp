"""Music engine v2 — chord-progression procedural score with intensity automation.

Intensity follows the footage energy curve so music swells where the edit peaks.
Provider hook (VIDMCP_MUSIC_PROVIDER) and MusicGen upgrade slot via models_registry.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vidmcp.audio.tracks import SR, _write_stereo, set_track
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.music")

# semitone offsets from root for each chord degree (triads, some 7ths)
_PROGRESSIONS: dict[str, list[list[int]]] = {
    "cinematic": [[0, 7, 12, 16], [-4, 3, 8, 12], [5, 12, 16, 21], [-2, 5, 10, 14]],  # I  vi  IV  V-ish
    "uplifting": [[0, 4, 7, 12], [5, 9, 12, 17], [-3, 4, 9, 12], [7, 11, 14, 19]],
    "tense": [[0, 3, 7, 10], [1, 4, 8, 11], [0, 3, 6, 10], [-2, 1, 5, 8]],
    "playful": [[0, 4, 7], [7, 11, 14], [5, 9, 12], [0, 4, 9]],
    "lofi": [[0, 3, 7, 10], [-4, 0, 3, 8], [5, 8, 12, 15], [3, 7, 10, 14]],
}

_ROOT_HZ = 110.0  # A2


def _note(freq: float, t: np.ndarray, amp: float) -> np.ndarray:
    ph = 2 * np.pi * freq * t
    return amp * (0.6 * np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph) + 0.05 * np.sin(0.5 * ph))


def synthesize_score(
    duration: float,
    *,
    style: str = "cinematic",
    bpm: float = 84.0,
    intensity_curve: list[float] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Render pad / pulse / keys stems (float32 stereo)."""
    rng = np.random.default_rng(seed)
    prog = _PROGRESSIONS.get(style, _PROGRESSIONS["cinematic"])
    n = int(duration * SR)
    t_all = np.arange(n, dtype=np.float32) / SR
    beat = 60.0 / bpm
    bar = beat * 4

    # per-sample intensity from per-second curve
    if intensity_curve:
        curve = np.asarray(intensity_curve, np.float32)
        idx = np.minimum((t_all).astype(np.int32), len(curve) - 1)
        inten = 0.35 + 0.65 * curve[idx]
    else:
        inten = 0.5 + 0.3 * np.sin(2 * np.pi * t_all / max(duration, 1) * 1.5 - np.pi / 2)

    pad = np.zeros(n, np.float32)
    keys = np.zeros(n, np.float32)
    pulse = np.zeros(n, np.float32)

    n_bars = int(np.ceil(duration / bar))
    for b in range(n_bars):
        chord = prog[b % len(prog)]
        s0 = int(b * bar * SR)
        s1 = min(int((b + 1) * bar * SR), n)
        if s0 >= n:
            break
        seg_t = t_all[s0:s1] - t_all[s0]
        env = np.minimum(seg_t / 0.8, 1.0) * np.minimum((seg_t[-1] - seg_t + 0.4) / 0.8, 1.0)
        for st in chord:
            f = _ROOT_HZ * (2 ** (st / 12.0))
            pad[s0:s1] += _note(f, seg_t, 0.10) * env
        # keys arpeggio on beats
        for k in range(4):
            ks = s0 + int(k * beat * SR)
            ke = min(ks + int(beat * 0.9 * SR), n)
            if ks >= n:
                break
            st = chord[k % len(chord)] + 12
            f = _ROOT_HZ * (2 ** (st / 12.0))
            kt = t_all[ks:ke] - t_all[ks]
            keys[ks:ke] += _note(f, kt, 0.09) * np.exp(-kt * 2.2)
        # pulse eighth notes
        for k in range(8):
            ps = s0 + int(k * beat / 2 * SR)
            pe = min(ps + int(0.09 * SR), n)
            if ps >= n:
                break
            pt = t_all[ps:pe] - t_all[ps]
            pulse[ps:pe] += np.sin(2 * np.pi * _ROOT_HZ * 2 * pt).astype(np.float32) * np.exp(-pt * 30) * 0.35

    pad *= inten
    keys *= 0.4 + 0.6 * inten
    pulse *= np.clip(inten - 0.35, 0, 1) * 1.4  # pulse only enters at higher energy
    noise_air = rng.normal(0, 0.004, n).astype(np.float32) * inten

    def st_split(x: np.ndarray, width: float) -> np.ndarray:
        d = int(SR * 0.012 * width)
        r = np.concatenate([np.zeros(d, np.float32), x[: n - d]]) if d else x
        return np.stack([x, r], axis=1)

    return {
        "pad": st_split(pad + noise_air, 1.0),
        "keys": st_split(keys, 0.6),
        "pulse": np.stack([pulse, pulse], axis=1),
    }


def generate_music_project(
    project: Any,
    prompt: str = "",
    style: str = "cinematic",
    bpm: float | None = None,
    duration_sec: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    m = project.manifest
    provider = os.environ.get("VIDMCP_MUSIC_PROVIDER", "").strip().lower()
    if provider and provider not in ("procedural", "builtin"):
        log.info("music_provider_requested_not_wired", provider=provider)

    if duration_sec is None:
        meta = m.source_meta or {}
        duration_sec = float(meta.get("duration") or (m.analysis or {}).get("duration_sec") or 30.0)
    # style inference from prompt
    pl = (prompt or "").lower()
    for key in _PROGRESSIONS:
        if key in pl:
            style = key
    if bpm is None:
        bpm = {"tense": 96, "playful": 104, "uplifting": 100, "lofi": 72}.get(style, 84)

    intensity = None
    try:
        from vidmcp.perception.indexer import load_index

        index = load_index(project) or {}
        intensity = index.get("energy")
    except Exception:  # noqa: BLE001
        pass

    stems = synthesize_score(duration_sec, style=style, bpm=float(bpm), intensity_curve=intensity, seed=seed)
    audio_dir = project.root / "audio"
    stem_paths = {}
    mix = np.zeros_like(stems["pad"])
    for name, buf in stems.items():
        p = audio_dir / f"music_{name}.wav"
        _write_stereo(buf, p)
        stem_paths[name] = project.rel(p)
        mix += buf
    peak = float(np.abs(mix).max())
    if peak > 0.9:
        mix *= 0.9 / peak
    full = audio_dir / f"music_{style}.wav"
    _write_stereo(mix, full)
    set_track(project, "bgm", project.rel(full), meta={"style": style, "bpm": bpm, "stems": stem_paths, "duck": True})
    return {
        "ok": True,
        "wav": project.rel(full),
        "style": style,
        "bpm": bpm,
        "duration_sec": round(duration_sec, 1),
        "stems": stem_paths,
        "paced_to_energy": intensity is not None,
        "provider": provider or "procedural",
    }


def beat_grid(bgm_path: Any, sr_hint: int = SR) -> dict[str, Any]:
    """Beat/tempo estimate (librosa if installed; onset autocorrelation fallback)."""
    from pathlib import Path

    p = Path(bgm_path)
    try:
        import librosa  # type: ignore

        y, sr = librosa.load(str(p), sr=22050, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(beats, sr=sr)
        return {"ok": True, "bpm": float(tempo), "beats_sec": [round(float(t), 3) for t in times][:400], "backend": "librosa"}
    except Exception:  # noqa: BLE001
        from vidmcp.audio.tracks import _load_stereo

        buf = _load_stereo(p).mean(axis=1)
        win = int(SR * 0.05)
        env = np.abs(buf[: (len(buf) // win) * win]).reshape(-1, win).mean(axis=1)
        d = np.clip(np.diff(env), 0, None)
        if len(d) < 40:
            return {"ok": False, "message": "Audio too short for beat analysis"}
        corr = np.correlate(d, d, mode="full")[len(d):]
        lo, hi = int(60 / 200 / 0.05), int(60 / 60 / 0.05)  # 200→60 bpm in envelope hops
        lag = int(np.argmax(corr[lo:hi])) + lo
        bpm = 60.0 / (lag * 0.05)
        thresh = d.mean() + d.std()
        beats = [round(float(i * 0.05), 3) for i in np.where(d > thresh)[0]]
        return {"ok": True, "bpm": round(bpm, 1), "beats_sec": beats[:400], "backend": "onset_autocorr"}
