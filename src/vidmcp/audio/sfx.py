"""Procedural SFX kit (original, copyright-safe) — whoosh/impact/riser placed on edit events."""

from __future__ import annotations

from typing import Any

import numpy as np

from vidmcp.audio.tracks import SR, _write_stereo, set_track
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.sfx")


def _env(n: int, attack: float, release: float, sr: int = SR) -> np.ndarray:
    e = np.ones(n, np.float32)
    a = min(int(attack * sr), n)
    r = min(int(release * sr), n)
    if a > 0:
        e[:a] = np.linspace(0, 1, a)
    if r > 0:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def whoosh(duration: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    noise = rng.normal(0, 1, n).astype(np.float32)
    # sweep a resonant band down 4kHz→300Hz via time-varying one-pole pair
    freqs = np.geomspace(4000, 300, n)
    out = np.zeros(n, np.float32)
    y1 = y2 = 0.0
    for i in range(n):
        f = freqs[i]
        q = 2.0
        w = 2 * np.pi * f / SR
        alpha = np.sin(w) / (2 * q)
        b0 = alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w)
        a2 = 1 - alpha
        x = noise[i]
        y = (b0 / a0) * x - (a1 / a0) * y1 - (a2 / a0) * y2
        y2, y1 = y1, y
        out[i] = y
    out *= _env(n, 0.12, 0.25)
    peak = float(np.abs(out).max())
    if peak > 0:
        out *= 0.8 / peak
    return np.stack([out, out], axis=1)


def impact(duration: float = 0.6, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (55 * np.exp(-t * 3)) * t) * np.exp(-t * 6)
    crack = rng.normal(0, 1, n).astype(np.float32) * np.exp(-t * 25) * 0.5
    out = (thump + crack).astype(np.float32) * _env(n, 0.002, 0.3)
    return np.stack([out, out], axis=1)


def riser(duration: float = 1.5, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    freq = np.geomspace(180, 900, n)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    tone = np.sin(phase) * 0.5 + rng.normal(0, 0.2, n)
    out = (tone * np.linspace(0.05, 1.0, n) ** 2).astype(np.float32) * _env(n, 0.05, 0.08)
    return np.stack([out * 0.9, out], axis=1)  # slight stereo tilt


GENERATORS = {"whoosh": whoosh, "impact": impact, "riser": riser}


def _auto_events(project: Any) -> list[dict[str, Any]]:
    """Derive SFX points from graphics layers, B-roll windows, and energy peaks."""
    m = project.manifest
    events: list[dict[str, Any]] = []
    for layer in m.layers.layers:
        meta = layer.meta or {}
        t = meta.get("t_start")
        if t is None:
            continue
        kind = "whoosh" if layer.kind.value in ("overlay", "broll") else "impact"
        events.append({"t": float(t), "kind": kind})
    try:
        from vidmcp.perception.indexer import load_index

        index = load_index(project) or {}
        for t in (index.get("energy_peaks_sec") or [])[:3]:
            events.append({"t": float(t), "kind": "riser", "offset": -1.2})
    except Exception:  # noqa: BLE001
        pass
    events.sort(key=lambda e: e["t"])
    # de-dup within 1s
    out: list[dict[str, Any]] = []
    for e in events:
        if out and abs(e["t"] - out[-1]["t"]) < 1.0:
            continue
        out.append(e)
    return out


def add_sfx_project(
    project: Any,
    auto: bool = True,
    events: list[dict[str, Any]] | None = None,
    gain_db: float = -18.0,
) -> dict[str, Any]:
    if events is None:
        events = _auto_events(project) if auto else []
    if not events:
        return {"ok": True, "n_placed": 0, "message": "No SFX events (add graphics/broll first or pass events)"}
    duration = max(float(e["t"]) for e in events) + 3.0
    buf = np.zeros((int(duration * SR), 2), np.float32)
    placed = []
    for i, e in enumerate(events):
        gen = GENERATORS.get(str(e.get("kind", "whoosh")))
        if gen is None:
            continue
        clip = gen(seed=i)
        t = float(e["t"]) + float(e.get("offset", 0.0))
        start = max(0, int(t * SR))
        end = min(len(buf), start + len(clip))
        buf[start:end] += clip[: end - start]
        placed.append({"t": round(t, 2), "kind": e.get("kind")})
    out = project.root / "audio" / "sfx.wav"
    _write_stereo(buf, out)
    set_track(project, "sfx", project.rel(out), gain_db=gain_db, meta={"events": placed})
    return {"ok": True, "n_placed": len(placed), "wav": project.rel(out), "events": placed[:10]}
