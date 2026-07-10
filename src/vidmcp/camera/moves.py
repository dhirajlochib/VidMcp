"""Camera moves — emphasis punch-ins, Ken Burns drift, slow zoom. Baked via crop-track renderer."""

from __future__ import annotations

from typing import Any

import numpy as np

from vidmcp.camera.reframe import render_crop_track
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video

log = get_logger("vidmcp.camera_moves")


def _ease_out(x: float) -> float:
    return 1 - (1 - x) ** 3


def punch_in_zoom_fn(punches: list[dict[str, Any]], max_zoom: float = 1.12):
    """Zoom track: quick eased punch at each event, hold, ease back."""

    def fn(t: float) -> float:
        z = 1.0
        for p in punches:
            t0 = float(p["t"])
            zin = float(p.get("zoom", max_zoom))
            attack, hold, release = 0.28, float(p.get("hold", 1.6)), 0.5
            if t0 <= t < t0 + attack:
                z = max(z, 1.0 + (zin - 1.0) * _ease_out((t - t0) / attack))
            elif t0 + attack <= t < t0 + attack + hold:
                z = max(z, zin)
            elif t0 + attack + hold <= t < t0 + attack + hold + release:
                k = 1 - _ease_out((t - t0 - attack - hold) / release)
                z = max(z, 1.0 + (zin - 1.0) * k)
        return z

    return fn


def drift_zoom_fn(duration: float, max_zoom: float = 1.08):
    def fn(t: float) -> float:
        return 1.0 + (max_zoom - 1.0) * min(t / max(duration, 1e-3), 1.0)

    return fn


def _emphasis_events(project: Any, min_gap: float = 6.0, limit: int = 8) -> list[dict[str, Any]]:
    """Energy peaks + kept dramatic pauses → punch-in points."""
    events: list[float] = []
    try:
        from vidmcp.perception.indexer import load_index

        index = load_index(project) or {}
        energy = index.get("energy") or []
        peaks = index.get("energy_peaks_sec") or []
        events.extend(float(t) for t in peaks if t < len(energy))
    except Exception:  # noqa: BLE001
        pass
    events.sort()
    picked: list[float] = []
    for t in events:
        if not picked or t - picked[-1] >= min_gap:
            picked.append(t)
        if len(picked) >= limit:
            break
    # alternate zoom in/out rhythm
    return [{"t": t, "zoom": 1.12 if i % 2 == 0 else 1.07} for i, t in enumerate(picked)]


def add_camera_moves_project(
    project: Any,
    style: str = "emphasis_punch",
    max_zoom: float = 1.12,
    events: list[dict[str, Any]] | None = None,
    render: bool = True,
) -> dict[str, Any]:
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    src = project.abs(m.renders[-1]["output_path"]) if m.renders else project.abs(m.source_video)
    if not src.exists():
        src = project.abs(m.source_video)
    meta = probe_video(src)

    if style == "emphasis_punch":
        punches = events or _emphasis_events(project)
        if not punches:
            return {"ok": True, "n_moves": 0, "message": "No emphasis events found (build_footage_index first)"}
        zoom_fn = punch_in_zoom_fn(punches, max_zoom)
        track_desc = punches
    elif style in ("slow_drift", "ken_burns"):
        zoom_fn = drift_zoom_fn(meta.duration_sec, max_zoom=min(max_zoom, 1.1))
        track_desc = [{"t": 0, "zoom": 1.0}, {"t": meta.duration_sec, "zoom": max_zoom}]
    else:
        return {"ok": False, "message": f"Unknown style '{style}'. Use emphasis_punch | slow_drift | ken_burns"}

    # keep framing centered on the existing reframe track when present, else center
    reframe = (m.analysis.get("reframe") or {}).get("16:9") or (m.analysis.get("reframe") or {}).get("9:16")
    if reframe:
        ts = np.array([p["t"] for p in reframe])
        xs = np.array([p["cx"] for p in reframe])
        ys = np.array([p["cy"] for p in reframe])

        def center_fn(t: float) -> tuple[float, float]:
            return float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys))
    else:
        def center_fn(t: float) -> tuple[float, float]:
            return 0.5, 0.45

    m.analysis["camera_moves"] = {"style": style, "events": track_desc}
    m.append_history("add_camera_moves", {"style": style, "n": len(track_desc)})
    project.save()

    out_info: dict[str, Any] = {"ok": True, "style": style, "n_moves": len(track_desc), "track": track_desc[:10]}
    if render:
        out = project.renders_dir / f"moves_{style}.mp4"
        info = render_crop_track(
            src, out,
            target_aspect=meta.width / max(meta.height, 1),
            center_fn=center_fn,
            zoom_fn=zoom_fn,
            out_size=(meta.width - meta.width % 2, meta.height - meta.height % 2),
        )
        rel = project.rel(out)
        m.renders.append({"render_id": out.stem, "output_path": rel, "kind": "camera_moves", **info})
        project.save()
        out_info["output_path"] = rel
    return out_info
