"""Content-type classification — adaptive recipe selection signal.

Types: talking_head | tutorial | vlog | product_demo | lecture | ad | unknown.
Heuristic over analysis + footage index; no model required.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.content_type")


def classify_signals(signals: dict[str, float]) -> tuple[str, float]:
    th = signals.get("talking_head_score", 0.0)
    face = signals.get("face_presence", 0.0)
    speech = signals.get("speech_ratio", 0.0)
    shots_pm = signals.get("shots_per_min", 0.0)
    motion = signals.get("motion", 0.0)
    duration = signals.get("duration_sec", 0.0)
    flat_visual = signals.get("visual_flatness", 0.0)  # low variance → screen content

    scores = {
        "talking_head": 0.5 * max(th, face) + 0.3 * speech + 0.2 * (1 - min(shots_pm / 12, 1)),
        "tutorial": 0.45 * flat_visual + 0.3 * speech + 0.25 * (1 - face),
        "vlog": 0.4 * min(shots_pm / 10, 1) + 0.3 * motion + 0.3 * face * 0.8,
        "product_demo": 0.45 * (1 - face) + 0.3 * (1 - flat_visual) + 0.25 * min(shots_pm / 8, 1),
        "lecture": 0.4 * speech + 0.3 * (1 if duration > 480 else duration / 480 * 0.5) + 0.3 * max(th, face) * 0.7,
        "ad": 0.5 * (1 if duration < 75 else 0) + 0.3 * min(shots_pm / 15, 1) + 0.2 * motion,
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    conf = float(scores[best])
    if conf < 0.35:
        return "unknown", conf
    return best, min(conf, 0.99)


def classify_project(project: Any) -> dict[str, Any]:
    m = project.manifest
    analysis = m.analysis or {}
    meta = m.source_meta or {}
    duration = float(meta.get("duration") or analysis.get("duration_sec") or 0)

    index = None
    try:
        from vidmcp.perception.indexer import load_index

        index = load_index(project)
    except Exception:  # noqa: BLE001
        pass

    face_presence = 0.0
    visual_flatness = 0.0
    speech_ratio = 0.0
    if index:
        visual = index.get("visual") or []
        if visual:
            face_presence = float(np.mean([1.0 if v.get("n_faces") else 0.0 for v in visual]))
            sharps = [v.get("sharpness", 0.0) for v in visual]
            brights = [v.get("brightness", 128.0) for v in visual]
            # screen recordings: uniform brightness, spiky sharpness (text)
            visual_flatness = float(1.0 - min(np.std(brights) / 40.0, 1.0))
        events = index.get("audio_events") or []
        if events:
            speech_ratio = float(np.mean([1.0 if e["tag"] == "speech" else 0.0 for e in events]))

    shots = analysis.get("shots") or []
    shots_pm = len(shots) / max(duration / 60.0, 0.5) if shots else 0.0
    motion = float((analysis.get("motion") or {}).get("mean_energy") or 0.0)

    signals = {
        "talking_head_score": float(analysis.get("talking_head_score") or 0.0),
        "face_presence": face_presence,
        "speech_ratio": speech_ratio,
        "shots_per_min": round(shots_pm, 2),
        "motion": min(motion, 1.0),
        "duration_sec": duration,
        "visual_flatness": visual_flatness,
    }
    ctype, conf = classify_signals(signals)
    m.analysis["content_type"] = {"type": ctype, "confidence": round(conf, 3)}
    project.save()
    return {"ok": True, "type": ctype, "confidence": round(conf, 3), "signals": signals}
