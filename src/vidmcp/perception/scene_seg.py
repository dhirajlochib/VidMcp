"""Shot boundary detection + scene clustering (PySceneDetect optional, histogram fallback)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video, sample_frames

log = get_logger("vidmcp.scene_seg")


def _hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.flatten()


def _shots_scenedetect(video: str) -> list[tuple[float, float]] | None:
    try:
        from scenedetect import AdaptiveDetector, detect  # type: ignore

        scenes = detect(video, AdaptiveDetector())
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes] or None
    except Exception as e:  # noqa: BLE001
        log.debug("scenedetect_unavailable", error=str(e))
        return None


def _shots_histogram(video: str, threshold: float = 0.45) -> list[tuple[float, float]]:
    frames = sample_frames(video, max_frames=400, target_fps=4.0, max_side=320)
    meta = probe_video(video)
    if not frames:
        return [(0.0, meta.duration_sec)]
    cuts = [0.0]
    prev = _hist(frames[0][2])
    for _, ts, img in frames[1:]:
        h = _hist(img)
        d = cv2.compareHist(prev.reshape(-1, 1), h.reshape(-1, 1), cv2.HISTCMP_BHATTACHARYYA)
        if d > threshold:
            cuts.append(ts)
        prev = h
    cuts.append(meta.duration_sec)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] > 0.2]


def _cluster_scenes(video: str, shots: list[tuple[float, float]], sim_threshold: float = 0.55) -> list[int]:
    """Greedy adjacent clustering by keyframe histogram correlation."""
    cap = cv2.VideoCapture(video)
    meta = probe_video(video)
    keyhists: list[np.ndarray] = []
    for a, b in shots:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((a + b) / 2 * meta.fps))
        ok, img = cap.read()
        keyhists.append(_hist(img) if ok else np.zeros(128, np.float32))
    cap.release()
    scene_ids = [0] * len(shots)
    sid = 0
    for i in range(1, len(shots)):
        corr = cv2.compareHist(
            keyhists[i - 1].reshape(-1, 1), keyhists[i].reshape(-1, 1), cv2.HISTCMP_CORREL
        )
        if corr < sim_threshold:
            sid += 1
        scene_ids[i] = sid
    return scene_ids


def detect_scenes_project(project: Any, backend: str = "auto", threshold: float = 0.45) -> dict[str, Any]:
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    video = str(project.abs(m.source_video))
    shots = None
    used = "histogram"
    if backend in ("auto", "scenedetect"):
        shots = _shots_scenedetect(video)
        if shots:
            used = "scenedetect"
    if not shots:
        shots = _shots_histogram(video, threshold)
    scene_ids = _cluster_scenes(video, shots) if len(shots) > 1 else [0] * len(shots)
    shot_list = [
        {"start": round(a, 3), "end": round(b, 3), "scene_id": scene_ids[i]}
        for i, (a, b) in enumerate(shots)
    ]
    m.analysis["shots"] = shot_list
    m.analysis["n_scenes"] = len(set(scene_ids))
    m.append_history("detect_scenes", {"n_shots": len(shots), "backend": used})
    project.save()
    return {
        "ok": True,
        "backend": used,
        "n_shots": len(shots),
        "n_scenes": len(set(scene_ids)),
        "shots": shot_list[:20],
    }
