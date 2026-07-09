"""Histogram-based shot boundary detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.utils.video_io import iter_frames, probe_video


def detect_shots(
    video_path: Path,
    *,
    threshold: float = 0.45,
    min_shot_len: int = 8,
    max_frames: int | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path)
    meta = probe_video(video_path)
    prev_hist = None
    cuts = [0]
    scores = []
    for idx, frame in iter_frames(video_path):
        if max_frames is not None and idx >= max_frames:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            # correlation distance
            corr = float(cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_CORREL))
            dist = 1.0 - corr
            scores.append({"frame": idx, "dist": dist})
            if dist >= threshold and (idx - cuts[-1]) >= min_shot_len:
                cuts.append(idx)
        prev_hist = hist
    # end
    last = scores[-1]["frame"] if scores else max(meta.frame_count - 1, 0)
    if cuts[-1] != last:
        cuts.append(last + 1)
    shots = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        shots.append(
            {
                "shot_index": i,
                "start_frame": a,
                "end_frame": b - 1,
                "start_sec": a / max(meta.fps, 1e-6),
                "end_sec": (b - 1) / max(meta.fps, 1e-6),
                "duration_sec": (b - a) / max(meta.fps, 1e-6),
            }
        )
    return {
        "ok": True,
        "shot_count": len(shots),
        "shots": shots,
        "fps": meta.fps,
        "threshold": threshold,
        "cut_frames": cuts[:-1],
    }
