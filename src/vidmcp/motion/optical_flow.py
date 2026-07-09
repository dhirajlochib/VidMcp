"""Optical flow utilities for motion-reactive VFX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.utils.video_io import iter_frames


def motion_energy_timeline(video_path: Path | str, *, max_frames: int = 120) -> dict[str, Any]:
    prev_gray = None
    energies: list[float] = []
    for idx, frame in iter_frames(video_path):
        if idx >= max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            energies.append(float(mag.mean()))
        prev_gray = gray
    return {
        "frame_count": len(energies) + (1 if prev_gray is not None else 0),
        "energy": energies,
        "mean_energy": float(np.mean(energies)) if energies else 0.0,
        "peak_frame": int(np.argmax(energies)) + 1 if energies else 0,
    }


def estimate_motion_field(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
