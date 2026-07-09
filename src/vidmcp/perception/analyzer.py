"""Video analysis: metadata, scene heuristics, suggested SAM prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video, sample_frames

log = get_logger("vidmcp.analyzer")


def _face_like_score(frame: np.ndarray) -> float:
    """Talking-head prior without Haar (OpenCV 5 headless has no CascadeClassifier).

    Combines: center-weighted warm skin-like chrominance + upper-third occupancy.
    Optional Haar path when classic OpenCV with cascades is installed.
    """
    h, w = frame.shape[:2]
    # Optional classic cascade
    try:
        cascade_path = getattr(cv2, "data", None) and (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade_path and hasattr(cv2, "CascadeClassifier"):
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if not face_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)
                )
                if len(faces) > 0:
                    areas = [fw * fh for (_, _, fw, fh) in faces]
                    return float(min(1.0, (max(areas) / (w * h)) * 8.0))
    except Exception:  # noqa: BLE001
        pass

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    yy, xx = np.mgrid[0:h, 0:w]
    # favor upper-center (typical talking head)
    cy, cx = h * 0.38, w * 0.5
    prior = np.exp(
        -(((xx - cx) ** 2) / (2 * (w * 0.2) ** 2) + ((yy - cy) ** 2) / (2 * (h * 0.22) ** 2))
    )
    weighted = (skin.astype(np.float32) / 255.0) * prior
    score = float(weighted.mean() * 12.0)
    return float(min(1.0, score))


def analyze_video(
    path: Path | str,
    *,
    preview_dir: Path | None = None,
    max_previews: int = 6,
    sample_fps: float = 2.0,
) -> dict[str, Any]:
    path = Path(path)
    meta = probe_video(path)
    frames = sample_frames(path, max_frames=max_previews, target_fps=sample_fps, max_side=960)

    face_scores = [_face_like_score(f) for _, _, f in frames] or [0.0]
    talking_head_score = float(np.mean(face_scores))

    # motion energy
    motion = 0.0
    if len(frames) >= 2:
        diffs = []
        for i in range(1, len(frames)):
            a = cv2.cvtColor(frames[i - 1][2], cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(frames[i][2], cv2.COLOR_BGR2GRAY)
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            diffs.append(float(np.mean(cv2.absdiff(a, b)) / 255.0))
        motion = float(np.mean(diffs))

    brightness = float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() / 255.0 for _, _, f in frames]))

    scene_hints: list[str] = []
    suggested: list[str] = []
    if talking_head_score > 0.25:
        scene_hints.append("talking_head_or_interview")
        suggested.extend(["person", "speaker", "face", "human"])
    else:
        scene_hints.append("general_scene")
        suggested.extend(["person", "subject", "object"])
    if motion < 0.05:
        scene_hints.append("low_motion")
    elif motion > 0.2:
        scene_hints.append("high_motion")
    if brightness < 0.3:
        scene_hints.append("low_light")
    elif brightness > 0.7:
        scene_hints.append("bright")

    thumbs: list[str] = []
    if preview_dir:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, ts, frame in frames:
            p = preview_dir / f"thumb_{idx:06d}_{ts:.2f}s.jpg"
            cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            thumbs.append(str(p))

    return {
        **meta.to_dict(),
        "talking_head_score": talking_head_score,
        "motion_energy": motion,
        "brightness": brightness,
        "scene_hints": scene_hints,
        "suggested_prompts": suggested,
        "thumbnail_paths": thumbs,
        "sample_count": len(frames),
    }
