"""Scopes for machines — waveform / vectorscope / histogram PNGs + numeric stats."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vidmcp.color.grade import skin_mask
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video, sample_frames

log = get_logger("vidmcp.scopes")


def waveform_png(img: np.ndarray, height: int = 256) -> np.ndarray:
    """Luma waveform: x = image column, y = luma, intensity = pixel count."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    wf = np.zeros((height, w), np.float32)
    ys = (255 - gray).astype(np.int32) * (height - 1) // 255
    for x in range(w):
        col = np.bincount(ys[:, x], minlength=height)
        wf[:, x] = col
    wf = np.clip(wf / max(wf.max() * 0.25, 1) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(wf, cv2.COLORMAP_BONE)


def vectorscope_png(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Cb/Cr density plot with skin-line reference."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].reshape(-1).astype(np.int32) * (size - 1) // 255
    cb = ycrcb[:, :, 2].reshape(-1).astype(np.int32) * (size - 1) // 255
    scope = np.zeros((size, size), np.float32)
    np.add.at(scope, (size - 1 - cr, cb), 1.0)
    scope = np.clip(scope / max(scope.max() * 0.15, 1) * 255, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(scope, cv2.COLORMAP_OCEAN)
    # skin line ~123° from Cb axis; draw reference
    center = size // 2
    end = (int(center + size * 0.32), int(center - size * 0.38))
    cv2.line(out, (center, center), end, (80, 200, 255), 1)
    return out


def frame_stats(img: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clip_hi = float((gray >= 250).mean())
    clip_lo = float((gray <= 4).mean())
    cast_a = float(lab[:, :, 1].mean() - 128)
    cast_b = float(lab[:, :, 2].mean() - 128)
    sm = skin_mask(img)
    skin_dev = 0.0
    if sm.max() > 0.5:
        skin_px = lab[sm > 0.5]
        if len(skin_px) > 100:
            # deviation from typical skin chroma center in Lab
            skin_dev = float(np.abs(skin_px[:, 1].mean() - 145) + np.abs(skin_px[:, 2].mean() - 145)) / 2
    return {
        "clip_high_pct": round(clip_hi * 100, 2),
        "clip_low_pct": round(clip_lo * 100, 2),
        "mean_luma": round(float(gray.mean()), 1),
        "mean_saturation": round(float(hsv[:, :, 1].mean()), 1),
        "cast_a": round(cast_a, 2),
        "cast_b": round(cast_b, 2),
        "cast_estimate": round(float(np.hypot(cast_a, cast_b)), 2),
        "skin_line_deviation": round(skin_dev, 2),
    }


def scopes_project(project: Any, frame: int | str = "auto", from_render: bool = True) -> dict[str, Any]:
    m = project.manifest
    video = None
    if from_render and m.renders:
        video = project.abs(m.renders[-1].get("output_path"))
        if not video.exists():
            video = None
    if video is None:
        if not m.source_video:
            return {"ok": False, "message": "No video to scope"}
        video = project.abs(m.source_video)
    meta = probe_video(video)
    if frame == "auto":
        idx = meta.frame_count // 2
    else:
        idx = int(frame)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        frames = sample_frames(video, max_frames=1)
        if not frames:
            return {"ok": False, "message": "Cannot read frame"}
        img = frames[0][2]
        idx = frames[0][0]

    wf = waveform_png(img)
    vs = vectorscope_png(img)
    out_dir = project.previews_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wf_path = out_dir / f"waveform_{idx}.png"
    vs_path = out_dir / f"vectorscope_{idx}.png"
    cv2.imwrite(str(wf_path), wf)
    cv2.imwrite(str(vs_path), vs)
    return {
        "ok": True,
        "frame": idx,
        "video": project.rel(video),
        "waveform_png": project.rel(wf_path),
        "vectorscope_png": project.rel(vs_path),
        "stats": frame_stats(img),
    }
