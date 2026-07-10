"""Replace background with plate using soft matte."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vidmcp.matte.fast_matte import segment_video_matte
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames, probe_video

log = get_logger("vidmcp.replace_bg")
ProgressFn = Callable[[float, str], None]


def _space_plate(h: int, w: int, t: float) -> np.ndarray:
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    pulse = 0.5 + 0.5 * math.sin(t * 0.35)
    r = 8 + 18 * y + 10 * pulse
    g = 10 + 12 * (1 - y) + 8 * math.sin(t * 0.2)
    b = 22 + 40 * (1 - y) + 20 * math.cos(t * 0.15)
    dist = np.sqrt((x - 0.45) ** 2 + (y - 0.42) ** 2 * 1.2)
    glow = np.clip(1.0 - dist * 1.8, 0, 1) ** 2
    r = r + glow * (40 + 30 * math.sin(t * 0.5))
    g = g + glow * (90 + 40 * math.sin(t * 0.5 + 1))
    b = b + glow * (60 + 20 * math.cos(t * 0.4))
    bg = np.stack([b, g, r], axis=-1)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(7 + int(t * 3) % 50)
    for _ in range(80):
        sx, sy = int(rng.integers(0, w)), int(rng.integers(0, h))
        c = int(140 + 80 * rng.random())
        cv2.circle(bg, (sx, sy), 1, (c, c, min(255, c + 40)), -1)
    return bg


def _blur_plate(frame: np.ndarray, radius: int = 45) -> np.ndarray:
    k = max(3, radius | 1)
    return cv2.GaussianBlur(frame, (k, k), 0)


def _composite(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = (mask.astype(np.float32) / 255.0)[..., None]
    rim = cv2.GaussianBlur(mask, (0, 0), 8).astype(np.float32) / 255.0
    edge = np.clip(rim - a[:, :, 0], 0, 1)[..., None]
    out = bg.astype(np.float32) * (1 - a) + fg.astype(np.float32) * a
    out = out + edge * np.array([80, 220, 180], dtype=np.float32) * 0.4
    return np.clip(out, 0, 255).astype(np.uint8)


def replace_background_video(
    video: Path | str,
    out_video: Path | str,
    *,
    plate: str = "space",
    plate_image: Path | str | None = None,
    matte_backend: str = "auto",
    mask_dir: Path | str | None = None,
    solid_color: tuple[int, int, int] = (5, 5, 7),
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    video = Path(video)
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video)

    if mask_dir is None:
        mask_dir = out_video.parent / "masks_replace"
        matte = segment_video_matte(video, mask_dir, backend=matte_backend, progress=progress)
    else:
        mask_dir = Path(mask_dir)
        matte = {"ok": True, "mask_dir": str(mask_dir), "backend": "provided", "coverage_mean": None}

    plate_img = None
    if plate == "image" and plate_image:
        plate_img = cv2.imread(str(plate_image))
        if plate_img is not None:
            plate_img = cv2.resize(plate_img, (meta.width, meta.height))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, meta.fps, (meta.width, meta.height))
    coverages = []
    for idx, frame in iter_frames(video):
        t = idx / max(meta.fps, 1e-6)
        mp = mask_dir / f"mask_{idx:06d}.png"
        if mp.exists():
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            elif mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        else:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        coverages.append(float(mask.mean()) / 255.0)

        if plate == "blur":
            bg = _blur_plate(frame)
        elif plate == "solid":
            bg = np.full_like(frame, solid_color[::-1])  # BGR from RGB-ish tuple
            bg[:] = (solid_color[2], solid_color[1], solid_color[0])
        elif plate == "image" and plate_img is not None:
            bg = plate_img
        else:
            bg = _space_plate(frame.shape[0], frame.shape[1], t)

        writer.write(_composite(frame, bg, mask))
        if progress and meta.frame_count:
            progress(0.5 + 0.5 * idx / meta.frame_count, f"composite {idx}")
    writer.release()

    return {
        "ok": True,
        "path": str(out_video),
        "mask_dir": str(mask_dir),
        "plate": plate,
        "matte": matte,
        "coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
    }
