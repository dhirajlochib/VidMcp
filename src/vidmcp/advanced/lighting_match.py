"""Match subject color/lighting to background plate via affine Lab + LUT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.perception.mask_ops import to_u8_mask
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames

log = get_logger("vidmcp.lighting")


def _stats_lab(img: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is not None:
        m = to_u8_mask(mask) > 127
        if m.sum() < 32:
            m = np.ones(mask.shape[:2], dtype=bool)
        pixels = lab[m]
    else:
        pixels = lab.reshape(-1, 3)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0) + 1e-6
    return mean, std


def match_frame(subject_bgr: np.ndarray, bg_bgr: np.ndarray, mask: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """Reinhard-style Lab transfer on subject pixels toward BG stats."""
    s_mean, s_std = _stats_lab(subject_bgr, mask)
    b_mean, b_std = _stats_lab(bg_bgr, None)
    lab = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = to_u8_mask(mask) > 127
    matched = lab.copy()
    for c in range(3):
        ch = (lab[:, :, c] - s_mean[c]) * (b_std[c] / s_std[c]) + b_mean[c]
        matched[:, :, c] = np.where(m, (1 - strength) * lab[:, :, c] + strength * ch, lab[:, :, c])
    matched = np.clip(matched, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)
    return out


def apply_lighting_match_to_project_render(
    source_video: Path,
    mask_dir: Path,
    bg_video_or_image: Path | None,
    out_path: Path,
    *,
    strength: float = 0.8,
    max_frames: int | None = None,
    progress=None,
) -> dict[str, Any]:
    """Write a video where subject region is color-matched toward background."""
    from vidmcp.compositor.alpha import load_mask_for_frame, over
    from vidmcp.utils.video_io import probe_video

    meta = probe_video(source_video)
    w, h, fps = meta.width, meta.height, meta.fps
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    bg_img = None
    bg_cap = None
    if bg_video_or_image:
        p = Path(bg_video_or_image)
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            bg_img = cv2.imread(str(p))
            if bg_img is not None:
                bg_img = cv2.resize(bg_img, (w, h))
        else:
            bg_cap = cv2.VideoCapture(str(p))

    n = 0
    for idx, frame in iter_frames(source_video):
        if max_frames is not None and idx >= max_frames:
            break
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        mask = load_mask_for_frame(str(mask_dir), idx, h, w)
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)

        if bg_img is not None:
            bg = bg_img
        elif bg_cap is not None:
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bg = bg_cap.read()
            if not ok:
                bg = frame.copy()
            else:
                bg = cv2.resize(bg, (w, h))
        else:
            # match to blurred surroundings outside mask
            inv = 255 - to_u8_mask(mask)
            bg = cv2.GaussianBlur(frame, (0, 0), 25)

        subject_matched = match_frame(frame, bg, mask, strength=strength)
        # composite: matched subject over bg
        canvas = over(bg, subject_matched, mask, 1.0)
        writer.write(canvas)
        n += 1
        if progress and meta.frame_count:
            progress(idx / max(meta.frame_count, 1), f"lighting {idx}")

    writer.release()
    if bg_cap is not None:
        bg_cap.release()
    return {"ok": True, "output_path": str(out_path), "frames": n, "strength": strength}
