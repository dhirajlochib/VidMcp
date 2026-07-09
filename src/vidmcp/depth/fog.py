"""Pseudo-depth from subject matte + depth-ordered particles/fog (occlude/disocclude)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.compositor.alpha import load_mask_for_frame, over
from vidmcp.perception.mask_ops import to_u8_mask
from vidmcp.utils.video_io import iter_frames, probe_video


def estimate_pseudo_depth(frame: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
    """0=near (subject), 1=far (background). Uses mask + vertical gradient prior."""
    h, w = frame.shape[:2]
    m = to_u8_mask(subject_mask).astype(np.float32) / 255.0
    # subject is near
    depth = 1.0 - m
    # slight vertical: bottom closer in talking-head
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    depth = np.clip(0.75 * depth + 0.25 * yy, 0, 1)
    depth = cv2.GaussianBlur(depth, (0, 0), 3)
    return depth


def apply_depth_ordered_particles(
    source_video: Path,
    mask_dir: Path,
    out_path: Path,
    *,
    density: float = 0.5,
    style: str = "fog",
    max_frames: int | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    """Render particles/fog only in far depth (behind subject) with proper occlusion."""
    meta = probe_video(source_video)
    w, h, fps = meta.width, meta.height, meta.fps
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(seed)

    # particle state in far field
    n_p = int(200 * density)
    parts = {
        "x": rng.uniform(0, w, n_p),
        "y": rng.uniform(0, h, n_p),
        "z": rng.uniform(0.55, 1.0, n_p),  # far
        "s": rng.uniform(1.0, 4.0, n_p),
        "vx": rng.normal(0, 0.4, n_p),
        "vy": rng.normal(-0.3, 0.5, n_p) if style != "rain" else rng.uniform(4, 10, n_p),
    }

    n = 0
    for idx, frame in iter_frames(source_video):
        if max_frames is not None and idx >= max_frames:
            break
        if frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
        mask = load_mask_for_frame(str(mask_dir), idx, h, w)
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
        depth = estimate_pseudo_depth(frame, mask)

        # draw far particles on transparent layer
        layer = np.zeros_like(frame)
        for i in range(n_p):
            parts["x"][i] = (parts["x"][i] + parts["vx"][i]) % w
            parts["y"][i] = (parts["y"][i] + parts["vy"][i]) % h
            x, y, z = int(parts["x"][i]), int(parts["y"][i]), parts["z"][i]
            # only draw if particle farther than local depth (behind surfaces)
            if depth[min(y, h - 1), min(x, w - 1)] < z - 0.05:
                continue  # particle would be in front of near surface — skip (occluded by subject region logic inverted)
            # We want particles BEHIND subject: subject depth~0, particles z~0.7-1
            # if local depth is small (subject near), don't draw particle on subject pixels
            if to_u8_mask(mask)[min(y, h - 1), min(x, w - 1)] > 127:
                continue
            rad = max(1, int(parts["s"][i] * (0.5 + 0.5 * z)))
            if style == "fog":
                col = (180, 180, 190)
                cv2.circle(layer, (x, y), rad + 2, col, -1, lineType=cv2.LINE_AA)
            elif style == "rain":
                cv2.line(layer, (x, y), (x, y + 12), (220, 210, 200), 1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(layer, (x, y), rad, (255, 100, 220), -1, lineType=cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 3 if style == "fog" else 1)
        # composite particles behind subject: on background only
        inv = 255 - to_u8_mask(mask)
        bg = frame.copy()
        # blend particles onto bg
        a = (inv.astype(np.float32) / 255.0) * 0.65
        a3 = a[..., None]
        bg = np.clip(bg.astype(np.float32) * (1 - a3 * 0.5) + layer.astype(np.float32) * a3, 0, 255).astype(np.uint8)
        # subject on top
        out = over(bg, frame, mask, 1.0)
        writer.write(out)
        n += 1
    writer.release()
    return {"ok": True, "output_path": str(out_path), "frames": n, "style": style, "density": density}
