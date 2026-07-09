"""Multi-cue pseudo-depth and flow-based plate warping (no heavy models required).

Cues: subject matte (near), vertical prior, defocus (Laplacian inverse), motion magnitude.
Optional: if `transformers` + depth model available, use pipeline (best-effort).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.compositor.alpha import load_mask_for_frame, over
from vidmcp.perception.mask_ops import to_u8_mask
from vidmcp.utils.video_io import iter_frames, probe_video


def multi_cue_depth(frame: np.ndarray, subject_mask: np.ndarray | None = None) -> np.ndarray:
    """Return float depth map 0=near, 1=far."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # defocus: low Laplacian variance → farther (background blur)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # local variance via blur of squares
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
    mean2 = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 3)
    var = np.clip(mean2 - mean**2, 0, None)
    var_n = var / (var.max() + 1e-6)
    defocus_far = 1.0 - var_n  # high var = sharp = near

    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    vertical = yy  # bottom closer for talking-head-ish

    if subject_mask is not None:
        m = to_u8_mask(subject_mask).astype(np.float32) / 255.0
        subject_near = 1.0 - m
    else:
        subject_near = np.ones((h, w), np.float32) * 0.5

    depth = 0.45 * subject_near + 0.25 * defocus_far + 0.20 * vertical + 0.10 * (np.abs(lap) / (np.abs(lap).max() + 1e-6))
    # wait — subject_near already 0 on subject; good
    depth = 0.50 * subject_near + 0.30 * defocus_far + 0.20 * vertical
    depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 2)
    depth = np.clip(depth, 0, 1)
    return depth


def try_midas_depth(frame: np.ndarray) -> np.ndarray | None:
    """Best-effort monocular depth via transformers if installed."""
    try:
        import torch
        from PIL import Image
        from transformers import pipeline

        # cache pipeline on function attr
        if not hasattr(try_midas_depth, "_pipe"):
            try_midas_depth._pipe = pipeline(  # type: ignore[attr-defined]
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
            )
        pipe = try_midas_depth._pipe  # type: ignore[attr-defined]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = pipe(Image.fromarray(rgb))
        depth = np.array(out["depth"], dtype=np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        # invert so larger = farther if model is inverse — Depth Anything: larger often closer
        # treat high value as near → convert to far
        depth = 1.0 - depth
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        return depth
    except Exception:
        return None


def compute_depth_sequence(
    video_path: Path,
    mask_dir: Path | None,
    *,
    out_dir: Path | None = None,
    max_frames: int = 120,
    prefer_midas: bool = True,
) -> dict[str, Any]:
    video_path = Path(video_path)
    meta = probe_video(video_path)
    depths = []
    backend = "multi_cue"
    for idx, frame in iter_frames(video_path):
        if idx >= max_frames:
            break
        mask = None
        if mask_dir:
            mask = load_mask_for_frame(str(mask_dir), idx, frame.shape[0], frame.shape[1])
        d = None
        if prefer_midas and idx == 0:
            d = try_midas_depth(frame)
            if d is not None:
                backend = "depth_anything_or_midas"
        if d is None:
            d = multi_cue_depth(frame, mask)
        elif mask is not None:
            # fuse midas with matte prior
            m = to_u8_mask(mask).astype(np.float32) / 255.0
            d = 0.7 * d + 0.3 * (1.0 - m)
        depths.append(d)
        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            vis = (np.clip(d, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"depth_{idx:06d}.png"), vis)
    mean_near = float(np.mean([1.0 - d.mean() for d in depths])) if depths else 0
    return {
        "ok": True,
        "backend": backend,
        "frame_count": len(depths),
        "mean_nearness": mean_near,
        "depth_dir": str(out_dir) if out_dir else None,
        "width": meta.width,
        "height": meta.height,
    }


def flow_warp_plate_composite(
    source_video: Path,
    plate_path: Path,
    mask_dir: Path,
    out_path: Path,
    *,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Warp plate each frame by dense optical flow from frame 0 (smoother than affine-only)."""
    meta = probe_video(source_video)
    w, h, fps = meta.width, meta.height, meta.fps
    plate = cv2.imread(str(plate_path))
    pcap = None
    if plate is None:
        pcap = cv2.VideoCapture(str(plate_path))
        ok, plate = pcap.read()
        if not ok:
            raise FileNotFoundError(plate_path)
        pcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    plate0 = cv2.resize(plate, (w, h))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    prev_gray = None
    # cumulative flow map: where each plate pixel moves
    flow_cum = np.zeros((h, w, 2), dtype=np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    n = 0
    for idx, frame in iter_frames(source_video):
        if max_frames is not None and idx >= max_frames:
            break
        if frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_cum += flow
        # inverse map for plate (stabilize opposite camera motion)
        map_x = (grid_x - flow_cum[:, :, 0]).astype(np.float32)
        map_y = (grid_y - flow_cum[:, :, 1]).astype(np.float32)
        if pcap is not None:
            pcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, pl = pcap.read()
            pl = cv2.resize(pl, (w, h)) if ok else plate0
        else:
            pl = plate0
        warped = cv2.remap(pl, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = load_mask_for_frame(str(mask_dir), idx, h, w)
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
        comp = over(warped, frame, mask, 1.0)
        writer.write(comp)
        prev_gray = gray
        n += 1
    writer.release()
    if pcap is not None:
        pcap.release()
    return {"ok": True, "output_path": str(out_path), "frames": n, "method": "dense_flow_warp"}
