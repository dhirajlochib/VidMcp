"""World-consistent background reproject: warp plate with estimated 2D affine camera motion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.compositor.alpha import load_mask_for_frame, over
from vidmcp.perception.mask_ops import to_u8_mask
from vidmcp.utils.video_io import iter_frames, probe_video


def estimate_camera_affines(video_path: Path, max_frames: int = 300) -> list[np.ndarray]:
    """Chain of 2x3 affines from frame 0 via ORB+partial affine on background-ish features."""
    affines = [np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)]
    prev_gray = None
    prev_pts = None
    orb = cv2.ORB_create(800)
    for idx, frame in iter_frames(video_path):
        if idx >= max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            prev_pts = orb.detect(gray, None)
            prev_pts, _ = orb.compute(gray, prev_pts) if prev_pts else (None, None)
            continue
        # optical flow based affine (more stable than ORB match for talking-head)
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8)
        if p0 is None:
            affines.append(affines[-1].copy())
            prev_gray = gray
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
        if p1 is None:
            affines.append(affines[-1].copy())
            prev_gray = gray
            continue
        good0 = p0[st.flatten() == 1]
        good1 = p1[st.flatten() == 1]
        if len(good0) < 6:
            affines.append(affines[-1].copy())
            prev_gray = gray
            continue
        A, _ = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC)
        if A is None:
            A = affines[-1].copy()
        # accumulate
        # convert to 3x3
        def to3(a):
            m = np.eye(3, dtype=np.float32)
            m[:2] = a
            return m

        cum = to3(affines[0] if False else affines[-1])  # noqa — use last cumulative
        # affines list stores cumulative from frame 0
        prev_cum = np.eye(3, dtype=np.float32)
        prev_cum[:2] = affines[-1]
        new_cum = to3(A) @ prev_cum
        affines.append(new_cum[:2].astype(np.float32))
        prev_gray = gray
    return affines


def reproject_background_plate(
    source_video: Path,
    plate_path: Path,
    mask_dir: Path,
    out_path: Path,
    *,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Warp a still/video plate by estimated camera motion; composite under subject."""
    meta = probe_video(source_video)
    w, h, fps = meta.width, meta.height, meta.fps
    affines = estimate_camera_affines(source_video, max_frames=max_frames or meta.frame_count or 300)

    plate = cv2.imread(str(plate_path))
    plate_is_video = plate is None
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
    n = 0
    for idx, frame in iter_frames(source_video):
        if max_frames is not None and idx >= max_frames:
            break
        if frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
        A = affines[min(idx, len(affines) - 1)]
        # invert warp so plate stabilizes opposite to camera
        try:
            A_inv = cv2.invertAffineTransform(A)
        except cv2.error:
            A_inv = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        if pcap is not None:
            pcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, pl = pcap.read()
            if not ok:
                pl = plate0
            else:
                pl = cv2.resize(pl, (w, h))
        else:
            pl = plate0
        warped = cv2.warpAffine(pl, A_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = load_mask_for_frame(str(mask_dir), idx, h, w)
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
        comp = over(warped, frame, mask, 1.0)
        writer.write(comp)
        n += 1
    writer.release()
    if pcap is not None:
        pcap.release()
    return {
        "ok": True,
        "output_path": str(out_path),
        "frames": n,
        "affine_count": len(affines),
        "method": "affine_optical_flow_reproject",
    }
