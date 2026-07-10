"""Temporal matte consistency — flow-guided alpha stabilization + dtSSD metric."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames

log = get_logger("vidmcp.matte_temporal")


def _flow_warp(prev: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = prev.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    return cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def dtssd(alphas: list[np.ndarray]) -> float:
    """Temporal alpha gradient error (flicker proxy). Lower = steadier."""
    if len(alphas) < 2:
        return 0.0
    acc = 0.0
    for a, b in zip(alphas[:-1], alphas[1:]):
        d = (b.astype(np.float32) - a.astype(np.float32)) / 255.0
        acc += float(np.sqrt((d * d).mean()))
    return acc / (len(alphas) - 1)


def _is_scene_cut(prev_gray: np.ndarray, gray: np.ndarray, threshold: float = 0.5) -> bool:
    h1 = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([gray], [0], None, [64], [0, 256])
    corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return corr < threshold


def stabilize_matte_project(
    project: Any,
    strength: float = 0.6,
    max_frames: int | None = None,
) -> dict[str, Any]:
    m = project.manifest
    seg = m.primary_segment()
    if seg is None or not m.source_video:
        return {"ok": False, "message": "Need source video + segment track first"}
    strength = float(np.clip(strength, 0.0, 1.0))

    src_dir = project.abs(seg.meta.get("alpha_dir") or seg.mask_dir)
    files = sorted(Path(src_dir).glob("mask_*.png"))
    if len(files) < 2:
        return {"ok": False, "message": f"Not enough masks in {src_dir}"}

    out_dir = project.masks_dir / f"{seg.id[:8]}_stable"
    out_dir.mkdir(parents=True, exist_ok=True)

    before: list[np.ndarray] = []
    after: list[np.ndarray] = []
    prev_gray: np.ndarray | None = None
    prev_out: np.ndarray | None = None
    n = 0
    for idx, frame in iter_frames(project.abs(m.source_video)):
        if max_frames is not None and idx >= max_frames:
            break
        if idx >= len(files):
            break
        alpha = cv2.imread(str(files[idx]), cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            continue
        if alpha.shape[:2] != frame.shape[:2]:
            alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = alpha.astype(np.float32)
        if prev_gray is not None and prev_out is not None and not _is_scene_cut(prev_gray, gray):
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            warped = _flow_warp(prev_out, flow)
            # trust the warp only where it disagrees mildly (flicker), not on true motion
            disagreement = np.abs(out - warped) / 255.0
            edge = cv2.Canny(alpha, 40, 120)
            band = cv2.dilate(edge, np.ones((7, 7), np.uint8)) > 0
            w_map = np.zeros_like(out)
            w_map[band] = strength * np.clip(1.0 - disagreement[band] * 2.0, 0.0, 1.0)
            out = out * (1.0 - w_map) + warped * w_map
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"mask_{idx:06d}.png"), out_u8)
        if len(before) < 300:
            before.append(alpha)
            after.append(out_u8)
        prev_gray = gray
        prev_out = out_u8.astype(np.float32)
        n += 1

    d_before = dtssd(before)
    d_after = dtssd(after)
    seg.meta["alpha_dir"] = project.rel(out_dir)
    seg.meta["stabilized"] = True
    m.append_history("stabilize_matte", {"dtssd_before": d_before, "dtssd_after": d_after})
    project.save()
    return {
        "ok": True,
        "segment_id": seg.id,
        "alpha_dir": project.rel(out_dir),
        "frames": n,
        "dtssd_before": round(d_before, 5),
        "dtssd_after": round(d_after, 5),
        "improved": d_after <= d_before,
    }
