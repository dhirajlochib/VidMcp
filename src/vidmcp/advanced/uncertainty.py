"""Per-frame matte uncertainty field: temporal variance + edge entropy.

High-uncertainty bands drive surgical refine (hair, hands, motion blur) without full resegment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import orjson

from vidmcp.perception.mask_ops import to_u8_mask
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.uncertainty")


def compute_uncertainty_field(
    mask_dir: Path,
    *,
    out_dir: Path | None = None,
    window: int = 3,
    max_frames: int = 500,
) -> dict[str, Any]:
    mask_dir = Path(mask_dir)
    files = sorted(mask_dir.glob("mask_*.png"))[:max_frames]
    if len(files) < 2:
        return {"ok": False, "message": "Need >=2 masks", "frames": len(files)}

    masks = []
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append(to_u8_mask(m).astype(np.float32) / 255.0)
    h, w = masks[0].shape
    # temporal variance over sliding window
    var_maps = []
    edge_ents = []
    frame_scores = []
    half = max(1, window // 2)
    for i in range(len(masks)):
        lo, hi = max(0, i - half), min(len(masks), i + half + 1)
        stack = np.stack(masks[lo:hi], axis=0)
        var = stack.var(axis=0)
        var_maps.append(var)
        # edge entropy: binary entropy near edges
        u8 = (masks[i] * 255).astype(np.uint8)
        edges = cv2.Canny(u8, 50, 150)
        ring = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2) > 0
        vals = masks[i][ring]
        if vals.size:
            # soft entropy of mask values near edges
            hist, _ = np.histogram(vals, bins=16, range=(0, 1), density=True)
            hist = hist + 1e-8
            hist = hist / hist.sum()
            ent = float(-(hist * np.log2(hist)).sum())
        else:
            ent = 0.0
        edge_ents.append(ent)
        score = float(var.mean() * 2.0 + ent * 0.15)
        frame_scores.append(score)

    mean_var = float(np.mean([v.mean() for v in var_maps]))
    # rank frames for refine
    order = np.argsort(frame_scores)[::-1]
    hot_frames = [int(i) for i in order[: min(8, len(order))] if frame_scores[i] > 0.02]

    # spatial hotspots on worst frame
    worst = int(order[0])
    var_u8 = np.clip(var_maps[worst] / (var_maps[worst].max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(var_u8, cv2.COLORMAP_JET)

    paths: dict[str, str] = {}
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        heat_path = out_dir / f"uncertainty_heat_{worst:06d}.png"
        cv2.imwrite(str(heat_path), heat)
        # binary high-uncertainty mask for refine ROI
        roi = (var_maps[worst] > np.percentile(var_maps[worst], 85)).astype(np.uint8) * 255
        roi_path = out_dir / f"uncertainty_roi_{worst:06d}.png"
        cv2.imwrite(str(roi_path), roi)
        meta_path = out_dir / "uncertainty.json"
        payload = {
            "mean_variance": mean_var,
            "hot_frames": hot_frames,
            "frame_scores": [float(s) for s in frame_scores],
            "edge_entropy": edge_ents,
            "worst_frame": worst,
        }
        meta_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        paths = {
            "heatmap": str(heat_path),
            "roi": str(roi_path),
            "meta": str(meta_path),
        }

    # suggested refine boxes from ROI contours on worst frame
    boxes = []
    if out_dir and paths.get("roi"):
        roi = cv2.imread(paths["roi"], cv2.IMREAD_GRAYSCALE)
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < 64:
                continue
            boxes.append(
                {
                    "frame_index": worst,
                    "box_xyxy": [x / w, y / h, (x + bw) / w, (y + bh) / h],
                    "reason": "high_uncertainty_roi",
                }
            )

    return {
        "ok": True,
        "mean_variance": mean_var,
        "hot_frames": hot_frames,
        "worst_frame": worst,
        "frame_score_max": float(max(frame_scores) if frame_scores else 0),
        "frame_score_mean": float(np.mean(frame_scores) if frame_scores else 0),
        "edge_entropy_mean": float(np.mean(edge_ents) if edge_ents else 0),
        "refine_hints": boxes,
        "paths": paths,
        "width": w,
        "height": h,
        "frame_count": len(masks),
    }
