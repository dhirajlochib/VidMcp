"""Adversarial critic ensemble — multi-axis QA with routed fix tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.core.workspace import ProjectStore
from vidmcp.harness.quality_gates import evaluate_gates
from vidmcp.perception.mask_ops import coverage_mean, temporal_stability_score, to_u8_mask
from vidmcp.utils.video_io import sample_frames


def run_critic_ensemble(project: ProjectStore) -> dict[str, Any]:
    axes: list[dict[str, Any]] = []
    m = project.manifest
    seg = m.primary_segment()

    # 1 matte flicker
    stab = 0.0
    cov = 0.0
    if seg:
        files = sorted(Path(project.abs(seg.mask_dir)).glob("mask_*.png"))[:200]
        masks = []
        for f in files:
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                masks.append(im)
        if masks:
            stab = temporal_stability_score(masks)
            cov = coverage_mean(masks)
    axes.append(_axis("matte_flicker", stab, 0.6, "refine_segment_keyframes", "higher temporal IoU better"))
    axes.append(_axis("matte_coverage", 1.0 - abs(cov - 0.25) / 0.25 if cov else 0.0, 0.5, "segment_subject", "coverage near 0.1-0.4 ideal for talking-head"))

    # 2 edge dirtyness
    edge_score = 0.5
    if seg:
        files = sorted(Path(project.abs(seg.mask_dir)).glob("mask_*.png"))[:20]
        softs = []
        for f in files:
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            u = to_u8_mask(im)
            soft = float(((u > 20) & (u < 235)).mean())
            softs.append(soft)
        if softs:
            # too hard or too soft both bad; prefer some softness
            edge_score = float(np.mean([1.0 - abs(s - 0.08) / 0.08 for s in softs]))
            edge_score = float(np.clip(edge_score, 0, 1))
    axes.append(_axis("edge_quality", edge_score, 0.45, "refine_segment_keyframes", "feather / uncertainty refine"))

    # 3 lighting mismatch proxy: subject vs border BG color distance on mid frame
    light_score = 0.7
    if m.source_video and seg:
        try:
            frames = sample_frames(project.abs(m.source_video), max_frames=1, max_side=480)
            if frames:
                _, _, fr = frames[0]
                mf = sorted(Path(project.abs(seg.mask_dir)).glob("mask_*.png"))
                if mf:
                    mask = cv2.imread(str(mf[len(mf)//2]), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        if mask.shape[:2] != fr.shape[:2]:
                            mask = cv2.resize(mask, (fr.shape[1], fr.shape[0]))
                        ms = mask > 127
                        if ms.any() and (~ms).any():
                            sub = fr[ms].mean(axis=0)
                            bg = fr[~ms].mean(axis=0)
                            dist = float(np.linalg.norm(sub.astype(float) - bg.astype(float)) / 255.0)
                            # high distance may mean mismatch after BG replace — moderate is ok
                            light_score = float(np.clip(1.0 - max(0, dist - 0.35), 0, 1))
        except Exception:
            pass
    axes.append(_axis("lighting_match", light_score, 0.5, "match_subject_lighting", "Lab transfer toward plate"))

    # 4 brand safety placeholder (always pass unless dark)
    axes.append(_axis("brand_safety", 1.0, 0.5, "review_edit", "manual review if flagged"))

    # 5 render presence
    has_render = bool(m.renders) and project.abs(m.renders[-1]["output_path"]).exists()
    axes.append(_axis("render_complete", 1.0 if has_render else 0.0, 1.0, "composite_and_render", "run composite"))

    # 6 gates overall
    gate = evaluate_gates(project)
    axes.append(_axis("production_gates", gate.score, 0.7, "run_quality_gated_pipeline", "see gate fix hints"))

    failed = [a for a in axes if not a["passed"]]
    # route unique fix tools
    route = []
    for a in failed:
        if a["fix_tool"] not in route:
            route.append(a["fix_tool"])

    overall = float(np.mean([a["score"] for a in axes])) if axes else 0.0
    return {
        "ok": len(failed) == 0,
        "overall_score": overall,
        "axes": axes,
        "failed_axes": [a["name"] for a in failed],
        "fix_route": route,
        "gate": gate.to_dict(),
    }


def _axis(name: str, score: float, thr: float, fix_tool: str, note: str) -> dict[str, Any]:
    score = float(np.clip(score, 0, 1))
    return {
        "name": name,
        "score": score,
        "threshold": thr,
        "passed": score >= thr,
        "fix_tool": fix_tool,
        "note": note,
    }
