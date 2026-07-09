"""Critic / reviewer agent — automated QA on mattes and renders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.core.workspace import ProjectStore
from vidmcp.models.project import ReviewNote
from vidmcp.perception.mask_ops import coverage_mean, temporal_stability_score, to_u8_mask
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.critic")


class CriticAgent:
    def review(self, project: ProjectStore) -> dict[str, Any]:
        notes: list[ReviewNote] = []
        m = project.manifest
        score = 1.0

        if not m.source_video:
            notes.append(ReviewNote(severity="error", code="no_source", message="No source video"))
            score = 0.0
        seg = m.primary_segment()
        if not seg:
            notes.append(ReviewNote(severity="error", code="no_segment", message="No segmentation track"))
            score = min(score, 0.2)
        else:
            mask_dir = project.abs(seg.mask_dir)
            masks = []
            files = sorted(Path(mask_dir).glob("mask_*.png"))[:200]
            for f in files:
                im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    masks.append(im)
            if not masks:
                notes.append(ReviewNote(severity="error", code="empty_masks", message="Mask directory empty"))
                score = min(score, 0.1)
            else:
                stab = temporal_stability_score(masks)
                cov = coverage_mean(masks)
                if stab < 0.55:
                    notes.append(
                        ReviewNote(
                            severity="warning",
                            code="temporal_instability",
                            message=f"Low temporal IoU stability ({stab:.2f})",
                            suggestion="Re-segment with higher conf or SAM 3.1 video predictor; enable temporal median",
                        )
                    )
                    score -= 0.2
                if cov < 0.02:
                    notes.append(
                        ReviewNote(
                            severity="warning",
                            code="thin_matte",
                            message=f"Very low subject coverage ({cov:.3f})",
                            suggestion="Adjust text prompt or lower conf_threshold",
                        )
                    )
                    score -= 0.15
                if cov > 0.85:
                    notes.append(
                        ReviewNote(
                            severity="warning",
                            code="over_segmented",
                            message=f"Subject covers most of frame ({cov:.2f}) — BG effects may be invisible",
                            suggestion="Tighten prompt or use box exemplar on speaker",
                        )
                    )
                    score -= 0.1
                # edge hardness
                hard = float(np.mean([(to_u8_mask(x) > 250).mean() + (to_u8_mask(x) < 5).mean() for x in masks[:20]]))
                if hard > 0.98:
                    notes.append(
                        ReviewNote(
                            severity="info",
                            code="hard_edges",
                            message="Masks are very binary — consider more feather for hair",
                            suggestion="Increase mask feather radius",
                        )
                    )

        if not m.layers.layers:
            notes.append(ReviewNote(severity="warning", code="no_layers", message="Empty layer stack"))
            score -= 0.2
        if not m.renders:
            notes.append(
                ReviewNote(
                    severity="info",
                    code="not_rendered",
                    message="No renders yet — run composite_and_render",
                    suggestion="Call composite_and_render",
                )
            )
        else:
            last = m.renders[-1]
            out = project.abs(last["output_path"])
            if not out.exists():
                notes.append(ReviewNote(severity="error", code="missing_render", message=f"Render missing: {out}"))
                score -= 0.3

        score = float(max(0.0, min(1.0, score)))
        passed = score >= 0.65 and not any(n.severity == "error" for n in notes)
        actions = []
        for n in notes:
            if n.suggestion:
                actions.append(n.suggestion)

        result = {
            "score": score,
            "passed": passed,
            "notes": [n.model_dump() for n in notes],
            "recommended_actions": list(dict.fromkeys(actions)),
        }
        m.reviews.append(result)
        from vidmcp.models.project import ProjectStatus

        m.status = ProjectStatus.REVIEWED
        m.append_history("review_edit", {"score": score, "passed": passed})
        project.save()
        return result
