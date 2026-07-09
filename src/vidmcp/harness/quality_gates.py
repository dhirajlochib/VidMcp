"""Hard quality gates — block/retry until matte + render meet production thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.config import Settings, get_settings
from vidmcp.core.workspace import ProjectStore
from vidmcp.perception.mask_ops import coverage_mean, temporal_stability_score, to_u8_mask


@dataclass
class GateCheck:
    name: str
    passed: bool
    value: float
    threshold: float
    severity: str  # block | warn
    message: str
    fix_hint: str = ""


@dataclass
class QualityGateResult:
    passed: bool
    score: float
    checks: list[GateCheck] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    refine_strategy: str | None = None  # lower_conf | alternate_prompt | more_feather | resegment

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                    "severity": c.severity,
                    "message": c.message,
                    "fix_hint": c.fix_hint,
                }
                for c in self.checks
            ],
            "recommended_actions": self.recommended_actions,
            "refine_strategy": self.refine_strategy,
        }


def _load_masks(mask_dir: Path, limit: int = 300) -> list[np.ndarray]:
    files = sorted(mask_dir.glob("mask_*.png"))[:limit]
    out = []
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            out.append(im)
    return out


def evaluate_gates(project: ProjectStore, settings: Settings | None = None) -> QualityGateResult:
    settings = settings or get_settings()
    checks: list[GateCheck] = []
    actions: list[str] = []
    m = project.manifest

    # source
    has_source = bool(m.source_video and project.abs(m.source_video).exists())
    checks.append(
        GateCheck(
            "source_present",
            has_source,
            1.0 if has_source else 0.0,
            1.0,
            "block",
            "Source video present" if has_source else "Missing source video",
            "import_video",
        )
    )

    seg = m.primary_segment()
    masks: list[np.ndarray] = []
    if seg:
        masks = _load_masks(project.abs(seg.mask_dir))

    has_masks = len(masks) > 0
    checks.append(
        GateCheck(
            "masks_present",
            has_masks,
            float(len(masks)),
            1.0,
            "block",
            f"{len(masks)} mask frames" if has_masks else "No masks",
            "segment_subject",
        )
    )

    stab = temporal_stability_score(masks) if masks else 0.0
    min_stab = settings.harness_min_temporal_stability
    checks.append(
        GateCheck(
            "temporal_stability",
            stab >= min_stab,
            stab,
            min_stab,
            "block",
            f"Temporal IoU stability={stab:.3f}",
            "refine_segment with alternate prompt or lower conf; enable temporal median",
        )
    )

    cov = coverage_mean(masks) if masks else 0.0
    cov_ok = 0.03 <= cov <= 0.80
    checks.append(
        GateCheck(
            "coverage_range",
            cov_ok,
            cov,
            0.03,
            "block" if cov < 0.02 or cov > 0.92 else "warn",
            f"Mean subject coverage={cov:.3f}",
            "tighten prompt" if cov > 0.8 else "lower conf or broaden prompt",
        )
    )

    # edge softness
    if masks:
        sample = masks[:: max(1, len(masks) // 15)][:15]
        soft = float(np.mean([((to_u8_mask(x) > 20) & (to_u8_mask(x) < 235)).mean() for x in sample]))
    else:
        soft = 0.0
    checks.append(
        GateCheck(
            "edge_softness",
            soft > 0.01 or not has_masks,
            soft,
            0.01,
            "warn",
            f"Soft edge fraction={soft:.3f}",
            "increase feather radius",
        )
    )

    has_layers = len(m.layers.layers) > 0
    checks.append(
        GateCheck(
            "layers_present",
            has_layers,
            float(len(m.layers.layers)),
            1.0,
            "block",
            f"{len(m.layers.layers)} layers",
            "apply_background_effects",
        )
    )

    has_render = bool(m.renders) and project.abs(m.renders[-1]["output_path"]).exists()
    checks.append(
        GateCheck(
            "render_present",
            has_render,
            1.0 if has_render else 0.0,
            1.0,
            "block",
            "Render exists" if has_render else "No render yet",
            "composite_and_render",
        )
    )

    # score: weighted
    weights = {
        "source_present": 0.1,
        "masks_present": 0.2,
        "temporal_stability": 0.25,
        "coverage_range": 0.15,
        "edge_softness": 0.05,
        "layers_present": 0.1,
        "render_present": 0.15,
    }
    score = 0.0
    for c in checks:
        w = weights.get(c.name, 0.05)
        score += w * (1.0 if c.passed else max(0.0, c.value / max(c.threshold, 1e-6)) * 0.4)
    score = float(min(1.0, score))

    block_fail = [c for c in checks if not c.passed and c.severity == "block"]
    passed = len(block_fail) == 0 and score >= settings.harness_min_review_score

    refine = None
    if not passed:
        if stab < min_stab:
            refine = "resegment_lower_conf"
            actions.append("Re-run segment_subject with lower conf_threshold and temporal smoothing")
        if cov < 0.03:
            refine = refine or "alternate_prompt"
            actions.append("Try alternate prompts from analyze_video.suggested_prompts")
        if cov > 0.85:
            actions.append("Use more specific prompt e.g. 'person speaking' or box refine")
        if not has_render:
            actions.append("composite_and_render")
        if soft < 0.01 and has_masks:
            actions.append("Increase mask feather (VIDMCP_DEFAULT_MASK_FEATHER)")

    for c in checks:
        if not c.passed and c.fix_hint:
            actions.append(c.fix_hint)
    # dedupe
    actions = list(dict.fromkeys(actions))

    return QualityGateResult(
        passed=passed,
        score=score,
        checks=checks,
        recommended_actions=actions,
        refine_strategy=refine,
    )
