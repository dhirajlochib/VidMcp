"""Harness contracts — required artifacts, gates, stop rules (2026 harness design).

Inspired by production harness research:
- Control: phase decomposition + adaptive topology
- Contracts: artifact requirements + validation gates + stop rules
- State: durable project/causal graph across steps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    INGEST = "ingest"
    PERCEIVE = "perceive"
    REFINE = "refine"
    SCENE = "scene"
    COMPOSE = "compose"
    VERIFY = "verify"
    SIGN = "sign"
    DONE = "done"


# Minimal tool packs (Vercel-style: fewer tools → better agent use)
TOOL_PACKS: dict[str, list[str]] = {
    "education": [
        "run_education_lesson",
        "transcribe_words",
        "render_speech_locked_scene",
        "segment_subject",
        "refine_segment_keyframes",
        "compute_uncertainty_field",
        "uncertainty_guided_refine",
        "composite_and_render",
        "run_critic_ensemble",
        "sign_render",
        "start_review_ui",
        "get_review_decisions",
        "get_backend_info",
        "platform_health",
        "enqueue_job",
        "queue_status",
    ],
    "creator_vfx": [
        "run_quality_gated_pipeline",
        "segment_subject",
        "segment_multi_objects",
        "apply_background_effects",
        "generate_meshy_plate",
        "apply_depth_fog_particles",
        "match_subject_lighting",
        "generate_edit_variants",
        "compare_renders",
        "run_critic_ensemble",
        "sign_render",
        "enhance_edit",
    ],
    "full": [],  # empty = no restriction
}


@dataclass
class PhaseContract:
    phase: Phase
    required_artifacts: list[str]
    tools: list[str]
    gate_metric: str | None = None
    gate_min: float | None = None
    max_retries: int = 2
    on_fail: str = "retry"  # retry | skip | stop
    parallelizable: bool = False


EDUCATION_CONTRACTS: list[PhaseContract] = [
    PhaseContract(Phase.INGEST, ["source_video"], ["import_video", "analyze_video", "attach_narration"], max_retries=1),
    PhaseContract(Phase.PERCEIVE, ["segment"], ["segment_subject", "transcribe_words"], gate_metric="coverage", gate_min=0.02),
    PhaseContract(
        Phase.REFINE,
        ["segment_stable"],
        ["compute_uncertainty_field", "uncertainty_guided_refine", "refine_segment_keyframes"],
        gate_metric="stability",
        gate_min=0.55,
        max_retries=2,
        on_fail="continue",  # type: ignore
    ),
    PhaseContract(
        Phase.SCENE,
        ["scene_plate"],
        ["render_speech_locked_scene", "render_math_scene", "generate_meshy_plate"],
        parallelizable=True,
    ),
    PhaseContract(Phase.COMPOSE, ["render"], ["composite_and_render", "match_subject_lighting"]),
    PhaseContract(
        Phase.VERIFY,
        ["critic_pass"],
        ["run_critic_ensemble", "evaluate_quality_gates", "apply_auto_heuristics"],
        gate_metric="critic_score",
        gate_min=0.6,
        max_retries=1,
        on_fail="continue",  # type: ignore
    ),
    PhaseContract(Phase.SIGN, ["provenance"], ["sign_render", "export_timeline"]),
]


@dataclass
class HarnessBudget:
    max_wall_sec: float = 600.0
    max_segment_frames: int | None = 90
    max_render_frames: int | None = 90
    max_tool_calls: int = 24
    allow_expensive_sam: bool = True


@dataclass
class HarnessPlan:
    product: str  # education | creator_vfx | full
    phases: list[PhaseContract] = field(default_factory=list)
    tool_allowlist: list[str] = field(default_factory=list)
    budget: HarnessBudget = field(default_factory=HarnessBudget)
    intent: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "intent": self.intent,
            "tool_allowlist": self.tool_allowlist,
            "budget": {
                "max_wall_sec": self.budget.max_wall_sec,
                "max_segment_frames": self.budget.max_segment_frames,
                "max_render_frames": self.budget.max_render_frames,
                "max_tool_calls": self.budget.max_tool_calls,
            },
            "phases": [
                {
                    "phase": p.phase.value,
                    "required_artifacts": p.required_artifacts,
                    "tools": p.tools,
                    "gate_metric": p.gate_metric,
                    "gate_min": p.gate_min,
                    "max_retries": p.max_retries,
                }
                for p in self.phases
            ],
            "notes": self.notes,
        }


def build_harness_plan(
    intent: str,
    *,
    product: str | None = None,
    fast: bool = True,
) -> HarnessPlan:
    text = (intent or "").lower()
    if product is None:
        if any(k in text for k in ("lesson", "teach", "math", "explain", "course", "tutor", "education")):
            product = "education"
        elif any(k in text for k in ("cyberpunk", "vfx", "particle", "background", "bokeh")):
            product = "creator_vfx"
        else:
            product = "education"  # product default for this project

    budget = HarnessBudget(
        max_wall_sec=180.0 if fast else 600.0,
        max_segment_frames=48 if fast else 180,
        max_render_frames=48 if fast else 180,
        max_tool_calls=16 if fast else 32,
    )
    if product == "education":
        phases = list(EDUCATION_CONTRACTS)
        allow = list(TOOL_PACKS["education"])
        notes = [
            "Education harness: minimal tool pack, speech-locked scene, early verify.",
            "Fast mode skips multi-variant and heavy broll.",
        ]
    elif product == "creator_vfx":
        phases = [
            PhaseContract(Phase.INGEST, ["source_video"], ["import_video", "analyze_video"]),
            PhaseContract(Phase.PERCEIVE, ["segment"], ["segment_subject", "segment_multi_objects"]),
            PhaseContract(Phase.REFINE, ["segment_stable"], ["uncertainty_guided_refine"], gate_metric="stability", gate_min=0.5),
            PhaseContract(Phase.SCENE, ["fx_layers"], ["apply_background_effects", "generate_meshy_plate", "apply_depth_fog_particles"]),
            PhaseContract(Phase.COMPOSE, ["render"], ["composite_and_render", "generate_edit_variants"]),
            PhaseContract(Phase.VERIFY, ["critic_pass"], ["run_critic_ensemble"], gate_metric="critic_score", gate_min=0.55),
            PhaseContract(Phase.SIGN, ["provenance"], ["sign_render"]),
        ]
        allow = list(TOOL_PACKS["creator_vfx"])
        notes = ["Creator VFX harness: effects + variants focus."]
    else:
        phases = list(EDUCATION_CONTRACTS)
        allow = []
        notes = ["Full tool surface — prefer specialized packs for reliability."]

    return HarnessPlan(product=product, phases=phases, tool_allowlist=allow, budget=budget, intent=intent, notes=notes)
