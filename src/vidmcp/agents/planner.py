"""Planner agent — decomposes natural language edit intent into a tool plan."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanStep:
    tool: str
    args: dict[str, Any]
    rationale: str


@dataclass
class EditPlan:
    intent: str
    steps: list[PlanStep] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    subject_prompt: str = "person"
    notes: list[str] = field(default_factory=list)


class PlannerAgent:
    """Rule + heuristic planner (host LLM still orchestrates; this is deterministic fallback)."""

    def plan(self, intent: str, analysis: dict[str, Any] | None = None) -> EditPlan:
        text = intent.lower()
        plan = EditPlan(intent=intent)
        analysis = analysis or {}

        # subject prompt
        if any(k in text for k in ("speaker", "person", "host", "me", "talking")):
            plan.subject_prompt = "person"
        if "face" in text:
            plan.subject_prompt = "face"
        suggested = analysis.get("suggested_prompts") or []
        if suggested and plan.subject_prompt not in suggested:
            plan.notes.append(f"analysis suggests prompts: {suggested}")

        # style
        if "cyberpunk" in text or "neon" in text:
            plan.style_tags += ["cyberpunk", "particles"]
        if "blur" in text or "bokeh" in text or "depth" in text:
            plan.style_tags.append("blur")
        if "particle" in text or "spark" in text or "dust" in text:
            plan.style_tags.append("particles")
        if "rain" in text:
            plan.style_tags.append("rain")
        if "replace" in text or "background" in text or "scene" in text:
            plan.style_tags.append("background")
        if not plan.style_tags:
            plan.style_tags = ["blur"]

        plan.steps = [
            PlanStep("analyze_video", {}, "Understand source geometry and scene type"),
            PlanStep(
                "segment_subject",
                {"prompt": plan.subject_prompt},
                "Text-promptable SAM matte with temporal tracking",
            ),
            PlanStep(
                "apply_background_effects",
                {"style_tags": plan.style_tags, "intent": intent},
                "Build non-destructive layer stack for behind-subject VFX",
            ),
        ]
        if any(k in text for k in ("b-roll", "broll", "plate", "city", "environment")):
            plan.steps.append(
                PlanStep("generate_broll", {"style": "cyberpunk_city" if "cyber" in text else "abstract"}, "Generate B-roll plate")
            )
        plan.steps.append(PlanStep("composite_and_render", {}, "Render layer stack with subject over FX"))
        plan.steps.append(PlanStep("review_edit", {}, "Critic pass for matte stability and artifacts"))
        return plan

    def effects_from_tags(self, style_tags: list[str], intent: str) -> list[dict[str, Any]]:
        effects: list[dict[str, Any]] = []
        tags = set(style_tags)
        if "cyberpunk" in tags:
            effects.append(
                {
                    "effect_type": "cyberpunk",
                    "kind": "background",
                    "intensity": 1.0,
                    "params": {"blur_radius": 21, "scanlines": True},
                    "name": "cyberpunk_bg",
                }
            )
        elif "blur" in tags:
            effects.append(
                {
                    "effect_type": "blur",
                    "kind": "background",
                    "intensity": 1.0,
                    "params": {"blur_radius": 35},
                    "name": "bg_blur",
                }
            )
        else:
            effects.append(
                {
                    "effect_type": "solid",
                    "kind": "background",
                    "intensity": 1.0,
                    "params": {"color": "#0a051e"},
                    "name": "solid_bg",
                }
            )

        if "particles" in tags or "rain" in tags or "cyberpunk" in tags:
            style = "rain" if "rain" in tags else ("sparks" if "spark" in intent.lower() else "neon_dust")
            effects.append(
                {
                    "effect_type": "particles",
                    "kind": "particles",
                    "intensity": 1.0,
                    "params": {"style": style, "density": 0.45, "seed": 7},
                    "name": f"particles_{style}",
                    "blend_mode": "screen",
                }
            )
        return effects
