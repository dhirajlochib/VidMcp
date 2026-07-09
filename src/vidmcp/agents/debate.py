"""Multi-agent strategy debate: propose N edit DAGs scored on quality/cost/risk."""

from __future__ import annotations

from typing import Any

from vidmcp.agents.planner import PlannerAgent


def propose_edit_strategies(
    intent: str,
    analysis: dict[str, Any] | None = None,
    *,
    n: int = 3,
) -> dict[str, Any]:
    planner = PlannerAgent()
    base = planner.plan(intent, analysis)
    strategies = []

    # A: quality-gated full pipeline
    strategies.append(
        {
            "id": "A_quality_max",
            "title": "Quality-max gated pipeline",
            "summary": "Multi-pass SAM + gates + critic ensemble + optional refine",
            "tools": [
                "run_quality_gated_pipeline",
                "compute_uncertainty_field",
                "refine_segment_keyframes",
                "run_critic_ensemble",
                "sign_render",
            ],
            "scores": {"quality": 0.92, "cost": 0.35, "risk": 0.25, "latency": 0.3},
            "plan_style_tags": base.style_tags,
        }
    )
    # B: education / math scene
    strategies.append(
        {
            "id": "B_education_scene",
            "title": "Talking-head + math scene + audio sync",
            "summary": "SAM subject, Manim/procedural plate, audio-reactive beats, lighting match",
            "tools": [
                "segment_subject",
                "render_math_scene",
                "sync_audio_semantics",
                "match_subject_lighting",
                "composite_and_render",
                "run_critic_ensemble",
            ],
            "scores": {"quality": 0.85, "cost": 0.45, "risk": 0.3, "latency": 0.4},
            "plan_style_tags": base.style_tags + ["math_scene"],
        }
    )
    # C: fast preview
    strategies.append(
        {
            "id": "C_fast_preview",
            "title": "Fast preview variant pack",
            "summary": "Single segment + A/B variants + compare; low cost exploration",
            "tools": [
                "segment_subject",
                "generate_edit_variants",
                "compare_renders",
                "evaluate_quality_gates",
            ],
            "scores": {"quality": 0.65, "cost": 0.85, "risk": 0.4, "latency": 0.9},
            "plan_style_tags": base.style_tags,
        }
    )
    # D: cinematic depth
    strategies.append(
        {
            "id": "D_cinematic_depth",
            "title": "Cinematic depth fog + reproject plate",
            "summary": "Depth-ordered particles, world-consistent BG, lighting match",
            "tools": [
                "segment_subject",
                "render_math_scene",
                "reproject_background",
                "apply_depth_fog_particles",
                "match_subject_lighting",
                "composite_and_render",
            ],
            "scores": {"quality": 0.88, "cost": 0.4, "risk": 0.35, "latency": 0.35},
            "plan_style_tags": ["blur", "particles"],
        }
    )

    strategies = strategies[: max(1, min(n, len(strategies)))]
    # overall = 0.5q + 0.2cost + 0.2(1-risk) + 0.1latency
    for s in strategies:
        sc = s["scores"]
        s["overall"] = round(
            0.5 * sc["quality"] + 0.2 * sc["cost"] + 0.2 * (1 - sc["risk"]) + 0.1 * sc["latency"],
            3,
        )
    strategies.sort(key=lambda x: x["overall"], reverse=True)
    winner = strategies[0]
    debate_log = [
        f"Planner intent tags: {base.style_tags}",
        f"Critic prefers higher quality unless latency-critical.",
        f"Winner: {winner['id']} (overall={winner['overall']})",
    ]
    return {
        "ok": True,
        "intent": intent,
        "strategies": strategies,
        "recommended": winner["id"],
        "debate_log": debate_log,
        "subject_prompt": base.subject_prompt,
    }
