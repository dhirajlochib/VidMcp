"""Apply mined failure heuristics automatically inside pipelines."""

from __future__ import annotations

from typing import Any

from vidmcp.config import get_settings
from vidmcp.core.workspace import ProjectStore, Workspace
from vidmcp.failure.mine import mine_workspace_failures, suggest_heuristics
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.auto_heuristics")


def collect_active_heuristics(workspace_root) -> list[dict[str, Any]]:
    mined = mine_workspace_failures(workspace_root)
    return suggest_heuristics(mined).get("suggestions") or []


def apply_heuristics_to_project(
    project: ProjectStore,
    service_module,
    adv_module,
    *,
    workspace: Workspace | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run post-pass fixes based on active heuristics + current critic state."""
    settings = get_settings()
    ws_root = workspace.root if workspace else settings.workspace_root
    heuristics = collect_active_heuristics(ws_root)
    actions_taken = []

    # always run critic to know axes
    critics = adv_module.critic_project(project, workspace_root=ws_root)
    failed = set(critics.get("failed_axes") or [])

    names = {h["heuristic"] for h in heuristics}

    # Avoid multi-minute re-SAM when matte is already temporally stable
    seg = project.manifest.primary_segment()
    stab = float((seg.meta or {}).get("temporal_stability") or 0) if seg else 0
    if stab >= 0.85 and "matte_flicker" not in failed:
        actions_taken.append({"heuristic": "skip_heavy_refine", "ok": True, "reason": f"stability={stab:.2f}"})
    elif "auto_refine_on_stab_below_0.65" in names or "matte_flicker" in failed or ("edge_quality" in failed and stab < 0.85):
        try:
            r = adv_module.uncertainty_guided_refine(project, service_module)
            actions_taken.append({"heuristic": "uncertainty_guided_refine", "ok": True})
        except Exception as e:  # noqa: BLE001
            actions_taken.append({"heuristic": "uncertainty_guided_refine", "ok": False, "error": str(e)})

    if "auto_lighting_match_after_bg_replace" in names or "lighting_match" in failed:
        try:
            r = adv_module.lighting_match_project(project, max_frames=max_frames)
            actions_taken.append({"heuristic": "match_subject_lighting", "ok": True, "path": r.get("project_relative")})
        except Exception as e:  # noqa: BLE001
            actions_taken.append({"heuristic": "match_subject_lighting", "ok": False, "error": str(e)})

    if "increase_feather_and_uncertainty_roi" in names:
        # re-refine with auto detect
        try:
            r = service_module.refine_segment_keyframes(project, auto_detect=True)
            actions_taken.append({"heuristic": "refine_segment_keyframes", "ok": True})
        except Exception as e:  # noqa: BLE001
            actions_taken.append({"heuristic": "refine_segment_keyframes", "ok": False, "error": str(e)})

    if "enforce_composite_before_sign" in names and not project.manifest.renders:
        try:
            r = service_module.composite(project, max_frames=max_frames)
            actions_taken.append({"heuristic": "composite_and_render", "ok": True})
        except Exception as e:  # noqa: BLE001
            actions_taken.append({"heuristic": "composite_and_render", "ok": False, "error": str(e)})

    # re-critic
    critics_after = adv_module.critic_project(project, workspace_root=ws_root)
    return {
        "ok": True,
        "heuristics_considered": list(names),
        "actions_taken": actions_taken,
        "critics_before": {"failed_axes": list(failed), "score": critics.get("overall_score")},
        "critics_after": {
            "failed_axes": critics_after.get("failed_axes"),
            "score": critics_after.get("overall_score"),
        },
    }
