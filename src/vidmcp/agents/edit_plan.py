"""Persisted executable edit plans — draft (no side effects) → revise → execute with checkpoints.

Plans live at plans/edit_<id>.json and commit to the causal edit graph. Execution runs
op-by-op so a killed run resumes from the last checkpoint, never from scratch.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import orjson

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.edit_plan")

# rough per-op cost estimates (seconds, proxy quality) for plan budgeting
OP_COST_S: dict[str, float] = {
    "build_footage_index": 20, "detect_scenes": 8, "plan_cuts": 4, "apply_cut_plan": 25,
    "segment_subject": 60, "refine_alpha": 40, "stabilize_matte": 30, "auto_color": 6,
    "apply_lut": 1, "match_color": 6, "generate_music": 8, "add_sfx": 4, "mixdown_audio": 10,
    "composite_and_render": 90, "add_camera_moves": 45, "smart_reframe": 45, "rewatch_render": 10,
    "export_multi": 60, "generate_thumbnails": 8, "generate_metadata": 4, "extract_clips": 5,
    "transcribe_and_caption": 30, "add_graphics": 2, "dub_video": 40, "time_warp": 30,
}

_INTENT_RECIPE_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("clip", "shorts", "podcast", "moments"), "podcast_clip_factory"),
    (("ad", "commercial", "15"), "ad_15s"),
    (("vlog", "cinematic", "travel"), "cinematic_vlog"),
    (("lecture", "course"), "lecture_to_shorts"),
    (("dub", "language", "spanish", "translate"), "multilang_release"),
    (("beat", "music video"), "music_video_beatcut"),
]


def _plans_dir(project: Any):
    d = project.root / "plans"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pick_recipe(intent: str, content_type: str) -> str:
    text = (intent or "").lower()
    for keys, recipe in _INTENT_RECIPE_HINTS:
        if any(k in text for k in keys):
            return recipe
    if content_type == "vlog":
        return "cinematic_vlog"
    if content_type == "lecture":
        return "lecture_to_shorts"
    if content_type == "ad":
        return "ad_15s"
    return "god_mode_talking_head"


def draft_edit_plan_project(
    project: Any,
    intent: str,
    duration_target_sec: float | None = None,
    deliverables: list[str] | None = None,
    tone: str | None = None,
) -> dict[str, Any]:
    from vidmcp.agents.creative import TONE_PROFILES
    from vidmcp.harness.content_type import classify_project
    from vidmcp.harness.recipe_schema import resolve_recipe

    m = project.manifest
    ct = classify_project(project)
    content_type = ct.get("type", "unknown")
    recipe_name = _pick_recipe(intent, content_type)
    recipe = resolve_recipe(recipe_name)

    # tone inference from intent
    text = (intent or "").lower()
    if tone is None:
        tone = next((t for t in TONE_PROFILES if t in text), None) or (
            "energetic" if any(k in text for k in ("punchy", "fast", "hype")) else "premium"
        )
    tone_params = TONE_PROFILES.get(tone, {})

    steps: list[dict[str, Any]] = []
    for s in recipe.steps:
        args = dict(s.args)
        # tone injection
        if s.tool == "apply_lut" and tone_params.get("lut"):
            args["lut"] = tone_params["lut"]
        if s.tool == "generate_music" and tone_params.get("bgm_style"):
            args["style"] = tone_params["bgm_style"]
        if s.tool == "plan_cuts" and tone_params.get("cut_aggressiveness") is not None:
            args["aggressiveness"] = tone_params["cut_aggressiveness"]
        if s.tool == "transcribe_and_caption" and tone_params.get("caption_style"):
            args["style"] = tone_params["caption_style"]
        steps.append({"tool": s.tool, "args": args, "id": s.id or s.tool, "if": s.when})

    if deliverables:
        targets = [d for d in deliverables if d in ("youtube_16x9", "reels_9x16", "square_1x1")]
        if targets and not any(st["tool"] == "export_multi" for st in steps):
            steps.append({"tool": "export_multi", "args": {"targets": targets}, "id": "export"})
        if "thumbnails" in deliverables:
            steps.append({"tool": "generate_thumbnails", "args": {"n": 3}, "id": "thumbs"})
        if "metadata" in deliverables:
            steps.append({"tool": "generate_metadata", "args": {}, "id": "meta"})

    est = sum(OP_COST_S.get(s["tool"], 15) for s in steps)
    plan_id = f"edit_{uuid4().hex[:8]}"
    plan = {
        "plan_id": plan_id,
        "intent": intent,
        "content_type": content_type,
        "tone": tone,
        "recipe_base": recipe_name,
        "duration_target_sec": duration_target_sec,
        "steps": steps,
        "est_cost_s": est,
        "created_at": time.time(),
        "checkpoints": {},
        "status": "draft",
    }
    (_plans_dir(project) / f"{plan_id}.json").write_bytes(orjson.dumps(plan, option=orjson.OPT_INDENT_2))
    try:
        from vidmcp.tools import advanced_service as adv

        adv.graph_commit(project, "draft_edit_plan", {"plan_id": plan_id, "recipe": recipe_name}, intent[:80])
    except Exception:  # noqa: BLE001
        pass
    m.append_history("draft_edit_plan", {"plan_id": plan_id, "n_steps": len(steps)})
    project.save()
    brief = (
        f"{content_type} footage → '{recipe_name}' base, tone={tone}. "
        f"{len(steps)} steps, ~{int(est / 60)}m{int(est % 60)}s estimated. "
        f"Steps: {' → '.join(s['id'] for s in steps)}"
    )
    return {"ok": True, "plan_id": plan_id, "brief": brief, "content_type": content_type,
            "tone": tone, "est_cost_s": est, "steps": steps}


def _load_plan(project: Any, plan_id: str) -> dict[str, Any] | None:
    p = _plans_dir(project) / f"{plan_id}.json"
    if not p.exists():
        return None
    return orjson.loads(p.read_bytes())


def _save_plan(project: Any, plan: dict[str, Any]) -> None:
    (_plans_dir(project) / f"{plan['plan_id']}.json").write_bytes(
        orjson.dumps(plan, option=orjson.OPT_INDENT_2)
    )


def revise_edit_plan_project(project: Any, plan_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """patch: {remove_ids?: [..], set_args?: {step_id: args}, add_steps?: [{step, after?: id}]}"""
    plan = _load_plan(project, plan_id)
    if plan is None:
        return {"ok": False, "message": f"Unknown plan {plan_id}"}
    steps = plan["steps"]
    diff: list[str] = []
    for rid in patch.get("remove_ids", []):
        before = len(steps)
        steps = [s for s in steps if s["id"] != rid]
        if len(steps) != before:
            diff.append(f"removed {rid}")
    for sid, args in (patch.get("set_args") or {}).items():
        for s in steps:
            if s["id"] == sid:
                s["args"] = {**s["args"], **args}
                diff.append(f"args({sid}) updated")
    for item in patch.get("add_steps", []):
        step = item.get("step") or item
        after = item.get("after")
        idx = next((i + 1 for i, s in enumerate(steps) if s["id"] == after), len(steps))
        steps.insert(idx, {"tool": step["tool"], "args": step.get("args", {}),
                           "id": step.get("id") or step["tool"], "if": step.get("if")})
        diff.append(f"added {step.get('id') or step['tool']}")
    plan["steps"] = steps
    plan["est_cost_s"] = sum(OP_COST_S.get(s["tool"], 15) for s in steps)
    _save_plan(project, plan)
    return {"ok": True, "plan_id": plan_id, "diff": diff, "n_steps": len(steps)}


def execute_edit_plan_project(
    project: Any,
    plan_id: str,
    until_step: str | None = None,
    max_repair_passes: int = 2,
) -> dict[str, Any]:
    from vidmcp.harness.expr import evaluate
    from vidmcp.harness.ops import _resolve_args, resolve_op
    from vidmcp.utils.compact import compact_result

    plan = _load_plan(project, plan_id)
    if plan is None:
        return {"ok": False, "message": f"Unknown plan {plan_id}"}
    checkpoints: dict[str, Any] = plan.get("checkpoints") or {}
    results: dict[str, Any] = {k: v for k, v in checkpoints.items()}
    executed, skipped, failures = [], [], []
    plan["status"] = "running"

    for spec in plan["steps"]:
        sid = spec["id"]
        if sid in checkpoints and (checkpoints[sid] or {}).get("ok"):
            skipped.append(sid)  # resume: already done
            continue
        cond = spec.get("if")
        if cond:
            try:
                if not evaluate(str(cond), results):
                    results[sid] = {"ok": True, "skipped": True}
                    checkpoints[sid] = results[sid]
                    continue
            except Exception:  # noqa: BLE001
                pass
        try:
            fn = resolve_op(spec["tool"])
            out = fn(project, **_resolve_args(dict(spec.get("args") or {}), results))
            if not isinstance(out, dict):
                out = {"ok": True, "value": out}
        except Exception as e:  # noqa: BLE001
            log.exception("plan_step_failed", step=sid)
            out = {"ok": False, "message": str(e)}
        results[sid] = out
        checkpoints[sid] = compact_result(out)
        plan["checkpoints"] = checkpoints
        _save_plan(project, plan)  # checkpoint after every step
        try:
            from vidmcp.tools import advanced_service as adv

            adv.graph_commit(project, f"plan:{sid}", {"ok": out.get("ok")}, spec["tool"])
        except Exception:  # noqa: BLE001
            pass
        (failures if out.get("ok") is False else executed).append(sid)
        if until_step and sid == until_step:
            break

    # gate-driven repair loop
    repairs = 0
    gate_report = None
    if not until_step and any(s["tool"] == "rewatch_render" for s in plan["steps"]):
        qc = results.get("qc") or {}
        while repairs < max_repair_passes and qc and qc.get("ok") is False and qc.get("defects"):
            repairs += 1
            worst = qc["defects"][0]
            log.info("plan_auto_repair", defect=worst.get("kind"), pass_n=repairs)
            repair_ops = {
                "lufs_offset": ("mixdown_audio", {"target": qc.get("loudness_target", "youtube")}),
                "luma_flicker": ("stabilize_matte", {}),
                "black_frames": ("composite_and_render", {}),
                "too_dark": ("auto_color", {}),
            }
            route = repair_ops.get(worst.get("kind"))
            if route is None:
                break
            try:
                fn = resolve_op(route[0])
                fn(project, **route[1])
                if route[0] != "composite_and_render":
                    resolve_op("composite_and_render")(project)
                from vidmcp.agents.rewatch import rewatch_project

                qc = rewatch_project(project)
                results["qc"] = qc
            except Exception as e:  # noqa: BLE001
                log.warning("repair_failed", error=str(e))
                break
        gate_report = {"repair_passes": repairs, "final_qc_ok": (results.get("qc") or {}).get("ok")}

    plan["status"] = "failed" if failures else "complete"
    _save_plan(project, plan)
    return {
        "ok": not failures,
        "plan_id": plan_id,
        "executed": executed,
        "resumed_from_checkpoint": skipped,
        "failures": failures,
        "gate_report": gate_report,
        "results": {k: compact_result(v) for k, v in results.items()},
    }
