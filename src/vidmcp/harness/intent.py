"""Intent routing + compact project briefs for agent context control."""

from __future__ import annotations

from typing import Any


def build_project_brief(project: Any, *, detail: bool = False) -> dict[str, Any]:
    """Compact project summary for agents (avoids full manifest dump)."""
    m = project.manifest
    meta = m.source_meta or {}
    analysis = m.analysis or {}
    last_render = m.renders[-1] if m.renders else None
    last_review = m.reviews[-1] if m.reviews else None
    hist = m.edit_history or []
    recent = []
    for h in hist[-5:]:
        if isinstance(h, dict):
            recent.append({"action": h.get("action"), "ts": h.get("ts")})
        else:
            recent.append(str(h)[:80])

    duration = meta.get("duration") or meta.get("duration_sec") or analysis.get("duration_sec")
    w = meta.get("width") or analysis.get("width")
    hgt = meta.get("height") or analysis.get("height")
    fps = meta.get("fps") or analysis.get("fps")

    next_steps = _suggest_next(m)
    brief: dict[str, Any] = {
        "ok": True,
        "project_id": m.id,
        "name": m.name,
        "status": m.status.value if hasattr(m.status, "value") else str(m.status),
        "source_video": m.source_video,
        "duration_sec": duration,
        "size": {"w": w, "h": hgt, "fps": fps} if (w or hgt or fps) else None,
        "n_segments": len(m.segments or []),
        "n_renders": len(m.renders or []),
        "n_reviews": len(m.reviews or []),
        "primary_segment_id": m.primary_segment_id,
        "last_render": _slim_render(last_render) if last_render else None,
        "last_review_score": (last_review or {}).get("score") if isinstance(last_review, dict) else None,
        "recent_actions": recent,
        "next_steps": next_steps,
        "root": str(project.root),
    }
    if detail:
        brief["tags"] = list(m.tags or [])
        brief["analysis_keys"] = sorted(analysis.keys()) if isinstance(analysis, dict) else []
        brief["audio_pipeline"] = (meta.get("audio_pipeline") or {}) if isinstance(meta, dict) else {}
        if m.segments:
            brief["segments"] = [
                {
                    "id": s.id,
                    "prompt": s.prompt,
                    "backend": s.backend,
                    "frame_count": s.frame_count,
                }
                for s in m.segments[:8]
            ]
    return brief


def _slim_render(r: dict[str, Any]) -> dict[str, Any]:
    return {
        k: r[k]
        for k in ("path", "preset", "created_at", "kind", "duration_sec", "lufs_out")
        if k in r
    }


def _suggest_next(m: Any) -> list[str]:
    status = m.status.value if hasattr(m.status, "value") else str(m.status)
    steps: list[str] = []
    if not m.source_video:
        steps.append("import_video")
        return steps
    if status in ("created",):
        steps.append("analyze_video or run_talking_head_polish / run_intent")
    if not m.segments and status not in ("rendered",):
        steps.append("segment_subject | replace_background | run_talking_head_polish")
    if m.segments and not m.renders:
        steps.append("composite_and_render | export_render | evaluate_quality_gates")
    if m.renders:
        steps.append("project_brief for paths; generate_thumbnail; export_edl")
    if not steps:
        steps.append("run_intent with a natural-language goal")
    return steps[:4]


def resolve_intent(
    intent: str,
    *,
    video_path: str | None = None,
    project_id: str | None = None,
    project_name: str = "intent_run",
    preset: str = "youtube_16x9",
    bg_mode: str = "none",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Map free-text intent → recipe / pipeline (no LLM required)."""
    text = (intent or "").strip().lower()
    if not text:
        return {"ok": False, "message": "intent is empty"}

    plan = _classify(text, preset=preset, bg_mode=bg_mode)
    if dry_run or not video_path:
        out = {
            "ok": True,
            "dry_run": True,
            "intent": intent,
            "plan": plan,
            "hint": "Pass video_path to execute, or dry_run=false with video_path",
        }
        if not video_path and not dry_run:
            out["message"] = "video_path required to execute (omit dry_run / set dry_run=true to only plan)"
            out["ok"] = True  # planning still useful
        return out

    return _execute(plan, video_path=video_path, project_name=project_name, project_id=project_id)


def _classify(text: str, *, preset: str, bg_mode: str) -> dict[str, Any]:
    # Education / math
    if any(k in text for k in ("math", "lesson", "teach", "education", "tutorial", "explain", "bayes", "theorem")):
        return {
            "kind": "education",
            "tool": "run_fast_education_harness",
            "recipe": None,
            "args": {"intent": text, "project_name": "edu_intent"},
        }
    # Named recipes
    if "cyberpunk" in text:
        return {"kind": "recipe", "tool": "apply_recipe", "recipe": "cyberpunk_talking_head", "args": {}}
    if "bokeh" in text or "cinematic" in text and "blur" in text:
        return {"kind": "recipe", "tool": "apply_recipe", "recipe": "cinematic_bokeh", "args": {}}
    if "rain" in text or "noir" in text:
        return {"kind": "recipe", "tool": "apply_recipe", "recipe": "rain_noir", "args": {}}
    if "product" in text and ("spotlight" in text or "hero" in text or "isolate" in text):
        return {"kind": "recipe", "tool": "apply_recipe", "recipe": "product_spotlight", "args": {}}
    if "green screen" in text or "greenscreen" in text:
        return {"kind": "recipe", "tool": "apply_recipe", "recipe": "green_screen_replace", "args": {}}
    if "multi" in text and ("subject" in text or "object" in text or "track" in text):
        return {"kind": "recipe", "tool": "apply_recipe", "recipe": "multi_subject_vfx", "args": {}}
    if any(k in text for k in ("vfx", "matte", "segment", "particle", "behind subject", "composite")):
        return {
            "kind": "harness",
            "tool": "run_quality_gated_pipeline",
            "recipe": None,
            "args": {"intent": text},
        }

    # Creator polish flags from intent language
    polish_bg = bg_mode
    if polish_bg == "none":
        if "space" in text or "galaxy" in text or "cosmos" in text:
            polish_bg = "space"
        elif "blur" in text or "bokeh" in text:
            polish_bg = "blur"
        elif "solid" in text or "studio" in text:
            polish_bg = "solid"

    polish_preset = preset
    if any(k in text for k in ("reel", "tiktok", "shorts", "9:16", "9x16", "vertical", "portrait")):
        polish_preset = "reels_9x16"
    elif any(k in text for k in ("square", "1:1", "1x1", "instagram feed")):
        polish_preset = "square_1x1"
    elif any(k in text for k in ("youtube", "16:9", "16x9", "landscape")):
        polish_preset = "youtube_16x9"

    smart_cut = any(k in text for k in ("hesitat", "filler", "um,", "dead air", "smart cut", "cut silence", "tight"))
    infographics = any(k in text for k in ("infographic", "lower third", "keyword card", "callout"))
    no_captions = any(k in text for k in ("no caption", "without caption", "skip caption"))
    no_bgm = any(k in text for k in ("no bgm", "without music", "no music", "silent bed"))

    # Default creator polish for talking-head / polish / publish language or generic
    return {
        "kind": "polish",
        "tool": "run_talking_head_polish",
        "recipe": "talking_head_polish",
        "args": {
            "preset": polish_preset,
            "bg_mode": polish_bg,
            "smart_cut": smart_cut,
            "infographics": infographics,
            "burn_captions": not no_captions,
            "mix_bgm": not no_bgm,
            "process_audio": True,
        },
    }


def _execute(
    plan: dict[str, Any],
    *,
    video_path: str,
    project_name: str,
    project_id: str | None,
) -> dict[str, Any]:
    from vidmcp.utils.compact import compact_result

    kind = plan.get("kind")
    try:
        if kind == "polish":
            from vidmcp.tools.creator import run_talking_head_polish

            args = dict(plan.get("args") or {})
            result = run_talking_head_polish(
                video_path,
                name=project_name or "talking_head_polish",
                preset=str(args.get("preset") or "youtube_16x9"),
                bg_mode=str(args.get("bg_mode") or "none"),
                burn_captions_flag=bool(args.get("burn_captions", True)),
                smart_cut=bool(args.get("smart_cut", False)),
                infographics=bool(args.get("infographics", False)),
                mix_bgm=bool(args.get("mix_bgm", True)),
                process_audio=bool(args.get("process_audio", True)),
            )
            return compact_result({"ok": True, "plan": plan, "result": result})

        if kind == "recipe":
            from vidmcp.config import get_settings
            from vidmcp.core.workspace import Workspace
            from vidmcp.harness.runtime import HarnessRuntime

            recipe = plan.get("recipe") or "talking_head_polish"
            rt = HarnessRuntime(Workspace(get_settings()))
            result = rt.apply_recipe(
                video_path=video_path,
                recipe_name=str(recipe),
                project_name=project_name,
            )
            return compact_result({"ok": True, "plan": plan, "result": result})

        if kind == "education":
            from vidmcp.config import get_settings
            from vidmcp.core.workspace import Workspace
            from vidmcp.harness.fast_runtime import FastEducationHarness

            args = plan.get("args") or {}
            result = FastEducationHarness(Workspace(get_settings())).run(
                video_path=video_path,
                intent=str(args.get("intent") or "education lesson"),
                project_name=project_name or "edu_intent",
                max_render_frames=48,
                force_fast=True,
            )
            return compact_result({"ok": True, "plan": plan, "result": result})

        if kind == "harness":
            from vidmcp.config import get_settings
            from vidmcp.core.workspace import Workspace
            from vidmcp.harness.runtime import HarnessRuntime

            args = plan.get("args") or {}
            rt = HarnessRuntime(Workspace(get_settings()))
            result = rt.run_quality_gated_pipeline(
                video_path=video_path,
                intent=str(args.get("intent") or "vfx edit"),
                project_name=project_name,
            )
            return compact_result({"ok": True, "plan": plan, "result": result})

        return {"ok": False, "message": f"Unknown plan kind: {kind}", "plan": plan}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": str(e), "plan": plan}
