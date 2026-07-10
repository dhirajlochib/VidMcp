"""VidMCP FastMCP server — tools, resources, prompts for AI video editing."""

from __future__ import annotations

import json
from typing import Any

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

from vidmcp import __version__
from vidmcp.agents.orchestrator import PipelineOrchestrator
from vidmcp.agents.planner import PlannerAgent
from vidmcp.harness.runtime import HarnessRuntime
from vidmcp.harness.quality_gates import evaluate_gates
from vidmcp.config import get_settings
from vidmcp.core.job_manager import get_job_manager
from vidmcp.core.workspace import Workspace
from vidmcp.effects.registry import get_effect_registry
from vidmcp.models.jobs import JobType
from vidmcp.models.schemas import (
    AnalyzeVideoResult,
    ApplyEffectsResult,
    CompositeResult,
    GenerateBrollResult,
    ReviewResult,
    SegmentSubjectResult,
)
from vidmcp.tools import service
from vidmcp.tools.advanced_mcp import register_advanced_tools
from vidmcp.utils.logging import get_logger, setup_logging

log = get_logger("vidmcp.server")

mcp = FastMCP(
    "VidMCP",
    instructions=(
        "VidMCP is an ADVANCED production AI video editing MCP. "
        "CONTEXT CONTROL: default tool pack is talking_head (small surface). "
        "Prefer run_intent(video_path, intent) or run_talking_head_polish for creator work; "
        "project_brief(project_id) instead of full get_project; list_tool_packs / set_tool_pack "
        "to switch packs (education|vfx|admin|all). Results are compact by default "
        "(VIDMCP_COMPACT=0 for full payloads; get_project(detail=true) for full manifest). "
        "Complex VFX: apply_recipe / run_quality_gated_pipeline; multi-object: segment_multi_objects; "
        "QA: evaluate_quality_gates / matte_diagnostics. Never overwrite source. "
        "Education: run_fast_education_harness / talking_head_math_lesson. Weights: ensure_sam_weights."
    ),
)


def _ws() -> Workspace:
    return Workspace(get_settings())


def _load(project_id: str):
    return _ws().load_project(project_id)


# v0.4 advanced platform tools
register_advanced_tools(
    mcp,
    load_project=_load,
    workspace_factory=_ws,
    get_settings=get_settings,
    service=service,
    log=log,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def create_project(name: str = "untitled", video_path: str | None = None) -> dict[str, Any]:
    """Create a new non-destructive editing project. Optionally import a source video path."""
    store = _ws().create_project(name=name)
    out: dict[str, Any] = {
        "ok": True,
        "project_id": store.manifest.id,
        "name": store.manifest.name,
        "root": str(store.root),
    }
    if video_path:
        rel = service.import_source(store, video_path)
        out["source_video"] = rel
    return out


@mcp.tool()
def list_projects() -> dict[str, Any]:
    """List all projects in the workspace with status and source paths."""
    from vidmcp.utils.compact import compact_result

    return compact_result({"ok": True, "projects": _ws().list_projects()})


@mcp.tool()
def get_project(project_id: str, detail: bool = False) -> dict[str, Any]:
    """Project info. Default is compact brief; detail=true returns full manifest (large)."""
    from vidmcp.harness.intent import build_project_brief
    from vidmcp.utils.compact import compact_result

    store = _load(project_id)
    if not detail:
        return compact_result(build_project_brief(store, detail=False))
    return compact_result(
        {
            "ok": True,
            "project": store.manifest.model_dump(mode="json"),
            "root": str(store.root),
        },
        force=False,  # respect VIDMCP_COMPACT only when user asked for detail
    )


@mcp.tool()
def import_video(
    project_id: str,
    video_path: str,
    bake_orientation: bool = True,
) -> dict[str, Any]:
    """Import (copy) a video into the project source folder.

    bake_orientation: when True (default), bake displaymatrix rotation so portrait
    phone clips are upright for all downstream tools.
    """
    store = _load(project_id)
    rel = service.import_source(store, video_path, bake_orientation=bake_orientation)
    meta = store.manifest.source_meta or {}
    return {
        "ok": True,
        "project_id": project_id,
        "source_video": rel,
        "oriented": meta.get("oriented"),
        "rotation_original": meta.get("rotation_original"),
        "width": meta.get("width"),
        "height": meta.get("height"),
    }


@mcp.tool()
def analyze_video(project_id: str) -> dict[str, Any]:
    """Analyze source video: metadata, talking-head score, motion, suggested SAM prompts, thumbnails.

    Call this first after import so the agent can choose subject prompts intelligently.
    """
    store = _load(project_id)
    jobs = get_job_manager()
    job = jobs.create(store, JobType.ANALYZE, {})
    job.mark_running("analyze")
    jobs.update(store, job)
    try:
        data = service.analyze(store)
        job.mark_succeeded(data)
        jobs.update(store, job)
        return AnalyzeVideoResult(
            ok=True,
            project_id=project_id,
            job_id=job.id,
            message="Analysis complete",
            duration_sec=float(data.get("duration_sec") or 0),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            fps=float(data.get("fps") or 0),
            frame_count=int(data.get("frame_count") or 0),
            has_audio=bool(data.get("has_audio")),
            scene_hints=list(data.get("scene_hints") or []),
            suggested_prompts=list(data.get("suggested_prompts") or []),
            thumbnail_paths=list(data.get("thumbnail_paths") or []),
            talking_head_score=float(data.get("talking_head_score") or 0),
            data=data,
        ).model_dump()
    except Exception as e:  # noqa: BLE001
        job.mark_failed(str(e))
        jobs.update(store, job)
        log.exception("analyze_failed", project_id=project_id)
        return {"ok": False, "project_id": project_id, "job_id": job.id, "message": str(e)}


@mcp.tool()
def segment_subject(
    project_id: str,
    prompt: str = "person",
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Run SAM 3 / SAM 3.1 (or fallback) text-promptable video segmentation + tracking.

    Produces a temporally stable mask sequence for the subject described by `prompt`.
    Use short noun phrases: 'person', 'speaker', 'red car', 'dog'.
    """
    store = _load(project_id)
    jobs = get_job_manager()
    job = jobs.create(store, JobType.SEGMENT, {"prompt": prompt, "conf": conf_threshold})
    job.mark_running("segment")
    jobs.update(store, job)

    def prog(p: float, msg: str) -> None:
        job.mark_progress(p, stage="segment", message=msg)
        jobs.update(store, job)

    try:
        data = service.segment(store, prompt=prompt, conf=conf_threshold, progress=prog)
        job.mark_succeeded(data)
        jobs.update(store, job)
        return SegmentSubjectResult(
            ok=True,
            project_id=project_id,
            job_id=job.id,
            message=f"Segmented '{prompt}' via {data.get('backend')}",
            segment_id=data.get("segment_id"),
            prompt=prompt,
            backend=str(data.get("backend")),
            mask_dir=data.get("mask_dir"),
            object_count=int(data.get("object_count") or 0),
            objects=list(data.get("objects") or []),
            coverage_mean=float(data.get("coverage_mean") or 0),
            temporal_stability=float(data.get("temporal_stability") or 0),
            data=data,
        ).model_dump()
    except Exception as e:  # noqa: BLE001
        job.mark_failed(str(e))
        jobs.update(store, job)
        log.exception("segment_failed", project_id=project_id)
        return {"ok": False, "project_id": project_id, "job_id": job.id, "message": str(e)}


@mcp.tool()
def apply_background_effects(
    project_id: str,
    effects: list[dict[str, Any]] | None = None,
    style_preset: str | None = None,
    intent: str = "",
) -> dict[str, Any]:
    """Add non-destructive background / particle / grade layers behind the segmented subject.

    Provide either:
    - style_preset: 'cyberpunk' | 'blur' | 'solid_dark' | 'particles_neon'
    - effects: list of {effect_type, kind, intensity, params, name, blend_mode}
    - intent: natural language used by planner if effects omitted

    Effect types: blur, solid, image_plate, cyberpunk, generative, particles, color_grade.
    """
    store = _load(project_id)
    specs = effects
    if specs is None and style_preset:
        preset_map = {
            "cyberpunk": ["cyberpunk", "particles"],
            "blur": ["blur"],
            "solid_dark": [],
            "particles_neon": ["blur", "particles"],
        }
        tags = preset_map.get(style_preset, [style_preset])
        if style_preset == "solid_dark":
            specs = [
                {
                    "effect_type": "solid",
                    "kind": "background",
                    "params": {"color": "#05010f"},
                    "name": "solid_dark",
                }
            ]
        else:
            specs = PlannerAgent().effects_from_tags(tags, intent or style_preset)
    jobs = get_job_manager()
    job = jobs.create(store, JobType.EFFECTS, {"preset": style_preset, "intent": intent})
    job.mark_running("effects")
    try:
        data = service.apply_effects(store, effect_specs=specs, intent=intent)
        job.mark_succeeded(data)
        jobs.update(store, job)
        return ApplyEffectsResult(
            ok=True,
            project_id=project_id,
            job_id=job.id,
            message="Effects layers applied",
            layer_ids=list(data.get("layer_ids") or []),
            effect_types=list(data.get("effect_types") or []),
            stack_version=int(data.get("stack_version") or 0),
            data=data,
        ).model_dump()
    except Exception as e:  # noqa: BLE001
        job.mark_failed(str(e))
        jobs.update(store, job)
        return {"ok": False, "project_id": project_id, "job_id": job.id, "message": str(e)}


@mcp.tool()
def generate_broll(
    project_id: str,
    style: str = "cyberpunk_city",
    prompt: str = "",
    duration_sec: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate a B-roll / background plate video matched to source resolution and duration.

    Styles: cyberpunk_city, particles_field, abstract.
    Future: hooks to diffusion / video generative models when VIDMCP_ENABLE_GENERATIVE=true.
    """
    store = _load(project_id)
    jobs = get_job_manager()
    job = jobs.create(store, JobType.BROLL, {"style": style, "prompt": prompt})
    job.mark_running("broll")
    try:
        data = service.generate_broll(
            store, style=style, prompt=prompt, duration_sec=duration_sec, seed=seed
        )
        job.mark_succeeded(data)
        jobs.update(store, job)
        return GenerateBrollResult(
            ok=True,
            project_id=project_id,
            job_id=job.id,
            message="B-roll generated",
            broll_path=data.get("broll_path"),
            layer_id=data.get("layer_id"),
            mode=str(data.get("mode") or "procedural"),
            duration_sec=float(data.get("duration_sec") or 0),
            data=data,
        ).model_dump()
    except Exception as e:  # noqa: BLE001
        job.mark_failed(str(e))
        jobs.update(store, job)
        return {"ok": False, "project_id": project_id, "job_id": job.id, "message": str(e)}


@mcp.tool()
def composite_and_render(
    project_id: str,
    output_name: str | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Composite the non-destructive layer stack and encode final video (subject over FX, original audio).

    max_frames: optional limit for previews / tests. Omit for full render.
    """
    store = _load(project_id)
    jobs = get_job_manager()
    job = jobs.create(store, JobType.COMPOSITE, {"output_name": output_name, "max_frames": max_frames})
    job.mark_running("composite")

    def prog(p: float, msg: str) -> None:
        job.mark_progress(p, stage="composite", message=msg)
        jobs.update(store, job)

    try:
        data = service.composite(store, output_name=output_name, max_frames=max_frames, progress=prog)
        job.mark_succeeded(data)
        jobs.update(store, job)
        return CompositeResult(
            ok=True,
            project_id=project_id,
            job_id=job.id,
            message="Render complete",
            output_path=data.get("output_path"),
            preview_path=data.get("preview_path"),
            duration_sec=float(data.get("duration_sec") or 0),
            render_id=data.get("render_id"),
            data=data,
        ).model_dump()
    except Exception as e:  # noqa: BLE001
        job.mark_failed(str(e))
        jobs.update(store, job)
        log.exception("composite_failed", project_id=project_id)
        return {"ok": False, "project_id": project_id, "job_id": job.id, "message": str(e)}


@mcp.tool()
def review_edit(project_id: str) -> dict[str, Any]:
    """Critic agent: QA matte stability, coverage, layer stack, and render presence.

    Returns score, pass/fail, notes, and recommended follow-up tool actions.
    """
    store = _load(project_id)
    jobs = get_job_manager()
    job = jobs.create(store, JobType.REVIEW, {})
    job.mark_running("review")
    try:
        data = service.review(store)
        job.mark_succeeded(data)
        jobs.update(store, job)
        return ReviewResult(
            ok=True,
            project_id=project_id,
            job_id=job.id,
            message="Review complete",
            score=float(data.get("score") or 0),
            passed=bool(data.get("passed")),
            notes=list(data.get("notes") or []),
            recommended_actions=list(data.get("recommended_actions") or []),
            data=data,
        ).model_dump()
    except Exception as e:  # noqa: BLE001
        job.mark_failed(str(e))
        jobs.update(store, job)
        return {"ok": False, "project_id": project_id, "job_id": job.id, "message": str(e)}


@mcp.tool()
def run_edit_pipeline(
    video_path: str,
    intent: str,
    project_name: str = "pipeline_edit",
    conf_threshold: float = 0.25,
    max_render_frames: int | None = None,
) -> dict[str, Any]:
    """One-shot multi-agent pipeline: plan → analyze → segment → effects → (broll) → composite → review.

    Example intent: 'Turn this talking-head video into a cyberpunk style with dramatic particle effects behind the speaker'
    """
    orch = PipelineOrchestrator(_ws())
    jobs = get_job_manager()
    # project created inside orchestrator — use a temp shell for job binding after
    try:
        result = orch.run(
            video_path=video_path,
            intent=intent,
            project_name=project_name,
            conf=conf_threshold,
            max_render_frames=max_render_frames,
        )
        result["ok"] = True
        result["message"] = "Pipeline complete"
        return result
    except Exception as e:  # noqa: BLE001
        log.exception("pipeline_failed")
        return {"ok": False, "message": str(e)}


@mcp.tool()
def list_effects() -> dict[str, Any]:
    """List registered VFX effect plugins available to apply_background_effects."""
    return {"ok": True, "effects": get_effect_registry().list_effects()}


@mcp.tool()
def get_job_status(job_id: str) -> dict[str, Any]:
    """Fetch progress/status for a long-running job id returned by other tools."""
    job = get_job_manager().get(job_id)
    if not job:
        return {"ok": False, "message": f"Unknown job: {job_id}"}
    return {"ok": True, "job": job.model_dump(mode="json")}


@mcp.tool()
def plan_edit(intent: str, project_id: str | None = None) -> dict[str, Any]:
    """Planner agent: convert natural language edit intent into an ordered tool plan (no side effects)."""
    analysis = None
    if project_id:
        try:
            analysis = _load(project_id).manifest.analysis
        except Exception:  # noqa: BLE001
            analysis = None
    plan = PlannerAgent().plan(intent, analysis)
    return {
        "ok": True,
        "intent": plan.intent,
        "subject_prompt": plan.subject_prompt,
        "style_tags": plan.style_tags,
        "steps": [{"tool": s.tool, "args": s.args, "rationale": s.rationale} for s in plan.steps],
        "notes": plan.notes,
    }




# ---------------------------------------------------------------------------
# Advanced harness tools (beyond commodity video MCPs)
# ---------------------------------------------------------------------------


@mcp.tool()
def segment_multi_objects(
    project_id: str,
    prompts: list[str],
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """SAM 3.1 Object Multiplex: detect+track MULTIPLE text concepts in one video pass.

    Example prompts: ["person", "microphone"] or ["red car", "traffic light"].
    Writes union masks + per-object mask dirs (obj_001, obj_002, ...).
    """
    store = _load(project_id)
    jobs = get_job_manager()
    job = jobs.create(store, JobType.SEGMENT, {"prompts": prompts, "conf": conf_threshold})
    job.mark_running("segment_multi")
    try:
        data = service.segment_multi(store, prompts=prompts, conf=conf_threshold)
        job.mark_succeeded(data)
        jobs.update(store, job)
        return {"ok": True, "project_id": project_id, "job_id": job.id, **data}
    except Exception as e:
        job.mark_failed(str(e))
        jobs.update(store, job)
        log.exception("segment_multi_failed")
        return {"ok": False, "project_id": project_id, "job_id": job.id, "message": str(e)}


@mcp.tool()
def run_quality_gated_pipeline(
    video_path: str,
    intent: str,
    project_name: str = "harness_edit",
    conf_threshold: float = 0.25,
    max_passes: int = 3,
    max_render_frames: int | None = None,
) -> dict[str, Any]:
    """ADVANCED multi-pass harness: plan → segment → effects → render → quality gates → auto-refine.

    Automatically retries segmentation with alternate prompts / lower conf until gates pass
    or max_passes exhausted. Emits edit_graph + telemetry. Prefer this over run_edit_pipeline
    for production-quality results.
    """
    try:
        rt = HarnessRuntime(_ws())
        result = rt.run_quality_gated_pipeline(
            video_path=video_path,
            intent=intent,
            project_name=project_name,
            conf=conf_threshold,
            max_passes=max_passes,
            max_render_frames=max_render_frames,
        )
        result["message"] = (
            "Quality-gated pipeline complete"
            if result.get("ok")
            else "Completed with gate failures — see final_gate.recommended_actions"
        )
        return result
    except Exception as e:
        log.exception("harness_pipeline_failed")
        return {"ok": False, "message": str(e)}


@mcp.tool()
def apply_recipe(
    video_path: str,
    recipe_name: str,
    project_name: str | None = None,
    max_render_frames: int | None = None,
) -> dict[str, Any]:
    """Run a named production recipe (cyberpunk_talking_head, cinematic_bokeh, product_spotlight, rain_noir, ...)."""
    try:
        rt = HarnessRuntime(_ws())
        return rt.apply_recipe(
            video_path=video_path,
            recipe_name=recipe_name,
            project_name=project_name,
            max_render_frames=max_render_frames,
        )
    except Exception as e:
        log.exception("recipe_failed")
        return {"ok": False, "message": str(e)}


@mcp.tool()
def list_recipes() -> dict[str, Any]:
    """List named production recipes available to apply_recipe."""
    from vidmcp.harness import recipes as recipe_mod

    return {"ok": True, "recipes": recipe_mod.list_recipes()}


@mcp.tool()
def generate_edit_variants(
    project_id: str,
    n: int = 3,
    max_render_frames: int | None = None,
) -> dict[str, Any]:
    """A/B style variants from ONE shared matte (VideoDiff-style). Requires prior segmentation."""
    try:
        rt = HarnessRuntime(_ws())
        return rt.generate_variants(project_id, n=n, max_render_frames=max_render_frames)
    except Exception as e:
        log.exception("variants_failed")
        return {"ok": False, "project_id": project_id, "message": str(e)}


@mcp.tool()
def evaluate_quality_gates(project_id: str) -> dict[str, Any]:
    """Production quality gates with fix hints for automated agent retry loops."""
    store = _load(project_id)
    try:
        gate = evaluate_gates(store, get_settings())
        return {"ok": True, "project_id": project_id, **gate.to_dict()}
    except Exception as e:
        return {"ok": False, "project_id": project_id, "message": str(e)}


@mcp.tool()
def matte_diagnostics(project_id: str) -> dict[str, Any]:
    """Deep matte QA: coverage curve, flicker events, hole ratio."""
    store = _load(project_id)
    try:
        data = service.matte_diagnostics(store)
        return {"ok": True, "project_id": project_id, **data}
    except Exception as e:
        return {"ok": False, "project_id": project_id, "message": str(e)}


@mcp.tool()
def compare_renders(
    project_id: str,
    render_a: str | None = None,
    render_b: str | None = None,
) -> dict[str, Any]:
    """Compare two project renders by mean frame difference (variant selection)."""
    store = _load(project_id)
    try:
        data = service.compare_renders(store, render_a=render_a, render_b=render_b)
        return {"ok": True, "project_id": project_id, **data}
    except Exception as e:
        return {"ok": False, "project_id": project_id, "message": str(e)}


@mcp.tool()
def analyze_motion(project_id: str, max_frames: int = 90) -> dict[str, Any]:
    """Optical-flow motion energy timeline for motion-reactive FX / cut points."""
    from vidmcp.motion.optical_flow import motion_energy_timeline

    store = _load(project_id)
    if not store.manifest.source_video:
        return {"ok": False, "message": "No source video"}
    try:
        data = motion_energy_timeline(store.abs(store.manifest.source_video), max_frames=max_frames)
        store.manifest.analysis.setdefault("motion", data)
        store.manifest.append_history("analyze_motion", {"mean_energy": data.get("mean_energy")})
        store.save()
        return {"ok": True, "project_id": project_id, **data}
    except Exception as e:
        return {"ok": False, "project_id": project_id, "message": str(e)}


@mcp.tool()
def get_backend_info() -> dict[str, Any]:
    """Report active SAM backend (MLX/official/ultra/mock), multiplex flags, device, harness."""
    import platform
    from vidmcp.perception.factory import get_perception_backend
    from vidmcp.perception.mlx_backend import MLXSam31Backend

    s = get_settings()
    b = get_perception_backend(s)
    mlx = MLXSam31Backend()
    return {
        "ok": True,
        "backend_name": getattr(b, "name", type(b).__name__),
        "available": b.is_available(),
        "sam_backend_config": s.sam_backend.value,
        "sam_use_multiplex": s.sam_use_multiplex,
        "sam_weights": str(s.sam_weights) if s.sam_weights else None,
        "device": s.device,
        "apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "mlx": {
            "available": mlx.is_available(),
            "model_id": getattr(s, "mlx_model_id", "mlx-community/sam3.1-bf16"),
            "detect_every": getattr(s, "mlx_detect_every", 8),
            "max_side": getattr(s, "mlx_max_side", 768),
            "recommended": "Set VIDMCP_SAM_BACKEND=mlx on M-series Macs",
        },
        "harness": {
            "max_passes": s.harness_max_passes,
            "min_review_score": s.harness_min_review_score,
            "min_temporal_stability": s.harness_min_temporal_stability,
            "auto_refine": s.harness_auto_refine,
            "variant_count": s.harness_variant_count,
            "fast_mode": getattr(s, "harness_fast_mode", True),
            "preview_frames": getattr(s, "harness_preview_frames", 48),
        },
    }




# ---------------------------------------------------------------------------
# Code → scene (Manim / procedural) + SAM keyframe refine
# ---------------------------------------------------------------------------


@mcp.tool()
def render_math_scene(
    project_id: str,
    prompt: str,
    engine: str = "auto",
    duration_sec: float | None = None,
    place_as_background: bool = True,
    source_code: str | None = None,
) -> dict[str, Any]:
    """Generate a math/education motion scene from text (or sandboxed Manim code) and place it BEHIND the subject.

    engine: auto | manim | procedural
    - auto uses Manim if installed, else high-quality procedural renderer
    - source_code: optional Manim construct body or full Scene class (sandboxed)
    Perfect for: proofs, function plots, geometry explainers behind a talking-head.
    """
    store = _load(project_id)
    jobs = get_job_manager()
    job = jobs.create(store, JobType.BROLL, {"prompt": prompt, "engine": engine})
    job.mark_running("math_scene")
    try:
        data = service.render_math_scene(
            store,
            prompt=prompt,
            source=source_code,
            engine=engine,
            duration_sec=duration_sec,
            place_as_background=place_as_background,
        )
        job.mark_succeeded(data)
        jobs.update(store, job)
        return {"ok": True, "project_id": project_id, "job_id": job.id, **data}
    except Exception as e:
        job.mark_failed(str(e))
        jobs.update(store, job)
        log.exception("math_scene_failed")
        return {"ok": False, "project_id": project_id, "job_id": job.id, "message": str(e)}


@mcp.tool()
def compile_scene_code(
    project_id: str,
    source_code: str,
    engine: str = "auto",
    class_name: str = "VidMCPScene",
    place_as_background: bool = True,
    prompt: str = "custom scene",
) -> dict[str, Any]:
    """Compile agent-authored scene code (Manim construct/class) under sandbox and render to plate.

    Forbidden: os/subprocess/network/eval/exec. Use for precise math animations from LLM-written code.
    """
    store = _load(project_id)
    try:
        data = service.render_math_scene(
            store,
            prompt=prompt,
            source=source_code,
            engine=engine,
            place_as_background=place_as_background,
        )
        return {"ok": True, "project_id": project_id, **data}
    except Exception as e:
        log.exception("compile_scene_failed")
        return {"ok": False, "project_id": project_id, "message": str(e)}


@mcp.tool()
def place_scene_as_background(project_id: str, scene_path: str, name: str = "math_scene_bg") -> dict[str, Any]:
    """Attach an existing scene/video plate as a non-destructive background layer (under subject matte)."""
    store = _load(project_id)
    try:
        data = service.place_scene_background(store, scene_path, name=name)
        return {"ok": True, "project_id": project_id, **data}
    except Exception as e:
        return {"ok": False, "project_id": project_id, "message": str(e)}


@mcp.tool()
def refine_segment_keyframes(
    project_id: str,
    keyframes: list[int] | None = None,
    hints: list[dict] | None = None,
    prompt: str | None = None,
    auto_detect: bool = True,
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Surgical matte repair: re-estimate masks at weak/key frames and temporally heal neighbors.

    hints: optional [{frame_index, prompt?, box_xyxy?, points?, point_labels?, reason?}]
    auto_detect: find flicker/coverage-collapse frames via matte diagnostics heuristics.
    Uses SAM 3.1 re-propagate when weights/backend available; else local heal (always works).
    Creates a NEW segment track (non-destructive) and points subject layer at it.
    """
    store = _load(project_id)
    jobs = get_job_manager()
    job = jobs.create(store, JobType.SEGMENT, {"refine": True, "keyframes": keyframes, "hints": hints})
    job.mark_running("keyframe_refine")
    try:
        data = service.refine_segment_keyframes(
            store,
            keyframes=keyframes,
            hints=hints,
            prompt=prompt,
            auto_detect=auto_detect,
            conf=conf_threshold,
        )
        job.mark_succeeded(data)
        jobs.update(store, job)
        return {"ok": True, "project_id": project_id, "job_id": job.id, **data}
    except Exception as e:
        job.mark_failed(str(e))
        jobs.update(store, job)
        log.exception("refine_failed")
        return {"ok": False, "project_id": project_id, "job_id": job.id, "message": str(e)}


@mcp.tool()
def ensure_sam_weights(
    repo_id: str = "facebook/sam3.1",
    download: bool = False,
    local_dir: str | None = None,
) -> dict[str, Any]:
    """Resolve SAM 3 / 3.1 checkpoint path (search env, ./weights, HF cache). Optionally download from Hugging Face.

    download=True requires HF access approval + HF_TOKEN. Does not download by default.
    """
    from pathlib import Path

    from vidmcp.perception.weights import describe_weights_status, try_download_sam_weights

    status = describe_weights_status(get_settings().sam_weights)
    if download and not status.get("found"):
        dl = try_download_sam_weights(
            repo_id=repo_id,
            local_dir=Path(local_dir) if local_dir else None,
        )
        status = {**status, "download": dl, **describe_weights_status(get_settings().sam_weights)}
    status["ok"] = True
    status["sam_backend"] = get_settings().sam_backend.value
    status["sam_use_multiplex"] = get_settings().sam_use_multiplex
    return status


@mcp.tool()
def talking_head_math_lesson(
    video_path: str,
    lesson_prompt: str,
    project_name: str = "math_lesson",
    style_preset: str = "cinematic_bokeh",
    max_render_frames: int | None = None,
) -> dict[str, Any]:
    """End-to-end: talking-head + math scene background + segment + optional refine + composite + gates.

    Combines SAM subject isolation with code/math scene generation — the education killer feature.
    """
    try:
        rt = HarnessRuntime(_ws())
        # recipe-like flow
        from vidmcp.tools import service as svc

        project = _ws().create_project(name=project_name)
        svc.import_source(project, video_path)
        svc.analyze(project)
        seg = svc.segment(project, prompt="person")
        # math plate behind
        scene = svc.render_math_scene(
            project,
            prompt=lesson_prompt,
            engine="auto",
            place_as_background=True,
        )
        # light style on top of scene plate
        from vidmcp.agents.planner import PlannerAgent

        tags = ["blur"] if "bokeh" in style_preset else ["cyberpunk", "particles"] if "cyber" in style_preset else ["blur"]
        specs = PlannerAgent().effects_from_tags(tags, lesson_prompt)
        # keep scene broll; add grade/particles only
        fx = svc.apply_effects(project, effect_specs=specs, intent=lesson_prompt, replace_existing=False)
        # refine if unstable
        if float(seg.get("temporal_stability") or 1) < 0.65:
            refine = svc.refine_segment_keyframes(project, auto_detect=True, prompt="person")
        else:
            refine = {"skipped": True}
        render = svc.composite(project, max_frames=max_render_frames)
        from vidmcp.harness.quality_gates import evaluate_gates

        gate = evaluate_gates(project, get_settings())
        review = svc.review(project)
        return {
            "ok": gate.passed,
            "project_id": project.manifest.id,
            "segment": seg,
            "scene": scene,
            "effects": fx,
            "refine": refine,
            "render": render,
            "gate": gate.to_dict(),
            "review": review,
            "message": "Talking-head math lesson composite complete",
        }
    except Exception as e:
        log.exception("math_lesson_failed")
        return {"ok": False, "message": str(e)}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("vidmcp://version")
def resource_version() -> str:
    """Server version and backend configuration."""
    s = get_settings()
    return json.dumps(
        {
            "name": "VidMCP",
            "version": __version__,
            "sam_backend": s.sam_backend.value,
            "workspace_root": str(s.workspace_root),
            "device": s.device,
            "sam_use_multiplex": s.sam_use_multiplex,
            "harness_max_passes": s.harness_max_passes,
            "tier": "advanced_harness",
        },
        indent=2,
    )


@mcp.resource("vidmcp://projects")
def resource_projects() -> str:
    """JSON list of projects in the workspace."""
    return json.dumps(_ws().list_projects(), indent=2)


@mcp.resource("vidmcp://project/{project_id}")
def resource_project(project_id: str) -> str:
    """Full manifest for a project."""
    store = _load(project_id)
    return json.dumps(store.manifest.model_dump(mode="json"), indent=2)


@mcp.resource("vidmcp://project/{project_id}/layers")
def resource_layers(project_id: str) -> str:
    """Layer stack snapshot."""
    store = _load(project_id)
    return json.dumps(store.manifest.layers.model_dump(mode="json"), indent=2)


@mcp.resource("vidmcp://project/{project_id}/status")
def resource_status(project_id: str) -> str:
    """High-level project status for agent polling."""
    store = _load(project_id)
    m = store.manifest
    return json.dumps(
        {
            "id": m.id,
            "status": m.status.value,
            "version": m.version,
            "segments": len(m.segments),
            "layers": len(m.layers.layers),
            "renders": len(m.renders),
            "updated_at": m.updated_at.isoformat(),
        },
        indent=2,
    )


@mcp.resource("vidmcp://effects")
def resource_effects() -> str:
    return json.dumps(get_effect_registry().list_effects(), indent=2)


@mcp.resource("vidmcp://recipes")
def resource_recipes() -> str:
    from vidmcp.harness.recipes import list_recipes as _lr

    return json.dumps(_lr(), indent=2)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt()
def cyberpunk_talking_head(video_path: str) -> str:
    """Prompt template: cyberpunk behind-subject edit for a talking-head video."""
    return f"""You are editing a video with VidMCP tools.

Source video: {video_path}

Goal: Turn this talking-head video into a cyberpunk style with dramatic particle effects behind the speaker.

Recommended tool sequence:
1. create_project(name="cyberpunk_talk", video_path="{video_path}")
2. analyze_video(project_id=...)
3. segment_subject(project_id=..., prompt="person")
4. apply_background_effects(project_id=..., style_preset="cyberpunk")
5. generate_broll(project_id=..., style="cyberpunk_city")  # optional plate under particles
6. composite_and_render(project_id=...)
7. review_edit(project_id=...) — if failed, re-segment or adjust effects then re-render

Alternatively call run_edit_pipeline with the same intent for a one-shot multi-agent run.
Report final output_path and review score to the user.
"""


@mcp.prompt()
def precise_bg_replace(video_path: str, subject_prompt: str = "person", bg_description: str = "blurred office") -> str:
    return f"""Use VidMCP for precise background replacement.

Video: {video_path}
Subject prompt: {subject_prompt}
Background intent: {bg_description}

Steps: create_project → import/analyze → segment_subject(prompt="{subject_prompt}") →
apply_background_effects (blur or generative with prompt) → composite_and_render → review_edit.
Prefer non-destructive layers; never overwrite source.
"""


@mcp.prompt()
def quality_gated_cyberpunk(video_path: str) -> str:
    """Advanced harness path with gates + variants."""
    return f"""Use VidMCP ADVANCED harness (not basic trim tools).

Video: {video_path}

Preferred path:
1. get_backend_info() — confirm SAM multiplex / mock
2. run_quality_gated_pipeline(video_path="{video_path}", intent="cyberpunk particles behind speaker", max_passes=3)
   OR apply_recipe(video_path="{video_path}", recipe_name="cyberpunk_talking_head")
3. If gates fail: matte_diagnostics + segment_multi_objects or re-segment
4. generate_edit_variants(project_id=..., n=3)
5. compare_renders(project_id=...)
6. report best render path + gate score

Do NOT stop at a single composite without evaluate_quality_gates.
"""


def create_server() -> FastMCP:
    setup_logging(get_settings().log_level)
    settings = get_settings()
    settings.ensure_dirs()
    from vidmcp.harness.packs import apply_tool_pack_filter

    pack_info = apply_tool_pack_filter(mcp, settings.tool_pack)
    log.info(
        "vidmcp_server_ready",
        version=__version__,
        tool_pack=pack_info.get("pack"),
        tools_kept=pack_info.get("kept"),
        tools_removed=pack_info.get("removed"),
        compact=settings.compact,
    )
    return mcp
