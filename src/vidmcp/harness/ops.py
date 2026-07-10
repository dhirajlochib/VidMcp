"""Server-side op batching — run a list of tool ops in one MCP call with conditions.

Op spec: {tool, args?, id?, if?}. `if` is an expression evaluated against results
so far (by op id), e.g. "seg.temporal_stability < 0.65". Cuts agent round-trips.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from typing import Any

from vidmcp.harness.expr import ExprError, evaluate
from vidmcp.utils.compact import compact_result
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.ops")

# op name -> "module:function". Functions take (project, **args) and return dict.
OP_TABLE: dict[str, str] = {
    # v1 service ops (wrapped below because they need the service module)
    "analyze_video": "vidmcp.tools.service:analyze",
    "segment_subject": "vidmcp.tools.service:segment",
    "apply_effects": "vidmcp.tools.service:apply_effects",
    "composite_and_render": "vidmcp.tools.service:composite",
    "review_edit": "vidmcp.tools.service:review",
    "render_math_scene": "vidmcp.tools.service:render_math_scene",
    "refine_segment_keyframes": "vidmcp.tools.service:refine_segment_keyframes",
    # creator ops
    "process_audio": "vidmcp.tools.creator:process_audio_project",
    "mix_bgm": "vidmcp.tools.creator:mix_bgm_project",
    "transcribe_and_caption": "vidmcp.tools.creator:transcribe_and_caption_project",
    "replace_background": "vidmcp.tools.creator:replace_background_project",
    "export_render": "vidmcp.tools.creator:export_render_project",
    "smart_cut_hesitations": "vidmcp.tools.creator:smart_cut_project",
    "add_speech_infographics": "vidmcp.tools.creator:add_infographics_project",
    "generate_thumbnail": "vidmcp.tools.creator:generate_thumbnail_project",
    # v2 ops (registered as they land)
    "refine_alpha": "vidmcp.matte.alpha_refine:refine_alpha_project",
    "stabilize_matte": "vidmcp.matte.temporal:stabilize_matte_project",
    "detect_scenes": "vidmcp.perception.scene_seg:detect_scenes_project",
    "build_footage_index": "vidmcp.perception.indexer:build_index_project",
    "search_footage": "vidmcp.perception.search:search_footage_project",
    "plan_cuts": "vidmcp.edit.cut_planner:plan_cuts_project",
    "apply_cut_plan": "vidmcp.edit.cut_planner:apply_cut_plan_project",
    "suggest_broll": "vidmcp.edit.broll_match:suggest_broll_project",
    "apply_lut": "vidmcp.color.lut:apply_lut_project",
    "auto_color": "vidmcp.color.auto_correct:auto_color_project",
    "match_color": "vidmcp.color.match:match_color_project",
    "color_scopes": "vidmcp.color.scopes:scopes_project",
    "smart_reframe": "vidmcp.camera.reframe:smart_reframe_project",
    "add_camera_moves": "vidmcp.camera.moves:add_camera_moves_project",
    "time_warp": "vidmcp.camera.timewarp:time_warp_project",
    "stabilize_video": "vidmcp.camera.stabilize:stabilize_video_project",
    "mixdown_audio": "vidmcp.audio.tracks:mixdown_project",
    "add_sfx": "vidmcp.audio.sfx:add_sfx_project",
    "generate_music": "vidmcp.audio.music:generate_music_project",
    "dub_video": "vidmcp.audio.dubbing:dub_video_project",
    "add_graphics": "vidmcp.graphics.engine:add_graphics_project",
    "classify_content": "vidmcp.harness.content_type:classify_project",
    "analyze_pacing": "vidmcp.agents.creative:analyze_pacing_project",
    "rewatch_render": "vidmcp.agents.rewatch:rewatch_project",
    "export_multi": "vidmcp.media.delivery:export_multi_project",
    "generate_thumbnails": "vidmcp.media.thumbs:generate_thumbnails_project",
    "generate_metadata": "vidmcp.media.metadata:generate_metadata_project",
    "extract_clips": "vidmcp.media.metadata:extract_clips_project",
}


def resolve_op(name: str) -> Callable[..., dict[str, Any]]:
    target = OP_TABLE.get(name)
    if target is None:
        raise KeyError(f"Unknown op '{name}'. Available: {sorted(OP_TABLE)}")
    mod_name, fn_name = target.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise KeyError(f"Op '{name}' target {target} not found")
    return fn


def _flatten_for_expr(result: Any) -> Any:
    """Results are dicts; expose them directly to the expression context."""
    return result if isinstance(result, dict) else {"value": result}


def _resolve_args(args: Any, results: dict[str, Any]) -> Any:
    """Template '$op_id.path' strings with values from earlier op results."""
    from vidmcp.harness.expr import _resolve

    if isinstance(args, dict):
        return {k: _resolve_args(v, results) for k, v in args.items()}
    if isinstance(args, list):
        return [_resolve_args(v, results) for v in args]
    if isinstance(args, str) and args.startswith("$"):
        val = _resolve(args[1:], {k: _flatten_for_expr(v) for k, v in results.items()})
        return val if val is not None else args
    return args


def run_ops(
    project: Any,
    ops: list[dict[str, Any]],
    *,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    order: list[str] = []
    failures: list[str] = []
    skipped: list[str] = []
    t0 = time.time()
    for i, spec in enumerate(ops):
        tool = str(spec.get("tool") or "").strip()
        op_id = str(spec.get("id") or f"{tool}_{i}")
        args = dict(spec.get("args") or {})
        cond = spec.get("if")
        if cond:
            try:
                ctx = {k: _flatten_for_expr(v) for k, v in results.items()}
                if not evaluate(str(cond), ctx):
                    skipped.append(op_id)
                    results[op_id] = {"ok": True, "skipped": True, "if": cond}
                    order.append(op_id)
                    continue
            except ExprError as e:
                results[op_id] = {"ok": False, "message": f"bad condition: {e}"}
                failures.append(op_id)
                order.append(op_id)
                if stop_on_error:
                    break
                continue
        try:
            fn = resolve_op(tool)
            out = fn(project, **_resolve_args(args, results))
            if not isinstance(out, dict):
                out = {"ok": True, "value": out}
            results[op_id] = out
            if out.get("ok") is False:
                failures.append(op_id)
                if stop_on_error:
                    order.append(op_id)
                    break
        except Exception as e:  # noqa: BLE001
            log.exception("op_failed", tool=tool)
            results[op_id] = {"ok": False, "message": str(e)}
            failures.append(op_id)
            if stop_on_error:
                order.append(op_id)
                break
        order.append(op_id)
    return {
        "ok": not failures,
        "results": {k: compact_result(v) for k, v in results.items()},
        "order": order,
        "skipped": skipped,
        "failures": failures,
        "timeline_ms": int((time.time() - t0) * 1000),
    }


def list_ops() -> list[str]:
    return sorted(OP_TABLE)
