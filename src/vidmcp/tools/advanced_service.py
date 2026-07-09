"""Service glue for v0.4 advanced platform tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from vidmcp.advanced.causal_graph import CausalGraphStore
from vidmcp.advanced.lighting_match import apply_lighting_match_to_project_render
from vidmcp.advanced.uncertainty import compute_uncertainty_field
from vidmcp.advanced.world_bg import reproject_background_plate
from vidmcp.agents.debate import propose_edit_strategies
from vidmcp.audio.course import compile_lesson
from vidmcp.audio.semantic import extract_audio_timeline, sync_audio_semantics
from vidmcp.core.workspace import ProjectStore, Workspace
from vidmcp.critics.ensemble import run_critic_ensemble
from vidmcp.depth.fog import apply_depth_ordered_particles
from vidmcp.dsl.viddsl import compile_viddsl, run_viddsl
from vidmcp.failure.mine import FailureStore, mine_workspace_failures, suggest_heuristics
from vidmcp.identity.lock import lock_identity_across_shots
from vidmcp.live.stream import get_live_registry
from vidmcp.provenance.sign import sign_project_render, verify_manifest
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.advanced_service")


def _graph(project: ProjectStore) -> tuple[CausalGraphStore, Any]:
    store = CausalGraphStore(project.root, project.manifest.id)
    g = store.load()
    return store, g


def graph_commit(project: ProjectStore, op: str, args: dict | None = None, message: str = "") -> dict[str, Any]:
    gs, g = _graph(project)
    node = g.commit(
        op,
        args or {},
        result_summary={},
        manifest_snapshot=project.manifest.model_dump(mode="json"),
        message=message or op,
    )
    gs.save(g)
    return {"ok": True, "node_id": node.id, "head": g.head, "branch": node.branch}


def graph_log(project: ProjectStore, limit: int = 30) -> dict[str, Any]:
    _, g = _graph(project)
    return {"ok": True, "head": g.head, "branches": g.branches, "log": g.log(limit=limit)}


def graph_branch(project: ProjectStore, name: str) -> dict[str, Any]:
    gs, g = _graph(project)
    src = g.branch(name)
    gs.save(g)
    return {"ok": True, "branch": name, "from_node": src, "branches": g.branches}


def graph_checkout(project: ProjectStore, node_id: str, *, restore_manifest: bool = True) -> dict[str, Any]:
    gs, g = _graph(project)
    node = g.checkout(node_id)
    restored = False
    if restore_manifest and node.manifest_snapshot:
        from vidmcp.models.project import ProjectManifest

        project.manifest = ProjectManifest.model_validate(node.manifest_snapshot)
        project.save()
        restored = True
    gs.save(g)
    return {"ok": True, "head": g.head, "node": node.model_dump(mode="json"), "manifest_restored": restored}


def graph_merge(project: ProjectStore, source_branch: str, target_branch: str = "main") -> dict[str, Any]:
    gs, g = _graph(project)
    result = g.merge(source_branch, target_branch)
    # apply merged layers if present
    head = g.nodes[g.head]
    if head.manifest_snapshot:
        from vidmcp.models.project import ProjectManifest

        project.manifest = ProjectManifest.model_validate(head.manifest_snapshot)
        project.save()
    gs.save(g)
    return {"ok": True, **result}


def uncertainty_for_project(project: ProjectStore) -> dict[str, Any]:
    seg = project.manifest.primary_segment()
    if not seg:
        return {"ok": False, "message": "No segment"}
    out = compute_uncertainty_field(
        project.abs(seg.mask_dir),
        out_dir=project.previews_dir / "uncertainty",
    )
    # relativize paths
    if out.get("paths"):
        out["paths"] = {k: project.rel(v) for k, v in out["paths"].items()}
    project.manifest.append_history("compute_uncertainty_field", {"hot_frames": out.get("hot_frames")})
    project.save()
    graph_commit(project, "compute_uncertainty_field", {"hot_frames": out.get("hot_frames")})
    return out


def uncertainty_guided_refine(project: ProjectStore, service_module) -> dict[str, Any]:
    unc = uncertainty_for_project(project)
    hints = unc.get("refine_hints") or []
    keyframes = unc.get("hot_frames") or [0]
    refined = service_module.refine_segment_keyframes(
        project,
        keyframes=keyframes[:5],
        hints=hints,
        auto_detect=False,
        prompt=None,
    )
    graph_commit(project, "uncertainty_guided_refine", {"keyframes": keyframes[:5]})
    return {"ok": True, "uncertainty": unc, "refine": refined}


def identity_lock_project(project: ProjectStore, other_shots: list[dict] | None = None) -> dict[str, Any]:
    shots = []
    if project.manifest.source_video and project.manifest.primary_segment():
        seg = project.manifest.primary_segment()
        shots.append(
            {
                "shot_id": project.manifest.id,
                "video_path": str(project.abs(project.manifest.source_video)),
                "mask_dir": str(project.abs(seg.mask_dir)),
                "label": seg.prompt,
            }
        )
    for s in other_shots or []:
        shots.append(s)
    result = lock_identity_across_shots(shots)
    project.manifest.append_history("lock_identity", result.get("registry"))
    project.save()
    return result


def audio_sync_project(project: ProjectStore, transcript: str | None = None, keywords: list[str] | None = None) -> dict[str, Any]:
    if not project.manifest.source_video:
        return {"ok": False, "message": "No source"}
    tl = extract_audio_timeline(
        project.abs(project.manifest.source_video),
        work_dir=project.tmp_dir / "audio",
    )
    sync = sync_audio_semantics(tl, transcript=transcript, keywords=keywords)
    project.manifest.analysis["audio"] = {"timeline": tl, "sync": sync}
    project.manifest.append_history("sync_audio_semantics", {"n_events": sync.get("n_events")})
    project.save()
    graph_commit(project, "sync_audio_semantics", {"n_events": sync.get("n_events")})
    return {"ok": True, "timeline": tl, "sync": sync}


def lighting_match_project(project: ProjectStore, strength: float = 0.8, max_frames: int | None = None) -> dict[str, Any]:
    if not project.manifest.source_video or not project.manifest.primary_segment():
        return {"ok": False, "message": "Need source + segment"}
    seg = project.manifest.primary_segment()
    # find broll/bg plate if any
    plate = None
    for L in project.manifest.layers.sorted_layers():
        if L.kind.value in ("broll", "background") and L.asset_path:
            plate = project.abs(L.asset_path)
            break
    out = project.renders_dir / f"lighting_match_{uuid4().hex[:8]}.mp4"
    result = apply_lighting_match_to_project_render(
        project.abs(project.manifest.source_video),
        project.abs(seg.mask_dir),
        plate,
        out,
        strength=strength,
        max_frames=max_frames,
    )
    project.manifest.renders.append(
        {
            "render_id": str(uuid4()),
            "output_path": project.rel(out),
            "kind": "lighting_match",
        }
    )
    project.manifest.append_history("match_subject_lighting", {"path": project.rel(out)})
    project.save()
    graph_commit(project, "match_subject_lighting", {"path": project.rel(out)})
    result["project_relative"] = project.rel(out)
    return result


def depth_fog_project(project: ProjectStore, style: str = "fog", density: float = 0.5, max_frames: int | None = None) -> dict[str, Any]:
    if not project.manifest.source_video or not project.manifest.primary_segment():
        return {"ok": False, "message": "Need source + segment"}
    seg = project.manifest.primary_segment()
    out = project.renders_dir / f"depth_fog_{uuid4().hex[:8]}.mp4"
    result = apply_depth_ordered_particles(
        project.abs(project.manifest.source_video),
        project.abs(seg.mask_dir),
        out,
        density=density,
        style=style,
        max_frames=max_frames,
    )
    project.manifest.renders.append(
        {"render_id": str(uuid4()), "output_path": project.rel(out), "kind": "depth_fog"}
    )
    project.save()
    graph_commit(project, "apply_depth_fog_particles", {"style": style})
    result["project_relative"] = project.rel(out)
    return result


def reproject_bg_project(project: ProjectStore, plate_path: str | None = None, max_frames: int | None = None) -> dict[str, Any]:
    if not project.manifest.source_video or not project.manifest.primary_segment():
        return {"ok": False, "message": "Need source + segment"}
    seg = project.manifest.primary_segment()
    plate = None
    if plate_path:
        plate = Path(plate_path)
        if not plate.is_absolute():
            plate = project.abs(plate_path)
    else:
        for L in project.manifest.layers.sorted_layers():
            if L.asset_path:
                plate = project.abs(L.asset_path)
                break
    if plate is None or not plate.exists():
        return {"ok": False, "message": "No plate — render_math_scene first or pass plate_path"}
    out = project.renders_dir / f"reproject_{uuid4().hex[:8]}.mp4"
    result = reproject_background_plate(
        project.abs(project.manifest.source_video),
        plate,
        project.abs(seg.mask_dir),
        out,
        max_frames=max_frames,
    )
    project.manifest.renders.append(
        {"render_id": str(uuid4()), "output_path": project.rel(out), "kind": "reproject_bg"}
    )
    project.save()
    graph_commit(project, "reproject_background", {})
    result["project_relative"] = project.rel(out)
    return result


def critic_project(project: ProjectStore, workspace_root: Path | None = None) -> dict[str, Any]:
    result = run_critic_ensemble(project)
    if not result.get("ok") and workspace_root:
        FailureStore(workspace_root).record(
            {
                "project_id": project.manifest.id,
                "failed_axes": result.get("failed_axes"),
                "fix_route": result.get("fix_route"),
                "overall_score": result.get("overall_score"),
            }
        )
    project.manifest.append_history("run_critic_ensemble", {"overall": result.get("overall_score")})
    project.save()
    graph_commit(project, "run_critic_ensemble", {"overall": result.get("overall_score")})
    return result


def debate(intent: str, project: ProjectStore | None = None) -> dict[str, Any]:
    analysis = project.manifest.analysis if project else None
    return propose_edit_strategies(intent, analysis)


def run_dsl(project: ProjectStore, source: str, service_module, max_render_frames: int | None = None) -> dict[str, Any]:
    prog = compile_viddsl(source)
    result = run_viddsl(prog, project=project, service_module=service_module, max_render_frames=max_render_frames)
    graph_commit(project, "run_viddsl", {"ops": len(prog.ops)})
    return {"ok": result.get("ok"), "program": prog.to_dict(), **result}


def live_start(**kw) -> dict[str, Any]:
    s = get_live_registry().create(**kw)
    s.start()
    return {"ok": True, **s.status()}


def live_process_file(session_id: str, video_path: str, out_path: str, max_frames: int = 60, effect: str = "blur") -> dict[str, Any]:
    s = get_live_registry().get(session_id)
    if not s:
        return {"ok": False, "message": "unknown session"}
    return s.process_video_file(video_path, out_path, max_frames=max_frames, effect=effect)


def sign(project: ProjectStore) -> dict[str, Any]:
    r = sign_project_render(project)
    graph_commit(project, "sign_render", {"path": r.get("manifest_path")})
    return r


def verify(manifest_path: str) -> dict[str, Any]:
    import orjson

    data = orjson.loads(Path(manifest_path).read_bytes())
    return verify_manifest(data)


def lesson(intent: str, duration_sec: float = 180.0) -> dict[str, Any]:
    return compile_lesson(intent, duration_sec=duration_sec)


def failures(workspace: Workspace) -> dict[str, Any]:
    mined = mine_workspace_failures(workspace.root)
    return {**mined, "heuristics": suggest_heuristics(mined)}


def word_timeline(project: ProjectStore, fallback_transcript: str | None = None, model_size: str = "base") -> dict[str, Any]:
    from vidmcp.audio.whisper_timeline import transcribe_words, words_to_keyword_events

    if not project.manifest.source_video:
        return {"ok": False, "message": "No source video"}
    result = transcribe_words(
        project.abs(project.manifest.source_video),
        work_dir=project.tmp_dir / "whisper",
        model_size=model_size,
        fallback_transcript=fallback_transcript,
    )
    project.manifest.analysis["words"] = result
    project.manifest.append_history("transcribe_words", {"backend": result.get("backend"), "n_words": len(result.get("words") or [])})
    project.save()
    graph_commit(project, "transcribe_words", {"backend": result.get("backend")})
    return result


def speech_locked_scene(
    project: ProjectStore,
    prompt: str,
    *,
    n_steps: int = 6,
    keywords: list[str] | None = None,
    place_as_background: bool = True,
    fallback_transcript: str | None = None,
) -> dict[str, Any]:
    from vidmcp.audio.speech_lock import plan_speech_locked_steps, render_speech_locked_scene
    from vidmcp.audio.whisper_timeline import transcribe_words, words_to_keyword_events
    from vidmcp.models.layers import Layer, LayerKind
    import shutil

    words_info = word_timeline(project, fallback_transcript=fallback_transcript)
    words = words_info.get("words") or []
    steps = plan_speech_locked_steps(words, n_steps=n_steps, cue_words=keywords)
    out = project.layers_dir / f"speech_scene_{uuid4().hex[:8]}.mp4"
    # geometry
    w, h, fps = 1280, 720, 24.0
    if project.manifest.source_video:
        from vidmcp.utils.video_io import probe_video

        meta = probe_video(project.abs(project.manifest.source_video))
        w, h, fps = meta.width, meta.height, meta.fps or 24
    rendered = render_speech_locked_scene(prompt, steps, out_path=out, width=w, height=h, fps=fps)
    layer_id = None
    if place_as_background:
        layer = Layer(
            name="speech_locked_scene",
            kind=LayerKind.BROLL,
            z_index=4,
            asset_path=project.rel(out),
            opacity=1.0,
            meta={"steps": steps, "prompt": prompt},
        )
        project.manifest.layers.add(layer)
        layer_id = layer.id
    kw_events = words_to_keyword_events(words, keywords or ["prove", "therefore", "equals", "first", "finally"])
    project.manifest.analysis["speech_lock"] = {"steps": steps, "keyword_events": kw_events}
    project.manifest.append_history("speech_locked_scene", {"layer_id": layer_id, "n_steps": len(steps)})
    project.save()
    graph_commit(project, "speech_locked_scene", {"n_steps": len(steps)})
    return {
        "ok": True,
        "layer_id": layer_id,
        "scene_path": project.rel(out),
        "absolute_path": str(out),
        "steps": steps,
        "words_backend": words_info.get("backend"),
        "keyword_events": kw_events,
        "n_words": len(words),
        **{k: rendered.get(k) for k in ("duration_sec", "frames")},
    }


def compute_depth(project: ProjectStore, max_frames: int = 60, prefer_midas: bool = True) -> dict[str, Any]:
    from vidmcp.depth.enhanced import compute_depth_sequence

    if not project.manifest.source_video:
        return {"ok": False, "message": "No source"}
    seg = project.manifest.primary_segment()
    mask_dir = project.abs(seg.mask_dir) if seg else None
    out = compute_depth_sequence(
        project.abs(project.manifest.source_video),
        mask_dir,
        out_dir=project.previews_dir / "depth",
        max_frames=max_frames,
        prefer_midas=prefer_midas,
    )
    if out.get("depth_dir"):
        out["depth_dir"] = project.rel(out["depth_dir"])
    project.manifest.analysis["depth"] = out
    project.save()
    graph_commit(project, "compute_depth", {"backend": out.get("backend")})
    return out


def flow_reproject(project: ProjectStore, plate_path: str | None = None, max_frames: int | None = None) -> dict[str, Any]:
    from vidmcp.depth.enhanced import flow_warp_plate_composite

    if not project.manifest.source_video or not project.manifest.primary_segment():
        return {"ok": False, "message": "Need source + segment"}
    seg = project.manifest.primary_segment()
    plate = None
    if plate_path:
        plate = Path(plate_path)
        if not plate.is_absolute():
            plate = project.abs(plate_path)
    else:
        for L in project.manifest.layers.sorted_layers():
            if L.asset_path:
                plate = project.abs(L.asset_path)
                break
    if plate is None or not Path(plate).exists():
        return {"ok": False, "message": "No plate for flow reproject"}
    out = project.renders_dir / f"flow_reproject_{uuid4().hex[:8]}.mp4"
    result = flow_warp_plate_composite(
        project.abs(project.manifest.source_video),
        Path(plate),
        project.abs(seg.mask_dir),
        out,
        max_frames=max_frames,
    )
    project.manifest.renders.append(
        {"render_id": str(uuid4()), "output_path": project.rel(out), "kind": "flow_reproject"}
    )
    project.save()
    graph_commit(project, "flow_reproject_background", {})
    result["project_relative"] = project.rel(out)
    return result


def detect_project_shots(project: ProjectStore, threshold: float = 0.45) -> dict[str, Any]:
    from vidmcp.advanced.shot_detect import detect_shots

    if not project.manifest.source_video:
        return {"ok": False, "message": "No source"}
    result = detect_shots(project.abs(project.manifest.source_video), threshold=threshold)
    project.manifest.analysis["shots"] = result
    project.manifest.append_history("detect_shots", {"shot_count": result.get("shot_count")})
    project.save()
    graph_commit(project, "detect_shots", {"shot_count": result.get("shot_count")})
    return result


def export_otio(project: ProjectStore) -> dict[str, Any]:
    from vidmcp.advanced.otio_export import export_timeline_json

    shots = (project.manifest.analysis or {}).get("shots", {}).get("shots")
    result = export_timeline_json(project, shots=shots)
    project.manifest.append_history("export_timeline", {"path": result.get("json_path")})
    project.save()
    return result


def apply_auto_heuristics(project: ProjectStore, service_module, max_frames: int | None = None) -> dict[str, Any]:
    from vidmcp.config import get_settings
    from vidmcp.core.workspace import Workspace
    from vidmcp.harness.auto_heuristics import apply_heuristics_to_project
    import vidmcp.tools.advanced_service as adv_mod

    return apply_heuristics_to_project(
        project,
        service_module,
        adv_mod,
        workspace=Workspace(get_settings()),
        max_frames=max_frames,
    )


def education_lesson(
    video_path: str,
    lesson_topic: str,
    project_name: str = "education_lesson",
    narration: str | None = None,
    max_render_frames: int | None = 90,
    style: str = "cinematic",
    n_steps: int = 6,
) -> dict[str, Any]:
    from vidmcp.education.pipeline import run_education_lesson

    return run_education_lesson(
        video_path=video_path,
        lesson_topic=lesson_topic,
        project_name=project_name,
        narration=narration,
        max_render_frames=max_render_frames,
        style=style,
        n_steps=n_steps,
    )


def health() -> dict[str, Any]:
    from vidmcp.tools.health import platform_health

    return platform_health()


def attach_narration(project: ProjectStore, narration: str, force: bool = True) -> dict[str, Any]:
    """Mux TTS narration onto project source (for silent uploads)."""
    from vidmcp.audio.media import ensure_video_with_narration
    import shutil

    if not project.manifest.source_video:
        return {"ok": False, "message": "No source"}
    src = project.abs(project.manifest.source_video)
    out = project.source_dir / "source_narrated.mp4"
    result = ensure_video_with_narration(src, narration=narration, out_path=out, force=force)
    if result.get("muxed"):
        # replace source pointer
        project.manifest.source_video = project.rel(out)
        project.manifest.append_history("attach_narration", {"path": project.rel(out)})
        project.save()
    return {**result, "source_video": project.manifest.source_video}


def enqueue_job(handler: str, payload: dict[str, Any], priority: int = 100) -> dict[str, Any]:
    from vidmcp.queue.worker import get_job_queue

    q = get_job_queue()
    return q.enqueue(handler, payload, priority=priority, project_id=payload.get("project_id"))


def queue_status(job_id: str | None = None, status: str | None = None) -> dict[str, Any]:
    from vidmcp.queue.worker import get_job_queue

    q = get_job_queue()
    if job_id:
        job = q.get(job_id)
        return {"ok": job is not None, "job": job}
    return {"ok": True, "jobs": q.list_jobs(status=status)}


def queue_worker_start(max_jobs: int | None = None, background: bool = True) -> dict[str, Any]:
    from vidmcp.queue.worker import get_job_queue

    q = get_job_queue()
    if background:
        q.start_background()
        return {"ok": True, "mode": "background"}
    n = q.run_worker(max_jobs=max_jobs or 1)
    return {"ok": True, "mode": "inline", "processed": n}


def diarize_project(project: ProjectStore, n_speakers: int = 2) -> dict[str, Any]:
    from vidmcp.audio.diarize import diarize_video

    if not project.manifest.source_video:
        return {"ok": False, "message": "No source"}
    words = (project.manifest.analysis or {}).get("words", {}).get("words")
    result = diarize_video(
        project.abs(project.manifest.source_video),
        work_dir=project.tmp_dir / "diarize",
        n_speakers=n_speakers,
        words=words,
    )
    project.manifest.analysis["diarization"] = result
    project.manifest.append_history("diarize_speakers", {"n_speakers": result.get("n_speakers"), "backend": result.get("backend")})
    project.save()
    graph_commit(project, "diarize_speakers", {"n_speakers": result.get("n_speakers")})
    return result


def meshy_plate(project: ProjectStore, prompt: str, place_as_background: bool = True, duration_sec: float = 4.0) -> dict[str, Any]:
    from vidmcp.integrations.meshy import text_to_3d_plate
    from vidmcp.models.layers import Layer, LayerKind
    import shutil

    w, h, fps = 1280, 720, 24.0
    if project.manifest.source_video:
        from vidmcp.utils.video_io import probe_video

        m = probe_video(project.abs(project.manifest.source_video))
        w, h, fps = m.width, m.height, m.fps or 24
    result = text_to_3d_plate(
        prompt,
        out_dir=project.layers_dir / "meshy",
        width=w,
        height=h,
        duration_sec=duration_sec,
        fps=fps,
    )
    plate = Path(result["plate_path"])
    dest = project.layers_dir / f"meshy_{uuid4().hex[:8]}.mp4"
    if plate.exists():
        shutil.copy2(plate, dest)
    layer_id = None
    if place_as_background and dest.exists():
        layer = Layer(name="meshy_3d_plate", kind=LayerKind.BROLL, z_index=3, asset_path=project.rel(dest))
        project.manifest.layers.add(layer)
        layer_id = layer.id
        project.save()
    graph_commit(project, "generate_meshy_plate", {"backend": result.get("backend")})
    return {**result, "layer_id": layer_id, "project_plate": project.rel(dest) if dest.exists() else None}


def remotion_scaffold(project: ProjectStore, prompt: str) -> dict[str, Any]:
    from vidmcp.integrations.remotion import scaffold_remotion_scene

    out = project.root / "scenes" / "remotion" / uuid4().hex[:8]
    result = scaffold_remotion_scene(prompt, out_dir=out)
    project.manifest.append_history("scaffold_remotion_scene", {"dir": project.rel(out)})
    project.save()
    result["project_dir_rel"] = project.rel(out)
    return result


def marketplace_list(workspace_root: Path | None = None) -> dict[str, Any]:
    from vidmcp.marketplace.registry import RecipeMarketplace
    from vidmcp.config import get_settings

    root = Path(workspace_root or get_settings().workspace_root)
    mp = RecipeMarketplace(root)
    return {"ok": True, "recipes": mp.list_all()}


def marketplace_publish(recipe: dict[str, Any], author: str = "local") -> dict[str, Any]:
    from vidmcp.marketplace.registry import RecipeMarketplace
    from vidmcp.config import get_settings

    return RecipeMarketplace(get_settings().workspace_root).publish(recipe, author=author)


def marketplace_install(path: str) -> dict[str, Any]:
    from vidmcp.marketplace.registry import RecipeMarketplace
    from vidmcp.config import get_settings

    return RecipeMarketplace(get_settings().workspace_root).install_from_path(Path(path))


def review_ui_start(port: int = 8765) -> dict[str, Any]:
    from vidmcp.review.app import start_review_server
    from vidmcp.config import get_settings

    return start_review_server(get_settings().workspace_root, port=port)


def review_ui_status() -> dict[str, Any]:
    from vidmcp.review.app import get_review_state

    return {"ok": True, **get_review_state()}


def review_decisions() -> dict[str, Any]:
    from vidmcp.review.app import get_review_state

    st = get_review_state()
    return {"ok": True, "decisions": st.get("decisions") or []}
