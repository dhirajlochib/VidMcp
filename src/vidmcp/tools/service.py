"""Domain service layer used by MCP tools and internal agents."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from vidmcp.config import get_settings
from vidmcp.core.workspace import ProjectStore
from vidmcp.effects.broll import generate_procedural_broll, match_source_geometry
from vidmcp.models.layers import BlendMode, EffectParams, Layer, LayerKind
from vidmcp.models.project import ProjectStatus, SegmentObject, SegmentTrack
from vidmcp.perception.analyzer import analyze_video
from vidmcp.perception.factory import get_perception_backend
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.service")
ProgressFn = Callable[[float, str], None]


def import_source(
    project: ProjectStore,
    video_path: str | Path,
    *,
    bake_orientation: bool = True,
) -> str:
    """Import video; optionally bake displaymatrix rotation so frames are upright."""
    from vidmcp.media.orient import bake_orientation as bake_orient
    from vidmcp.media.orient import display_size
    from vidmcp.utils.video_io import probe_video

    settings = get_settings()
    dest = project.import_video(video_path, copy=settings.import_copy_into_workspace)
    meta = probe_video(dest)
    dw, dh = display_size(meta)
    project.manifest.source_meta = {
        **(project.manifest.source_meta or {}),
        "width": meta.width,
        "height": meta.height,
        "display_width": dw,
        "display_height": dh,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "duration_sec": meta.duration_sec,
        "codec": meta.codec,
        "has_audio": meta.has_audio,
        "rotation": meta.rotation,
        "rotation_original": meta.rotation,
        "oriented": False,
    }
    if bake_orientation and int(meta.rotation or 0) % 360 != 0:
        oriented = project.source_dir / "source_oriented.mp4"
        info = bake_orient(dest, oriented)
        project.manifest.source_video = project.rel(oriented)
        project.manifest.source_meta.update(
            {
                "width": info.get("output_width"),
                "height": info.get("output_height"),
                "display_width": info.get("output_width"),
                "display_height": info.get("output_height"),
                "oriented": True,
                "rotation": 0,
            }
        )
        project.manifest.append_history("bake_orientation", info)
        dest = oriented
    project.save()
    return project.rel(dest)


def analyze(project: ProjectStore) -> dict[str, Any]:
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    src = project.abs(project.manifest.source_video)
    preview_dir = project.previews_dir / "analysis"
    result = analyze_video(
        src,
        preview_dir=preview_dir,
        sample_fps=get_settings().frame_sample_fps,
    )
    # store relative thumbnails
    thumbs = []
    for t in result.get("thumbnail_paths", []):
        thumbs.append(project.rel(t))
    result["thumbnail_paths"] = thumbs
    project.manifest.source_meta = {
        k: result[k]
        for k in ("width", "height", "fps", "frame_count", "duration_sec", "codec", "has_audio", "bitrate")
        if k in result
    }
    project.manifest.analysis = result
    project.manifest.status = ProjectStatus.ANALYZED
    project.manifest.append_history("analyze_video", {"talking_head_score": result.get("talking_head_score")})
    project.save()
    return result


def segment(
    project: ProjectStore,
    *,
    prompt: str = "person",
    conf: float | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    settings = get_settings()
    conf = settings.conf_threshold if conf is None else conf
    backend = get_perception_backend(settings)
    src = project.abs(project.manifest.source_video)
    seg_id = str(uuid4())
    out_dir = project.masks_dir / seg_id
    result = backend.segment_video(
        src,
        prompt,
        output_dir=out_dir,
        conf=conf,
        progress=progress,
        feather=settings.default_mask_feather,
    )
    objects = [
        SegmentObject(
            object_id=o.object_id,
            label=o.label,
            confidence_mean=o.confidence_mean,
            frame_span=o.frame_span,
            area_ratio_mean=o.area_ratio_mean,
            is_primary=(o.object_id == 1),
        )
        for o in result.objects
    ]
    track = SegmentTrack(
        id=seg_id,
        prompt=prompt,
        backend=result.backend,
        mask_dir=project.rel(result.mask_dir),
        mask_video=project.rel(result.mask_video) if result.mask_video else None,
        objects=objects,
        conf_threshold=conf,
        frame_count=result.frame_count,
        fps=result.fps,
        width=result.width,
        height=result.height,
        meta={
            **result.meta,
            "temporal_stability": result.temporal_stability,
            "coverage_mean": result.coverage_mean,
        },
    )
    project.manifest.segments.append(track)
    project.manifest.primary_segment_id = track.id
    project.manifest.status = ProjectStatus.SEGMENTED
    project.manifest.append_history(
        "segment_subject",
        {"segment_id": track.id, "prompt": prompt, "backend": result.backend},
    )
    # ensure subject layer exists
    if not any(L.kind == LayerKind.SUBJECT for L in project.manifest.layers.layers):
        project.manifest.layers.add(
            Layer(
                name="subject",
                kind=LayerKind.SUBJECT,
                z_index=100,
                segment_track_id=track.id,
                mask_path=track.mask_dir,
            )
        )
    project.save()
    return {
        "segment_id": track.id,
        "prompt": prompt,
        "backend": result.backend,
        "mask_dir": track.mask_dir,
        "object_count": len(objects),
        "objects": [o.model_dump() for o in objects],
        "coverage_mean": result.coverage_mean,
        "temporal_stability": result.temporal_stability,
        "frame_count": result.frame_count,
    }


def apply_effects(
    project: ProjectStore,
    *,
    effect_specs: list[dict[str, Any]] | None = None,
    intent: str = "",
    replace_existing: bool = True,
) -> dict[str, Any]:
    from vidmcp.agents.planner import PlannerAgent

    if effect_specs is None:
        # derive from intent
        planner = PlannerAgent()
        plan = planner.plan(intent or "blur background behind person")
        effect_specs = planner.effects_from_tags(plan.style_tags, intent)

    if replace_existing:
        # keep subject/source; drop fx layers
        keep = [L for L in project.manifest.layers.layers if L.kind in (LayerKind.SUBJECT, LayerKind.SOURCE)]
        project.manifest.layers.layers = keep

    # base source layer if missing
    if not any(L.kind == LayerKind.SOURCE for L in project.manifest.layers.layers):
        project.manifest.layers.add(Layer(name="source", kind=LayerKind.SOURCE, z_index=0, opacity=0.0, enabled=False))

    if not any(L.kind == LayerKind.SUBJECT for L in project.manifest.layers.layers):
        project.manifest.layers.add(Layer(name="subject", kind=LayerKind.SUBJECT, z_index=100))

    layer_ids: list[str] = []
    types: list[str] = []
    z_bg = 10
    z_fx = 50
    for spec in effect_specs:
        kind_s = spec.get("kind", "background")
        kind = LayerKind.BACKGROUND if kind_s == "background" else (
            LayerKind.PARTICLES if kind_s == "particles" else (
                LayerKind.GRADE if kind_s == "grade" else LayerKind.OVERLAY
            )
        )
        blend = BlendMode.NORMAL
        if spec.get("blend_mode") == "screen":
            blend = BlendMode.SCREEN
        elif spec.get("blend_mode") == "add":
            blend = BlendMode.ADD
        z = z_bg if kind == LayerKind.BACKGROUND else z_fx
        z_bg += 1
        z_fx += 1
        effect = EffectParams(
            effect_type=spec["effect_type"],
            intensity=float(spec.get("intensity", 1.0)),
            params=dict(spec.get("params") or {}),
        )
        layer = Layer(
            name=spec.get("name") or spec["effect_type"],
            kind=kind,
            z_index=z,
            opacity=float(spec.get("opacity", 1.0)),
            blend_mode=blend,
            effect=effect,
            segment_track_id=project.manifest.primary_segment_id,
        )
        project.manifest.layers.add(layer)
        layer_ids.append(layer.id)
        types.append(effect.effect_type)

    project.manifest.status = ProjectStatus.EFFECTS_APPLIED
    project.manifest.append_history("apply_background_effects", {"layer_ids": layer_ids, "types": types})
    project.save()
    return {
        "layer_ids": layer_ids,
        "effect_types": types,
        "stack_version": project.manifest.layers.version,
        "layers": [L.model_dump(mode="json") for L in project.manifest.layers.sorted_layers()],
    }


def generate_broll(
    project: ProjectStore,
    *,
    style: str = "cyberpunk_city",
    prompt: str = "",
    duration_sec: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    src = project.abs(project.manifest.source_video)
    geom = match_source_geometry(src)
    dur = duration_sec or float(geom["duration_sec"] or 5.0)
    out = project.layers_dir / f"broll_{style}_{uuid4().hex[:8]}.mp4"
    path = generate_procedural_broll(
        out,
        width=int(geom["width"]),
        height=int(geom["height"]),
        fps=float(geom["fps"] or 30),
        duration_sec=dur,
        style=style,
        prompt=prompt,
        seed=seed,
    )
    layer = Layer(
        name=f"broll_{style}",
        kind=LayerKind.BROLL,
        z_index=5,
        asset_path=project.rel(path),
        opacity=1.0,
    )
    project.manifest.layers.add(layer)
    project.manifest.append_history("generate_broll", {"layer_id": layer.id, "style": style})
    project.save()
    return {
        "broll_path": project.rel(path),
        "layer_id": layer.id,
        "mode": "procedural",
        "duration_sec": dur,
        "style": style,
    }


def composite(
    project: ProjectStore,
    *,
    output_name: str | None = None,
    max_frames: int | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    from vidmcp.compositor.engine import CompositorEngine

    engine = CompositorEngine(project)
    return engine.render(output_name=output_name, max_frames=max_frames, progress=progress)


def review(project: ProjectStore) -> dict[str, Any]:
    from vidmcp.agents.critic import CriticAgent

    return CriticAgent().review(project)


def segment_multi(
    project: ProjectStore,
    *,
    prompts: list[str],
    conf: float | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Multi-concept segmentation via SAM 3.1 Object Multiplex (or mock multi-object)."""
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    settings = get_settings()
    conf = settings.conf_threshold if conf is None else conf
    backend = get_perception_backend(settings)
    src = project.abs(project.manifest.source_video)
    seg_id = str(uuid4())
    out_dir = project.masks_dir / seg_id
    primary = prompts[0] if prompts else "person"

    if hasattr(backend, "segment_multi"):
        result = backend.segment_multi(  # type: ignore[attr-defined]
            src,
            prompts,
            output_dir=out_dir,
            conf=conf,
            progress=progress,
            feather=settings.default_mask_feather,
            primary_prompt=primary,
            keep_per_object=True,
        )
    else:
        result = backend.segment_video(
            src,
            primary,
            output_dir=out_dir,
            conf=conf,
            progress=progress,
            feather=settings.default_mask_feather,
            prompts=prompts,
        )

    objects = [
        SegmentObject(
            object_id=o.object_id,
            label=o.label,
            confidence_mean=o.confidence_mean,
            frame_span=o.frame_span,
            area_ratio_mean=o.area_ratio_mean,
            is_primary=(o.object_id == 1),
        )
        for o in result.objects
    ]
    track = SegmentTrack(
        id=seg_id,
        prompt=", ".join(prompts),
        backend=result.backend,
        mask_dir=project.rel(result.mask_dir),
        objects=objects,
        conf_threshold=conf,
        frame_count=result.frame_count,
        fps=result.fps,
        width=result.width,
        height=result.height,
        meta={
            **result.meta,
            "temporal_stability": result.temporal_stability,
            "coverage_mean": result.coverage_mean,
            "multi_prompts": prompts,
        },
    )
    project.manifest.segments.append(track)
    project.manifest.primary_segment_id = track.id
    project.manifest.status = ProjectStatus.SEGMENTED
    project.manifest.append_history(
        "segment_multi_objects",
        {"segment_id": track.id, "prompts": prompts, "backend": result.backend, "n_objects": len(objects)},
    )
    if not any(L.kind == LayerKind.SUBJECT for L in project.manifest.layers.layers):
        project.manifest.layers.add(
            Layer(
                name="subject",
                kind=LayerKind.SUBJECT,
                z_index=100,
                segment_track_id=track.id,
                mask_path=track.mask_dir,
            )
        )
    project.save()
    return {
        "segment_id": track.id,
        "prompts": prompts,
        "backend": result.backend,
        "mask_dir": track.mask_dir,
        "object_count": len(objects),
        "objects": [o.model_dump() for o in objects],
        "coverage_mean": result.coverage_mean,
        "temporal_stability": result.temporal_stability,
        "frame_count": result.frame_count,
        "multiplex": bool(result.meta.get("multiplex") or result.meta.get("multiplex_sim")),
        "meta": result.meta,
    }


def matte_diagnostics(project: ProjectStore) -> dict[str, Any]:
    """Deep matte QA: per-frame coverage curve, flicker events, hole estimate."""
    import cv2
    import numpy as np

    from vidmcp.perception.mask_ops import to_u8_mask

    seg = project.manifest.primary_segment()
    if not seg:
        raise RuntimeError("No segment track")
    mask_dir = project.abs(seg.mask_dir)
    files = sorted(Path(mask_dir).glob("mask_*.png"))
    coverages = []
    flicker = []
    prev = None
    holes = []
    for i, f in enumerate(files):
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        u = to_u8_mask(m)
        cov = float((u > 127).mean())
        coverages.append(cov)
        # holes: background islands inside bbox of subject
        n, labels, stats, _ = cv2.connectedComponentsWithStats((u <= 127).astype(np.uint8), 8)
        # rough: small bg components not on border
        hole_area = 0
        h, w = u.shape
        for lab in range(1, n):
            x, y, bw, bh, area = stats[lab]
            if area < 0.02 * h * w and x > 2 and y > 2 and x + bw < w - 2 and y + bh < h - 2:
                # check if surrounded by fg — approximate by mean of ring
                hole_area += area
        holes.append(float(hole_area / (h * w)))
        if prev is not None:
            inter = np.logical_and(prev > 127, u > 127).sum()
            union = np.logical_or(prev > 127, u > 127).sum()
            iou = float(inter / union) if union else 1.0
            if iou < 0.5:
                flicker.append({"frame": i, "iou": iou})
        prev = u

    out = {
        "frame_count": len(coverages),
        "coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
        "coverage_std": float(np.std(coverages)) if coverages else 0.0,
        "coverage_min": float(np.min(coverages)) if coverages else 0.0,
        "coverage_max": float(np.max(coverages)) if coverages else 0.0,
        "coverage_curve": coverages[:: max(1, len(coverages) // 50)],
        "flicker_events": flicker[:50],
        "flicker_count": len(flicker),
        "hole_ratio_mean": float(np.mean(holes)) if holes else 0.0,
        "segment_id": seg.id,
        "prompt": seg.prompt,
        "backend": seg.backend,
    }
    project.manifest.append_history("matte_diagnostics", {"flicker_count": out["flicker_count"]})
    project.save()
    return out


def compare_renders(project: ProjectStore, render_a: str | None = None, render_b: str | None = None) -> dict[str, Any]:
    """Compare two renders (or last two) via frame-diff energy — for variant selection."""
    import cv2
    import numpy as np

    renders = project.manifest.renders
    if len(renders) < 1:
        raise RuntimeError("Need at least one render")
    if render_a is None:
        render_a = renders[-1]["output_path"]
    if render_b is None:
        if len(renders) < 2:
            raise RuntimeError("Need two renders to compare — run generate_edit_variants first")
        render_b = renders[-2]["output_path"]
    pa, pb = project.abs(render_a), project.abs(render_b)
    ca, cb = cv2.VideoCapture(str(pa)), cv2.VideoCapture(str(pb))
    diffs = []
    for _ in range(24):
        oa, fa = ca.read()
        ob, fb = cb.read()
        if not oa or not ob:
            break
        if fa.shape != fb.shape:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
        d = float(np.mean(cv2.absdiff(fa, fb)) / 255.0)
        diffs.append(d)
    ca.release()
    cb.release()
    return {
        "render_a": render_a,
        "render_b": render_b,
        "mean_diff": float(np.mean(diffs)) if diffs else 0.0,
        "max_diff": float(np.max(diffs)) if diffs else 0.0,
        "samples": len(diffs),
        "interpretation": "higher mean_diff ⇒ more visually distinct variants",
    }


def render_math_scene(
    project: ProjectStore,
    *,
    prompt: str | None = None,
    source: str | None = None,
    engine: str = "auto",
    duration_sec: float | None = None,
    place_as_background: bool = True,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Compile+render a math/education scene and optionally place as BG broll layer."""
    from vidmcp.scenes.engine import SceneEngine

    scenes_root = project.root / "scenes"
    engine_impl = SceneEngine(scenes_root)
    # match source geometry when available
    w, h = width or 1280, height or 720
    fps = 24.0
    if project.manifest.source_video:
        try:
            from vidmcp.utils.video_io import probe_video

            meta = probe_video(project.abs(project.manifest.source_video))
            w, h = width or meta.width, height or meta.height
            fps = meta.fps or 24.0
            if duration_sec is None:
                duration_sec = min(float(meta.duration_sec or 5.0), 30.0)
        except Exception:  # noqa: BLE001
            pass

    result = engine_impl.compile_and_render(
        prompt=prompt,
        source=source,
        engine=engine if engine in ("auto", "manim", "procedural") else "auto",  # type: ignore[arg-type]
        width=w,
        height=h,
        fps=fps,
        duration_sec=duration_sec,
    )
    # copy into layers
    dest = project.layers_dir / f"scene_{result.scene_id[:8]}.mp4"
    import shutil

    shutil.copy2(result.output_path, dest)
    layer_id = None
    if place_as_background:
        layer = Layer(
            name=f"scene_{result.engine}",
            kind=LayerKind.BROLL,
            z_index=4,
            asset_path=project.rel(dest),
            opacity=1.0,
            meta={"scene_id": result.scene_id, "engine": result.engine, "prompt": prompt or ""},
        )
        project.manifest.layers.add(layer)
        layer_id = layer.id
        # ensure subject on top
        if not any(L.kind == LayerKind.SUBJECT for L in project.manifest.layers.layers):
            project.manifest.layers.add(Layer(name="subject", kind=LayerKind.SUBJECT, z_index=100))
    project.manifest.append_history(
        "render_math_scene",
        {"scene_id": result.scene_id, "engine": result.engine, "layer_id": layer_id},
    )
    project.save()
    return {
        "scene_id": result.scene_id,
        "engine": result.engine,
        "scene_path": project.rel(dest),
        "absolute_path": str(dest),
        "source_path": project.rel(result.source_path) if result.source_path else None,
        "layer_id": layer_id,
        "placed_as_background": place_as_background,
        "prompt": prompt or "",
        "meta": result.meta,
    }


def place_scene_background(project: ProjectStore, scene_path: str, name: str = "math_scene_bg") -> dict[str, Any]:
    path = project.abs(scene_path) if not Path(scene_path).is_absolute() else Path(scene_path)
    if not path.exists():
        # try relative to project
        alt = project.root / scene_path
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(scene_path)
    # store under layers if external
    if project.root not in path.resolve().parents and path.resolve() != project.root:
        dest = project.layers_dir / path.name
        import shutil

        shutil.copy2(path, dest)
        path = dest
    layer = Layer(
        name=name,
        kind=LayerKind.BROLL,
        z_index=4,
        asset_path=project.rel(path),
        opacity=1.0,
    )
    project.manifest.layers.add(layer)
    project.manifest.append_history("place_scene_as_background", {"layer_id": layer.id, "path": project.rel(path)})
    project.save()
    return {"layer_id": layer.id, "asset_path": project.rel(path)}


def refine_segment_keyframes(
    project: ProjectStore,
    *,
    keyframes: list[int] | None = None,
    hints: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
    auto_detect: bool = True,
    conf: float | None = None,
    prefer_sam: bool = True,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Surgical matte refine at weak/key frames."""
    from vidmcp.perception.keyframe_refine import (
        KeyframeHint,
        detect_weak_keyframes,
        refine_masks_local,
        refine_with_sam_backend,
    )

    seg = project.manifest.primary_segment()
    if not seg:
        raise RuntimeError("No segment to refine — run segment_subject first")
    if not project.manifest.source_video:
        raise RuntimeError("No source video")
    settings = get_settings()
    conf = settings.conf_threshold if conf is None else conf
    src = project.abs(project.manifest.source_video)
    mask_dir = project.abs(seg.mask_dir)
    new_id = str(uuid4())
    out_dir = project.masks_dir / f"{new_id}_refined"

    hint_objs = [
        KeyframeHint(
            frame_index=int(h["frame_index"]),
            prompt=h.get("prompt"),
            box_xyxy=h.get("box_xyxy"),
            points=h.get("points"),
            point_labels=h.get("point_labels"),
            reason=h.get("reason") or "",
        )
        for h in (hints or [])
        if "frame_index" in h
    ]
    if keyframes is None and auto_detect and not hint_objs:
        keyframes = detect_weak_keyframes(mask_dir)
    elif keyframes is None and hint_objs:
        keyframes = [h.frame_index for h in hint_objs]

    use_prompt = prompt or seg.prompt.split(",")[0].strip() or "person"
    backend = get_perception_backend(settings)
    used_sam = False
    if prefer_sam and backend.name not in ("mock",) and hasattr(backend, "segment_multi"):
        try:
            if not hint_objs:
                hint_objs = [KeyframeHint(frame_index=k, prompt=use_prompt) for k in (keyframes or [0])]
            result = refine_with_sam_backend(
                backend,
                src,
                output_dir=out_dir,
                prompt=use_prompt,
                hints=hint_objs,
                conf=conf,
                feather=settings.default_mask_feather,
                progress=progress,
                previous_mask_dir=mask_dir,
            )
            used_sam = True
        except Exception as e:  # noqa: BLE001
            log.warning("sam_refine_failed_local_fallback", error=str(e))
            result = refine_masks_local(
                src,
                mask_dir,
                output_dir=out_dir,
                keyframes=keyframes,
                hints=hint_objs,
                prompt=use_prompt,
                feather=settings.default_mask_feather,
                progress=progress,
            )
    else:
        result = refine_masks_local(
            src,
            mask_dir,
            output_dir=out_dir,
            keyframes=keyframes,
            hints=hint_objs,
            prompt=use_prompt,
            feather=settings.default_mask_feather,
            progress=progress,
        )

    # register new segment track, keep old history
    objects = [
        SegmentObject(
            object_id=1,
            label=use_prompt,
            confidence_mean=conf,
            frame_span=(0, max(result.meta.get("n_masks", 1) - 1, 0)),
            area_ratio_mean=result.coverage_after,
            is_primary=True,
        )
    ]
    track = SegmentTrack(
        id=new_id,
        prompt=use_prompt,
        backend=result.backend,
        mask_dir=project.rel(result.mask_dir),
        objects=objects,
        conf_threshold=conf,
        frame_count=int(result.meta.get("n_masks") or seg.frame_count),
        fps=seg.fps,
        width=seg.width,
        height=seg.height,
        meta={
            "refined_from": seg.id,
            "keyframes": result.keyframes,
            "stability_delta": result.temporal_stability_after - result.temporal_stability_before,
            "used_sam": used_sam,
            **result.meta,
        },
    )
    project.manifest.segments.append(track)
    project.manifest.primary_segment_id = track.id
    # update subject layer mask path
    for L in project.manifest.layers.layers:
        if L.kind == LayerKind.SUBJECT:
            L.mask_path = track.mask_dir
            L.segment_track_id = track.id
    project.manifest.append_history(
        "refine_segment_keyframes",
        {
            "segment_id": track.id,
            "from": seg.id,
            "keyframes": result.keyframes,
            "stability_before": result.temporal_stability_before,
            "stability_after": result.temporal_stability_after,
        },
    )
    project.save()
    d = result.to_dict()
    d.update(
        {
            "segment_id": track.id,
            "previous_segment_id": seg.id,
            "mask_dir": track.mask_dir,
            "used_sam": used_sam,
            "improvement": {
                "temporal_stability": result.temporal_stability_after - result.temporal_stability_before,
                "coverage": result.coverage_after - result.coverage_before,
            },
        }
    )
    return d
