"""End-to-end education product path: talking-head + speech-locked lesson plate."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from vidmcp.audio.course import compile_lesson
from vidmcp.audio.media import ensure_video_with_narration
from vidmcp.config import get_settings
from vidmcp.core.workspace import Workspace
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.education")
ProgressFn = Callable[[float, str], None]


def run_education_lesson(
    *,
    video_path: str | Path,
    lesson_topic: str,
    project_name: str = "education_lesson",
    narration: str | None = None,
    max_render_frames: int | None = 90,
    conf: float = 0.25,
    n_steps: int = 6,
    style: str = "cinematic",
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    from vidmcp.tools import advanced_service as adv
    from vidmcp.tools import service

    def report(p: float, msg: str) -> None:
        if progress:
            progress(p, msg)
        log.info("education_progress", p=p, msg=msg)

    settings = get_settings()
    ws = Workspace(settings)
    video_path = Path(video_path)

    lesson = compile_lesson(
        lesson_topic,
        duration_sec=max(30.0, float(max_render_frames or 90) / 12.0 * 4),
        n_beats=n_steps,
    )
    narr = narration or " ".join(b["narration_cue"] for b in lesson["beats"][:n_steps])

    report(0.05, "ensure narration audio")
    media_dir = settings.workspace_root / "_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    media = ensure_video_with_narration(
        video_path,
        narration=narr,
        out_path=media_dir / f"{project_name}_narrated.mp4",
        force=False,
    )
    src = Path(media["path"])

    report(0.1, "create project")
    project = ws.create_project(name=project_name)
    service.import_source(project, src)
    service.analyze(project)
    adv.graph_commit(project, "education_start", {"topic": lesson_topic})

    report(0.2, "segment speaker")
    seg = service.segment(project, prompt="person", conf=conf)

    report(0.35, "transcribe words")
    words = adv.word_timeline(project, fallback_transcript=narr, model_size="base")

    report(0.45, "speech-locked scene")
    keywords = lesson.get("keywords") or ["first", "therefore", "prove", "equals", "finally"]
    scene = adv.speech_locked_scene(
        project,
        lesson_topic,
        n_steps=n_steps,
        keywords=[str(k) for k in keywords][:12],
        fallback_transcript=narr,
        place_as_background=True,
    )

    report(0.55, "style layers")
    if style == "cyberpunk":
        specs = [
            {"effect_type": "cyberpunk", "kind": "background", "params": {"blur_radius": 15}, "name": "cyber"},
            {
                "effect_type": "particles",
                "kind": "particles",
                "params": {"style": "neon_dust", "density": 0.3},
                "blend_mode": "screen",
                "name": "dust",
            },
        ]
    elif style == "clean":
        specs = [
            {
                "effect_type": "color_grade",
                "kind": "grade",
                "params": {"contrast": 1.05, "saturation": 1.0, "background_only": True},
                "name": "clean",
            }
        ]
    else:
        specs = [
            {"effect_type": "blur", "kind": "background", "params": {"blur_radius": 21}, "name": "bokeh"},
            {
                "effect_type": "color_grade",
                "kind": "grade",
                "params": {
                    "contrast": 1.1,
                    "saturation": 1.05,
                    "temperature": 0.05,
                    "background_only": True,
                },
                "name": "grade",
            },
        ]
    fx = service.apply_effects(project, effect_specs=specs, intent=lesson_topic, replace_existing=False)

    report(0.65, "refine if unstable")
    refine: dict[str, Any] = {"skipped": True}
    if float(seg.get("temporal_stability") or 1) < 0.7:
        refine = adv.uncertainty_guided_refine(project, service)

    report(0.75, "lighting + composite")
    light = adv.lighting_match_project(project, max_frames=max_render_frames)
    render = service.composite(project, max_frames=max_render_frames)
    adv.graph_commit(project, "education_composite", {"path": render.get("output_path")})

    report(0.88, "critic + heuristics")
    auto = adv.apply_auto_heuristics(project, service, max_frames=max_render_frames)
    critics = auto.get("critics_after") or adv.critic_project(project, workspace_root=settings.workspace_root)

    report(0.95, "sign + timeline")
    signed = adv.sign(project)
    timeline = adv.export_otio(project)
    report(1.0, "education lesson complete")

    return {
        "ok": True,
        "project_id": project.manifest.id,
        "lesson": lesson,
        "media": media,
        "segment": {
            "segment_id": seg.get("segment_id"),
            "backend": seg.get("backend"),
            "temporal_stability": seg.get("temporal_stability"),
        },
        "asr": {
            "backend": words.get("backend"),
            "n_words": len(words.get("words") or []),
            "text": (words.get("text") or "")[:300],
        },
        "scene": {
            "path": scene.get("scene_path"),
            "steps": len(scene.get("steps") or []),
            "keyword_events": len(scene.get("keyword_events") or []),
        },
        "effects": fx.get("effect_types"),
        "refine": refine,
        "lighting": light.get("project_relative"),
        "render": render,
        "critics": critics,
        "provenance": signed.get("manifest_path"),
        "timeline": timeline.get("json_path"),
        "message": f"Education lesson ready: {lesson_topic}",
    }
