"""Creator pipeline service — process_audio, bgm, captions, export, polish recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from vidmcp.audio.bgm import mix_bgm as mix_bgm_files
from vidmcp.audio.media import has_audio_stream
from vidmcp.audio.process import process_audio as process_audio_file
from vidmcp.audio.whisper_timeline import transcribe_words
from vidmcp.captions.burn import burn_captions, words_to_cues, write_ass
from vidmcp.compositor.ffmpeg_ops import mux_audio_replace
from vidmcp.core.workspace import ProjectStore, Workspace
from vidmcp.edit.edl import export_edl
from vidmcp.edit.infographics import burn_infographics, derive_beats_from_transcript
from vidmcp.edit.smart_cut import apply_smart_cuts, plan_smart_cuts
from vidmcp.matte.replace_bg import replace_background_video
from vidmcp.media.export import export_render as export_render_file
from vidmcp.media.orient import bake_orientation, display_size
from vidmcp.models.project import ProjectStatus
from vidmcp.tools.service import analyze, import_source
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video

log = get_logger("vidmcp.creator")


def _audio_dir(project: ProjectStore) -> Path:
    d = project.root / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pipeline(project: ProjectStore) -> dict[str, Any]:
    pipe = project.manifest.source_meta.get("audio_pipeline")
    if isinstance(pipe, dict):
        return pipe
    pipe = {}
    project.manifest.source_meta["audio_pipeline"] = pipe
    return pipe


def process_audio_project(
    project: ProjectStore,
    *,
    strength: float = 0.7,
    target_lufs: float = -14.0,
) -> dict[str, Any]:
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    src = project.abs(project.manifest.source_video)
    out = _audio_dir(project) / "vocals_clean.wav"
    result = process_audio_file(src, out, strength=strength, target_lufs=target_lufs)
    pipe = _pipeline(project)
    pipe["vocals"] = project.rel(out)
    pipe["lufs_out"] = result.get("lufs_out")
    project.manifest.append_history("process_audio", {"path": pipe["vocals"]})
    project.save()
    result["project_id"] = project.manifest.id
    return result


def mix_bgm_project(
    project: ProjectStore,
    *,
    bgm_path: str | None = None,
    volume: float = 0.35,
    style: str = "cinematic",
    duck: bool = True,
) -> dict[str, Any]:
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    pipe = _pipeline(project)
    vocals = pipe.get("vocals")
    if vocals:
        vox = project.abs(vocals)
    else:
        # extract first
        from vidmcp.audio.media import extract_wav

        vox = _audio_dir(project) / "vocals_raw.wav"
        src = project.abs(project.manifest.source_video)
        if has_audio_stream(src):
            import subprocess

            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "48000", str(vox)],
                check=True,
                capture_output=True,
            )
        else:
            raise RuntimeError("No audio to mix")
    out = _audio_dir(project) / "final_mix.wav"
    result = mix_bgm_files(
        vox,
        out,
        bgm_wav=Path(bgm_path) if bgm_path else None,
        bgm_volume=volume,
        style=style,
        duck=duck,
    )
    pipe["mix"] = project.rel(out)
    pipe["bgm"] = project.rel(result.get("bgm_path", "")) if result.get("bgm_path") else None
    project.manifest.append_history("mix_bgm", {"path": pipe["mix"], "volume": volume})
    project.save()
    result["project_id"] = project.manifest.id
    return result


def transcribe_and_caption_project(
    project: ProjectStore,
    *,
    burn: bool = True,
    style: str = "brand",
    language: str | None = None,
    model_size: str = "base",
    fallback_transcript: str | None = None,
) -> dict[str, Any]:
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    src = project.abs(project.manifest.source_video)
    # prefer processed video if last render? use source for timing
    work = project.tmp_dir / "caption"
    work.mkdir(parents=True, exist_ok=True)
    tr = transcribe_words(
        src,
        work_dir=work,
        model_size=model_size,
        language=language,
        fallback_transcript=fallback_transcript,
        auto_narrate_if_silent=False,
    )
    words = tr.get("words") or []
    cues = words_to_cues(words)
    cap_dir = project.root / "captions"
    cap_dir.mkdir(exist_ok=True)
    words_path = project.root / "transcripts" / "words.json"
    words_path.parent.mkdir(exist_ok=True)
    words_path.write_text(json.dumps(tr, indent=2), encoding="utf-8")
    ass_path = cap_dir / "captions.ass"
    meta = probe_video(src)
    write_ass(cues, ass_path, style=style, play_res_x=meta.width or 1920, play_res_y=meta.height or 1080)

    burned = None
    if burn and cues:
        # video base: last visual render or source
        base = src
        if project.manifest.renders:
            cand = project.abs(project.manifest.renders[-1].get("path", ""))
            if cand.exists():
                base = cand
        out = project.renders_dir / f"captioned_{uuid4().hex[:8]}.mp4"
        br = burn_captions(base, out, cues=cues, ass_path=ass_path, style=style)
        # attach best audio
        pipe = _pipeline(project)
        mix = pipe.get("mix") or pipe.get("vocals")
        if mix:
            muxed = project.renders_dir / f"captioned_mix_{uuid4().hex[:8]}.mp4"
            mux_audio_replace(out, project.abs(mix), muxed)
            burned = str(muxed)
            project.manifest.renders.append({"path": project.rel(muxed), "kind": "captioned"})
        else:
            burned = br["path"]
            project.manifest.renders.append({"path": project.rel(out), "kind": "captioned"})

    project.manifest.append_history(
        "transcribe_and_caption",
        {"backend": tr.get("backend"), "n_words": len(words), "burned": burned},
    )
    project.save()
    return {
        "ok": True,
        "project_id": project.manifest.id,
        "backend": tr.get("backend"),
        "text": tr.get("text"),
        "words_path": project.rel(words_path),
        "ass_path": project.rel(ass_path),
        "n_cues": len(cues),
        "burned_path": burned,
        "warnings": [] if words else ["No words — install faster-whisper for ASR"],
    }


def replace_background_project(
    project: ProjectStore,
    *,
    plate: str = "space",
    matte_backend: str = "auto",
    plate_image: str | None = None,
) -> dict[str, Any]:
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    src = project.abs(project.manifest.source_video)
    out = project.renders_dir / f"bg_{plate}_{uuid4().hex[:8]}.mp4"
    mask_dir = project.masks_dir / f"fast_{uuid4().hex[:8]}"
    result = replace_background_video(
        src,
        out,
        plate=plate,
        plate_image=plate_image,
        matte_backend=matte_backend,
        mask_dir=mask_dir,
    )
    # mux audio
    pipe = _pipeline(project)
    audio = pipe.get("mix") or pipe.get("vocals")
    final = out
    if audio:
        muxed = project.renders_dir / f"bg_mix_{uuid4().hex[:8]}.mp4"
        mux_audio_replace(out, project.abs(audio), muxed)
        final = muxed
    elif has_audio_stream(src):
        muxed = project.renders_dir / f"bg_srcaudio_{uuid4().hex[:8]}.mp4"
        mux_audio_replace(out, src, muxed)
        final = muxed

    project.manifest.renders.append(
        {
            "path": project.rel(final),
            "kind": "replace_background",
            "plate": plate,
            "coverage_mean": result.get("coverage_mean"),
        }
    )
    project.manifest.append_history("replace_background", {"path": project.rel(final), "plate": plate})
    project.save()
    result["path"] = str(final)
    result["project_id"] = project.manifest.id
    return result


def export_render_project(
    project: ProjectStore,
    *,
    render_path: str | None = None,
    preset: str = "youtube_16x9",
    loudnorm: bool = True,
) -> dict[str, Any]:
    if render_path:
        video = project.abs(render_path)
    elif project.manifest.renders:
        video = project.abs(project.manifest.renders[-1]["path"])
    elif project.manifest.source_video:
        video = project.abs(project.manifest.source_video)
    else:
        raise RuntimeError("No render or source")
    pipe = _pipeline(project)
    audio = None
    if pipe.get("mix"):
        audio = project.abs(pipe["mix"])
    elif pipe.get("vocals"):
        audio = project.abs(pipe["vocals"])
    out = project.renders_dir / f"export_{preset}_{uuid4().hex[:8]}.mp4"
    result = export_render_file(video, out, preset=preset, audio=audio, loudnorm=loudnorm)
    project.manifest.renders.append({"path": project.rel(out), "kind": "export", "preset": preset})
    project.manifest.append_history("export_render", {"path": project.rel(out), "preset": preset})
    project.save()
    result["project_id"] = project.manifest.id
    return result


def smart_cut_project(project: ProjectStore, *, aggressiveness: float = 0.5) -> dict[str, Any]:
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    src = project.abs(project.manifest.source_video)
    meta = probe_video(src)
    words_path = project.root / "transcripts" / "words.json"
    words: list[dict] = []
    if words_path.exists():
        data = json.loads(words_path.read_text())
        words = data.get("words") or []
    else:
        tr = transcribe_words(src, work_dir=project.tmp_dir / "cut_asr", auto_narrate_if_silent=False)
        words = tr.get("words") or []
        words_path.parent.mkdir(exist_ok=True)
        words_path.write_text(json.dumps(tr, indent=2), encoding="utf-8")

    ranges = plan_smart_cuts(words, duration_sec=meta.duration_sec, aggressiveness=aggressiveness)
    out = project.renders_dir / f"smartcut_{uuid4().hex[:8]}.mp4"
    pipe = _pipeline(project)
    audio = project.abs(pipe["mix"]) if pipe.get("mix") else None
    out_audio = _audio_dir(project) / "smartcut_mix.wav" if audio else None
    result = apply_smart_cuts(src, out, ranges, audio=audio, out_audio=out_audio)
    if out_audio and Path(out_audio).exists():
        muxed = project.renders_dir / f"smartcut_mix_{uuid4().hex[:8]}.mp4"
        mux_audio_replace(out, out_audio, muxed)
        result["path"] = str(muxed)
        pipe["mix"] = project.rel(out_audio)
        project.manifest.renders.append({"path": project.rel(muxed), "kind": "smart_cut"})
    else:
        project.manifest.renders.append({"path": project.rel(out), "kind": "smart_cut"})
    project.manifest.append_history("smart_cut_hesitations", {"removed_sec": result.get("removed_sec")})
    project.save()
    result["project_id"] = project.manifest.id
    return result


def add_infographics_project(project: ProjectStore, *, topic: str = "auto") -> dict[str, Any]:
    if not project.manifest.source_video:
        raise RuntimeError("Import a video first")
    # base visual
    if project.manifest.renders:
        base = project.abs(project.manifest.renders[-1]["path"])
    else:
        base = project.abs(project.manifest.source_video)
    words_path = project.root / "transcripts" / "words.json"
    text = ""
    words = []
    if words_path.exists():
        data = json.loads(words_path.read_text())
        text = data.get("text") or ""
        words = data.get("words") or []
    beats = derive_beats_from_transcript(text, words)
    out = project.renders_dir / f"info_{uuid4().hex[:8]}.mp4"
    result = burn_infographics(base, out, beats)
    pipe = _pipeline(project)
    if pipe.get("mix"):
        muxed = project.renders_dir / f"info_mix_{uuid4().hex[:8]}.mp4"
        mux_audio_replace(out, project.abs(pipe["mix"]), muxed)
        result["path"] = str(muxed)
        project.manifest.renders.append({"path": project.rel(muxed), "kind": "infographics"})
    else:
        project.manifest.renders.append({"path": project.rel(out), "kind": "infographics"})
    project.manifest.append_history("add_speech_infographics", {"n_beats": len(beats)})
    project.save()
    result["project_id"] = project.manifest.id
    result["beats"] = beats
    return result


def generate_thumbnail_project(project: ProjectStore, title: str | None = None) -> dict[str, Any]:
    import cv2

    if project.manifest.renders:
        video = project.abs(project.manifest.renders[-1]["path"])
    elif project.manifest.source_video:
        video = project.abs(project.manifest.source_video)
    else:
        raise RuntimeError("No video")
    meta = probe_video(video)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(meta.frame_count * 0.35)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not grab frame")
    # title from transcript
    if not title:
        wp = project.root / "transcripts" / "words.json"
        if wp.exists():
            title = (json.loads(wp.read_text()).get("text") or "VidMCP")[:48]
        else:
            title = project.manifest.name or "VidMCP"
    from PIL import Image, ImageDraw, ImageFont
    from vidmcp.captions.fonts import resolve_font

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_path = resolve_font()
    font = ImageFont.truetype(str(font_path), 48) if font_path else ImageFont.load_default()
    draw.rectangle((0, img.height - 120, img.width, img.height), fill=(5, 5, 7, 200))
    draw.text((40, img.height - 90), title, fill=(239, 239, 235), font=font)
    out = project.previews_dir / f"thumb_{uuid4().hex[:8]}.jpg"
    out.parent.mkdir(exist_ok=True)
    img.save(out, quality=92)
    return {"ok": True, "path": str(out), "title": title, "project_id": project.manifest.id}


def run_talking_head_polish(
    video_path: str | Path,
    *,
    workspace: Workspace | None = None,
    name: str = "talking_head_polish",
    preset: str = "youtube_16x9",
    bg_mode: str = "none",
    strength: float = 0.7,
    bgm_volume: float = 0.35,
    burn_captions_flag: bool = True,
    smart_cut: bool = False,
    aggressiveness: float = 0.45,
) -> dict[str, Any]:
    """One-shot creator pipeline."""
    from vidmcp.config import get_settings
    from vidmcp.core.workspace import Workspace as WS

    ws = workspace or WS(get_settings())
    project = ws.create_project(name)
    import_source(project, video_path, bake_orientation=True)
    analyze(project)
    warnings: list[str] = []
    steps: dict[str, Any] = {}

    try:
        steps["process_audio"] = process_audio_project(project, strength=strength)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"process_audio: {e}")
    try:
        steps["mix_bgm"] = mix_bgm_project(project, volume=bgm_volume)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"mix_bgm: {e}")

    if smart_cut:
        try:
            steps["smart_cut"] = smart_cut_project(project, aggressiveness=aggressiveness)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"smart_cut: {e}")

    if bg_mode and bg_mode != "none":
        try:
            steps["replace_background"] = replace_background_project(project, plate=bg_mode)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"replace_background: {e}")

    try:
        steps["caption"] = transcribe_and_caption_project(project, burn=burn_captions_flag)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"caption: {e}")

    try:
        steps["export"] = export_render_project(project, preset=preset, loudnorm=True)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"export: {e}")

    project.manifest.status = ProjectStatus.RENDERED
    project.save()
    final = None
    if project.manifest.renders:
        final = project.abs(project.manifest.renders[-1]["path"])
    return {
        "ok": True,
        "project_id": project.manifest.id,
        "final_path": str(final) if final else None,
        "steps": {k: {kk: vv for kk, vv in v.items() if kk != "raw"} if isinstance(v, dict) else v for k, v in steps.items()},
        "warnings": warnings,
    }
