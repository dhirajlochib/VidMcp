"""Register VidMCP 2.0 tools — matte, understanding, color, camera, audio, graphics,
recipes v2, cognition, delivery, infra. See UPGRADE_ROADMAP.md."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def register_v2_tools(
    mcp: Any,
    *,
    load_project: Callable[[str], Any],
    workspace_factory: Callable[[], Any],
    log: Any,
) -> None:
    from vidmcp.utils.compact import compact_result as cc

    def _safe(fn: Callable[..., dict], *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return cc(fn(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            log.exception("v2_tool_failed", tool=fn.__name__)
            return {"ok": False, "message": str(e)}

    # --- W10 infra ---

    @mcp.tool()
    def ensure_model(name: str, download: bool = False) -> dict[str, Any]:
        """Resolve/download an optional model (matting/ASR/stems/depth/...). See list_models."""
        from vidmcp.models_registry import ensure_model as _em

        return _safe(_em, name, download)

    @mcp.tool()
    def list_models() -> dict[str, Any]:
        """Registry of optional quality-upgrade models with install status + hints."""
        from vidmcp.models_registry import list_models as _lm

        return _safe(_lm)

    @mcp.tool()
    def run_ops(project_id: str, ops: list[dict], stop_on_error: bool = False) -> dict[str, Any]:
        """Batch ops in ONE call with conditions + '$id.field' templating. Spec: {tool, args?, id?, if?}."""
        from vidmcp.harness.ops import run_ops as _ro

        return _safe(_ro, load_project(project_id), ops, stop_on_error=stop_on_error)

    @mcp.tool()
    def list_ops() -> dict[str, Any]:
        """Ops available to run_ops / recipes v2 / edit plans."""
        from vidmcp.harness.ops import list_ops as _lo

        return {"ok": True, "ops": _lo()}

    # --- W1 matte ---

    @mcp.tool()
    def refine_alpha(
        project_id: str, segment_id: str | None = None, backend: str = "auto",
        band_px: int = 16, max_frames: int | None = None,
    ) -> dict[str, Any]:
        """Hair-level alpha from binary masks (RVM onnx or guided filter). Compositor auto-uses it."""
        from vidmcp.matte.alpha_refine import refine_alpha_project

        return _safe(refine_alpha_project, load_project(project_id), segment_id, backend, band_px, max_frames)

    @mcp.tool()
    def stabilize_matte(project_id: str, strength: float = 0.6, max_frames: int | None = None) -> dict[str, Any]:
        """Flow-guided temporal alpha stabilization; reports dtSSD flicker before/after."""
        from vidmcp.matte.temporal import stabilize_matte_project

        return _safe(stabilize_matte_project, load_project(project_id), strength, max_frames)

    @mcp.tool()
    def set_composite_realism(
        project_id: str, enabled: bool = True,
        light_wrap: bool = True, decontaminate: bool = True,
        contact_shadow: bool = True, grain_match: bool = True,
    ) -> dict[str, Any]:
        """Enable realism composite (light wrap/spill/shadow/grain) for the next render."""
        project = load_project(project_id)
        m = project.manifest
        if enabled:
            m.source_meta["composite_realism"] = {
                "light_wrap": light_wrap, "decontaminate": decontaminate,
                "contact_shadow": contact_shadow, "grain_match": grain_match,
            }
        else:
            m.source_meta.pop("composite_realism", None)
        project.save()
        return {"ok": True, "realism": m.source_meta.get("composite_realism")}

    # --- W4 understanding ---

    @mcp.tool()
    def detect_scenes(project_id: str, backend: str = "auto", threshold: float = 0.45) -> dict[str, Any]:
        """Shot boundaries + scene clustering (PySceneDetect or histogram)."""
        from vidmcp.perception.scene_seg import detect_scenes_project

        return _safe(detect_scenes_project, load_project(project_id), backend, threshold)

    @mcp.tool()
    def build_footage_index(
        project_id: str, include: list[str] | None = None,
        model_size: str = "base", fallback_transcript: str | None = None,
    ) -> dict[str, Any]:
        """ONE analysis pass: visual stats+faces, audio events, words/sentences, energy/emotion curve."""
        from vidmcp.perception.indexer import build_index_project

        return _safe(build_index_project, load_project(project_id), include, model_size, fallback_transcript)

    @mcp.tool()
    def search_footage(project_id: str, query: str, top_k: int = 5) -> dict[str, Any]:
        """Semantic search this project's footage ('find the part where I laugh')."""
        from vidmcp.perception.search import search_footage_project

        return _safe(search_footage_project, load_project(project_id), query, top_k)

    @mcp.tool()
    def search_library(query: str, top_k: int = 8) -> dict[str, Any]:
        """Search ALL indexed projects (B-roll retrieval across your library)."""
        from vidmcp.perception.search import search_library as _sl

        return _safe(_sl, workspace_factory(), query, top_k)

    @mcp.tool()
    def plan_cuts(
        project_id: str, aggressiveness: float = 0.5, keep_dramatic_pauses: bool = True,
    ) -> dict[str, Any]:
        """Cut plan v2: retakes, contextual fillers, dead-vs-dramatic pauses. Review before apply."""
        from vidmcp.edit.cut_planner import plan_cuts_project

        return _safe(plan_cuts_project, load_project(project_id),
                     aggressiveness=aggressiveness, keep_dramatic_pauses=keep_dramatic_pauses)

    @mcp.tool()
    def apply_cut_plan(project_id: str, plan_id: str, room_tone: bool = True) -> dict[str, Any]:
        """Execute a cut plan with room-tone fill (no dead digital silence at cuts)."""
        from vidmcp.edit.cut_planner import apply_cut_plan_project

        return _safe(apply_cut_plan_project, load_project(project_id), plan_id, room_tone)

    @mcp.tool()
    def suggest_broll(project_id: str, top_k: int = 3) -> dict[str, Any]:
        """Match transcript beats to library clips (or plate prompts) for B-roll inserts."""
        from vidmcp.edit.broll_match import suggest_broll_project

        return _safe(suggest_broll_project, load_project(project_id), top_k=top_k)

    @mcp.tool()
    def insert_broll(project_id: str, suggestions: list[dict], transition: str = "cut") -> dict[str, Any]:
        """Place accepted B-roll suggestions as time-windowed layers."""
        from vidmcp.edit.broll_match import insert_broll_project

        return _safe(insert_broll_project, load_project(project_id), suggestions, transition)

    # --- W2 color ---

    @mcp.tool()
    def apply_lut(project_id: str, lut: str = "filmic_soft", intensity: float = 1.0,
                  background_only: bool = False) -> dict[str, Any]:
        """Apply a .cube LUT or builtin look (teal_orange, noir, filmic_soft, ...) as a grade layer."""
        from vidmcp.color.lut import apply_lut_project

        return _safe(apply_lut_project, load_project(project_id), lut, intensity, background_only)

    @mcp.tool()
    def list_luts() -> dict[str, Any]:
        """Builtin looks + workspace/luts/*.cube."""
        from vidmcp.color.lut import list_luts as _ll
        from vidmcp.config import get_settings

        return _safe(_ll, get_settings().workspace_root)

    @mcp.tool()
    def auto_color(project_id: str, wb: bool = True, exposure: bool = True, per_shot: bool = True) -> dict[str, Any]:
        """Auto white balance + exposure per shot, skin-protected, subject-anchored."""
        from vidmcp.color.auto_correct import auto_color_project

        return _safe(auto_color_project, load_project(project_id), wb, exposure, per_shot)

    @mcp.tool()
    def match_color(project_id: str, reference: str = "shot:0", strength: float = 0.8) -> dict[str, Any]:
        """Match all shots to a hero shot or reference image/video ('grade like this')."""
        from vidmcp.color.match import match_color_project

        return _safe(match_color_project, load_project(project_id), reference, strength)

    @mcp.tool()
    def color_scopes(project_id: str, frame: str = "auto") -> dict[str, Any]:
        """Waveform + vectorscope PNGs and clip/cast/skin stats for QC."""
        from vidmcp.color.scopes import scopes_project

        return _safe(scopes_project, load_project(project_id), frame)

    @mcp.tool()
    def apply_style(
        project_id: str, reference: str | None = None, lut: str | None = None,
        grain: float = 1.2, intensity: float = 0.85,
    ) -> dict[str, Any]:
        """Look emulation layer: reference grade fingerprint + grain + vignette + halation."""
        from vidmcp.effects.style import apply_style_project

        return _safe(apply_style_project, load_project(project_id),
                     reference=reference, lut=lut, grain=grain, intensity=intensity)

    # --- W3 camera ---

    @mcp.tool()
    def smart_reframe(project_id: str, target: str = "9:16", mode: str = "track_subject",
                      render: bool = True) -> dict[str, Any]:
        """Saliency-tracked reframe (matte>face>center) with smoothed crop path. No stretch."""
        from vidmcp.camera.reframe import smart_reframe_project

        return _safe(smart_reframe_project, load_project(project_id), target, mode, render)

    @mcp.tool()
    def add_camera_moves(project_id: str, style: str = "emphasis_punch", max_zoom: float = 1.12,
                         render: bool = True) -> dict[str, Any]:
        """Emphasis punch-ins on energy peaks, or slow drift / ken burns."""
        from vidmcp.camera.moves import add_camera_moves_project

        return _safe(add_camera_moves_project, load_project(project_id), style, max_zoom, render=render)

    @mcp.tool()
    def time_warp(project_id: str, ramps: list[dict], quality: str = "flow",
                  allow_speech_warp: bool = False) -> dict[str, Any]:
        """Speed ramps [{t, speed}] — pitch-safe audio, flow-interpolated slow-mo, speech protected."""
        from vidmcp.camera.timewarp import time_warp_project

        return _safe(time_warp_project, load_project(project_id), ramps, quality, allow_speech_warp)

    @mcp.tool()
    def stabilize_video(project_id: str, strength: float = 0.6) -> dict[str, Any]:
        """Two-pass vidstab (or deshake) stabilization."""
        from vidmcp.camera.stabilize import stabilize_video_project

        return _safe(stabilize_video_project, load_project(project_id), strength)

    # --- W5 audio ---

    @mcp.tool()
    def audio_tracks(project_id: str) -> dict[str, Any]:
        """Named track graph (voice/bgm/sfx/ambience/dub_*)."""
        from vidmcp.audio.tracks import audio_tracks_project

        return _safe(audio_tracks_project, load_project(project_id))

    @mcp.tool()
    def edit_audio_track(project_id: str, track: str, ops: list[dict]) -> dict[str, Any]:
        """Track ops: set_gain{db} | set_src{src} | set_offset{sec} | enable_duck{value}."""
        from vidmcp.audio.tracks import edit_audio_track_project

        return _safe(edit_audio_track_project, load_project(project_id), track, ops)

    @mcp.tool()
    def mixdown_audio(project_id: str, target: str = "youtube", duck_floor_db: float = -13.0) -> dict[str, Any]:
        """Mix all tracks: VAD-lookahead ducking, BGM widening, limiter, platform LUFS target."""
        from vidmcp.audio.tracks import mixdown_project

        return _safe(mixdown_project, load_project(project_id), target, duck_floor_db=duck_floor_db)

    @mcp.tool()
    def add_sfx(project_id: str, auto: bool = True, events: list[dict] | None = None,
                gain_db: float = -18.0) -> dict[str, Any]:
        """Procedural whoosh/impact/riser placed on graphics beats + energy peaks."""
        from vidmcp.audio.sfx import add_sfx_project

        return _safe(add_sfx_project, load_project(project_id), auto, events, gain_db)

    @mcp.tool()
    def generate_music(project_id: str, prompt: str = "", style: str = "cinematic",
                       bpm: float | None = None, duration_sec: float | None = None) -> dict[str, Any]:
        """Score matched to the footage energy curve (stems: pad/keys/pulse). Sets bgm track."""
        from vidmcp.audio.music import generate_music_project

        return _safe(generate_music_project, load_project(project_id), prompt, style, bpm, duration_sec)

    @mcp.tool()
    def dub_video(project_id: str, language: str, translated_segments: list[dict] | None = None,
                  voice: str = "neutral", voice_clone_consent: bool = False) -> dict[str, Any]:
        """Multi-language dub: segment TTS with duration fitting → dub track + render variant."""
        from vidmcp.audio.dubbing import dub_video_project

        return _safe(dub_video_project, load_project(project_id), language,
                     translated_segments, voice, voice_clone_consent)

    # --- W6 graphics / brand ---

    @mcp.tool()
    def add_graphics(project_id: str, items: list[dict], brand: str = "default") -> dict[str, Any]:
        """Brand mograph overlays: lower_third, title_card, stat_counter, charts... lock='speech:<kw>'."""
        from vidmcp.graphics.engine import add_graphics_project

        return _safe(add_graphics_project, load_project(project_id), items, brand)

    @mcp.tool()
    def list_graphic_templates() -> dict[str, Any]:
        """Available mograph templates + field schemas."""
        from vidmcp.graphics.engine import list_graphic_templates as _lt

        return _safe(_lt)

    @mcp.tool()
    def get_brand_kit(name: str = "default") -> dict[str, Any]:
        """Persistent brand kit (fonts/colors/logo/lut/caption style)."""
        from vidmcp.graphics.brand import get_brand_kit as _gb

        return {"ok": True, "kit": _gb(name)}

    @mcp.tool()
    def set_brand_kit(kit: dict, name: str = "default") -> dict[str, Any]:
        """Save brand kit — auto-injected into graphics, captions, thumbnails."""
        from vidmcp.graphics.brand import set_brand_kit as _sb

        return _safe(_sb, kit, name)

    @mcp.tool()
    def extract_brand_from_video(video_path: str, name: str = "extracted") -> dict[str, Any]:
        """Pull dominant palette from an existing branded video into a kit."""
        from vidmcp.graphics.brand import extract_brand_from_video as _eb

        return _safe(_eb, video_path, name)

    # --- W7 recipes v2 ---

    @mcp.tool()
    def list_recipes_v2() -> dict[str, Any]:
        """Composable v2 recipes (god_mode_talking_head, podcast_clip_factory, ad_15s, ...)."""
        from vidmcp.harness.recipe_schema import list_recipes_v2 as _lr

        return {"ok": True, "recipes": _lr()}

    @mcp.tool()
    def compose_recipes(names: list[str], overrides: dict | None = None) -> dict[str, Any]:
        """Chain/mix recipes into one (duplicate renders deduped, last wins)."""
        from vidmcp.harness.recipe_schema import compose_recipes as _cr

        return _safe(_cr, names, overrides)

    @mcp.tool()
    def validate_recipe(recipe: dict) -> dict[str, Any]:
        """Validate a v2 recipe dict: unknown ops, missing models."""
        from vidmcp.harness.recipe_schema import validate_recipe as _vr

        return _safe(_vr, recipe)

    @mcp.tool()
    def run_recipe_v2(project_id: str, recipe: str, content_type: str | None = None) -> dict[str, Any]:
        """Execute a v2 recipe (adapts to detected content type) via the op batcher."""
        from vidmcp.harness.recipe_schema import run_recipe_v2 as _rr

        return _safe(_rr, load_project(project_id), recipe, content_type)

    @mcp.tool()
    def classify_content(project_id: str) -> dict[str, Any]:
        """Detect content type: talking_head|tutorial|vlog|product_demo|lecture|ad."""
        from vidmcp.harness.content_type import classify_project

        return _safe(classify_project, load_project(project_id))

    # --- W8 cognition ---

    @mcp.tool()
    def draft_edit_plan(project_id: str, intent: str, duration_target_sec: float | None = None,
                        deliverables: list[str] | None = None, tone: str | None = None) -> dict[str, Any]:
        """Draft a full edit plan (NO side effects): recipe base + tone params + cost estimate."""
        from vidmcp.agents.edit_plan import draft_edit_plan_project

        return _safe(draft_edit_plan_project, load_project(project_id), intent,
                     duration_target_sec, deliverables, tone)

    @mcp.tool()
    def revise_edit_plan(project_id: str, plan_id: str, patch: dict) -> dict[str, Any]:
        """Patch a drafted plan: {remove_ids, set_args: {id: args}, add_steps: [{step, after}]}."""
        from vidmcp.agents.edit_plan import revise_edit_plan_project

        return _safe(revise_edit_plan_project, load_project(project_id), plan_id, patch)

    @mcp.tool()
    def execute_edit_plan(project_id: str, plan_id: str, until_step: str | None = None) -> dict[str, Any]:
        """Run a plan with per-step checkpoints (resumable) + auto-repair from rewatch QC."""
        from vidmcp.agents.edit_plan import execute_edit_plan_project
        from vidmcp.config import get_settings

        return _safe(execute_edit_plan_project, load_project(project_id), plan_id, until_step,
                     getattr(get_settings(), "max_repair_passes", 2))

    @mcp.tool()
    def set_creative_profile(project_id: str, tone: str = "premium",
                             pacing_template: str = "steady_educational") -> dict[str, Any]:
        """Tone (energetic|calm|dramatic|playful|premium) → concrete edit params; pacing target curve."""
        from vidmcp.agents.creative import set_creative_profile_project

        return _safe(set_creative_profile_project, load_project(project_id), tone, pacing_template)

    @mcp.tool()
    def analyze_pacing(project_id: str) -> dict[str, Any]:
        """Measured pacing curve vs target: hook strength, sag regions with fixes."""
        from vidmcp.agents.creative import analyze_pacing_project

        return _safe(analyze_pacing_project, load_project(project_id))

    @mcp.tool()
    def rewatch_render(project_id: str, depth: str = "mechanical", target: str = "youtube") -> dict[str, Any]:
        """Re-watch QC: black/frozen/flicker frames, LUFS/TP, dead air — with repair routes."""
        from vidmcp.agents.rewatch import rewatch_project

        return _safe(rewatch_project, load_project(project_id), depth, target)

    # --- W9 delivery ---

    @mcp.tool()
    def export_multi(project_id: str, targets: list[str] | None = None, hw: bool = True) -> dict[str, Any]:
        """One call → all platform variants (codec/size/loudness per target, reframe-aware)."""
        from vidmcp.media.delivery import export_multi_project

        return _safe(export_multi_project, load_project(project_id), targets, hw)

    @mcp.tool()
    def generate_thumbnails(project_id: str, n: int = 3, title_variants: list[str] | None = None,
                            brand: str = "default") -> dict[str, Any]:
        """A/B thumbnails: face/sharpness/color-scored frames + brand title styles + contact sheet."""
        from vidmcp.media.thumbs import generate_thumbnails_project

        return _safe(generate_thumbnails_project, load_project(project_id), n, title_variants, brand)

    @mcp.tool()
    def generate_metadata(project_id: str, platform: str = "youtube") -> dict[str, Any]:
        """Publish pack: chapters, title candidates, description, tags, SRT/VTT sidecars."""
        from vidmcp.media.metadata import generate_metadata_project

        return _safe(generate_metadata_project, load_project(project_id), platform)

    @mcp.tool()
    def extract_clips(project_id: str, n: int = 3, max_sec: float = 45.0) -> dict[str, Any]:
        """Rank self-contained high-energy moments for short-form clips (hook-scored)."""
        from vidmcp.media.metadata import extract_clips_project

        return _safe(extract_clips_project, load_project(project_id), n, max_sec)

    @mcp.tool()
    def freeze_frame(project_id: str, t: float, hold_sec: float = 1.5) -> dict[str, Any]:
        """Insert a freeze frame at t (audio padded)."""
        from vidmcp.camera.timewarp import freeze_frame_project

        return _safe(freeze_frame_project, load_project(project_id), t, hold_sec)
