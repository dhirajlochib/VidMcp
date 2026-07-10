"""Register v0.4 advanced MCP tools onto a FastMCP instance."""

from __future__ import annotations

from typing import Any, Callable


def register_advanced_tools(
    mcp: Any,
    *,
    load_project: Callable[[str], Any],
    workspace_factory: Callable[[], Any],
    get_settings: Callable[[], Any],
    service: Any,
    log: Any,
) -> None:
    from vidmcp.tools import advanced_service as adv

    @mcp.tool()
    def graph_commit(project_id: str, op: str, message: str = "", args: dict | None = None) -> dict[str, Any]:
        """Commit an operation onto the causal edit graph (content-addressed time travel)."""
        return adv.graph_commit(load_project(project_id), op, args, message)

    @mcp.tool()
    def graph_log(project_id: str, limit: int = 30) -> dict[str, Any]:
        """Show causal graph history (newest first) and branch heads."""
        return adv.graph_log(load_project(project_id), limit=limit)

    @mcp.tool()
    def graph_branch(project_id: str, name: str) -> dict[str, Any]:
        """Create a named branch from current HEAD for experimental edits."""
        return adv.graph_branch(load_project(project_id), name)

    @mcp.tool()
    def graph_checkout(project_id: str, node_id: str, restore_manifest: bool = True) -> dict[str, Any]:
        """Time-travel: checkout a historical graph node (optionally restore manifest snapshot)."""
        return adv.graph_checkout(load_project(project_id), node_id, restore_manifest=restore_manifest)

    @mcp.tool()
    def graph_merge(project_id: str, source_branch: str, target_branch: str = "main") -> dict[str, Any]:
        """Merge branches with layer-aware conflict report."""
        return adv.graph_merge(load_project(project_id), source_branch, target_branch)

    @mcp.tool()
    def compute_uncertainty_field(project_id: str) -> dict[str, Any]:
        """Compute matte uncertainty (temporal variance + edge entropy); emit heatmap + refine ROI boxes."""
        return adv.uncertainty_for_project(load_project(project_id))

    @mcp.tool()
    def uncertainty_guided_refine(project_id: str) -> dict[str, Any]:
        """Surgical refine only on high-uncertainty frames/ROIs (cheap quality upgrade)."""
        return adv.uncertainty_guided_refine(load_project(project_id), service)

    @mcp.tool()
    def lock_identity_across_shots(project_id: str, other_shots: list[dict] | None = None) -> dict[str, Any]:
        """Cross-shot identity lock via appearance signatures; stable global IDs across cuts."""
        return adv.identity_lock_project(load_project(project_id), other_shots)

    @mcp.tool()
    def sync_audio_semantics(
        project_id: str,
        transcript: str | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """Audio energy/onsets + optional keywords → scene step beats and particle modulation events."""
        return adv.audio_sync_project(load_project(project_id), transcript=transcript, keywords=keywords)

    @mcp.tool()
    def match_subject_lighting(
        project_id: str,
        strength: float = 0.8,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """Lab color transfer so subject lighting matches background plate (kills sticker look)."""
        return adv.lighting_match_project(load_project(project_id), strength=strength, max_frames=max_frames)

    @mcp.tool()
    def apply_depth_fog_particles(
        project_id: str,
        style: str = "fog",
        density: float = 0.5,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """Depth-ordered fog/particles that occlude correctly (behind subject only)."""
        return adv.depth_fog_project(load_project(project_id), style=style, density=density, max_frames=max_frames)

    @mcp.tool()
    def reproject_background(
        project_id: str,
        plate_path: str | None = None,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """World-consistent BG: warp plate by estimated camera affine so parallax survives motion."""
        return adv.reproject_bg_project(load_project(project_id), plate_path=plate_path, max_frames=max_frames)

    @mcp.tool()
    def run_critic_ensemble(project_id: str) -> dict[str, Any]:
        """Multi-axis adversarial critics (matte/edge/lighting/render/gates) with fix_route tool list."""
        return adv.critic_project(load_project(project_id), workspace_root=get_settings().workspace_root)

    @mcp.tool()
    def propose_edit_strategies(intent: str, project_id: str | None = None, n: int = 3) -> dict[str, Any]:
        """Multi-agent debate: propose scored strategies (quality/cost/risk/latency); returns recommended winner."""
        proj = load_project(project_id) if project_id else None
        return adv.debate(intent, proj)

    @mcp.tool()
    def compile_viddsl(source: str) -> dict[str, Any]:
        """Parse/validate VidDSL recipe language into an op program (no side effects)."""
        from vidmcp.dsl.viddsl import compile_viddsl as _c

        try:
            prog = _c(source)
            return {"ok": True, "program": prog.to_dict()}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    @mcp.tool()
    def run_viddsl(project_id: str, source: str, max_render_frames: int | None = None) -> dict[str, Any]:
        """Execute VidDSL program on a project (track/scene/composite/gate/refine/sign/...)."""
        return adv.run_dsl(load_project(project_id), source, service, max_render_frames=max_render_frames)

    @mcp.tool()
    def start_live_session(mode: str = "mock_matte", width: int = 1280, height: int = 720) -> dict[str, Any]:
        """Start low-latency live/streaming session (ring-buffer matte path for OBS-like loops)."""
        return adv.live_start(mode=mode, width=width, height=height)

    @mcp.tool()
    def live_process_video(
        session_id: str,
        video_path: str,
        output_name: str = "live_out.mp4",
        max_frames: int = 60,
        effect: str = "blur",
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Run live pipeline over a file (benchmark latency) into workspace or cwd."""
        from pathlib import Path

        if project_id:
            proj = load_project(project_id)
            out = str(proj.renders_dir / output_name)
        else:
            out = str(get_settings().workspace_root / output_name)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        return adv.live_process_file(session_id, video_path, out, max_frames=max_frames, effect=effect)

    @mcp.tool()
    def sign_render(project_id: str) -> dict[str, Any]:
        """Write HMAC-SHA256 signed provenance manifest for latest render (reproducible audit trail)."""
        return adv.sign(load_project(project_id))

    @mcp.tool()
    def verify_render_manifest(manifest_path: str) -> dict[str, Any]:
        """Verify a signed provenance manifest."""
        return adv.verify(manifest_path)

    @mcp.tool()
    def compile_lesson(intent: str, duration_sec: float = 180.0) -> dict[str, Any]:
        """Course compiler: intent → beats, scene prompts, FX cues, VidDSL scaffold."""
        return adv.lesson(intent, duration_sec=duration_sec)

    @mcp.tool()
    def mine_failures() -> dict[str, Any]:
        """Mine historical critic/gate failures and suggest new auto-heuristics."""
        return adv.failures(workspace_factory())


    @mcp.tool()
    def transcribe_words(
        project_id: str,
        fallback_transcript: str | None = None,
        model_size: str = "base",
    ) -> dict[str, Any]:
        """Word-level ASR timeline (faster-whisper/openai-whisper if installed; else energy-aligned fallback)."""
        return adv.word_timeline(load_project(project_id), fallback_transcript=fallback_transcript, model_size=model_size)

    @mcp.tool()
    def render_speech_locked_scene(
        project_id: str,
        prompt: str,
        n_steps: int = 6,
        keywords: list[str] | None = None,
        fallback_transcript: str | None = None,
        place_as_background: bool = True,
    ) -> dict[str, Any]:
        """Advance educational scene beats locked to word timestamps / cue words (true speech–scene coupling)."""
        return adv.speech_locked_scene(
            load_project(project_id),
            prompt,
            n_steps=n_steps,
            keywords=keywords,
            place_as_background=place_as_background,
            fallback_transcript=fallback_transcript,
        )

    @mcp.tool()
    def compute_depth_field(
        project_id: str,
        max_frames: int = 60,
        prefer_midas: bool = True,
    ) -> dict[str, Any]:
        """Multi-cue depth maps (matte+defocus+vertical); uses Depth-Anything if transformers installed."""
        return adv.compute_depth(load_project(project_id), max_frames=max_frames, prefer_midas=prefer_midas)

    @mcp.tool()
    def flow_reproject_background(
        project_id: str,
        plate_path: str | None = None,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        """Dense optical-flow plate warp (smoother world-consistent BG than affine-only)."""
        return adv.flow_reproject(load_project(project_id), plate_path=plate_path, max_frames=max_frames)

    @mcp.tool()
    def detect_shots(project_id: str, threshold: float = 0.45) -> dict[str, Any]:
        """Histogram shot-boundary detection for multi-shot identity + per-shot graphs."""
        return adv.detect_project_shots(load_project(project_id), threshold=threshold)

    @mcp.tool()
    def export_timeline(project_id: str) -> dict[str, Any]:
        """Export OTIO-compatible JSON timeline (+ real .otio if opentimelineio installed)."""
        return adv.export_otio(load_project(project_id))

    @mcp.tool()
    def apply_auto_heuristics(project_id: str, max_frames: int | None = None) -> dict[str, Any]:
        """Apply mined failure heuristics automatically (refine/lighting/composite) and re-score critics."""
        return adv.apply_auto_heuristics(load_project(project_id), service, max_frames=max_frames)

    @mcp.tool()
    def enhance_edit(
        project_id: str,
        max_frames: int | None = None,
        speech_prompt: str | None = None,
        fallback_transcript: str | None = None,
    ) -> dict[str, Any]:
        """One-call enhancement pass: uncertainty refine → speech scene (optional) → depth fog → lighting → flow reproject → auto heuristics → critic → sign."""
        project = load_project(project_id)
        steps = {}
        try:
            steps["uncertainty"] = adv.uncertainty_guided_refine(project, service)
        except Exception as e:
            steps["uncertainty"] = {"ok": False, "message": str(e)}
        if speech_prompt:
            steps["speech_scene"] = adv.speech_locked_scene(
                project, speech_prompt, fallback_transcript=fallback_transcript
            )
        try:
            steps["depth"] = adv.compute_depth(project, max_frames=max_frames or 40)
        except Exception as e:
            steps["depth"] = {"ok": False, "message": str(e)}
        try:
            steps["fog"] = adv.depth_fog_project(project, max_frames=max_frames)
        except Exception as e:
            steps["fog"] = {"ok": False, "message": str(e)}
        try:
            steps["lighting"] = adv.lighting_match_project(project, max_frames=max_frames)
        except Exception as e:
            steps["lighting"] = {"ok": False, "message": str(e)}
        try:
            steps["flow_reproject"] = adv.flow_reproject(project, max_frames=max_frames)
        except Exception as e:
            steps["flow_reproject"] = {"ok": False, "message": str(e)}
        try:
            steps["composite"] = service.composite(project, max_frames=max_frames)
        except Exception as e:
            steps["composite"] = {"ok": False, "message": str(e)}
        steps["auto_heuristics"] = adv.apply_auto_heuristics(project, service, max_frames=max_frames)
        steps["critics"] = adv.critic_project(project, workspace_root=get_settings().workspace_root)
        steps["provenance"] = adv.sign(project)
        steps["timeline"] = adv.export_otio(project)
        return {
            "ok": True,
            "project_id": project_id,
            "steps": {k: (v if isinstance(v, dict) else str(v)) for k, v in steps.items()},
            "message": "enhance_edit complete",
        }


    @mcp.tool()
    def platform_health() -> dict[str, Any]:
        """Report installed backends (Whisper/SAM/Manim/Blender/FFmpeg) and readiness for education vs GPU paths."""
        return adv.health()

    @mcp.tool()
    def attach_narration(project_id: str, narration: str, force: bool = True) -> dict[str, Any]:
        """Mux TTS narration (macOS say / espeak) onto a silent source video for ASR + speech-lock demos."""
        return adv.attach_narration(load_project(project_id), narration, force=force)

    @mcp.tool()
    def run_education_lesson(
        video_path: str,
        lesson_topic: str,
        project_name: str = "education_lesson",
        narration: str | None = None,
        max_render_frames: int | None = 72,
        style: str = "cinematic",
        n_steps: int = 6,
    ) -> dict[str, Any]:
        """EDUCATION PRODUCT PATH: narrate → segment speaker → Whisper words → speech-locked lesson plate → grade → composite → critic → sign.

        style: cinematic | cyberpunk | clean
        """
        return adv.education_lesson(
            video_path=video_path,
            lesson_topic=lesson_topic,
            project_name=project_name,
            narration=narration,
            max_render_frames=max_render_frames,
            style=style,
            n_steps=n_steps,
        )


    @mcp.tool()
    def enqueue_job(handler: str, payload: dict, priority: int = 100) -> dict[str, Any]:
        """Enqueue durable background job (handlers: composite, segment, enhance, education_lesson)."""
        return adv.enqueue_job(handler, payload, priority=priority)

    @mcp.tool()
    def queue_status(job_id: str | None = None, status: str | None = None) -> dict[str, Any]:
        """Inspect queue job by id or list jobs filtered by status (pending/running/done/failed)."""
        return adv.queue_status(job_id=job_id, status=status)

    @mcp.tool()
    def start_queue_worker(background: bool = True, max_jobs: int | None = None) -> dict[str, Any]:
        """Start file-queue worker (background thread or process N jobs inline)."""
        return adv.queue_worker_start(max_jobs=max_jobs, background=background)

    @mcp.tool()
    def diarize_speakers(project_id: str, n_speakers: int = 2) -> dict[str, Any]:
        """Multi-speaker diarization (pyannote if available; else spectral/energy clustering)."""
        return adv.diarize_project(load_project(project_id), n_speakers=n_speakers)

    @mcp.tool()
    def generate_meshy_plate(
        project_id: str,
        prompt: str,
        place_as_background: bool = True,
        duration_sec: float = 4.0,
    ) -> dict[str, Any]:
        """Text→3D plate via Meshy API (MESHY_API_KEY) or procedural 3D-ish fallback; places under subject."""
        return adv.meshy_plate(load_project(project_id), prompt, place_as_background=place_as_background, duration_sec=duration_sec)

    @mcp.tool()
    def scaffold_remotion_scene(project_id: str, prompt: str) -> dict[str, Any]:
        """Generate a Remotion/JS composition scaffold for web-native explainers."""
        return adv.remotion_scaffold(load_project(project_id), prompt)

    # --- Creator 1.1 tools (audio / captions / BG / export / polish) ---
    from vidmcp.tools import creator as cr

    @mcp.tool()
    def process_audio(
        project_id: str,
        strength: float = 0.7,
        target_lufs: float = -14.0,
    ) -> dict[str, Any]:
        """Denoise + enhance vocals + loudnorm. Writes audio/vocals_clean.wav on the project."""
        return cr.process_audio_project(load_project(project_id), strength=strength, target_lufs=target_lufs)

    @mcp.tool()
    def mix_bgm(
        project_id: str,
        bgm_path: str | None = None,
        volume: float = 0.35,
        style: str = "cinematic",
        duck: bool = True,
    ) -> dict[str, Any]:
        """Mix copyright-safe ambient BGM under vocals (or user bgm_path). Ducking optional."""
        return cr.mix_bgm_project(
            load_project(project_id),
            bgm_path=bgm_path,
            volume=volume,
            style=style,
            duck=duck,
        )

    @mcp.tool()
    def transcribe_and_caption(
        project_id: str,
        burn: bool = True,
        style: str = "brand",
        language: str | None = None,
        model_size: str = "base",
        fallback_transcript: str | None = None,
    ) -> dict[str, Any]:
        """ASR word timeline + ASS captions; optionally burn into a new render (brand|karaoke|minimal)."""
        return cr.transcribe_and_caption_project(
            load_project(project_id),
            burn=burn,
            style=style,
            language=language,
            model_size=model_size,
            fallback_transcript=fallback_transcript,
        )

    @mcp.tool()
    def replace_background(
        project_id: str,
        plate: str = "space",
        matte_backend: str = "auto",
        plate_image: str | None = None,
    ) -> dict[str, Any]:
        """Cut out subject (MediaPipe/SAM) and composite over plate: space|blur|solid|image."""
        return cr.replace_background_project(
            load_project(project_id),
            plate=plate,
            matte_backend=matte_backend,
            plate_image=plate_image,
        )

    @mcp.tool()
    def export_render(
        project_id: str,
        render_path: str | None = None,
        preset: str = "youtube_16x9",
        loudnorm: bool = True,
    ) -> dict[str, Any]:
        """Export last render (or path) to preset: youtube_16x9 | reels_9x16 | square_1x1 | source."""
        return cr.export_render_project(
            load_project(project_id),
            render_path=render_path,
            preset=preset,
            loudnorm=loudnorm,
        )

    @mcp.tool()
    def smart_cut_hesitations(project_id: str, aggressiveness: float = 0.5) -> dict[str, Any]:
        """Remove dead air / long fillers (um, basically, …) using transcript + energy gaps."""
        return cr.smart_cut_project(load_project(project_id), aggressiveness=aggressiveness)

    @mcp.tool()
    def add_speech_infographics(project_id: str, topic: str = "auto") -> dict[str, Any]:
        """Burn keyword/number infographic cards locked to transcript timing."""
        return cr.add_infographics_project(load_project(project_id), topic=topic)

    @mcp.tool()
    def run_talking_head_polish(
        video_path: str,
        preset: str = "youtube_16x9",
        bg_mode: str = "none",
        strength: float = 0.7,
        bgm_volume: float = 0.35,
        burn_captions: bool = True,
        smart_cut: bool = False,
        project_name: str = "talking_head_polish",
    ) -> dict[str, Any]:
        """ONE-SHOT: orient → denoise → BGM → optional BG → captions → export preset.

        bg_mode: none | space | blur | solid
        preset: youtube_16x9 | reels_9x16 | square_1x1 | source
        """
        return cr.run_talking_head_polish(
            video_path,
            name=project_name,
            preset=preset,
            bg_mode=bg_mode,
            strength=strength,
            bgm_volume=bgm_volume,
            burn_captions_flag=burn_captions,
            smart_cut=smart_cut,
        )

    @mcp.tool()
    def export_edl(project_id: str) -> dict[str, Any]:
        """Export edit decision list JSON (source, audio pipeline, renders, history)."""
        from vidmcp.edit.edl import export_edl as _export_edl

        return _export_edl(load_project(project_id))

    @mcp.tool()
    def generate_thumbnail(project_id: str, title: str | None = None) -> dict[str, Any]:
        """Grab mid-frame thumbnail with title overlay from transcript."""
        return cr.generate_thumbnail_project(load_project(project_id), title=title)

    @mcp.tool()
    def list_tool_packs() -> dict[str, Any]:
        """List agent tool packs (talking_head, education, vfx_matte)."""
        from vidmcp.harness.packs import list_packs

        return {"ok": True, "packs": list_packs()}

    @mcp.tool()
    def list_marketplace_recipes() -> dict[str, Any]:
        """List builtin + installed marketplace recipe plugins."""
        return adv.marketplace_list()

    @mcp.tool()
    def publish_recipe(recipe: dict, author: str = "local") -> dict[str, Any]:
        """Publish a recipe dict into the local marketplace registry."""
        return adv.marketplace_publish(recipe, author=author)

    @mcp.tool()
    def install_recipe_file(path: str) -> dict[str, Any]:
        """Install a recipe JSON file into the marketplace."""
        return adv.marketplace_install(path)

    @mcp.tool()
    def start_review_ui(port: int = 8765) -> dict[str, Any]:
        """Start local human-in-the-loop review UI at http://127.0.0.1:<port>/"""
        return adv.review_ui_start(port=port)

    @mcp.tool()
    def review_ui_status() -> dict[str, Any]:
        """Review UI status + recent approve/reject decisions."""
        return adv.review_ui_status()

    @mcp.tool()
    def get_review_decisions() -> dict[str, Any]:
        """Fetch human review decisions for the agent to continue/revise."""
        return adv.review_decisions()


    @mcp.tool()
    def plan_harness(intent: str, product: str | None = None, fast: bool = True) -> dict[str, Any]:
        """Build a phase-contract harness plan (education/creator) with tool allowlist + budgets. No side effects."""
        from vidmcp.harness.contracts import build_harness_plan

        plan = build_harness_plan(intent, product=product, fast=fast)
        return {"ok": True, **plan.to_dict()}

    @mcp.tool()
    def run_fast_education_harness(
        video_path: str,
        intent: str,
        project_name: str = "edu_fast",
        max_render_frames: int | None = 48,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """FAST education product harness: minimal tools, budgets, skip refine if matte OK, speech-locked scene, critic, sign.

        Prefer this over run_ultimate_pipeline for teaching/talking-head lessons (much lower latency).
        """
        from vidmcp.harness.fast_runtime import FastEducationHarness

        try:
            return FastEducationHarness(workspace_factory()).run(
                video_path=video_path,
                intent=intent,
                project_name=project_name,
                max_render_frames=max_render_frames,
                n_steps=n_steps,
                force_fast=True,
            )
        except Exception as e:
            log.exception("fast_edu_harness_failed")
            return {"ok": False, "message": str(e)}

    @mcp.tool()
    def list_tool_packs() -> dict[str, Any]:
        """List reduced tool packs for reliable agent use (education vs creator_vfx)."""
        from vidmcp.harness.contracts import TOOL_PACKS

        return {
            "ok": True,
            "packs": {k: v for k, v in TOOL_PACKS.items()},
            "note": "Agents should use a pack allowlist to avoid 76-tool confusion and loops.",
        }

    @mcp.tool()
    def run_ultimate_pipeline(
        video_path: str,
        intent: str,
        project_name: str = "ultimate",
        max_render_frames: int | None = 48,
        include_math_scene: bool = True,
        lesson_prompt: str | None = None,
    ) -> dict[str, Any]:
        """FULL platform showcase: debate → segment → uncertainty refine → scene → audio → fog → lighting → composite → critics → sign → mine.

        This is the kitchen-sink advanced path demonstrating the entire stack.
        """
        try:
            from vidmcp.harness.runtime import HarnessRuntime

            ws = workspace_factory()
            settings = get_settings()
            # debate
            deb = adv.debate(intent, None)
            # create + import
            project = ws.create_project(name=project_name)
            service.import_source(project, video_path)
            service.analyze(project)
            adv.graph_commit(project, "import_analyze", {"intent": intent})
            # segment
            seg = service.segment(project, prompt="person")
            adv.graph_commit(project, "segment_subject", {"segment_id": seg.get("segment_id")})
            # uncertainty refine
            uref = adv.uncertainty_guided_refine(project, service)
            # math scene
            scene = None
            if include_math_scene:
                scene = service.render_math_scene(
                    project,
                    prompt=lesson_prompt or intent,
                    engine="procedural",
                    place_as_background=True,
                )
            # audio
            audio = adv.audio_sync_project(project, keywords=["prove", "therefore", "equals"])
            # effects light
            from vidmcp.agents.planner import PlannerAgent

            tags = PlannerAgent().plan(intent).style_tags or ["blur"]
            specs = PlannerAgent().effects_from_tags(tags, intent)
            service.apply_effects(project, effect_specs=specs, replace_existing=False)
            # depth fog optional
            fog = adv.depth_fog_project(project, style="fog", density=0.35, max_frames=max_render_frames)
            # lighting
            light = adv.lighting_match_project(project, max_frames=max_render_frames)
            # main composite
            render = service.composite(project, max_frames=max_render_frames)
            adv.graph_commit(project, "composite_and_render", {"path": render.get("output_path")})
            # critics
            critics = adv.critic_project(project, workspace_root=settings.workspace_root)
            # sign
            signed = adv.sign(project)
            # mine
            mined = adv.failures(ws)
            return {
                "ok": bool(critics.get("ok") or (critics.get("overall_score") or 0) > 0.5),
                "project_id": project.manifest.id,
                "debate": deb,
                "segment": seg,
                "uncertainty_refine": {"hot_frames": uref.get("uncertainty", {}).get("hot_frames")},
                "scene": scene,
                "audio_events": (audio.get("sync") or {}).get("n_events"),
                "fog": fog.get("project_relative"),
                "lighting": light.get("project_relative"),
                "render": render,
                "critics": critics,
                "provenance": signed,
                "failure_heuristics": mined.get("heuristics"),
                "message": "Ultimate advanced pipeline complete",
            }
        except Exception as e:
            log.exception("ultimate_pipeline_failed")
            return {"ok": False, "message": str(e)}
