"""Fast education-first harness runtime — fewer steps, budgets, phase contracts.

Design principles applied:
1. Minimal tool pack (education ~15 tools not 76)
2. Phase contracts with gates + retry budgets
3. Skip expensive work when quality already OK
4. Durable state via project manifest + causal graph
5. Early stop when wall clock / tool budget hit
"""

from __future__ import annotations

import time
from typing import Any, Callable

from vidmcp.config import get_settings
from vidmcp.core.workspace import Workspace
from vidmcp.harness.contracts import Phase, build_harness_plan
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.fast_harness")
ProgressFn = Callable[[float, str], None]


class FastEducationHarness:
    def __init__(self, workspace: Workspace | None = None):
        self.settings = get_settings()
        self.workspace = workspace or Workspace(self.settings)

    def run(
        self,
        *,
        video_path: str,
        intent: str,
        project_name: str = "edu_fast",
        max_render_frames: int | None = None,
        n_steps: int | None = None,
        progress: ProgressFn | None = None,
        force_fast: bool = True,
    ) -> dict[str, Any]:
        from vidmcp.tools import advanced_service as adv
        from vidmcp.tools import service
        from vidmcp.audio.course import compile_lesson
        from vidmcp.audio.media import ensure_video_with_narration
        from vidmcp.critics.ensemble import run_critic_ensemble
        from vidmcp.harness.quality_gates import evaluate_gates

        t0 = time.perf_counter()
        plan = build_harness_plan(intent, product="education", fast=force_fast or self.settings.harness_fast_mode)
        max_frames = max_render_frames
        if max_frames is None:
            max_frames = plan.budget.max_render_frames or self.settings.harness_preview_frames
        n_steps = n_steps or self.settings.education_default_steps
        tool_calls = 0
        phase_log: list[dict[str, Any]] = []

        def report(p: float, msg: str) -> None:
            if progress:
                progress(p, msg)
            log.info("fast_harness", p=round(p, 3), msg=msg)

        def budget_ok() -> bool:
            return (time.perf_counter() - t0) < plan.budget.max_wall_sec and tool_calls < plan.budget.max_tool_calls

        # --- INGEST ---
        report(0.05, "ingest")
        lesson = compile_lesson(intent, duration_sec=max(20.0, (max_frames or 48) / 12 * 3), n_beats=n_steps)
        narr = " ".join(b["narration_cue"] for b in lesson["beats"][:n_steps])
        media_dir = self.settings.workspace_root / "_media"
        media_dir.mkdir(parents=True, exist_ok=True)
        media = ensure_video_with_narration(
            video_path,
            narration=narr,
            out_path=media_dir / f"{project_name}_narrated.mp4",
            force=False,
        )
        tool_calls += 1
        project = self.workspace.create_project(name=project_name)
        service.import_source(project, media["path"])
        service.analyze(project)
        adv.graph_commit(project, "fast_ingest", {"intent": intent})
        tool_calls += 2
        phase_log.append({"phase": "ingest", "ok": True, "sec": time.perf_counter() - t0})

        # --- PERCEIVE ---
        report(0.2, "perceive")
        if not budget_ok():
            return self._timeout(project.manifest.id, plan, phase_log, tool_calls, t0)
        seg = service.segment(project, prompt="person", conf=self.settings.conf_threshold)
        # Note: full-video MLX track is expensive; max_frames applied at composite; perception may still scan full clip unless backend honors kwargs
        tool_calls += 1
        words = adv.word_timeline(project, fallback_transcript=narr)
        tool_calls += 1
        stab = float(seg.get("temporal_stability") or 0)
        phase_log.append({"phase": "perceive", "ok": True, "stability": stab, "backend": seg.get("backend")})

        # --- REFINE (skip if already good — minimal intervention) ---
        report(0.4, "refine_if_needed")
        refine = {"skipped": True}
        if stab < 0.65 and budget_ok():
            try:
                refine = adv.uncertainty_guided_refine(project, service)
                tool_calls += 2
                refine["skipped"] = False
            except Exception as e:  # noqa: BLE001
                refine = {"skipped": False, "error": str(e)}
        phase_log.append({"phase": "refine", **{k: refine.get(k) for k in ("skipped", "error")}})

        # --- SCENE (speech-locked only — one plate, not multi variants) ---
        report(0.55, "scene")
        if not budget_ok():
            return self._timeout(project.manifest.id, plan, phase_log, tool_calls, t0)
        scene = adv.speech_locked_scene(
            project,
            intent,
            n_steps=n_steps,
            keywords=list(lesson.get("keywords") or ["first", "therefore", "prove", "finally"])[:10],
            fallback_transcript=narr,
            place_as_background=True,
        )
        tool_calls += 1
        # light grade only — skip cyberpunk particles unless intent asks
        if any(k in intent.lower() for k in ("neon", "cyber", "particle")):
            service.apply_effects(
                project,
                effect_specs=[
                    {"effect_type": "cyberpunk", "kind": "background", "params": {"blur_radius": 12}, "name": "bg"}
                ],
                replace_existing=False,
            )
            tool_calls += 1
        else:
            service.apply_effects(
                project,
                effect_specs=[
                    {
                        "effect_type": "color_grade",
                        "kind": "grade",
                        "params": {"contrast": 1.08, "saturation": 1.05, "background_only": True},
                        "name": "soft",
                    }
                ],
                replace_existing=False,
            )
            tool_calls += 1
        phase_log.append({"phase": "scene", "ok": True, "scene_path": scene.get("scene_path")})

        # --- COMPOSE ---
        report(0.75, "compose")
        if not budget_ok():
            return self._timeout(project.manifest.id, plan, phase_log, tool_calls, t0)
        render = service.composite(project, max_frames=max_frames)
        tool_calls += 1
        phase_log.append({"phase": "compose", "ok": True, "render": render.get("output_path")})

        # --- VERIFY ---
        report(0.9, "verify")
        critics = run_critic_ensemble(project)
        gates = evaluate_gates(project, self.settings)
        tool_calls += 2
        # one auto-heuristic pass only if failed
        auto = {"skipped": True}
        if not critics.get("ok") and budget_ok():
            auto = adv.apply_auto_heuristics(project, service, max_frames=max_frames)
            tool_calls += 1
            auto["skipped"] = False
            critics = run_critic_ensemble(project)
        phase_log.append(
            {
                "phase": "verify",
                "critic_score": critics.get("overall_score"),
                "failed_axes": critics.get("failed_axes"),
                "gate_score": gates.score,
            }
        )

        # --- SIGN ---
        report(0.97, "sign")
        signed = adv.sign(project)
        tool_calls += 1
        phase_log.append({"phase": "sign", "ok": True})

        elapsed = time.perf_counter() - t0
        return {
            "ok": bool(critics.get("ok") or (critics.get("overall_score") or 0) >= 0.55),
            "project_id": project.manifest.id,
            "product": "education",
            "plan": plan.to_dict(),
            "lesson": {"title": lesson.get("title"), "beats": len(lesson.get("beats") or [])},
            "segment": {"backend": seg.get("backend"), "stability": stab},
            "scene": scene.get("scene_path"),
            "render": render,
            "critics": {
                "score": critics.get("overall_score"),
                "failed_axes": critics.get("failed_axes"),
                "fix_route": critics.get("fix_route"),
            },
            "gates": gates.to_dict(),
            "provenance": signed.get("manifest_path"),
            "metrics": {
                "wall_sec": round(elapsed, 2),
                "tool_calls": tool_calls,
                "max_frames": max_frames,
                "budget_wall_sec": plan.budget.max_wall_sec,
            },
            "phase_log": phase_log,
            "message": f"Fast education harness complete in {elapsed:.1f}s",
        }

    def _timeout(self, project_id: str, plan, phase_log, tool_calls, t0) -> dict[str, Any]:
        return {
            "ok": False,
            "project_id": project_id,
            "message": "Harness budget exceeded (wall clock or tool calls)",
            "plan": plan.to_dict(),
            "phase_log": phase_log,
            "metrics": {"wall_sec": round(time.perf_counter() - t0, 2), "tool_calls": tool_calls},
        }
