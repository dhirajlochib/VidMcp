"""HarnessRuntime — multi-pass quality-gated editing loop (the advanced agent core)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vidmcp.agents.planner import PlannerAgent
from vidmcp.config import Settings, get_settings
from vidmcp.core.workspace import Workspace
from vidmcp.harness.edit_graph import EditGraph
from vidmcp.harness.quality_gates import QualityGateResult, evaluate_gates
from vidmcp.harness.recipes import get_recipe
from vidmcp.harness.telemetry import TelemetryRun
from vidmcp.harness.variants import pick_variants
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.harness")
ProgressFn = Callable[[float, str], None]


class HarnessRuntime:
    """
    Production agent harness above raw MCP tools:

    - Declarative edit graphs
    - Quality gates with auto-refine strategies
    - Multi-pass loops until score threshold
    - Recipe execution
    - A/B variant generation under one segmentation
    - Full telemetry dump per run
    """

    def __init__(self, workspace: Workspace | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.workspace = workspace or Workspace(self.settings)
        self.planner = PlannerAgent()

    # ------------------------------------------------------------------
    # Quality-gated pipeline
    # ------------------------------------------------------------------
    def run_quality_gated_pipeline(
        self,
        *,
        video_path: str,
        intent: str,
        project_name: str = "harness_edit",
        conf: float | None = None,
        max_passes: int | None = None,
        max_render_frames: int | None = None,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        from vidmcp.tools import service

        conf = self.settings.conf_threshold if conf is None else conf
        max_passes = max_passes or self.settings.harness_max_passes
        plan = self.planner.plan(intent)
        project = self.workspace.create_project(name=project_name)
        tel = TelemetryRun(project.manifest.id, kind="quality_gated")
        tel.event("plan", steps=[s.tool for s in plan.steps], style_tags=plan.style_tags)

        def report(p: float, msg: str) -> None:
            if progress:
                progress(p, msg)
            tel.event("progress", p=p, msg=msg)

        report(0.02, "import")
        service.import_source(project, video_path)
        report(0.06, "analyze")
        analysis = service.analyze(project)
        prompts = [plan.subject_prompt] + list(analysis.get("suggested_prompts") or [])
        # unique preserve order
        seen: set[str] = set()
        prompt_candidates = []
        for p in prompts:
            if p and p not in seen:
                seen.add(p)
                prompt_candidates.append(p)

        last_gate: QualityGateResult | None = None
        segment_info: dict[str, Any] = {}
        effects_info: dict[str, Any] = {}
        render_info: dict[str, Any] = {}
        pass_logs: list[dict[str, Any]] = []

        for attempt in range(1, max_passes + 1):
            report(0.1 + 0.25 * (attempt - 1) / max_passes, f"pass {attempt}/{max_passes}: segment")
            prompt = prompt_candidates[min(attempt - 1, len(prompt_candidates) - 1)]
            # adaptive conf
            use_conf = conf * (0.85 ** (attempt - 1))
            segment_info = service.segment(
                project,
                prompt=prompt,
                conf=use_conf,
                progress=lambda p, m: report(0.1 + 0.2 * p, m),
            )
            tel.event("segment", pass_=attempt, prompt=prompt, conf=use_conf, **{
                k: segment_info.get(k) for k in ("temporal_stability", "coverage_mean", "backend")
            })

            # early matte gate (before expensive render)
            pre = evaluate_gates(project, self.settings)
            # render may be missing — that's ok for pre-check on matte only
            matte_ok = all(
                c.passed
                for c in pre.checks
                if c.name in ("masks_present", "temporal_stability", "coverage_range") and c.severity == "block"
            )
            if not matte_ok and attempt < max_passes and self.settings.harness_auto_refine:
                pass_logs.append({"pass": attempt, "stage": "pre_matte", "gate": pre.to_dict(), "prompt": prompt})
                tel.event("matte_gate_fail", pass_=attempt, strategy=pre.refine_strategy)
                continue

            report(0.45, f"pass {attempt}: effects")
            effect_specs = self.planner.effects_from_tags(plan.style_tags, intent)
            effects_info = service.apply_effects(project, effect_specs=effect_specs, intent=intent)

            if any(s.tool == "generate_broll" for s in plan.steps) or "cyberpunk" in plan.style_tags:
                report(0.55, "broll")
                try:
                    service.generate_broll(
                        project,
                        style="cyberpunk_city" if "cyberpunk" in plan.style_tags else "abstract",
                        prompt=intent,
                    )
                except Exception as e:  # noqa: BLE001
                    tel.event("broll_skip", error=str(e))

            report(0.6, f"pass {attempt}: composite")
            render_info = service.composite(
                project,
                max_frames=max_render_frames,
                progress=lambda p, m: report(0.6 + 0.25 * p, m),
            )

            report(0.9, f"pass {attempt}: quality gates")
            last_gate = evaluate_gates(project, self.settings)
            pass_logs.append(
                {
                    "pass": attempt,
                    "prompt": prompt,
                    "conf": use_conf,
                    "gate": last_gate.to_dict(),
                    "segment_id": segment_info.get("segment_id"),
                    "render": render_info.get("output_path"),
                }
            )
            tel.event("gate", pass_=attempt, score=last_gate.score, passed=last_gate.passed)
            if last_gate.passed:
                break
            if not self.settings.harness_auto_refine:
                break

        review = service.review(project)
        tel.set_metric("final_gate_score", last_gate.score if last_gate else 0)
        tel.set_metric("passes", len(pass_logs))
        tel.set_metric("review_score", review.get("score"))
        tel_path = project.root / "jobs" / f"telemetry_{tel.id}.json"
        tel.finish()
        tel.save(tel_path)

        graph = EditGraph.from_intent_plan(
            intent,
            [{"tool": s.tool, "args": s.args} for s in plan.steps],
            project_id=project.manifest.id,
        )

        return {
            "ok": bool(last_gate and last_gate.passed),
            "project_id": project.manifest.id,
            "passes": pass_logs,
            "final_gate": last_gate.to_dict() if last_gate else None,
            "segment": segment_info,
            "effects": effects_info,
            "render": render_info,
            "review": review,
            "plan": {
                "intent": intent,
                "subject_prompt": plan.subject_prompt,
                "style_tags": plan.style_tags,
                "steps": [{"tool": s.tool, "rationale": s.rationale} for s in plan.steps],
            },
            "edit_graph": graph.model_dump(mode="json"),
            "telemetry_path": project.rel(tel_path),
            "prompt_candidates": prompt_candidates,
        }

    # ------------------------------------------------------------------
    # Recipes
    # ------------------------------------------------------------------
    def apply_recipe(
        self,
        *,
        video_path: str,
        recipe_name: str,
        project_name: str | None = None,
        max_render_frames: int | None = None,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        from vidmcp.tools import service

        recipe = get_recipe(recipe_name)
        # Creator polish path (audio/captions/BG/export) — fully productized
        if recipe.get("creator_pipeline"):
            from vidmcp.tools.creator import run_talking_head_polish

            return run_talking_head_polish(
                video_path,
                workspace=self.workspace,
                name=project_name or recipe["name"],
                preset=str(recipe.get("preset") or "youtube_16x9"),
                bg_mode=str(recipe.get("bg_mode") or "none"),
                process_audio=bool(recipe.get("process_audio", True)),
                mix_bgm=bool(recipe.get("mix_bgm", True)),
                burn_captions_flag=bool(recipe.get("captions", True)),
                smart_cut=bool(recipe.get("smart_cut", False)),
                aggressiveness=float(recipe.get("aggressiveness", 0.45)),
                infographics=bool(recipe.get("infographics", False)),
                infographic_topic=str(recipe.get("infographic_topic") or "auto"),
                thumbnail=bool(recipe.get("thumbnail", False)),
            )

        project = self.workspace.create_project(name=project_name or recipe["name"])
        tel = TelemetryRun(project.manifest.id, kind=f"recipe:{recipe_name}")

        def report(p: float, msg: str) -> None:
            if progress:
                progress(p, msg)

        report(0.05, "import+analyze")
        service.import_source(project, video_path, bake_orientation=True)
        analysis = service.analyze(project)

        prompts = recipe.get("multi_prompts") or [recipe.get("subject_prompt", "person")]
        alt = recipe.get("alternate_prompts") or []
        conf = float(recipe.get("conf", self.settings.conf_threshold))
        max_passes = int(recipe.get("max_passes", self.settings.harness_max_passes))

        segment_info: dict[str, Any] = {}
        for attempt in range(1, max_passes + 1):
            prompt = prompts[0] if not alt else ([prompts[0]] + alt)[min(attempt - 1, len(alt))]
            if len(prompts) > 1:
                segment_info = service.segment_multi(project, prompts=prompts, conf=conf * (0.9 ** (attempt - 1)))
            else:
                segment_info = service.segment(project, prompt=prompt, conf=conf * (0.9 ** (attempt - 1)))
            pre = evaluate_gates(project, self.settings)
            matte_ok = all(
                c.passed
                for c in pre.checks
                if c.name in ("masks_present", "temporal_stability", "coverage_range") and c.severity == "block"
            )
            if matte_ok or attempt == max_passes:
                break

        if recipe.get("effects"):
            effects = service.apply_effects(project, effect_specs=recipe["effects"], intent=recipe["description"])
        else:
            specs = self.planner.effects_from_tags(recipe.get("style_tags") or ["blur"], recipe["description"])
            effects = service.apply_effects(project, effect_specs=specs, intent=recipe["description"])

        broll = None
        if recipe.get("generate_broll"):
            broll = service.generate_broll(
                project, style=recipe.get("broll_style") or "abstract", prompt=recipe["description"]
            )

        render = service.composite(project, max_frames=max_render_frames)
        gate = evaluate_gates(project, self.settings)
        review = service.review(project)
        tel.event("recipe_done", recipe=recipe_name, gate=gate.score)
        tel.finish()
        tel.save(project.root / "jobs" / f"telemetry_{tel.id}.json")

        return {
            "ok": gate.passed,
            "project_id": project.manifest.id,
            "recipe": recipe,
            "analysis_hints": analysis.get("scene_hints"),
            "segment": segment_info,
            "effects": effects,
            "broll": broll,
            "render": render,
            "gate": gate.to_dict(),
            "review": review,
        }

    # ------------------------------------------------------------------
    # Variants (single matte, many looks)
    # ------------------------------------------------------------------
    def generate_variants(
        self,
        project_id: str,
        *,
        n: int | None = None,
        max_render_frames: int | None = None,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        from vidmcp.tools import service

        project = self.workspace.load_project(project_id)
        if not project.manifest.primary_segment():
            raise RuntimeError("Segment subject before generating variants")
        variants = pick_variants(n or self.settings.harness_variant_count)
        results = []
        for i, v in enumerate(variants):
            if progress:
                progress(i / max(len(variants), 1), f"variant {v['id']}")
            if v.get("effects"):
                specs = v["effects"]
            else:
                specs = self.planner.effects_from_tags(v.get("style_tags") or ["blur"], v.get("label", ""))
            service.apply_effects(project, effect_specs=specs, intent=v.get("label", ""), replace_existing=True)
            if v.get("broll"):
                try:
                    service.generate_broll(project, style=v["broll"], prompt=v.get("label", ""))
                except Exception:  # noqa: BLE001
                    pass
            render = service.composite(
                project,
                output_name=f"variant_{v['id']}.mp4",
                max_frames=max_render_frames,
            )
            results.append({"variant": v, "render": render})
        gate = evaluate_gates(project, self.settings)
        return {
            "ok": True,
            "project_id": project_id,
            "variants": results,
            "gate": gate.to_dict(),
            "message": f"Generated {len(results)} stylistic variants from shared matte",
        }

    # ------------------------------------------------------------------
    # Multi-object segment wrapper
    # ------------------------------------------------------------------
    def segment_multi_objects(
        self,
        project_id: str,
        prompts: list[str],
        conf: float | None = None,
    ) -> dict[str, Any]:
        from vidmcp.tools import service

        project = self.workspace.load_project(project_id)
        return service.segment_multi(project, prompts=prompts, conf=conf)
