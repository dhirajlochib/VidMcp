"""Internal multi-agent pipeline orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vidmcp.agents.critic import CriticAgent
from vidmcp.agents.planner import PlannerAgent
from vidmcp.core.events import Event, get_event_bus
from vidmcp.core.workspace import Workspace
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.orchestrator")
ProgressFn = Callable[[float, str], None]


class PipelineOrchestrator:
    """
    Coordinates Planner → Perception → VFX → Compositor → Critic.

    Host LLMs can call individual MCP tools; this orchestrator supports
    one-shot `run_edit_pipeline` for complex intents.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.planner = PlannerAgent()
        self.critic = CriticAgent()
        self.bus = get_event_bus()

    def run(
        self,
        *,
        video_path: str,
        intent: str,
        project_name: str = "pipeline_edit",
        conf: float = 0.25,
        progress: ProgressFn | None = None,
        max_render_frames: int | None = None,
    ) -> dict[str, Any]:
        # Lazy imports to avoid cycles
        from vidmcp.tools import service

        def report(p: float, msg: str) -> None:
            if progress:
                progress(p, msg)
            log.info("pipeline_progress", progress=p, message=msg)

        report(0.02, "create project")
        project = self.workspace.create_project(name=project_name)
        service.import_source(project, video_path)

        report(0.08, "analyze")
        analysis = service.analyze(project)
        plan = self.planner.plan(intent, analysis)
        self.bus.publish(
            Event(type="plan.created", project_id=project.manifest.id, payload={"steps": [s.tool for s in plan.steps]})
        )

        report(0.15, "segment subject")
        seg = service.segment(project, prompt=plan.subject_prompt, conf=conf, progress=lambda p, m: report(0.15 + 0.35 * p, m))

        report(0.55, "apply effects")
        effect_specs = self.planner.effects_from_tags(plan.style_tags, intent)
        effects = service.apply_effects(project, effect_specs=effect_specs, intent=intent)

        if any(s.tool == "generate_broll" for s in plan.steps):
            report(0.62, "generate broll")
            style = "cyberpunk_city" if "cyberpunk" in plan.style_tags else "abstract"
            service.generate_broll(project, style=style, prompt=intent)

        report(0.7, "composite")
        render = service.composite(
            project,
            progress=lambda p, m: report(0.7 + 0.22 * p, m),
            max_frames=max_render_frames,
        )

        report(0.95, "review")
        review = self.critic.review(project)
        report(1.0, "pipeline complete")

        return {
            "project_id": project.manifest.id,
            "plan": {
                "intent": plan.intent,
                "subject_prompt": plan.subject_prompt,
                "style_tags": plan.style_tags,
                "steps": [{"tool": s.tool, "rationale": s.rationale, "args": s.args} for s in plan.steps],
            },
            "analysis": analysis,
            "segment": seg,
            "effects": effects,
            "render": render,
            "review": review,
        }
