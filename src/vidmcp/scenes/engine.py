"""Unified scene engine: Manim (preferred) or procedural fallback → project layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from vidmcp.scenes.manim_backend import (
    build_manim_source_from_prompt,
    inject_user_construct,
    manim_available,
    render_manim,
)
from vidmcp.scenes.procedural_backend import render_procedural_math_scene
from vidmcp.scenes.blender_backend import blender_available, render_blender_scene
from vidmcp.scenes.sandbox import ensure_safe_prompt_slug, validate_scene_source
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.scenes.engine")

EngineName = Literal["auto", "manim", "procedural", "blender"]


@dataclass
class SceneRenderResult:
    scene_id: str
    engine: str
    output_path: Path
    source_path: Path | None = None
    prompt: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "engine": self.engine,
            "output_path": str(self.output_path),
            "source_path": str(self.source_path) if self.source_path else None,
            "prompt": self.prompt,
            "meta": self.meta,
        }


class SceneEngine:
    def __init__(self, work_root: Path):
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)

    def compile_and_render(
        self,
        *,
        prompt: str | None = None,
        source: str | None = None,
        engine: EngineName = "auto",
        width: int = 1280,
        height: int = 720,
        fps: float = 24.0,
        duration_sec: float | None = None,
        manim_quality: str = "l",
        class_name: str = "VidMCPScene",
    ) -> SceneRenderResult:
        if not prompt and not source:
            raise ValueError("Provide prompt and/or source code")
        scene_id = str(uuid4())
        out_dir = self.work_root / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = ensure_safe_prompt_slug(prompt or "custom_scene")

        use = engine
        if use == "auto":
            use = "manim" if manim_available() else "procedural"

        if use == "manim":
            try:
                if source:
                    validate_scene_source(source)
                    code = inject_user_construct(source, class_name=class_name)
                else:
                    code = build_manim_source_from_prompt(prompt or "Math scene", class_name=class_name)
                src_path = out_dir / "scene.py"
                src_path.write_text(code, encoding="utf-8")
                mp4 = render_manim(code, out_dir=out_dir, class_name=class_name, quality=manim_quality)
                return SceneRenderResult(
                    scene_id=scene_id,
                    engine="manim",
                    output_path=mp4,
                    source_path=src_path,
                    prompt=prompt or "",
                    meta={"slug": slug, "class_name": class_name},
                )
            except Exception as e:  # noqa: BLE001
                log.warning("manim_failed_fallback_procedural", error=str(e))
                if engine == "manim":
                    raise
                use = "procedural"


        if use == "blender" or (engine == "auto" and blender_available() and "3d" in (prompt or "").lower()):
            try:
                bp = render_blender_scene(prompt or "Abstract sculpture", out_dir=out_dir, frames=max(24, int((duration_sec or 2) * 24)))
                if bp is not None:
                    return SceneRenderResult(
                        scene_id=scene_id,
                        engine="blender",
                        output_path=bp,
                        source_path=out_dir / "blender_scene.py",
                        prompt=prompt or "",
                        meta={"slug": slug, "blender": True},
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("blender_scene_failed", error=str(e))
                if use == "blender":
                    use = "procedural"

        # procedural
        mp4 = out_dir / "render.mp4"
        render_procedural_math_scene(
            prompt or "Mathematical visualization",
            out_path=mp4,
            width=width,
            height=height,
            fps=fps,
            duration_sec=duration_sec,
        )
        src_path = None
        if source:
            # store user source even if procedural used
            try:
                validate_scene_source(source)
                src_path = out_dir / "scene_user.py"
                src_path.write_text(source, encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        return SceneRenderResult(
            scene_id=scene_id,
            engine="procedural",
            output_path=mp4,
            source_path=src_path,
            prompt=prompt or "",
            meta={"slug": slug, "fallback": engine == "auto"},
        )
