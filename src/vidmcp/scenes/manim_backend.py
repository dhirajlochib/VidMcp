"""Manim Community backend for code → math/motion video."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from vidmcp.scenes.sandbox import validate_scene_source
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.scenes.manim")


def manim_available() -> bool:
    try:
        import manim  # noqa: F401

        return True
    except ImportError:
        return shutil.which("manim") is not None


def build_manim_source_from_prompt(prompt: str, class_name: str = "VidMCPScene") -> str:
    """Generate a safe Manim script from a natural-language math/education prompt."""
    # Escape for embedding in Python string
    safe = prompt.replace("\\", "\\\\").replace('"""', "'")
    return textwrap.dedent(
        f'''
        from manim import *

        class {class_name}(Scene):
            def construct(self):
                title = Text("""{safe[:120]}""", font_size=36)
                title.to_edge(UP)
                self.play(Write(title), run_time=1.2)

                # Core mathematical demonstration scaffold
                axes = Axes(
                    x_range=[-3, 3, 1],
                    y_range=[-2, 2, 1],
                    x_length=8,
                    y_length=4.5,
                    tips=False,
                ).scale(0.85).shift(DOWN * 0.3)
                graph = axes.plot(lambda x: 0.25 * x ** 2 - 0.5, color=BLUE)
                formula = MathTex(r"f(x)=\\\\frac{{1}}{{4}}x^2-\\\\frac{{1}}{{2}}", font_size=40)
                formula.next_to(axes, DOWN, buff=0.35)

                self.play(Create(axes), run_time=0.8)
                self.play(Create(graph), Write(formula), run_time=1.4)
                self.play(Indicate(formula), run_time=0.6)

                note = Text("VidMCP · code-generated scene", font_size=22, color=GRAY_B)
                note.to_edge(DOWN)
                self.play(FadeIn(note), run_time=0.5)
                self.wait(0.8)
        '''
    ).strip() + "\n"


def inject_user_construct(user_body: str, class_name: str = "VidMCPScene") -> str:
    """Wrap a user-provided construct body or full scene class into a file."""
    validate_scene_source(user_body)
    if "class " in user_body and "Scene" in user_body:
        # full script — still validate
        if "from manim import" not in user_body and "import manim" not in user_body:
            return "from manim import *\n\n" + user_body
        return user_body
    # treat as construct() body
    body = textwrap.indent(user_body.strip(), "        ")
    return textwrap.dedent(
        f"""
        from manim import *

        class {class_name}(Scene):
            def construct(self):
        """
    ).rstrip() + "\n" + body + "\n"


def render_manim(
    source: str,
    *,
    out_dir: Path,
    class_name: str = "VidMCPScene",
    quality: str = "l",  # l=480p15, m=720p30, h=1080p60
    timeout_sec: float = 180.0,
) -> Path:
    validate_scene_source(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "scene.py"
    script.write_text(source, encoding="utf-8")

    media_dir = out_dir / "media"
    cmd = [
        sys.executable,
        "-m",
        "manim",
        "render",
        str(script),
        class_name,
        f"-q{quality}",
        "--media_dir",
        str(media_dir),
        "-o",
        "scene_out",
        "--disable_caching",
    ]
    log.info("manim_render_start", cmd=" ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(out_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "MANIM_DISABLE_CACHING": "1"},
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Manim timed out after {timeout_sec}s") from e
    except FileNotFoundError as e:
        raise RuntimeError("Manim not installed. pip install 'manim'") from e

    if proc.returncode != 0:
        raise RuntimeError(
            f"Manim failed ({proc.returncode}):\n{(proc.stderr or proc.stdout)[-2000:]}"
        )

    # find mp4
    videos = list(media_dir.rglob("*.mp4"))
    if not videos:
        raise RuntimeError("Manim produced no mp4")
    # prefer scene_out
    videos.sort(key=lambda p: ("scene_out" not in p.name, -p.stat().st_mtime))
    final = out_dir / "render.mp4"
    shutil.copy2(videos[0], final)
    log.info("manim_render_done", path=str(final))
    return final
