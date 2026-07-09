#!/usr/bin/env python3
"""Talking-head + math scene + keyframe refine demo (v0.3)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["VIDMCP_SAM_BACKEND"] = "mock"
os.environ["VIDMCP_WORKSPACE_ROOT"] = str(ROOT / "workspaces")

from vidmcp.config import get_settings, reset_settings  # noqa: E402
from vidmcp.core.workspace import Workspace  # noqa: E402
from vidmcp.harness.quality_gates import evaluate_gates  # noqa: E402
from vidmcp.perception.weights import describe_weights_status  # noqa: E402
from vidmcp.tools import service  # noqa: E402
from vidmcp.utils.logging import setup_logging  # noqa: E402


def synth(path: Path, n: int = 36) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (640, 360))
    rng = np.random.default_rng(3)
    for i in range(n):
        bg = rng.integers(30, 55, (360, 640, 3), dtype=np.uint8)
        cx = 320 + int(10 * np.sin(i / 5))
        cv2.ellipse(bg, (cx, 240), (90, 100), 0, 0, 360, (55, 50, 110), -1)
        cv2.circle(bg, (cx, 150), 55, (190, 170, 150), -1)
        wr.write(bg)
    wr.release()


def main() -> int:
    reset_settings()
    setup_logging("INFO")
    print("SAM weights status:", describe_weights_status())
    sample = ROOT / "examples" / "fixtures" / "math_talk.mp4"
    synth(sample)
    ws = Workspace(get_settings())
    project = ws.create_project("math_lesson_demo")
    service.import_source(project, sample)
    service.analyze(project)
    seg = service.segment(project, prompt="person")
    print("segment stability:", seg.get("temporal_stability"))
    refined = service.refine_segment_keyframes(project, auto_detect=True, prompt="person")
    print("refine delta stability:", refined["improvement"]["temporal_stability"])
    scene = service.render_math_scene(
        project,
        prompt="Animate the Pythagorean theorem a^2 + b^2 = c^2",
        engine="procedural",
        place_as_background=True,
    )
    print("scene engine:", scene["engine"], "path:", scene["scene_path"])
    service.apply_effects(
        project,
        effect_specs=[
            {
                "effect_type": "color_grade",
                "kind": "grade",
                "params": {"contrast": 1.1, "saturation": 1.05, "background_only": True},
                "name": "grade",
            }
        ],
        replace_existing=False,
    )
    render = service.composite(project, max_frames=36)
    gate = evaluate_gates(project, get_settings())
    print("render:", render.get("absolute_output"))
    print("gate score:", gate.score, "passed:", gate.passed)
    return 0 if gate.score > 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
