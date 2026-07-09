#!/usr/bin/env python3
"""Education product path demo — TTS narration + Whisper + speech-locked scene + composite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("VIDMCP_SAM_BACKEND", "mock")
os.environ.setdefault("VIDMCP_WORKSPACE_ROOT", str(ROOT / "workspaces"))

from vidmcp.config import get_settings, reset_settings  # noqa: E402
from vidmcp.education.pipeline import run_education_lesson  # noqa: E402
from vidmcp.tools.health import platform_health  # noqa: E402
from vidmcp.utils.logging import setup_logging  # noqa: E402


def synth_talking_head(path: Path, n: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (640, 360))
    rng = np.random.default_rng(21)
    for i in range(n):
        bg = rng.integers(30, 55, (360, 640, 3), dtype=np.uint8)
        cx = 320 + int(8 * np.sin(i / 6))
        cv2.ellipse(bg, (cx, 235), (85, 95), 0, 0, 360, (55, 50, 105), -1)
        cv2.circle(bg, (cx, 148), 52, (185, 165, 145), -1)
        wr.write(bg)
    wr.release()


def main() -> int:
    reset_settings()
    setup_logging("INFO")
    health = platform_health()
    print("=== PLATFORM HEALTH ===")
    for k, v in health["checks"].items():
        print(f"  {k}: {v}")
    print("readiness:", health["readiness"])
    print("next:", health["next_steps"])

    sample = ROOT / "examples" / "fixtures" / "edu_talk.mp4"
    synth_talking_head(sample, n=48)
    topic = "Pythagorean theorem"
    narration = (
        "First we draw a right triangle. "
        "Therefore the square on the legs equals the square on the hypotenuse. "
        "We prove a squared plus b squared equals c squared. "
        "Finally we recap the identity."
    )
    print("\n=== RUN EDUCATION LESSON ===")
    result = run_education_lesson(
        video_path=sample,
        lesson_topic=topic,
        project_name="edu_demo",
        narration=narration,
        max_render_frames=48,
        style="cinematic",
        n_steps=5,
    )
    print("ok:", result.get("ok"))
    print("project:", result.get("project_id"))
    print("ASR:", result.get("asr"))
    print("scene steps:", result.get("scene"))
    print("render:", (result.get("render") or {}).get("absolute_output"))
    print("critics:", result.get("critics"))
    print("provenance:", result.get("provenance"))
    print("timeline:", result.get("timeline"))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
