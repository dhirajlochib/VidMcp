#!/usr/bin/env python3
"""End-to-end sample: synthetic talking-head-ish clip → cyberpunk behind-subject edit."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidmcp.agents.orchestrator import PipelineOrchestrator  # noqa: E402
from vidmcp.config import get_settings, reset_settings  # noqa: E402
from vidmcp.core.workspace import Workspace  # noqa: E402
from vidmcp.utils.logging import setup_logging  # noqa: E402


def make_synthetic_talking_head(path: Path, *, n_frames: int = 48, fps: float = 12.0) -> Path:
    """Create a simple moving 'person' ellipse on a textured background."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        bg = rng.integers(30, 70, (h, w, 3), dtype=np.uint8)
        # room-ish gradient
        for y in range(h):
            bg[y, :, :] = np.clip(bg[y, :, :] + y // 8, 0, 255)
        cx = int(w * 0.5 + 18 * np.sin(i / 8))
        cy = int(h * 0.48 + 6 * np.cos(i / 10))
        # torso
        cv2.ellipse(bg, (cx, cy + 70), (70, 90), 0, 0, 360, (40, 40, 90), -1)
        # head
        cv2.circle(bg, (cx, cy), 48, (180, 160, 140), -1)
        cv2.circle(bg, (cx - 15, cy - 5), 6, (30, 30, 30), -1)
        cv2.circle(bg, (cx + 15, cy - 5), 6, (30, 30, 30), -1)
        writer.write(bg)
    writer.release()
    return path


def main() -> int:
    reset_settings()
    setup_logging("INFO")
    # force mock for portable demo
    import os

    os.environ["VIDMCP_SAM_BACKEND"] = "mock"
    os.environ["VIDMCP_WORKSPACE_ROOT"] = str(ROOT / "workspaces")
    reset_settings()

    sample = ROOT / "examples" / "fixtures" / "synthetic_talk.mp4"
    make_synthetic_talking_head(sample, n_frames=36, fps=12)

    print("Running pipeline on", sample)
    orch = PipelineOrchestrator(Workspace(get_settings()))
    result = orch.run(
        video_path=str(sample),
        intent="Turn this talking-head video into a cyberpunk style with dramatic particle effects behind the speaker",
        project_name="cyberpunk_demo",
        max_render_frames=36,
    )
    print("OK:", result.get("ok", True))
    print("project_id:", result["project_id"])
    print("backend segment:", result["segment"].get("backend"))
    print("render:", result["render"].get("absolute_output"))
    print("review score:", result["review"].get("score"), "passed:", result["review"].get("passed"))
    print("plan steps:", [s["tool"] for s in result["plan"]["steps"]])
    return 0 if result["review"].get("score", 0) >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
