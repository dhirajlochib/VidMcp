#!/usr/bin/env python3
"""v0.5 enhancement demo: speech lock, depth, flow, shots, OTIO, auto-heuristics."""

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
from vidmcp.tools import advanced_service as adv  # noqa: E402
from vidmcp.tools import service  # noqa: E402
from vidmcp.utils.logging import setup_logging  # noqa: E402


def synth(path: Path, n: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (640, 360))
    rng = np.random.default_rng(11)
    for i in range(n):
        # two "shots"
        base = 35 if i < 24 else 70
        bg = rng.integers(base, base + 25, (360, 640, 3), dtype=np.uint8)
        cx = 320 + int(6 * np.sin(i / 5))
        cv2.ellipse(bg, (cx, 230), (80, 90), 0, 0, 360, (55, 50, 105), -1)
        cv2.circle(bg, (cx, 145), 50, (185, 165, 145), -1)
        wr.write(bg)
    wr.release()


def main() -> int:
    reset_settings()
    setup_logging("INFO")
    sample = ROOT / "examples" / "fixtures" / "v05.mp4"
    synth(sample)
    ws = Workspace(get_settings())
    project = ws.create_project("v05_enhance")
    service.import_source(project, sample)
    service.analyze(project)
    service.segment(project, prompt="person")

    words = adv.word_timeline(
        project,
        fallback_transcript="first we define the problem therefore we prove a equals b finally it follows",
    )
    print("ASR backend:", words.get("backend"), "words:", len(words.get("words") or []))

    speech = adv.speech_locked_scene(
        project,
        "Pythagorean theorem proof",
        n_steps=5,
        keywords=["first", "therefore", "prove", "equals", "finally"],
        fallback_transcript="first we define therefore we prove equals finally",
    )
    print("speech steps:", len(speech.get("steps") or []), "path:", speech.get("scene_path"))

    shots = adv.detect_project_shots(project, threshold=0.35)
    print("shots:", shots.get("shot_count"), shots.get("cut_frames"))

    depth = adv.compute_depth(project, max_frames=24, prefer_midas=False)
    print("depth backend:", depth.get("backend"))

    fog = adv.depth_fog_project(project, style="fog", max_frames=24)
    light = adv.lighting_match_project(project, max_frames=24)
    flow = adv.flow_reproject(project, max_frames=24)
    print("flow:", flow.get("project_relative"))

    service.composite(project, max_frames=24)
    auto = adv.apply_auto_heuristics(project, service, max_frames=24)
    print("auto heuristics actions:", [a.get("heuristic") for a in auto.get("actions_taken") or []])
    print("critics after:", auto.get("critics_after"))

    signed = adv.sign(project)
    timeline = adv.export_otio(project)
    print("signed:", signed.get("manifest_path"), "valid:", signed.get("verify", {}).get("valid"))
    print("timeline:", timeline.get("json_path"))
    print("project:", project.manifest.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
