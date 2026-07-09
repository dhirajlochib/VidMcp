#!/usr/bin/env python3
"""VidMCP v0.4 ultimate platform demo — all advanced systems."""

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


def synth(path: Path, n: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (640, 360))
    rng = np.random.default_rng(9)
    for i in range(n):
        bg = rng.integers(28, 50, (360, 640, 3), dtype=np.uint8)
        cx = 320 + int(8 * np.sin(i / 6))
        cv2.ellipse(bg, (cx, 230), (85, 95), 0, 0, 360, (55, 50, 105), -1)
        cv2.circle(bg, (cx, 145), 52, (185, 165, 145), -1)
        wr.write(bg)
    wr.release()


def main() -> int:
    reset_settings()
    setup_logging("INFO")
    sample = ROOT / "examples" / "fixtures" / "ultimate.mp4"
    synth(sample)

    # Use MCP-level ultimate via service composition
    from vidmcp.server import create_server

    # direct service path for offline demo
    ws = Workspace(get_settings())
    deb = adv.debate("cyberpunk math lesson with particles behind speaker")
    print("DEBATE winner:", deb["recommended"])
    for s in deb["strategies"]:
        print(f"  {s['id']}: overall={s['overall']} q={s['scores']['quality']}")

    project = ws.create_project("ultimate_demo")
    service.import_source(project, sample)
    service.analyze(project)
    adv.graph_commit(project, "start", {"intent": "ultimate"})
    seg = service.segment(project, prompt="person")
    print("segment stab:", round(seg.get("temporal_stability", 0), 3))
    uref = adv.uncertainty_guided_refine(project, service)
    print("uncertainty hot frames:", uref.get("uncertainty", {}).get("hot_frames"))
    scene = service.render_math_scene(project, prompt="Prove a^2+b^2=c^2", engine="procedural")
    print("scene:", scene.get("engine"), scene.get("scene_path"))
    audio = adv.audio_sync_project(project, transcript="therefore we prove the theorem equals truth", keywords=["prove", "equals", "therefore"])
    print("audio events:", audio.get("sync", {}).get("n_events"))
    service.apply_effects(
        project,
        effect_specs=[{"effect_type": "cyberpunk", "kind": "background", "params": {}, "name": "cp"}],
        replace_existing=False,
    )
    fog = adv.depth_fog_project(project, style="fog", density=0.4, max_frames=40)
    light = adv.lighting_match_project(project, max_frames=40)
    render = service.composite(project, max_frames=40)
    critics = adv.critic_project(project, workspace_root=get_settings().workspace_root)
    signed = adv.sign(project)
    lesson = adv.lesson("Bayes theorem for creators", duration_sec=120)
    viddsl = lesson["viddsl"]
    print("--- VidDSL scaffold ---\n", viddsl)
    # live latency
    live = adv.live_start()
    live_out = project.renders_dir / "live_bench.mp4"
    live_r = adv.live_process_file(live["id"], str(sample), str(live_out), max_frames=30, effect="cyberpunk")
    mined = adv.failures(ws)
    glog = adv.graph_log(project)

    print("\n=== RESULTS ===")
    print("project:", project.manifest.id)
    print("render:", render.get("absolute_output"))
    print("critics overall:", critics.get("overall_score"), "failed:", critics.get("failed_axes"))
    print("fix_route:", critics.get("fix_route"))
    print("provenance:", signed.get("manifest_path"), "valid:", signed.get("verify", {}).get("valid"))
    print("live p95 ms:", live_r.get("p95_latency_ms"))
    print("graph commits:", len(glog.get("log") or []))
    print("heuristics:", [h["heuristic"] for h in mined.get("heuristics", {}).get("suggestions", [])])
    print("lesson beats:", len(lesson["beats"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
