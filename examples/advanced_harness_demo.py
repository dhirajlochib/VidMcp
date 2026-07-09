#!/usr/bin/env python3
"""Advanced harness demo: quality-gated pipeline + variants + diagnostics."""

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
from vidmcp.harness.runtime import HarnessRuntime  # noqa: E402
from vidmcp.tools import service  # noqa: E402
from vidmcp.utils.logging import setup_logging  # noqa: E402


def synth(path: Path, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 360
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (w, h))
    rng = np.random.default_rng(1)
    for i in range(n):
        bg = rng.integers(25, 60, (h, w, 3), dtype=np.uint8)
        cx = int(w * 0.5 + 12 * np.sin(i / 6))
        cv2.ellipse(bg, (cx, 220), (80, 100), 0, 0, 360, (50, 45, 100), -1)
        cv2.circle(bg, (cx, 140), 55, (185, 165, 145), -1)
        wr.write(bg)
    wr.release()


def main() -> int:
    reset_settings()
    setup_logging("INFO")
    sample = ROOT / "examples" / "fixtures" / "harness_talk.mp4"
    synth(sample)
    rt = HarnessRuntime(Workspace(get_settings()), get_settings())
    print("=== Quality-gated pipeline ===")
    result = rt.run_quality_gated_pipeline(
        video_path=str(sample),
        intent="cyberpunk with dramatic particles behind the speaker",
        max_render_frames=30,
        max_passes=2,
    )
    pid = result["project_id"]
    print("project:", pid)
    print("ok:", result["ok"], "passes:", len(result["passes"]))
    print("gate score:", (result.get("final_gate") or {}).get("score"))
    print("render:", (result.get("render") or {}).get("absolute_output"))
    print("telemetry:", result.get("telemetry_path"))

    print("=== Matte diagnostics ===")
    store = Workspace(get_settings()).load_project(pid)
    diag = service.matte_diagnostics(store)
    print("flicker_count:", diag["flicker_count"], "coverage_mean:", diag["coverage_mean"])

    print("=== Variants ===")
    variants = rt.generate_variants(pid, n=2, max_render_frames=20)
    print("n variants:", len(variants["variants"]))
    if len(store.manifest.renders) >= 2:
        cmp_ = service.compare_renders(store)
        print("variant mean_diff:", cmp_["mean_diff"])

    print("=== Recipe ===")
    rec = rt.apply_recipe(video_path=str(sample), recipe_name="rain_noir", max_render_frames=20)
    print("recipe ok:", rec.get("ok"), "render:", (rec.get("render") or {}).get("absolute_output"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
