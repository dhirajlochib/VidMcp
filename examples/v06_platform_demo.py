#!/usr/bin/env python3
"""v0.6 platform demo: queue, diarize, meshy, remotion, marketplace, review UI."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["VIDMCP_SAM_BACKEND"] = "mock"
os.environ["VIDMCP_WORKSPACE_ROOT"] = str(ROOT / "workspaces")

from vidmcp.config import get_settings, reset_settings  # noqa: E402
from vidmcp.core.workspace import Workspace  # noqa: E402
from vidmcp.marketplace.registry import RecipeMarketplace  # noqa: E402
from vidmcp.tools import advanced_service as adv  # noqa: E402
from vidmcp.tools import service  # noqa: E402
from vidmcp.utils.logging import setup_logging  # noqa: E402


def synth(path: Path, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (480, 270))
    for i in range(n):
        f = np.full((270, 480, 3), 50, dtype=np.uint8)
        cv2.circle(f, (240, 120), 40, (190, 170, 150), -1)
        wr.write(f)
    wr.release()


def main() -> int:
    reset_settings()
    setup_logging("INFO")
    sample = ROOT / "examples" / "fixtures" / "v06.mp4"
    synth(sample)
    ws = Workspace(get_settings())
    project = ws.create_project("v06_platform")
    service.import_source(project, sample)
    service.segment(project, prompt="person")

    # marketplace
    mp = RecipeMarketplace(get_settings().workspace_root)
    inst = mp.install_from_path(ROOT / "examples" / "marketplace_recipes" / "neon_classroom.json")
    print("installed recipe:", inst)

    # meshy plate (fallback)
    meshy = adv.meshy_plate(project, "floating neon torus math sculpture", duration_sec=2.0)
    print("meshy backend:", meshy.get("backend"), meshy.get("project_plate"))

    # remotion scaffold
    rem = adv.remotion_scaffold(project, "Explain Bayes theorem visually")
    print("remotion:", rem.get("project_dir_rel"))

    # diarize
    dia = adv.diarize_project(project, n_speakers=2)
    print("diarize:", dia.get("backend"), "speakers:", dia.get("speakers"))

    # queue composite
    q = adv.enqueue_job("composite", {"project_id": project.manifest.id, "max_frames": 20})
    print("enqueued:", q.get("job_id"))
    w = adv.queue_worker_start(background=False, max_jobs=1)
    print("worker:", w)
    st = adv.queue_status(job_id=q["job_id"])
    print("job status:", (st.get("job") or {}).get("status"))

    # review UI
    ui = adv.review_ui_start(port=18766)
    print("review UI:", ui.get("url"))
    time.sleep(0.3)
    print("review running:", adv.review_ui_status().get("running"))

    service.composite(project, max_frames=16)
    signed = adv.sign(project)
    print("signed:", signed.get("manifest_path"))
    print("project:", project.manifest.id)
    print("Open review UI while server lives:", ui.get("url"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
