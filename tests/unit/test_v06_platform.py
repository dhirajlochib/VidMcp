from pathlib import Path

import cv2
import numpy as np

from vidmcp.audio.diarize import diarize_video
from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace
from vidmcp.integrations.meshy import text_to_3d_plate
from vidmcp.integrations.remotion import scaffold_remotion_scene
from vidmcp.marketplace.registry import RecipeMarketplace
from vidmcp.queue.worker import JobQueue
from vidmcp.review.app import start_review_server, stop_review_server, get_review_state
from vidmcp.tools import advanced_service as adv
from vidmcp.tools import service


def test_job_queue_inline(tmp_path: Path, monkeypatch):
    from vidmcp.config import reset_settings

    monkeypatch.setenv("VIDMCP_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    reset_settings()
    settings = Settings(workspace_root=tmp_path, sam_backend="mock")
    ws = Workspace(settings)
    store = ws.create_project("q")
    vid = tmp_path / "v.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 90))
    for i in range(12):
        f = np.full((90, 160, 3), 40, dtype=np.uint8)
        cv2.circle(f, (80, 45), 20, (180, 160, 140), -1)
        wr.write(f)
    wr.release()
    service.import_source(store, vid)
    q = JobQueue(tmp_path)
    from vidmcp.queue import worker as wmod

    wmod._register_default_handlers(q)
    # segment via queue needs project
    service.segment(store, prompt="person")
    job = q.enqueue("composite", {"project_id": store.manifest.id, "max_frames": 8})
    done = q.process_one()
    assert done is not None
    assert done["status"] == "done"


def test_marketplace_and_meshy_remotion(tmp_path: Path):
    mp = RecipeMarketplace(tmp_path)
    r = mp.publish(
        {
            "name": "test_pack",
            "description": "x",
            "subject_prompt": "person",
            "style_tags": ["blur"],
            "effects": [{"effect_type": "blur", "kind": "background", "params": {"blur_radius": 10}, "name": "b"}],
        },
        author="unit",
    )
    assert r["ok"]
    names = {x["name"] for x in mp.list_all()}
    assert "test_pack" in names

    plate = text_to_3d_plate("neon cube", out_dir=tmp_path / "m", width=320, height=180, duration_sec=0.5, fps=10)
    assert Path(plate["plate_path"]).exists()

    sc = scaffold_remotion_scene("Fourier intro", out_dir=tmp_path / "rem")
    assert Path(sc["entry"]).exists()


def test_diarize_no_audio(tmp_path: Path):
    vid = tmp_path / "s.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 90))
    for i in range(10):
        wr.write(np.zeros((90, 160, 3), dtype=np.uint8))
    wr.release()
    r = diarize_video(vid, work_dir=tmp_path / "d", n_speakers=2, words=[{"word": "hi", "start": 0, "end": 0.2}])
    assert r["ok"]
    assert r["n_speakers"] >= 1


def test_review_ui_start_stop(tmp_path: Path):
    # use high port
    st = start_review_server(tmp_path, port=18765)
    assert st.get("ok")
    assert get_review_state().get("running")
    stop_review_server()
