from pathlib import Path

import cv2
import numpy as np

from vidmcp.advanced.shot_detect import detect_shots
from vidmcp.audio.speech_lock import plan_speech_locked_steps, render_speech_locked_scene
from vidmcp.audio.whisper_timeline import transcribe_words, words_to_keyword_events
from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace
from vidmcp.depth.enhanced import multi_cue_depth
from vidmcp.dsl.viddsl import compile_viddsl
from vidmcp.tools import advanced_service as adv
from vidmcp.tools import service


def test_word_fallback_and_keywords(tmp_path: Path):
    vid = tmp_path / "a.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 90))
    for i in range(20):
        f = np.zeros((90, 160, 3), dtype=np.uint8)
        wr.write(f)
    wr.release()
    # may fail audio extract on silent mp4v without audio - handle
    r = transcribe_words(vid, work_dir=tmp_path / "w", fallback_transcript="prove therefore equals done")
    # if no audio stream, ok false - still test keyword helper
    words = r.get("words") or [
        {"word": "prove", "start": 0.1, "end": 0.3, "prob": 1},
        {"word": "equals", "start": 0.5, "end": 0.7, "prob": 1},
    ]
    ev = words_to_keyword_events(words, ["prove", "equals"])
    assert len(ev) >= 1
    steps = plan_speech_locked_steps(words, n_steps=3)
    assert len(steps) == 3
    out = tmp_path / "sc.mp4"
    ren = render_speech_locked_scene("Pythagoras proof", steps, out_path=out, width=320, height=180, fps=10)
    assert Path(ren["output_path"]).exists()


def test_depth_and_shots(tmp_path: Path):
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    cv2.circle(frame, (80, 45), 25, (200, 180, 160), -1)
    mask = np.zeros((90, 160), dtype=np.uint8)
    cv2.circle(mask, (80, 45), 25, 255, -1)
    d = multi_cue_depth(frame, mask)
    assert d.shape == (90, 160)
    assert d[45, 80] < d[5, 5]  # subject nearer than corner

    vid = tmp_path / "s.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 90))
    for i in range(30):
        f = np.full((90, 160, 3), 30 if i < 15 else 200, dtype=np.uint8)
        wr.write(f)
    wr.release()
    shots = detect_shots(vid, threshold=0.3, min_shot_len=3)
    assert shots["shot_count"] >= 1


def test_enhance_edit_stack(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    ws = Workspace(Settings(workspace_root=tmp_path, sam_backend="mock"))
    store = ws.create_project("e")
    vid = tmp_path / "t.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (320, 180))
    for i in range(24):
        f = np.full((180, 320, 3), 45, dtype=np.uint8)
        cv2.circle(f, (160, 90), 30, (190, 170, 150), -1)
        wr.write(f)
    wr.release()
    service.import_source(store, vid)
    service.segment(store, prompt="person")
    service.render_math_scene(store, prompt="unit circle", engine="procedural", duration_sec=1.0)
    otio = adv.export_otio(store)
    assert otio.get("ok")
    shots = adv.detect_project_shots(store, threshold=0.5)
    assert shots.get("ok")
    depth = adv.compute_depth(store, max_frames=12, prefer_midas=False)
    assert depth.get("ok")
    flow = adv.flow_reproject(store, max_frames=12)
    assert flow.get("ok")
    ah = adv.apply_auto_heuristics(store, service, max_frames=12)
    assert ah.get("ok")
