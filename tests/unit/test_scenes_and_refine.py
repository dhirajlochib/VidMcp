from pathlib import Path

import cv2
import numpy as np

from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace
from vidmcp.perception.keyframe_refine import detect_weak_keyframes, refine_masks_local
from vidmcp.scenes.engine import SceneEngine
from vidmcp.scenes.sandbox import SandboxError, validate_scene_source
from vidmcp.tools import service


def test_sandbox_blocks_os():
    try:
        validate_scene_source("import os\nos.system('x')")
        assert False, "should block"
    except SandboxError:
        pass


def test_procedural_scene(tmp_path: Path):
    eng = SceneEngine(tmp_path / "scenes")
    r = eng.compile_and_render(prompt="Animate Pythagoras theorem", engine="procedural", duration_sec=1.0, fps=10)
    assert r.output_path.exists()
    assert r.engine == "procedural"


def test_keyframe_refine(tmp_path: Path):
    # fake masks with a flicker
    md = tmp_path / "masks"
    md.mkdir()
    for i in range(12):
        m = np.zeros((64, 64), dtype=np.uint8)
        if i != 5:
            m[20:40, 20:40] = 255
        else:
            m[5:15, 5:15] = 255  # jump
        cv2.imwrite(str(md / f"mask_{i:06d}.png"), m)
    kfs = detect_weak_keyframes(md, max_keyframes=3)
    assert 5 in kfs or kfs[0] == 0
    # synthetic video
    vid = tmp_path / "v.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for i in range(12):
        f = np.full((64, 64, 3), 40, dtype=np.uint8)
        cv2.circle(f, (30, 30), 12, (180, 160, 140), -1)
        wr.write(f)
    wr.release()
    out = tmp_path / "out"
    res = refine_masks_local(vid, md, output_dir=out, keyframes=kfs, prompt="person")
    assert res.mask_dir.exists()
    assert res.temporal_stability_after >= 0.0


def test_math_scene_on_project(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    ws = Workspace(Settings(workspace_root=tmp_path, sam_backend="mock"))
    store = ws.create_project("m")
    # minimal source
    vid = tmp_path / "s.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    for i in range(10):
        f = np.full((180, 320, 3), 50, dtype=np.uint8)
        cv2.circle(f, (160, 90), 30, (200, 180, 160), -1)
        wr.write(f)
    wr.release()
    service.import_source(store, vid)
    out = service.render_math_scene(store, prompt="Plot a parabola", engine="procedural", duration_sec=1.0)
    assert out["layer_id"]
    assert (store.root / out["scene_path"]).exists()
