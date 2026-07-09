from pathlib import Path

import cv2
import numpy as np
import pytest

from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace
from vidmcp.dsl.viddsl import compile_viddsl
from vidmcp.tools import advanced_service as adv
from vidmcp.tools import service


def _vid(path: Path, n: int = 20):
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (320, 180))
    for i in range(n):
        f = np.full((180, 320, 3), 50, dtype=np.uint8)
        cv2.circle(f, (160 + i % 2, 80), 28, (190, 170, 150), -1)
        wr.write(f)
    wr.release()


@pytest.mark.integration
def test_viddsl_run_and_sign(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    settings = Settings(workspace_root=tmp_path, sam_backend="mock")
    ws = Workspace(settings)
    store = ws.create_project("dsl")
    v = tmp_path / "t.mp4"
    _vid(v)
    service.import_source(store, v)
    src = '''
    track person as S
    scene procedural("unit circle") as B
    composite B under S
    gate score >= 0.3
    sign
    '''
    # gate metric stability maps to score in runner
    src = src.replace("score", "stability")
    out = adv.run_dsl(store, src, service, max_render_frames=12)
    assert out.get("ok") is not False
    assert any(s["op"] == "sign" for s in out["steps"])


@pytest.mark.integration
def test_ultimate_pieces(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    settings = Settings(workspace_root=tmp_path, sam_backend="mock")
    ws = Workspace(settings)
    store = ws.create_project("ult")
    v = tmp_path / "t.mp4"
    _vid(v, 24)
    service.import_source(store, v)
    service.segment(store, prompt="person")
    service.render_math_scene(store, prompt="Fourier series", engine="procedural", duration_sec=1.0)
    adv.audio_sync_project(store, keywords=["wave"])
    fog = adv.depth_fog_project(store, max_frames=12)
    assert Path(fog["output_path"]).exists()
    light = adv.lighting_match_project(store, max_frames=12)
    assert Path(light["output_path"]).exists()
    service.composite(store, max_frames=12)
    signed = adv.sign(store)
    assert signed.get("ok")
    deb = adv.debate("cyberpunk math lesson behind speaker")
    assert deb["recommended"]
    mined = adv.failures(ws)
    assert "heuristics" in mined
