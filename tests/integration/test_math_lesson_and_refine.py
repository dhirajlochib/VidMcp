from pathlib import Path

import cv2
import numpy as np
import pytest

from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace
from vidmcp.tools import service


def _synth(path: Path, n: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (480, 270))
    for i in range(n):
        f = np.full((270, 480, 3), 45, dtype=np.uint8)
        cv2.ellipse(f, (240, 180), (60, 70), 0, 0, 360, (50, 40, 90), -1)
        cv2.circle(f, (240 + i % 3, 110), 40, (185, 165, 145), -1)
        wr.write(f)
    wr.release()


@pytest.mark.integration
def test_segment_refine_scene_composite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    settings = Settings(workspace_root=tmp_path, sam_backend="mock")
    ws = Workspace(settings)
    store = ws.create_project("lesson")
    vid = tmp_path / "talk.mp4"
    _synth(vid)
    service.import_source(store, vid)
    seg = service.segment(store, prompt="person")
    assert seg["segment_id"]
    refined = service.refine_segment_keyframes(store, auto_detect=True, prompt="person")
    assert refined["segment_id"] != seg["segment_id"]
    scene = service.render_math_scene(store, prompt="Explain the unit circle", engine="procedural", duration_sec=1.5)
    assert scene["placed_as_background"]
    service.apply_effects(
        store,
        effect_specs=[{"effect_type": "blur", "kind": "background", "params": {"blur_radius": 15}, "name": "soft"}],
        replace_existing=False,
    )
    render = service.composite(store, max_frames=20)
    assert Path(render["absolute_output"]).exists()
