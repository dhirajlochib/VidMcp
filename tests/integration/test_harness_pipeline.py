from pathlib import Path

import cv2
import numpy as np
import pytest

from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace
from vidmcp.harness.runtime import HarnessRuntime


def _synth(path: Path, n: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 320, 180
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), 50, dtype=np.uint8)
        cv2.circle(frame, (160 + (i % 5), 90), 35, (190, 170, 150), -1)
        cv2.ellipse(frame, (160, 140), (50, 40), 0, 0, 360, (40, 40, 100), -1)
        wr.write(frame)
    wr.release()


@pytest.mark.integration
def test_quality_gated_and_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    monkeypatch.setenv("VIDMCP_WORKSPACE_ROOT", str(tmp_path))
    video = tmp_path / "t.mp4"
    _synth(video)
    settings = Settings(
        workspace_root=tmp_path,
        sam_backend="mock",
        harness_max_passes=2,
        harness_min_review_score=0.5,
        harness_min_temporal_stability=0.3,
    )
    rt = HarnessRuntime(Workspace(settings), settings)
    result = rt.run_quality_gated_pipeline(
        video_path=str(video),
        intent="cyberpunk particles behind the speaker",
        max_render_frames=12,
        max_passes=2,
    )
    assert result["project_id"]
    assert result.get("final_gate") is not None
    assert "edit_graph" in result
    # variants
    v = rt.generate_variants(result["project_id"], n=2, max_render_frames=8)
    assert v["ok"]
    assert len(v["variants"]) == 2


@pytest.mark.integration
def test_recipe_and_multi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    video = tmp_path / "t.mp4"
    _synth(video, 16)
    settings = Settings(workspace_root=tmp_path, sam_backend="mock")
    rt = HarnessRuntime(Workspace(settings), settings)
    out = rt.apply_recipe(video_path=str(video), recipe_name="cinematic_bokeh", max_render_frames=10)
    assert out["project_id"]
    assert out["render"]["absolute_output"]
    multi = rt.segment_multi_objects(out["project_id"], ["person", "microphone"])
    assert multi["object_count"] >= 1
