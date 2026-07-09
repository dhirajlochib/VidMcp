from pathlib import Path

import cv2
import numpy as np
import pytest

from vidmcp.agents.orchestrator import PipelineOrchestrator
from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace


def _synth(path: Path, n: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 320, 180
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (160 + i, 80), 30, (200, 180, 160), -1)
        wr.write(frame)
    wr.release()


@pytest.mark.integration
def test_full_pipeline_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    monkeypatch.setenv("VIDMCP_WORKSPACE_ROOT", str(tmp_path))
    video = tmp_path / "in.mp4"
    _synth(video)
    settings = Settings(workspace_root=tmp_path, sam_backend="mock")
    orch = PipelineOrchestrator(Workspace(settings))
    result = orch.run(
        video_path=str(video),
        intent="cyberpunk particles behind the speaker",
        max_render_frames=12,
    )
    assert result["segment"]["backend"] == "mock"
    assert result["render"]["absolute_output"]
    assert Path(result["render"]["absolute_output"]).exists()
    assert "score" in result["review"]
