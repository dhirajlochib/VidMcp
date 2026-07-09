from pathlib import Path

import cv2
import numpy as np
import pytest

from vidmcp.config import Settings, reset_settings
from vidmcp.education.pipeline import run_education_lesson
from vidmcp.tools.health import platform_health


def _vid(path: Path, n: int = 24):
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (320, 180))
    for i in range(n):
        f = np.full((180, 320, 3), 50, dtype=np.uint8)
        cv2.circle(f, (160, 90), 28, (190, 170, 150), -1)
        wr.write(f)
    wr.release()


@pytest.mark.integration
def test_health_and_education(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    monkeypatch.setenv("VIDMCP_WORKSPACE_ROOT", str(tmp_path))
    reset_settings()
    h = platform_health()
    assert h["ok"]
    assert h["checks"]["ffmpeg"]
    v = tmp_path / "t.mp4"
    _vid(v)
    out = run_education_lesson(
        video_path=v,
        lesson_topic="Unit circle",
        project_name="edu_test",
        narration="First we draw a circle. Therefore sine and cosine appear. Finally we recap.",
        max_render_frames=20,
        n_steps=3,
        style="clean",
    )
    assert out["ok"]
    assert out["project_id"]
    assert out["render"]["absolute_output"]
    assert Path(out["render"]["absolute_output"]).exists()
