import platform
from pathlib import Path

from vidmcp.config import Settings, SamBackend
from vidmcp.harness.contracts import build_harness_plan, TOOL_PACKS
from vidmcp.perception.factory import get_perception_backend
from vidmcp.perception.mlx_backend import MLXSam31Backend


def test_education_plan_minimal_tools():
    plan = build_harness_plan("teach Bayes theorem with a talking head", product="education", fast=True)
    assert plan.product == "education"
    assert "run_education_lesson" in plan.tool_allowlist or "render_speech_locked_scene" in plan.tool_allowlist
    assert plan.budget.max_render_frames is not None
    assert plan.budget.max_render_frames <= 90
    assert len(plan.phases) >= 5


def test_tool_packs_exist():
    assert "education" in TOOL_PACKS
    assert len(TOOL_PACKS["education"]) < 30  # reduced surface


def test_mlx_backend_available_on_arm():
    b = MLXSam31Backend()
    # on this CI/dev machine we installed mlx-vlm
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert b.is_available() is True


def test_factory_auto_prefers_mlx_on_apple(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "auto")
    monkeypatch.setenv("VIDMCP_WORKSPACE_ROOT", str(tmp_path))
    from vidmcp.config import reset_settings

    reset_settings()
    s = Settings(workspace_root=tmp_path, sam_backend=SamBackend.AUTO)
    b = get_perception_backend(s)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # should pick mlx when package present
        assert "mlx" in b.name or b.name == "mock"  # mock only if mlx import fails
        if MLXSam31Backend().is_available():
            assert b.name == "mlx_sam3.1"
