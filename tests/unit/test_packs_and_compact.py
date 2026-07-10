"""Unit tests for agent context control: packs, compact, intent, brief."""

from __future__ import annotations

from types import SimpleNamespace

from vidmcp.config import Settings, reset_settings
from vidmcp.harness.intent import build_project_brief, resolve_intent
from vidmcp.harness.packs import ALWAYS_TOOLS, PACKS, allowed_tool_names, get_pack, list_packs
from vidmcp.harness.recipes import get_recipe, list_recipes
from vidmcp.utils.compact import compact_result


def test_packs_defined():
    for name in ("talking_head", "education", "vfx", "admin", "all"):
        assert name in PACKS
    packs = list_packs()
    assert len(packs) >= 5
    assert "list_tool_packs" in ALWAYS_TOOLS
    assert "run_intent" in ALWAYS_TOOLS
    assert "project_brief" in ALWAYS_TOOLS


def test_allowed_tool_names_talking_head(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDMCP_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDMCP_TOOL_PACK", "talking_head")
    reset_settings()
    names = allowed_tool_names("talking_head")
    assert names is not None
    assert "run_talking_head_polish" in names
    assert "run_intent" in names
    assert "segment_multi_objects" not in names
    assert allowed_tool_names("all") is None
    info = get_pack("vfx_matte")
    assert info["name"] == "vfx"


def test_compact_result_drops_heavy_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDMCP_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDMCP_COMPACT", "1")
    monkeypatch.setenv("VIDMCP_MAX_RESULT_CHARS", "4000")
    reset_settings()
    fat = {
        "ok": True,
        "project_id": "abc",
        "words": [{"w": i} for i in range(100)],
        "edit_history": [{"a": 1}] * 50,
        "layers": {"stack": [1, 2, 3]},
        "final_path": "/tmp/out.mp4",
        "steps": {
            "export": {"ok": True, "path": "/tmp/x.mp4", "raw": "HUGE", "duration_sec": 1.2},
        },
    }
    slim = compact_result(fat, force=True)
    assert isinstance(slim, dict)
    assert "words" not in slim
    assert slim.get("n_words") == 100
    assert "layers" not in slim
    assert slim.get("layers_present") is True
    assert "raw" not in (slim.get("steps") or {}).get("export", {})
    assert slim["steps"]["export"]["path"] == "/tmp/x.mp4"


def test_compact_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDMCP_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDMCP_COMPACT", "0")
    reset_settings()
    data = {"words": [1, 2, 3], "ok": True}
    assert compact_result(data) is data or compact_result(data)["words"] == [1, 2, 3]
    full = compact_result(data, force=False)
    assert full == data


def test_intent_dry_run_polish():
    plan = resolve_intent(
        "polish this talking head for reels with space background and cut fillers",
        dry_run=True,
    )
    assert plan["ok"]
    assert plan["plan"]["kind"] == "polish"
    assert plan["plan"]["args"]["preset"] == "reels_9x16"
    assert plan["plan"]["args"]["bg_mode"] == "space"
    assert plan["plan"]["args"]["smart_cut"] is True


def test_intent_dry_run_education():
    plan = resolve_intent("teach Bayes theorem math lesson", dry_run=True)
    assert plan["ok"]
    assert plan["plan"]["kind"] == "education"


def test_intent_dry_run_cyberpunk():
    plan = resolve_intent("cyberpunk neon city behind speaker", dry_run=True)
    assert plan["ok"]
    assert plan["plan"]["kind"] == "recipe"
    assert plan["plan"]["recipe"] == "cyberpunk_talking_head"


def test_project_brief_compact():
    m = SimpleNamespace(
        id="p1",
        name="demo",
        status=SimpleNamespace(value="analyzed"),
        source_video="source/source.mp4",
        source_meta={"duration": 12.5, "width": 1920, "height": 1080, "fps": 30},
        analysis={},
        segments=[],
        renders=[{"path": "renders/out.mp4", "preset": "youtube_16x9"}],
        reviews=[],
        edit_history=[{"action": "import_video", "ts": "t"}],
        primary_segment_id=None,
        tags=["x"],
    )
    project = SimpleNamespace(manifest=m, root="/tmp/ws/p1")
    brief = build_project_brief(project)
    assert brief["ok"]
    assert brief["project_id"] == "p1"
    assert brief["n_renders"] == 1
    assert "layers" not in brief
    assert brief["last_render"]["path"] == "renders/out.mp4"
    assert brief["next_steps"]


def test_creator_recipes_exist():
    names = {r["name"] for r in list_recipes()}
    for n in (
        "talking_head_polish",
        "talking_head_reels",
        "talking_head_space",
        "talking_head_tight",
        "talking_head_infographics",
    ):
        assert n in names
        r = get_recipe(n)
        assert r.get("creator_pipeline") is True


def test_settings_tool_pack_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDMCP_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDMCP_TOOL_PACK", "creator")
    reset_settings()
    s = Settings()
    assert s.tool_pack == "talking_head"
