from pathlib import Path

import cv2
import numpy as np

from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace
from vidmcp.harness.quality_gates import evaluate_gates
from vidmcp.harness.recipes import get_recipe, list_recipes
from vidmcp.models.layers import Layer, LayerKind
from vidmcp.models.project import SegmentObject, SegmentTrack


def test_recipes_exist():
    names = {r["name"] for r in list_recipes()}
    assert "cyberpunk_talking_head" in names
    r = get_recipe("cinematic_bokeh")
    assert r["subject_prompt"]


def test_gates_on_empty_project(tmp_path: Path):
    ws = Workspace(Settings(workspace_root=tmp_path))
    store = ws.create_project("g")
    gate = evaluate_gates(store, Settings(workspace_root=tmp_path))
    assert gate.passed is False
    assert any(c.name == "source_present" for c in gate.checks)


def test_edit_graph_waves():
    from vidmcp.harness.edit_graph import EditGraph

    g = EditGraph.from_intent_plan("x", [{"tool": "a"}, {"tool": "b"}, {"tool": "c"}])
    waves = g.topological_wave()
    assert len(waves) == 3
