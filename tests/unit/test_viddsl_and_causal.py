from pathlib import Path

import cv2
import numpy as np

from vidmcp.advanced.causal_graph import CausalGraph
from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace
from vidmcp.dsl.viddsl import compile_viddsl
from vidmcp.tools import advanced_service as adv
from vidmcp.tools import service


def test_viddsl_parse():
    src = '''
    track person as S
    scene procedural("Pythagoras") as B
    composite B under S
    gate stability >= 0.5
    sign
    '''
    prog = compile_viddsl(src)
    ops = [o.op for o in prog.ops]
    assert ops == ["track", "scene", "composite", "gate", "sign"]


def test_causal_branch_merge(tmp_path: Path):
    g = CausalGraph(project_id="p1")
    g.ensure_root()
    g.commit("segment", {"prompt": "person"}, message="seg")
    g.branch("exp")
    g.commit("effect", {"type": "cyberpunk"}, message="fx", branch="exp")
    # merge needs snapshots with layers - just ensure merge doesn't crash
    g.nodes[g.branches["exp"]].manifest_snapshot = {
        "layers": {"layers": [{"id": "a", "effect": {"effect_type": "blur"}}], "version": 1}
    }
    g.nodes[g.branches["main"]].manifest_snapshot = {
        "layers": {"layers": [{"id": "b", "effect": {"effect_type": "solid"}}], "version": 1}
    }
    # set head to main for merge target
    g.head = g.branches["main"]
    r = g.merge("exp", "main")
    assert "merge_node" in r


def test_uncertainty_and_critics(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VIDMCP_SAM_BACKEND", "mock")
    ws = Workspace(Settings(workspace_root=tmp_path, sam_backend="mock"))
    store = ws.create_project("u")
    vid = tmp_path / "v.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 90))
    for i in range(16):
        f = np.full((90, 160, 3), 40, dtype=np.uint8)
        cv2.circle(f, (80, 45), 20, (180, 160, 140), -1)
        wr.write(f)
    wr.release()
    service.import_source(store, vid)
    service.segment(store, prompt="person")
    unc = adv.uncertainty_for_project(store)
    assert unc.get("ok")
    crit = adv.critic_project(store, workspace_root=tmp_path)
    assert "axes" in crit
    assert "fix_route" in crit
