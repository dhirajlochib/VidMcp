from pathlib import Path

from vidmcp.config import Settings
from vidmcp.core.workspace import Workspace


def test_create_and_load(tmp_path: Path):
    ws = Workspace(Settings(workspace_root=tmp_path))
    store = ws.create_project(name="demo")
    assert (store.root / "manifest.json").exists()
    loaded = ws.load_project(store.manifest.id)
    assert loaded.manifest.name == "demo"
    assert loaded.manifest.id == store.manifest.id
