"""Edit Decision List export / apply."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vidmcp.core.workspace import ProjectStore


def export_edl(project: ProjectStore, path: Path | str | None = None) -> dict[str, Any]:
    m = project.manifest
    edl = {
        "version": 1,
        "project_id": m.id,
        "source_video": m.source_video,
        "source_meta": m.source_meta,
        "audio_pipeline": (m.analysis or {}).get("audio_pipeline") or m.source_meta.get("audio_pipeline"),
        "segments": [s.model_dump(mode="json") if hasattr(s, "model_dump") else s for s in m.segments],
        "layers": m.layers.model_dump(mode="json") if hasattr(m.layers, "model_dump") else {},
        "renders": m.renders,
        "edit_history": m.edit_history,
        "tags": m.tags,
    }
    path = Path(path) if path else project.root / "edl.json"
    path.write_text(json.dumps(edl, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(path), "edl": edl}
