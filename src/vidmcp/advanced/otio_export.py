"""Export a lightweight OpenTimelineIO-compatible JSON timeline (no otio dep required).

Can be imported by tools that speak OTIO JSON, or converted when opentimelineio is installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from vidmcp.core.workspace import ProjectStore


def export_timeline_json(project: ProjectStore, *, shots: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    m = project.manifest
    fps = float((m.source_meta or {}).get("fps") or 24)
    duration = float((m.source_meta or {}).get("duration_sec") or 0)
    tracks = []
    # video track clips from shots or single clip
    clips = []
    if shots:
        for s in shots:
            clips.append(
                {
                    "name": f"shot_{s['shot_index']}",
                    "source_range": {
                        "start_sec": s["start_sec"],
                        "duration_sec": s["duration_sec"],
                    },
                    "media_ref": m.source_video,
                }
            )
    else:
        clips.append(
            {
                "name": "source",
                "source_range": {"start_sec": 0, "duration_sec": duration},
                "media_ref": m.source_video,
            }
        )
    tracks.append({"name": "Video", "kind": "Video", "clips": clips})
    # FX metadata track
    fx_clips = []
    for L in m.layers.sorted_layers():
        fx_clips.append(
            {
                "name": L.name,
                "kind": L.kind.value,
                "effect": L.effect.model_dump() if L.effect else None,
                "z_index": L.z_index,
            }
        )
    tracks.append({"name": "Layers", "kind": "Metadata", "clips": fx_clips})
    if m.renders:
        tracks.append(
            {
                "name": "Renders",
                "kind": "Video",
                "clips": [
                    {"name": r.get("render_id", "render"), "media_ref": r.get("output_path")} for r in m.renders[-5:]
                ],
            }
        )
    timeline = {
        "OTIO_SCHEMA": "Timeline.1",
        "name": m.name,
        "metadata": {
            "vidmcp_project_id": m.id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "fps": fps,
            "primary_segment": m.primary_segment_id,
        },
        "tracks": tracks,
    }
    out = project.root / "exports" / "timeline.otio.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(timeline, option=orjson.OPT_INDENT_2))

    # optional real OTIO if installed
    otio_path = None
    try:
        import opentimelineio as otio

        tl = otio.schema.Timeline(name=m.name)
        track = otio.schema.Track(name="Video", kind=otio.schema.TrackKind.Video)
        if m.source_video:
            media = otio.schema.ExternalReference(target_url=str(project.abs(m.source_video)))
            clip = otio.schema.Clip(
                name="source",
                media_reference=media,
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, fps),
                    duration=otio.opentime.RationalTime(int(duration * fps), fps),
                ),
            )
            track.append(clip)
        tl.tracks.append(track)
        otio_path = project.root / "exports" / "timeline.otio"
        otio.adapters.write_to_file(tl, str(otio_path))
    except Exception:
        pass

    return {
        "ok": True,
        "json_path": project.rel(out),
        "absolute_json": str(out),
        "otio_path": project.rel(otio_path) if otio_path else None,
        "track_count": len(tracks),
        "clip_count": len(clips),
    }
