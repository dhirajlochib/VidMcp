"""Stabilization — ffmpeg vidstab two-pass when available, deshake fallback."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import Any

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video

log = get_logger("vidmcp.stabilize")


@lru_cache(maxsize=1)
def _has_filter(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
        return name in out.stdout
    except Exception:  # noqa: BLE001
        return False


def stabilize_video_project(project: Any, strength: float = 0.6) -> dict[str, Any]:
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    src = project.abs(m.source_video)
    out = project.renders_dir / "stabilized.mp4"
    shakiness = max(1, min(10, int(3 + strength * 6)))
    smoothing = max(5, int(10 + strength * 20))

    if _has_filter("vidstabdetect"):
        trf = project.tmp_dir / "transforms.trf"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy=15:result={trf}",
             "-f", "null", "-"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", f"vidstabtransform=input={trf}:smoothing={smoothing}:crop=black:zoom=2,unsharp=5:5:0.6",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "copy", str(out)],
            check=True, capture_output=True,
        )
        backend = "vidstab"
    elif _has_filter("deshake"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-vf", "deshake", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "copy", str(out)],
            check=True, capture_output=True,
        )
        backend = "deshake"
    else:
        return {"ok": False, "message": "No stabilization filter in this ffmpeg build (need vidstab or deshake)"}

    meta = probe_video(out)
    rel = project.rel(out)
    m.renders.append({"render_id": "stabilized", "output_path": rel, "kind": "stabilize"})
    m.append_history("stabilize_video", {"backend": backend, "strength": strength})
    project.save()
    return {"ok": True, "output_path": rel, "backend": backend, "duration_sec": round(meta.duration_sec, 2)}
