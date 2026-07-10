"""Orientation-safe import: bake displaymatrix rotation into pixels."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import VideoMeta, probe_video

log = get_logger("vidmcp.orient")


def display_size(meta: VideoMeta) -> tuple[int, int]:
    """Width/height as displayed (after rotation metadata)."""
    w, h = meta.width, meta.height
    rot = int(meta.rotation or 0) % 360
    if rot in (90, -90, 270, -270):
        return h, w
    return w, h


def rotation_vf(rotation: int) -> str | None:
    """FFmpeg vf for baking rotation. Returns None if no transform needed."""
    rot = int(rotation or 0) % 360
    if rot == 0:
        return None
    if rot in (90, -270):
        return "transpose=1"  # 90 clockwise
    if rot in (-90, 270):
        return "transpose=2"  # 90 counter-clockwise
    if rot in (180, -180):
        return "hflip,vflip"
    # normalize odd values
    if rot > 180:
        rot -= 360
    if rot == 90:
        return "transpose=1"
    if rot == -90:
        return "transpose=2"
    return None


def bake_orientation(
    src: Path | str,
    dst: Path | str,
    *,
    max_side: int | None = None,
    crf: int = 18,
) -> dict[str, Any]:
    """
    Bake rotation into pixel data and write upright video.
    Clears displaymatrix so players don't double-rotate.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    meta = probe_video(src)
    rot = int(meta.rotation or 0)
    vf_parts: list[str] = []
    rot_filter = rotation_vf(rot)
    if rot_filter:
        vf_parts.append(rot_filter)
    if max_side and max(meta.width, meta.height) > max_side:
        vf_parts.append(f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease")

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    # force no rotate metadata
    cmd.extend(
        [
            "-metadata:s:v:0",
            "rotate=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="replace")[-800:]
        log.error("bake_orientation_failed", error=err)
        # fallback: copy if no rotation needed
        if not rot_filter:
            import shutil

            shutil.copy2(src, dst)
        else:
            raise RuntimeError(f"bake_orientation failed: {err}") from e

    out_meta = probe_video(dst)
    dw, dh = display_size(meta)
    return {
        "ok": True,
        "path": str(dst),
        "rotation_original": rot,
        "source_width": meta.width,
        "source_height": meta.height,
        "display_width": dw,
        "display_height": dh,
        "output_width": out_meta.width,
        "output_height": out_meta.height,
        "oriented": bool(rot_filter),
    }
