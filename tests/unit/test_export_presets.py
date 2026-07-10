"""Export aspect presets."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vidmcp.media.export import export_render
from vidmcp.utils.video_io import probe_video


def _solid(path: Path, w: int, h: int, d: float = 0.4):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s={w}x{h}:d={d}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_export_reels_and_youtube(tmp_path: Path):
    portrait = tmp_path / "p.mp4"
    landscape = tmp_path / "l.mp4"
    _solid(portrait, 100, 200)
    _solid(landscape, 200, 100)
    out_r = tmp_path / "reels.mp4"
    r = export_render(portrait, out_r, preset="reels_9x16")
    assert r["ok"]
    m = probe_video(out_r)
    assert m.width == 1080 and m.height == 1920

    out_y = tmp_path / "yt.mp4"
    r2 = export_render(landscape, out_y, preset="youtube_16x9")
    m2 = probe_video(out_y)
    assert m2.width == 1920 and m2.height == 1080
