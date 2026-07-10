"""Orientation bake + display size."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vidmcp.media.orient import bake_orientation, display_size, rotation_vf
from vidmcp.utils.video_io import VideoMeta, probe_video


def test_rotation_vf():
    assert rotation_vf(0) is None
    assert rotation_vf(90) == "transpose=1"
    assert rotation_vf(-90) == "transpose=2"
    assert rotation_vf(180) == "hflip,vflip"


def test_display_size_swap():
    m = VideoMeta(
        path="x",
        width=1920,
        height=1080,
        fps=30,
        frame_count=30,
        duration_sec=1.0,
        codec="h264",
        has_audio=False,
        rotation=-90,
    )
    assert display_size(m) == (1080, 1920)


def test_bake_orientation_no_rot(tmp_path: Path):
    src = tmp_path / "src.mp4"
    # 2 solid frames 320x240
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    dst = tmp_path / "out.mp4"
    info = bake_orientation(src, dst)
    assert info["ok"]
    assert Path(info["path"]).exists()
    meta = probe_video(dst)
    assert meta.width == 320
    assert meta.height == 240
