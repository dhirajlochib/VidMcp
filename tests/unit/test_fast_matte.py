"""Fast matte center prior."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vidmcp.matte.fast_matte import segment_video_matte


def test_center_matte(tmp_path: Path):
    src = tmp_path / "v.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x240:d=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    masks = tmp_path / "masks"
    r = segment_video_matte(src, masks, backend="center")
    assert r["ok"]
    assert r["coverage_mean"] > 0.1
    assert list(masks.glob("mask_*.png"))
