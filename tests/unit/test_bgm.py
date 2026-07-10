"""BGM generate + mix."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vidmcp.audio.bgm import generate_ambient_bgm, mix_bgm


def test_generate_and_mix(tmp_path: Path):
    bgm = tmp_path / "bgm.wav"
    generate_ambient_bgm(bgm, 2.0, style="cinematic")
    assert bgm.exists() and bgm.stat().st_size > 1000

    vox = tmp_path / "vox.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:duration=2",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(vox),
        ],
        check=True,
        capture_output=True,
    )
    out = tmp_path / "mix.wav"
    r = mix_bgm(vox, out, bgm_wav=bgm, bgm_volume=0.4, duck=False)
    assert r["ok"]
    assert Path(r["path"]).exists()
