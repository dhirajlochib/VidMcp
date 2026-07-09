"""B-roll generation — procedural plates and optional generative hook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video

log = get_logger("vidmcp.broll")


def generate_procedural_broll(
    out_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    duration_sec: float,
    style: str = "cyberpunk_city",
    prompt: str = "",
    seed: int = 0,
) -> Path:
    """Generate a full-frame B-roll plate video (no subject)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, int(duration_sec * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    rng = np.random.default_rng(seed or abs(hash(prompt or style)) % (2**31))

    for i in range(n_frames):
        t = i / max(fps, 1e-6)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if style in ("cyberpunk_city", "cyberpunk"):
            # animated gradient + grid
            for y in range(height):
                v = int(20 + 40 * (y / height) + 20 * np.sin(t * 2 + y * 0.01))
                frame[y, :, 0] = min(255, v + 40)  # B
                frame[y, :, 2] = min(255, v + 20)  # R
            spacing = 40
            offset = int((t * 30) % spacing)
            for x in range(-offset, width, spacing):
                cv2.line(frame, (x, 0), (x + height // 2, height), (255, 100, 0), 1)
            for y in range(0, height, spacing):
                cv2.line(frame, (0, y), (width, y), (80, 0, 80), 1)
            # floating neon orbs
            for k in range(8):
                cx = int((width * (0.1 + 0.1 * k) + 50 * np.sin(t + k)) % width)
                cy = int((height * 0.3 + 80 * np.cos(t * 0.7 + k)) % height)
                cv2.circle(frame, (cx, cy), 12, (255, 0, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 30, (100, 0, 80), 2, lineType=cv2.LINE_AA)
            frame = cv2.GaussianBlur(frame, (0, 0), 1.2)
        elif style == "particles_field":
            noise = rng.integers(0, 40, (height, width, 3), dtype=np.uint8)
            frame[:] = (10, 5, 20)
            frame = cv2.add(frame, noise)
            for _ in range(120):
                x = int(rng.integers(0, width))
                y = int((rng.integers(0, height) + int(t * 40)) % height)
                cv2.circle(frame, (x, y), 1, (200, 200, 255), -1)
        else:
            # abstract
            hue = int((t * 20 + abs(hash(style)) % 180) % 180)
            frame[:] = (hue, 200, 80)
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            frame = cv2.GaussianBlur(frame, (0, 0), 15)

        writer.write(frame)
    writer.release()
    # remux to h264 if ffmpeg available
    try:
        import subprocess

        h264 = out_path.with_name(out_path.stem + "_h264.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(out_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(h264),
            ],
            check=True,
            capture_output=True,
        )
        h264.replace(out_path)
    except Exception as e:  # noqa: BLE001
        log.warning("broll_h264_remux_failed", error=str(e))
    log.info("broll_generated", path=str(out_path), frames=n_frames, style=style)
    return out_path


def match_source_geometry(source_video: Path) -> dict[str, Any]:
    m = probe_video(source_video)
    return {"width": m.width, "height": m.height, "fps": m.fps, "duration_sec": m.duration_sec}
