"""Procedural math/motion scene renderer (no Manim required).

Produces clean educational plates: equations, axes, animated curves, geometric proofs.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.scenes.procedural")


def _parse_duration(prompt: str, default: float = 5.0) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*s(?:ec)?", prompt.lower())
    if m:
        return float(m.group(1))
    return default


def render_procedural_math_scene(
    prompt: str,
    *,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: float = 24.0,
    duration_sec: float | None = None,
    seed: int = 0,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = duration_sec or _parse_duration(prompt, 5.0)
    n = max(1, int(duration_sec * fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    rng = np.random.default_rng(seed or abs(hash(prompt)) % (2**31))

    # theme from prompt keywords
    p = prompt.lower()
    if "pythag" in p or "triangle" in p:
        mode = "pythagoras"
    elif "circle" in p or "pi" in p or "π" in p:
        mode = "circle"
    elif "fourier" in p or "wave" in p:
        mode = "wave"
    elif "matrix" in p or "linear" in p:
        mode = "matrix"
    else:
        mode = "function"

    title = prompt.strip()[:64] if prompt.strip() else "Mathematical Scene"

    for i in range(n):
        t = i / max(fps, 1e-6)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # deep slate bg
        frame[:] = (28, 22, 18)
        # subtle grid
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (45, 38, 32), 1)
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (45, 38, 32), 1)

        # title bar
        cv2.putText(
            frame,
            title,
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (230, 220, 200),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "VidMCP procedural scene",
            (40, height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 110, 100),
            1,
            cv2.LINE_AA,
        )

        cx, cy = width // 2, height // 2 + 20
        progress = min(1.0, t / max(duration_sec * 0.7, 0.1))

        if mode == "pythagoras":
            # animated right triangle + squares on sides
            scale = int(160 * progress) + 20
            a, b = scale, int(scale * 0.75)
            pts = np.array([[cx - a // 2, cy + b // 2], [cx - a // 2 + a, cy + b // 2], [cx - a // 2, cy + b // 2 - b]], np.int32)
            cv2.polylines(frame, [pts], True, (80, 200, 255), 3, cv2.LINE_AA)
            # square on hypotenuse (simplified)
            if progress > 0.4:
                cv2.putText(frame, "a^2 + b^2 = c^2", (cx - 140, cy + b // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 180), 2, cv2.LINE_AA)
            # grow squares
            if progress > 0.2:
                cv2.rectangle(frame, (pts[0][0] - b, pts[0][1]), (pts[0][0], pts[0][1] + b), (255, 120, 80), 2)
            if progress > 0.35:
                cv2.rectangle(frame, (pts[0][0], pts[0][1]), (pts[0][0] + a, pts[0][1] + a), (80, 160, 255), 2)

        elif mode == "circle":
            r = int(120 + 40 * math.sin(t * 2))
            cv2.circle(frame, (cx, cy), r, (255, 180, 80), 3, cv2.LINE_AA)
            ang = t * 2.2
            px = int(cx + r * math.cos(ang))
            py = int(cy + r * math.sin(ang))
            cv2.line(frame, (cx, cy), (px, py), (180, 255, 200), 2, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "x = r cos t   y = r sin t", (cx - 200, cy + r + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 255), 2, cv2.LINE_AA)

        elif mode == "wave":
            pts = []
            for x in range(0, width, 3):
                y = int(cy + 80 * math.sin(0.02 * x + t * 3) * progress + 30 * math.sin(0.05 * x - t * 2))
                pts.append([x, y])
            arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [arr], False, (255, 200, 80), 2, cv2.LINE_AA)
            cv2.putText(frame, "fourier-ish superposition", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 200, 220), 2, cv2.LINE_AA)

        elif mode == "matrix":
            # animated matrix grid
            m = 3
            cell = 70
            x0, y0 = cx - m * cell // 2, cy - m * cell // 2
            for r in range(m):
                for c in range(m):
                    x1, y1 = x0 + c * cell, y0 + r * cell
                    val = math.sin(t + r + c) * progress
                    color = (int(80 + 100 * abs(val)), int(120 + 80 * abs(val)), 255)
                    cv2.rectangle(frame, (x1, y1), (x1 + cell - 8, y1 + cell - 8), color, -1)
                    cv2.putText(frame, f"{val:.1f}", (x1 + 12, y1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, "A v = λ v", (cx - 80, y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 230, 255), 2, cv2.LINE_AA)

        else:  # function plot
            # axes
            cv2.line(frame, (80, cy), (width - 80, cy), (100, 100, 110), 2)
            cv2.line(frame, (cx, 100), (cx, height - 80), (100, 100, 110), 2)
            pts = []
            x_max = int((width - 160) * progress)
            for x in range(0, max(x_max, 1), 2):
                xn = (x - (width // 2 - 80)) / 80.0
                yn = 0.35 * xn ** 2 - 0.4
                px = 80 + x
                py = int(cy - yn * 80)
                pts.append([px, py])
            if len(pts) > 1:
                arr = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [arr], False, (80, 200, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "y = 0.35 x^2 - 0.4", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 255), 2, cv2.LINE_AA)

        # scanline polish
        frame[::4, :, :] = (frame[::4, :, :].astype(np.float32) * 0.92).astype(np.uint8)
        writer.write(frame)

    writer.release()
    # remux h264 if possible
    try:
        import subprocess

        h264 = out_path.with_name(out_path.stem + "_h264.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(h264)],
            check=True,
            capture_output=True,
        )
        h264.replace(out_path)
    except Exception as e:  # noqa: BLE001
        log.warning("procedural_h264_failed", error=str(e))
    log.info("procedural_scene_done", path=str(out_path), mode=mode, frames=n)
    return out_path
