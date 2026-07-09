"""Lock procedural/Manim scene beats to word timestamps or lesson beats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.scenes.procedural_backend import render_procedural_math_scene
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.speech_lock")


def plan_speech_locked_steps(
    words: list[dict[str, Any]],
    *,
    n_steps: int = 6,
    cue_words: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return scene steps with start times derived from words / cues."""
    cue_words = [c.lower() for c in (cue_words or ["first", "second", "therefore", "prove", "equals", "finally", "so"])]
    cue_hits = []
    for w in words:
        wl = w["word"].lower().strip(".,")
        if any(c in wl for c in cue_words):
            cue_hits.append(float(w["start"]))
    duration = float(words[-1]["end"]) if words else 5.0
    if len(cue_hits) >= n_steps:
        times = cue_hits[:n_steps]
    elif words:
        idxs = np.linspace(0, len(words) - 1, num=n_steps).astype(int)
        times = [float(words[i]["start"]) for i in idxs]
    else:
        times = [duration * i / n_steps for i in range(n_steps)]
    steps = []
    for i, t in enumerate(times):
        t_end = times[i + 1] if i + 1 < len(times) else duration
        steps.append(
            {
                "step": i,
                "t_start": t,
                "t_end": t_end,
                "label": f"step_{i}",
                "progress": (i + 1) / n_steps,
            }
        )
    return steps


def render_speech_locked_scene(
    prompt: str,
    steps: list[dict[str, Any]],
    *,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Render a multi-phase educational plate that advances with step times."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max((s.get("t_end") or 0) for s in steps) if steps else 5.0
    duration = max(duration, 1.0)
    n = max(1, int(duration * fps))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    title = prompt[:56]
    for i in range(n):
        t = i / fps
        # active step
        step_i = 0
        for s in steps:
            if t >= float(s["t_start"]):
                step_i = int(s["step"])
        progress = (step_i + 1) / max(len(steps), 1)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (26, 20, 16)
        for x in range(0, width, 48):
            cv2.line(frame, (x, 0), (x, height), (40, 34, 28), 1)
        cv2.putText(frame, title, (36, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 220, 200), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Speech-locked step {step_i + 1}/{len(steps)}",
            (36, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (120, 200, 255),
            2,
            cv2.LINE_AA,
        )
        # progress bar
        bw = int((width - 80) * progress)
        cv2.rectangle(frame, (40, height - 50), (width - 40, height - 30), (60, 60, 70), -1)
        cv2.rectangle(frame, (40, height - 50), (40 + bw, height - 30), (80, 180, 255), -1)
        # animated diagram grows with steps
        cx, cy = width // 2, height // 2 + 10
        scale = int(40 + 120 * progress)
        if "pythag" in prompt.lower() or "triangle" in prompt.lower():
            pts = np.array(
                [[cx - scale, cy + scale // 2], [cx + scale, cy + scale // 2], [cx - scale, cy - scale // 2]],
                np.int32,
            )
            cv2.polylines(frame, [pts], True, (80, 200, 255), 3, cv2.LINE_AA)
            if progress > 0.4:
                cv2.putText(frame, "a^2 + b^2 = c^2", (cx - 120, cy + scale // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 255, 180), 2, cv2.LINE_AA)
        else:
            # function draw
            pts = []
            for x in range(0, int((width - 100) * progress), 3):
                xn = (x - width // 2) / 90.0
                yn = 0.3 * xn * xn - 0.3
                pts.append([80 + x, int(cy - yn * 70)])
            if len(pts) > 1:
                cv2.polylines(frame, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, (255, 180, 80), 3, cv2.LINE_AA)
        # highlight current step label
        for s in steps:
            color = (255, 220, 120) if int(s["step"]) == step_i else (100, 100, 110)
            y = 130 + int(s["step"]) * 28
            if y < height - 70:
                cv2.putText(frame, f'{s["step"]+1}. t={s["t_start"]:.1f}s', (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    try:
        import subprocess

        h264 = out_path.with_name(out_path.stem + "_h264.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(h264)],
            check=True,
            capture_output=True,
        )
        h264.replace(out_path)
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "output_path": str(out_path), "steps": steps, "duration_sec": duration, "frames": n}
