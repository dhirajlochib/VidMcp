"""Meshy AI 3D → rendered plate for behind-subject composite.

Uses MESHY_API_KEY when present; otherwise generates a procedural 3D-ish plate
as a stand-in so pipelines stay runnable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import httpx

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.meshy")

API = "https://api.meshy.ai"


def meshy_available() -> bool:
    return bool(os.environ.get("MESHY_API_KEY") or os.environ.get("MESHY_KEY"))


def _headers() -> dict[str, str]:
    key = os.environ.get("MESHY_API_KEY") or os.environ.get("MESHY_KEY") or ""
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def text_to_3d_plate(
    prompt: str,
    *,
    out_dir: Path,
    width: int = 1280,
    height: int = 720,
    duration_sec: float = 4.0,
    fps: float = 24.0,
    poll: bool = True,
    timeout_sec: float = 300.0,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not meshy_available():
        plate = _procedural_3d_plate(prompt, out_dir / "meshy_fallback.mp4", width, height, duration_sec, fps)
        return {
            "ok": True,
            "backend": "procedural_3d_fallback",
            "plate_path": str(plate),
            "prompt": prompt,
            "note": "Set MESHY_API_KEY for real Meshy text-to-3D",
        }

    # Best-effort Meshy text-to-3D preview API (endpoints may evolve)
    try:
        with httpx.Client(timeout=60.0) as client:
            # Common pattern: create task
            r = client.post(
                f"{API}/openapi/v2/text-to-3d",
                headers=_headers(),
                json={"mode": "preview", "prompt": prompt},
            )
            if r.status_code >= 400:
                # try v1
                r = client.post(
                    f"{API}/v2/text-to-3d",
                    headers=_headers(),
                    json={"mode": "preview", "prompt": prompt},
                )
            r.raise_for_status()
            data = r.json()
            task_id = data.get("result") or data.get("id") or data.get("task_id")
            thumbnail = None
            if poll and task_id:
                t0 = time.time()
                while time.time() - t0 < timeout_sec:
                    s = client.get(f"{API}/openapi/v2/text-to-3d/{task_id}", headers=_headers())
                    if s.status_code >= 400:
                        s = client.get(f"{API}/v2/text-to-3d/{task_id}", headers=_headers())
                    js = s.json()
                    status = (js.get("status") or js.get("task_status") or "").upper()
                    if status in ("SUCCEEDED", "SUCCESS", "DONE", "COMPLETED"):
                        thumbnail = (
                            js.get("thumbnail_url")
                            or js.get("video_url")
                            or (js.get("model_urls") or {}).get("glb")
                        )
                        break
                    if status in ("FAILED", "ERROR"):
                        raise RuntimeError(f"Meshy task failed: {js}")
                    time.sleep(3.0)
            # If we only have a still, animate orbit plate
            if thumbnail and str(thumbnail).startswith("http"):
                img_path = out_dir / "meshy_thumb.jpg"
                img = client.get(thumbnail)
                if img.status_code < 400:
                    img_path.write_bytes(img.content)
                    plate = _orbit_still(img_path, out_dir / "meshy_plate.mp4", width, height, duration_sec, fps)
                    return {
                        "ok": True,
                        "backend": "meshy",
                        "task_id": task_id,
                        "plate_path": str(plate),
                        "thumbnail": thumbnail,
                        "prompt": prompt,
                    }
            # fallback procedural but note API accepted
            plate = _procedural_3d_plate(prompt, out_dir / "meshy_fallback.mp4", width, height, duration_sec, fps)
            return {
                "ok": True,
                "backend": "meshy_partial_procedural_plate",
                "task_id": task_id,
                "plate_path": str(plate),
                "raw": data,
                "prompt": prompt,
            }
    except Exception as e:  # noqa: BLE001
        log.warning("meshy_failed", error=str(e))
        plate = _procedural_3d_plate(prompt, out_dir / "meshy_fallback.mp4", width, height, duration_sec, fps)
        return {
            "ok": True,
            "backend": "procedural_3d_fallback",
            "plate_path": str(plate),
            "error": str(e),
            "prompt": prompt,
        }


def _orbit_still(img_path: Path, out: Path, w: int, h: int, duration: float, fps: float) -> Path:
    img = cv2.imread(str(img_path))
    if img is None:
        return _procedural_3d_plate("asset", out, w, h, duration, fps)
    img = cv2.resize(img, (w, h))
    n = max(1, int(duration * fps))
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        # ken burns zoom/pan
        scale = 1.0 + 0.08 * np.sin(i / max(n, 1) * np.pi)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        M[0, 2] += 12 * np.sin(i / 10)
        frame = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        wr.write(frame)
    wr.release()
    return out


def _procedural_3d_plate(prompt: str, out: Path, w: int, h: int, duration: float, fps: float) -> Path:
    n = max(1, int(duration * fps))
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        t = i / max(fps, 1)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (18, 12, 28)
        # faux 3D grid floor
        for z in range(1, 20):
            y = int(h * 0.55 + z * 12 + 20 * np.sin(t + z * 0.2))
            cv2.line(frame, (0, y), (w, y), (60, 40, 90), 1)
        # rotating cube-ish
        cx, cy = w // 2, int(h * 0.42)
        s = 80
        ang = t * 1.5
        pts = []
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            x = cx + int(s * (dx * np.cos(ang) - dy * np.sin(ang) * 0.5))
            y = cy + int(s * (dx * np.sin(ang) * 0.3 + dy * np.cos(ang) * 0.6))
            pts.append([x, y])
        cv2.fillPoly(frame, [np.array(pts, np.int32)], (100, 80, 255))
        cv2.polylines(frame, [np.array(pts, np.int32)], True, (200, 220, 255), 2)
        cv2.putText(frame, prompt[:40], (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 210, 230), 2, cv2.LINE_AA)
        cv2.putText(frame, "Meshy plate (fallback/3D-ish)", (30, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 120, 160), 1, cv2.LINE_AA)
        wr.write(frame)
    wr.release()
    return out
