"""Saliency-driven smart reframe — 9:16⇄16:9⇄1:1 without stretch.

Saliency fusion: subject matte centroid > faces > motion > center prior.
Crop path smoothed by a 1-Euro filter with velocity clamp (virtual camera operator).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames, probe_video, sample_frames

log = get_logger("vidmcp.reframe")

ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5}


class OneEuro:
    """1-Euro filter — smooth when slow, responsive when fast."""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 0.3, beta: float = 0.02):
        self.freq, self.min_cutoff, self.beta = freq, min_cutoff, beta
        self._x: float | None = None
        self._dx = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        self._dx += self._alpha(1.0) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff) * (x - self._x)
        return self._x


def _face_center(gray: np.ndarray) -> tuple[float, float] | None:
    from vidmcp.perception.faces import detect_faces

    faces = detect_faces(gray)
    if not faces:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (x + w / 2, y + h / 3)  # bias toward eyes


def compute_saliency_track(project: Any, mode: str = "track_subject") -> list[dict[str, Any]]:
    """Sampled (t, cx, cy) normalized [0,1] saliency centers."""
    m = project.manifest
    video = project.abs(m.source_video)
    meta = probe_video(video)
    seg = m.primary_segment()
    mask_dir = None
    if seg:
        mask_dir = Path(project.abs((seg.meta or {}).get("alpha_dir") or seg.mask_dir))
    shots = m.analysis.get("shots") or []
    cut_times = {round(s["start"], 1) for s in shots}
    frames = sample_frames(video, max_frames=min(300, max(30, int(meta.duration_sec * 3))), max_side=480)
    track: list[dict[str, Any]] = []
    for idx, ts, img in frames:
        h, w = img.shape[:2]
        cx, cy, source = 0.5, 0.45, "center"
        # matte centroid (strongest signal)
        if mask_dir is not None:
            mp = mask_dir / f"mask_{idx:06d}.png"
            if mp.exists():
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.max() > 20:
                    ys, xs = np.nonzero(mask > 64)
                    if len(xs) > 50:
                        cx, cy = float(xs.mean()) / mask.shape[1], float(ys.mean()) / mask.shape[0]
                        source = "matte"
        if source == "center":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fc = _face_center(gray)
            if fc:
                cx, cy, source = fc[0] / w, fc[1] / h, "face"
        track.append({"t": round(ts, 3), "cx": round(cx, 4), "cy": round(cy, 4), "src": source,
                      "cut": round(ts, 1) in cut_times})
    return track


def smooth_track(track: list[dict[str, Any]], fps_hint: float = 3.0, max_vel: float = 0.25) -> list[dict[str, Any]]:
    """1-Euro + dead-zone + velocity clamp; reset at shot cuts."""
    fx = OneEuro(freq=fps_hint)
    fy = OneEuro(freq=fps_hint)
    out: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for p in track:
        if p.get("cut"):
            fx = OneEuro(freq=fps_hint)
            fy = OneEuro(freq=fps_hint)
            prev = None
        cx, cy = fx(p["cx"]), fy(p["cy"])
        if prev is not None:
            dt = max(p["t"] - prev["t"], 1e-3)
            # dead zone: don't chase tiny drift
            if abs(cx - prev["cx"]) < 0.02:
                cx = prev["cx"]
            if abs(cy - prev["cy"]) < 0.02:
                cy = prev["cy"]
            # velocity clamp
            vmax = max_vel * dt
            cx = prev["cx"] + np.clip(cx - prev["cx"], -vmax, vmax)
            cy = prev["cy"] + np.clip(cy - prev["cy"], -vmax, vmax)
        cur = {"t": p["t"], "cx": round(float(cx), 4), "cy": round(float(cy), 4)}
        out.append(cur)
        prev = cur
    return out


def _track_fn(track: list[dict[str, Any]]) -> Callable[[float], tuple[float, float]]:
    ts = np.array([p["t"] for p in track])
    xs = np.array([p["cx"] for p in track])
    ys = np.array([p["cy"] for p in track])

    def fn(t: float) -> tuple[float, float]:
        if len(ts) == 0:
            return 0.5, 0.45
        return float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys))

    return fn


def render_crop_track(
    video: Path,
    out: Path,
    *,
    target_aspect: float,
    center_fn: Callable[[float], tuple[float, float]],
    zoom_fn: Callable[[float], float] | None = None,
    out_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Bake a moving crop (and optional zoom) into a new video, preserving audio."""
    meta = probe_video(video)
    src_aspect = meta.width / max(meta.height, 1)
    if target_aspect <= src_aspect:
        crop_h0, crop_w0 = meta.height, int(meta.height * target_aspect)
    else:
        crop_w0, crop_h0 = meta.width, int(meta.width / target_aspect)
    if out_size is None:
        out_size = (crop_w0 - crop_w0 % 2, crop_h0 - crop_h0 % 2)

    tmp = out.parent / f"_{out.stem}_video.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, out_size)
    for idx, frame in iter_frames(video):
        t = idx / max(meta.fps, 1e-6)
        z = float(zoom_fn(t)) if zoom_fn else 1.0
        cw, ch = int(crop_w0 / z), int(crop_h0 / z)
        cx, cy = center_fn(t)
        x0 = int(np.clip(cx * meta.width - cw / 2, 0, meta.width - cw))
        y0 = int(np.clip(cy * meta.height - ch / 2, 0, meta.height - ch))
        crop = frame[y0 : y0 + ch, x0 : x0 + cw]
        writer.write(cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA))
    writer.release()
    # h264 + carry audio from source
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp), "-i", str(video),
         "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)],
        check=True, capture_output=True,
    )
    tmp.unlink(missing_ok=True)
    return {"width": out_size[0], "height": out_size[1], "fps": meta.fps}


def smart_reframe_project(
    project: Any,
    target: str = "9:16",
    mode: str = "track_subject",
    render: bool = True,
) -> dict[str, Any]:
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    aspect = ASPECTS.get(target)
    if aspect is None:
        return {"ok": False, "message": f"Unknown target '{target}'. Use: {sorted(ASPECTS)}"}
    raw = compute_saliency_track(project, mode=mode)
    if mode == "static":
        med_x = float(np.median([p["cx"] for p in raw])) if raw else 0.5
        med_y = float(np.median([p["cy"] for p in raw])) if raw else 0.45
        smoothed = [{"t": 0.0, "cx": med_x, "cy": med_y}]
    else:
        smoothed = smooth_track(raw)
    m.analysis.setdefault("reframe", {})[target] = smoothed
    m.append_history("smart_reframe", {"target": target, "n_keyframes": len(smoothed)})
    project.save()
    out_info: dict[str, Any] = {"ok": True, "target": target, "n_keyframes": len(smoothed),
                                "sources": {p["src"] for p in raw} and sorted({p["src"] for p in raw})}
    if render:
        src = project.abs(m.renders[-1]["output_path"]) if m.renders else project.abs(m.source_video)
        if not Path(src).exists():
            src = project.abs(m.source_video)
        out = project.renders_dir / f"reframe_{target.replace(':', 'x')}.mp4"
        info = render_crop_track(Path(src), out, target_aspect=aspect, center_fn=_track_fn(smoothed))
        rel = project.rel(out)
        m.renders.append({"render_id": out.stem, "output_path": rel, "kind": "reframe", **info})
        project.save()
        out_info["output_path"] = rel
        out_info.update(info)
    return out_info
