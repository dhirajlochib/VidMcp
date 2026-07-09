"""Video I/O utilities — metadata, frame sampling, mask sequences."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.video_io")


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec: str
    has_audio: bool
    bitrate: int | None = None
    rotation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return json.loads(out)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("ffprobe_failed", path=str(path), error=str(e))
        return {}


def probe_video(path: Path | str) -> VideoMeta:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    data = _run_ffprobe(path)
    vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    astreams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})

    if vstreams:
        vs = vstreams[0]
        width = int(vs.get("width") or 0)
        height = int(vs.get("height") or 0)
        duration = float(vs.get("duration") or fmt.get("duration") or 0.0)
        nb = vs.get("nb_frames")
        frame_count = int(nb) if nb and str(nb).isdigit() else 0

        def _parse_rate(raw: object) -> float:
            if raw is None:
                return 0.0
            if isinstance(raw, (int, float)):
                return float(raw)
            s = str(raw)
            if "/" in s:
                num, den = s.split("/", 1)
                return float(num) / max(float(den), 1e-6)
            try:
                return float(s)
            except ValueError:
                return 0.0

        # Prefer avg_frame_rate over r_frame_rate (r_frame_rate is often a max/VFR flag, e.g. 120)
        avg = _parse_rate(vs.get("avg_frame_rate"))
        rfr = _parse_rate(vs.get("r_frame_rate"))
        fps = 0.0
        if frame_count > 0 and duration > 0.05:
            # Ground truth for most files: frames / duration
            fps = frame_count / duration
        elif 1.0 <= avg <= 120.0:
            fps = avg
        elif 1.0 <= rfr <= 120.0:
            fps = rfr
        else:
            fps = avg or rfr or 30.0

        # If r_frame_rate is wildly higher than avg (common on phone/screen recordings), trust avg/duration
        if avg >= 1.0 and rfr >= 1.0 and rfr > avg * 1.35:
            fps = avg if duration <= 0 else (frame_count / duration if frame_count else avg)

        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(round(duration * fps))
        if duration <= 0 and frame_count > 0 and fps > 0:
            duration = frame_count / fps

        codec = str(vs.get("codec_name") or "unknown")
        rotation = 0
        for side in vs.get("side_data_list") or []:
            if "rotation" in side:
                rotation = int(side["rotation"])
        bitrate = int(fmt["bit_rate"]) if fmt.get("bit_rate") else None
        return VideoMeta(
            path=str(path.resolve()),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration if duration else (frame_count / max(fps, 1e-6)),
            codec=codec,
            has_audio=bool(astreams),
            bitrate=bitrate,
            rotation=rotation,
        )

    # OpenCV fallback
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frame_count / max(fps, 1e-6)
    return VideoMeta(
        path=str(path.resolve()),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
        codec="unknown",
        has_audio=False,
    )


def sample_frames(
    path: Path | str,
    *,
    max_frames: int = 12,
    target_fps: float | None = None,
    max_side: int = 960,
) -> list[tuple[int, float, np.ndarray]]:
    """Return list of (frame_index, timestamp_sec, BGR uint8 image)."""
    path = Path(path)
    meta = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    if target_fps and target_fps > 0:
        step = max(1, int(round(meta.fps / target_fps)))
    else:
        step = max(1, meta.frame_count // max(max_frames, 1))

    frames: list[tuple[int, float, np.ndarray]] = []
    idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ts = idx / max(meta.fps, 1e-6)
        frames.append((idx, ts, frame))
        idx += step
        if idx >= meta.frame_count:
            break
    cap.release()
    return frames


def iter_frames(path: Path | str) -> Iterator[tuple[int, np.ndarray]]:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def write_mask_sequence(
    masks: list[np.ndarray] | Iterator[np.ndarray],
    out_dir: Path,
    *,
    prefix: str = "mask",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i, mask in enumerate(masks):
        if mask.dtype != np.uint8:
            m = (np.clip(mask, 0, 1) * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
        else:
            m = mask
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        path = out_dir / f"{prefix}_{i:06d}.png"
        cv2.imwrite(str(path), m)
        count += 1
    log.info("wrote_mask_sequence", dir=str(out_dir), count=count)
    return out_dir


def masks_to_video(
    mask_dir: Path,
    out_path: Path,
    *,
    fps: float,
    pattern: str = "mask_%06d.png",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(Path(mask_dir) / pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "gray",
        "-crf",
        "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def ensure_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
