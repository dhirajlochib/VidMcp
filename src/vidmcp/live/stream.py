"""Live / streaming mode — ring-buffer session skeleton for low-latency composite.

Full real-time SAM multiplex requires GPU; this provides session state, ring buffer,
and a process_frame path using mock/local matte for OBS-style loops.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from vidmcp.perception.mask_ops import feather_mask, morph_clean
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.live")


@dataclass
class LiveSession:
    id: str
    mode: str = "mock_matte"  # mock_matte | passthrough | sam_session
    effect: str = "blur"
    use_perception: bool = False
    width: int = 1280
    height: int = 720
    ring_size: int = 8
    running: bool = False
    frames_processed: int = 0
    last_latency_ms: float = 0.0
    created_at: float = field(default_factory=time.time)
    _buffer: deque = field(default_factory=lambda: deque(maxlen=8))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _subtractor: Any = field(default=None, repr=False)

    def start(self) -> None:
        self.running = True
        self._subtractor = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=16, detectShadows=False)
        log.info("live_session_start", id=self.id)

    def stop(self) -> None:
        self.running = False
        log.info("live_session_stop", id=self.id, frames=self.frames_processed)

    def process_frame_bgr(self, frame: np.ndarray, *, effect: str = "blur") -> np.ndarray:
        t0 = time.perf_counter()
        if not self.running:
            self.start()
        h, w = frame.shape[:2]
        if self.mode == "passthrough":
            out = frame
        else:
            fg = self._subtractor.apply(frame)
            m = morph_clean((fg > 40).astype(np.uint8) * 255, 3, 5)
            m = feather_mask(m, 5)
            if effect == "blur":
                bg = cv2.GaussianBlur(frame, (0, 0), 25)
            elif effect == "cyberpunk":
                bg = cv2.GaussianBlur(frame, (0, 0), 15)
                bg = cv2.convertScaleAbs(bg, alpha=1.2, beta=-10)
                bg[:, :, 0] = np.clip(bg[:, :, 0].astype(int) + 40, 0, 255).astype(np.uint8)
            else:
                bg = np.full_like(frame, (30, 15, 40))
            a = m.astype(np.float32)[..., None] / 255.0
            out = (frame.astype(np.float32) * a + bg.astype(np.float32) * (1 - a)).astype(np.uint8)
        with self._lock:
            self._buffer.append(out)
            self.frames_processed += 1
            self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return out

    def process_video_file(self, path: str, out_path: str, *, max_frames: int = 90, effect: str = "blur") -> dict[str, Any]:
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        self.start()
        n = 0
        lat = []
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            out = self.process_frame_bgr(frame, effect=effect)
            wr.write(out)
            lat.append(self.last_latency_ms)
            n += 1
        cap.release()
        wr.release()
        self.stop()
        return {
            "ok": True,
            "session_id": self.id,
            "frames": n,
            "output_path": out_path,
            "mean_latency_ms": float(np.mean(lat)) if lat else 0.0,
            "p95_latency_ms": float(np.percentile(lat, 95)) if lat else 0.0,
        }

    def status(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "running": self.running,
            "frames_processed": self.frames_processed,
            "last_latency_ms": self.last_latency_ms,
            "mode": self.mode,
            "ring_size": self.ring_size,
            "use_perception": self.use_perception,
            "effect": self.effect,
        }


class _Registry:
    def __init__(self) -> None:
        self.sessions: dict[str, LiveSession] = {}

    def create(self, **kw: Any) -> LiveSession:
        s = LiveSession(id=str(uuid4())[:8], **kw)
        self.sessions[s.id] = s
        return s

    def get(self, sid: str) -> LiveSession | None:
        return self.sessions.get(sid)


_reg: _Registry | None = None


def get_live_registry() -> _Registry:
    global _reg
    if _reg is None:
        _reg = _Registry()
    return _reg
