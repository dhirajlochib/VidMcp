"""Face detection helper — survives OpenCV wheels that ship without cascade XMLs."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.faces")

_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)
_detector: cv2.CascadeClassifier | None | bool = None  # False = unavailable


def _cascade_path() -> Path | None:
    # 1) wheel-shipped
    try:
        p = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if p.exists():
            return p
    except Exception:  # noqa: BLE001
        pass
    # 2) cached download
    cache = Path.home() / ".cache" / "vidmcp" / "models" / "haarcascade_frontalface_default.xml"
    if cache.exists() and cache.stat().st_size > 10000:
        return cache
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_CASCADE_URL, cache)
        if cache.stat().st_size > 10000:
            log.info("cascade_downloaded", path=str(cache))
            return cache
    except Exception as e:  # noqa: BLE001
        log.debug("cascade_download_failed", error=str(e))
    return None


def detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) boxes; empty list when no detector is available."""
    global _detector
    if _detector is None:
        p = _cascade_path()
        if p is None:
            _detector = False
        else:
            det = cv2.CascadeClassifier(str(p))
            _detector = det if not det.empty() else False
            if _detector is False:
                log.warning("face_detector_unavailable")
    if _detector is False:
        return []
    try:
        faces = _detector.detectMultiScale(gray, 1.15, 4, minSize=(36, 36))
        return [tuple(int(v) for v in f) for f in faces]
    except Exception:  # noqa: BLE001
        return []
