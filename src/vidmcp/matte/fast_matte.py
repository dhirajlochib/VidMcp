"""Fast person matte via MediaPipe selfie or coarse center prior."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames, probe_video, write_mask_sequence

log = get_logger("vidmcp.matte")
ProgressFn = Callable[[float, str], None]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)


def _model_path() -> Path:
    cache = Path.home() / ".cache" / "vidmcp" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / "selfie_segmenter.tflite"
    if not p.exists() or p.stat().st_size < 1000:
        log.info("downloading_selfie_segmenter", path=str(p))
        urllib.request.urlretrieve(_MODEL_URL, p)
    return p


def _soft(mask: np.ndarray) -> np.ndarray:
    m = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    m = cv2.GaussianBlur(m, (0, 0), 2.5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.erode(m, k, iterations=1)
    m = cv2.GaussianBlur(m, (0, 0), 2.0)
    return m.astype(np.float32) / 255.0


def _polarity_fix(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    cy0, cy1 = int(h * 0.12), int(h * 0.98)
    cx0, cx1 = int(w * 0.15), int(w * 0.85)
    center = float(mask[cy0:cy1, cx0:cx1].mean())
    edge = float(
        np.concatenate([mask[:40, :].ravel(), mask[-40:, :].ravel(), mask[:, :40].ravel(), mask[:, -40:].ravel()]).mean()
    )
    if edge > center:
        mask = 1.0 - mask
    if float(mask[cy0:cy1, cx0:cx1].mean()) < 0.2:
        mask = 1.0 - mask
    return mask


def _mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401

        return True
    except Exception:
        return False


def segment_video_matte(
    video: Path | str,
    out_mask_dir: Path | str,
    *,
    backend: str = "auto",
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Write soft person masks as mask_000000.png ...
    backend: auto|mediapipe|center (center = ellipse prior for tests)
    """
    video = Path(video)
    out_mask_dir = Path(out_mask_dir)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video)
    backend = backend or "auto"
    if backend == "auto":
        backend = "mediapipe" if _mediapipe_available() else "center"

    masks: list[np.ndarray] = []
    coverages: list[float] = []

    if backend == "mediapipe":
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model = _model_path()
        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True,
        )
        segmenter = vision.ImageSegmenter.create_from_options(options)
        prev: np.ndarray | None = None
        alpha = 0.55
        for idx, frame in iter_frames(video):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(idx * 1000 / max(meta.fps, 1e-6))
            result = segmenter.segment_for_video(mp_image, ts_ms)
            cat = result.category_mask.numpy_view()
            if cat.ndim == 3:
                cat = cat[:, :, 0]
            mask = cat.astype(np.float32)
            if mask.max() > 1:
                mask = (mask > 0).astype(np.float32)
            if mask.mean() > 0.55:
                mask = 1.0 - mask
            mask = _polarity_fix(mask)
            mask = _soft(mask)
            if prev is not None:
                mask = alpha * mask + (1 - alpha) * prev
            prev = mask
            u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            masks.append(u8)
            coverages.append(float(mask.mean()))
            if progress and meta.frame_count:
                progress(idx / meta.frame_count, f"matte {idx}")
        segmenter.close()
    else:
        # center ellipse prior — deterministic, works offline for tests
        for idx, frame in iter_frames(video):
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(
                mask,
                (w // 2, int(h * 0.52)),
                (int(w * 0.22), int(h * 0.38)),
                0,
                0,
                360,
                1.0,
                -1,
            )
            mask = _soft(mask)
            u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
            masks.append(u8)
            coverages.append(float(mask.mean()))

    write_mask_sequence(masks, out_mask_dir, prefix="mask")
    cov = float(np.mean(coverages)) if coverages else 0.0
    return {
        "ok": True,
        "mask_dir": str(out_mask_dir),
        "backend": backend,
        "frame_count": len(masks),
        "coverage_mean": cov,
        "warnings": [] if backend == "mediapipe" else ["Used center prior matte (install mediapipe for real cutout)"],
    }
