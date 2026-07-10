"""Alpha refinement — turn binary-ish masks into soft hair-level alpha.

Backends: rvm (onnx, via models_registry) > guided (dependency-free, always works).
Trimap band comes from mask morphology; only the unknown band is re-estimated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.models_registry import ensure_model, model_path
from vidmcp.perception.mask_ops import to_u8_mask
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames

log = get_logger("vidmcp.alpha_refine")


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-4) -> np.ndarray:
    """Edge-preserving filter (He et al.) — numpy/cv2 only. guide, src float32 in [0,1]."""
    r = int(radius)
    mean_i = cv2.boxFilter(guide, -1, (r, r))
    mean_p = cv2.boxFilter(src, -1, (r, r))
    corr_ip = cv2.boxFilter(guide * src, -1, (r, r))
    corr_ii = cv2.boxFilter(guide * guide, -1, (r, r))
    var_i = corr_ii - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    return mean_a * guide + mean_b


def trimap_from_mask(mask_u8: np.ndarray, band_px: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (definite_fg, definite_bg, unknown) boolean maps."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px, band_px))
    binm = (mask_u8 > 127).astype(np.uint8) * 255
    fg = cv2.erode(binm, k) > 127
    bg = cv2.dilate(binm, k) < 128
    unknown = ~(fg | bg)
    return fg, bg, unknown


def refine_frame_alpha(frame_bgr: np.ndarray, mask_u8: np.ndarray, band_px: int = 16) -> np.ndarray:
    """Guided-filter alpha in the unknown band. Returns float32 alpha [0,1]."""
    fg, bg, unknown = trimap_from_mask(mask_u8, band_px)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    coarse = mask_u8.astype(np.float32) / 255.0
    filtered = guided_filter(gray, coarse, radius=max(4, band_px // 2))
    alpha = coarse.copy()
    alpha[unknown] = np.clip(filtered[unknown], 0.0, 1.0)
    alpha[fg] = 1.0
    alpha[bg] = 0.0
    return alpha


class _RVMBackend:
    """RobustVideoMatting ONNX — recurrent, temporally consistent by design."""

    def __init__(self) -> None:
        self._sess = None
        self._rec: list[np.ndarray] | None = None

    def available(self) -> bool:
        return bool(ensure_model("rvm").get("found"))

    def _session(self):
        if self._sess is None:
            import onnxruntime as ort

            self._sess = ort.InferenceSession(str(model_path("rvm")), providers=["CPUExecutionProvider"])
            self._rec = [np.zeros([1, 1, 1, 1], dtype=np.float32)] * 4
        return self._sess

    def alpha(self, frame_bgr: np.ndarray, downsample: float = 0.4) -> np.ndarray:
        sess = self._session()
        src = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = np.transpose(src, (2, 0, 1))[None]
        io = {
            "src": src,
            "r1i": self._rec[0], "r2i": self._rec[1], "r3i": self._rec[2], "r4i": self._rec[3],
            "downsample_ratio": np.array([downsample], dtype=np.float32),
        }
        fgr, pha, *rec = sess.run(None, io)
        self._rec = rec
        return np.clip(pha[0, 0], 0.0, 1.0)


def edge_quality_score(alpha: np.ndarray) -> float:
    """Fraction of edge pixels that are soft (0<a<1) — proxy for hair-level detail."""
    edges = cv2.Canny((alpha * 255).astype(np.uint8), 40, 120) > 0
    if edges.sum() == 0:
        return 0.0
    band = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    vals = alpha[band]
    soft = ((vals > 0.05) & (vals < 0.95)).mean() if vals.size else 0.0
    return float(soft)


def refine_alpha_project(
    project: Any,
    segment_id: str | None = None,
    backend: str = "auto",
    band_px: int = 16,
    max_frames: int | None = None,
) -> dict[str, Any]:
    m = project.manifest
    seg = None
    if segment_id:
        seg = next((s for s in m.segments if s.id == segment_id), None)
    seg = seg or m.primary_segment()
    if seg is None:
        return {"ok": False, "message": "No segment track — run segment_subject first"}
    if not m.source_video:
        return {"ok": False, "message": "No source video"}

    mask_dir = project.abs(seg.mask_dir)
    mask_files = sorted(Path(mask_dir).glob("mask_*.png"))
    if not mask_files:
        return {"ok": False, "message": f"No masks in {mask_dir}"}

    rvm = _RVMBackend()
    use_rvm = backend in ("auto", "rvm") and rvm.available()
    chosen = "rvm" if use_rvm else "guided"

    out_dir = project.masks_dir / f"{seg.id[:8]}_alpha"
    out_dir.mkdir(parents=True, exist_ok=True)
    qualities: list[float] = []
    n = 0
    for idx, frame in iter_frames(project.abs(m.source_video)):
        if max_frames is not None and idx >= max_frames:
            break
        if idx >= len(mask_files):
            break
        mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = to_u8_mask(mask)
        if use_rvm:
            try:
                pha = rvm.alpha(frame)
                pha = cv2.resize(pha, (frame.shape[1], frame.shape[0]))
                # constrain RVM output to the SAM subject region (keeps prompt targeting)
                region = cv2.dilate(mask, np.ones((band_px, band_px), np.uint8)) > 32
                alpha = np.where(region, pha, 0.0).astype(np.float32)
            except Exception as e:  # noqa: BLE001
                log.warning("rvm_failed_fallback_guided", error=str(e))
                use_rvm = False
                chosen = "guided"
                alpha = refine_frame_alpha(frame, mask, band_px)
        else:
            alpha = refine_frame_alpha(frame, mask, band_px)
        cv2.imwrite(str(out_dir / f"mask_{idx:06d}.png"), (alpha * 255).astype(np.uint8))
        if idx % 10 == 0:
            qualities.append(edge_quality_score(alpha))
        n += 1

    seg.meta["alpha_dir"] = project.rel(out_dir)
    seg.meta["alpha_backend"] = chosen
    m.append_history("refine_alpha", {"segment_id": seg.id, "backend": chosen, "frames": n})
    project.save()
    return {
        "ok": True,
        "segment_id": seg.id,
        "alpha_dir": project.rel(out_dir),
        "backend": chosen,
        "frames": n,
        "edge_quality": round(float(np.mean(qualities)) if qualities else 0.0, 4),
    }
