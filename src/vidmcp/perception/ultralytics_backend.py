"""Ultralytics SAM 3 / SAM3 video semantic predictor backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.perception.base import ObjectTrack, PerceptionBackend, ProgressFn, SegmentationResult
from vidmcp.perception.mask_ops import (
    coverage_mean,
    feather_mask,
    temporal_stability_score,
    to_u8_mask,
)
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video, write_mask_sequence

log = get_logger("vidmcp.perception.ultralytics")


class UltralyticsSam3Backend(PerceptionBackend):
    name = "ultralytics"

    def __init__(self, weights: Path | str | None = None, device: str = "auto"):
        self.weights = Path(weights) if weights else Path("sam3.pt")
        self.device = device
        self._predictor = None

    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False
        # weights may be missing — still "available" as package, but segment will fail clearly
        return True

    def _build_predictor(self) -> Any:
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model=str(self.weights),
            quantize=16,
            save=False,
            verbose=False,
        )
        if self.device and self.device != "auto":
            overrides["device"] = self.device
        return SAM3SemanticPredictor(overrides=overrides)

    def segment_video(
        self,
        video_path: Path,
        prompt: str,
        *,
        output_dir: Path,
        conf: float = 0.25,
        progress: ProgressFn | None = None,
        feather: int = 3,
        **kwargs: Any,
    ) -> SegmentationResult:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = probe_video(video_path)

        if not self.weights.exists():
            raise FileNotFoundError(
                f"SAM 3 weights not found at {self.weights}. "
                "Request access at https://huggingface.co/facebook/sam3 and set VIDMCP_SAM_WEIGHTS."
            )

        # Prefer video semantic predictor when present; else frame-wise with tracking note
        try:

            return self._segment_video_native(
                video_path, prompt, output_dir=output_dir, conf=conf, progress=progress, feather=feather, meta=meta
            )
        except Exception as e:  # noqa: BLE001
            log.warning("video_predictor_unavailable_fallback_frames", error=str(e))
            return self._segment_framewise(
                video_path, prompt, output_dir=output_dir, conf=conf, progress=progress, feather=feather, meta=meta
            )

    def _segment_video_native(
        self,
        video_path: Path,
        prompt: str,
        *,
        output_dir: Path,
        conf: float,
        progress: ProgressFn | None,
        feather: int,
        meta: Any,
    ) -> SegmentationResult:
        from ultralytics.models.sam import SAM3VideoSemanticPredictor  # type: ignore

        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=str(self.weights),
            quantize=16,
            save=False,
            verbose=False,
        )
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        # API surface may evolve — support common call patterns
        if progress:
            progress(0.05, "ultralytics video predictor init")
        results = None
        for attempt in (
            lambda: predictor(source=str(video_path), text=[prompt]),
            lambda: predictor.predict(source=str(video_path), text=[prompt]),
            lambda: predictor(str(video_path), texts=[prompt]),
        ):
            try:
                results = attempt()
                break
            except TypeError:
                continue
        if results is None:
            raise RuntimeError("Ultralytics SAM3 video API call failed for all known signatures")

        masks = self._results_to_masks(results, meta.height, meta.width)
        masks = [feather_mask(m, feather) for m in masks]
        write_mask_sequence(masks, output_dir, prefix="mask")
        if progress:
            progress(1.0, "ultralytics video segment done")
        return self._pack(prompt, output_dir, masks, meta, conf)

    def _segment_framewise(
        self,
        video_path: Path,
        prompt: str,
        *,
        output_dir: Path,
        conf: float,
        progress: ProgressFn | None,
        feather: int,
        meta: Any,
    ) -> SegmentationResult:
        predictor = self._build_predictor()
        cap = cv2.VideoCapture(str(video_path))
        masks: list[np.ndarray] = []
        total = max(meta.frame_count, 1)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # write temp frame for predictor APIs that need paths — use ndarray when supported
            try:
                predictor.set_image(frame)
                res = predictor(text=[prompt])
            except Exception:
                tmp = output_dir / f"_frame_{idx:06d}.jpg"
                cv2.imwrite(str(tmp), frame)
                predictor.set_image(str(tmp))
                res = predictor(text=[prompt])
                tmp.unlink(missing_ok=True)
            m = self._first_mask_from_result(res, frame.shape[0], frame.shape[1])
            masks.append(feather_mask(m, feather))
            idx += 1
            if progress and idx % max(1, total // 20) == 0:
                progress(idx / total, f"ultralytics frame {idx}/{total}")
        cap.release()
        write_mask_sequence(masks, output_dir, prefix="mask")
        return self._pack(prompt, output_dir, masks, meta, conf, extra={"mode": "framewise"})

    def _first_mask_from_result(self, res: Any, h: int, w: int) -> np.ndarray:
        try:
            # ultralytics Results list
            r0 = res[0] if isinstance(res, (list, tuple)) else res
            if hasattr(r0, "masks") and r0.masks is not None:
                data = r0.masks.data
                if hasattr(data, "cpu"):
                    data = data.cpu().numpy()
                if len(data) == 0:
                    return np.zeros((h, w), dtype=np.uint8)
                # union all instances for concept
                union = np.zeros((h, w), dtype=np.float32)
                for m in data:
                    mm = m
                    if mm.shape != (h, w):
                        mm = cv2.resize(mm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                    union = np.maximum(union, mm.astype(np.float32))
                return to_u8_mask(union)
        except Exception as e:  # noqa: BLE001
            log.warning("mask_extract_failed", error=str(e))
        return np.zeros((h, w), dtype=np.uint8)

    def _results_to_masks(self, results: Any, h: int, w: int) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        if results is None:
            return masks
        iterable = results if isinstance(results, (list, tuple)) else [results]
        for r in iterable:
            masks.append(self._first_mask_from_result([r], h, w))
        return masks

    def _pack(
        self,
        prompt: str,
        output_dir: Path,
        masks: list[np.ndarray],
        meta: Any,
        conf: float,
        extra: dict | None = None,
    ) -> SegmentationResult:
        cov = coverage_mean(masks)
        stab = temporal_stability_score(masks)
        obj = ObjectTrack(
            object_id=1,
            label=prompt,
            confidence_mean=conf,
            frame_span=(0, max(len(masks) - 1, 0)),
            area_ratio_mean=cov,
        )
        return SegmentationResult(
            prompt=prompt,
            backend=self.name,
            mask_dir=output_dir,
            masks=masks,
            objects=[obj],
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            frame_count=len(masks),
            temporal_stability=stab,
            coverage_mean=cov,
            meta=extra or {},
        )
