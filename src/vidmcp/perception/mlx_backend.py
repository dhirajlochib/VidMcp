"""Apple Silicon SAM 3.1 via mlx-vlm (mlx-community/sam3.1-bf16)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.perception.base import ObjectTrack, PerceptionBackend, ProgressFn, SegmentationResult
from vidmcp.perception.mask_ops import (
    coverage_mean,
    feather_mask,
    temporal_median,
    temporal_stability_score,
    to_u8_mask,
)
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video, write_mask_sequence

log = get_logger("vidmcp.perception.mlx")

DEFAULT_MODEL = "mlx-community/sam3.1-bf16"


class MLXSam31Backend(PerceptionBackend):
    name = "mlx_sam3.1"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        score_threshold: float = 0.25,
        resolution: int = 768,
        detect_every: int = 4,
        max_side: int | None = 768,
        max_frames: int | None = None,
    ):
        self.model_id = model_id
        self.score_threshold = score_threshold
        self.resolution = resolution
        self.detect_every = max(1, detect_every)
        self.max_side = max_side
        self.max_frames = max_frames
        self._predictor = None

    def is_available(self) -> bool:
        try:
            import mlx  # noqa: F401
            import mlx_vlm  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        import mlx.core as mx
        from mlx_vlm.models.sam3.generate import Sam3Predictor
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
        from mlx_vlm.utils import get_model_path, load_model

        log.info("mlx_sam_loading", model=self.model_id)
        mp = get_model_path(self.model_id)
        model = load_model(mp)
        processor = Sam31Processor.from_pretrained(str(mp))
        if self.resolution and self.resolution != 1008:
            try:
                processor.image_size = self.resolution
            except Exception:
                pass
        self._predictor = Sam3Predictor(model, processor, score_threshold=self.score_threshold)
        self._model = model
        self._mx = mx
        log.info("mlx_sam_ready", model=self.model_id)
        return self._predictor

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
        prompts = kwargs.get("prompts") or [prompt]
        if isinstance(prompts, str):
            prompts = [prompts]
        return self.segment_multi(
            video_path,
            list(prompts),
            output_dir=output_dir,
            conf=conf,
            progress=progress,
            feather=feather,
            primary_prompt=prompt,
            max_frames=kwargs.get("max_frames", self.max_frames),
        )

    def segment_multi(
        self,
        video_path: Path,
        prompts: list[str],
        *,
        output_dir: Path,
        conf: float = 0.25,
        progress: ProgressFn | None = None,
        feather: int = 3,
        primary_prompt: str | None = None,
        max_frames: int | None = None,
        **kwargs: Any,
    ) -> SegmentationResult:
        import mlx.core as mx
        from mlx_vlm.generate import wired_limit
        from mlx_vlm.models.sam3.generate import DetectionResult
        from mlx_vlm.models.sam3_1.generate import (
            SimpleTracker,
            _detect_with_backbone,
            _get_backbone_features,
        )
        from PIL import Image

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = probe_video(video_path)
        prompts = [p for p in prompts if p] or [primary_prompt or "person"]
        primary_prompt = primary_prompt or prompts[0]
        max_frames = max_frames if max_frames is not None else self.max_frames

        predictor = self._ensure_predictor()
        predictor.score_threshold = conf

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or meta.frame_count or 0)
        if max_frames is not None:
            total = min(total, max_frames)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or meta.width)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or meta.height)

        id_tracker = SimpleTracker()
        backbone_cache = None
        encoder_cache: dict = {}
        detect_count = 0
        latest = DetectionResult(boxes=np.array([]), masks=np.array([]), scores=np.array([]), labels=[])
        masks: list[np.ndarray] = []

        if progress:
            progress(0.02, "mlx SAM 3.1 tracking")

        with wired_limit(predictor.model):
            fi = 0
            while fi < total:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if fi % self.detect_every == 0:
                    frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    if self.max_side:
                        w0, h0 = frame_pil.size
                        scale = min(1.0, self.max_side / max(w0, h0))
                        if scale < 1.0:
                            frame_pil = frame_pil.resize((int(w0 * scale), int(h0 * scale)))
                    inputs = predictor.processor.preprocess_image(frame_pil)
                    pixel_values = mx.array(inputs["pixel_values"])
                    if detect_count % max(1, self.detect_every) == 0 or backbone_cache is None:
                        backbone_cache = _get_backbone_features(predictor.model, pixel_values)
                        encoder_cache.clear()
                    result = _detect_with_backbone(
                        predictor,
                        backbone_cache,
                        prompts,
                        frame_pil.size,
                        conf,
                        encoder_cache=encoder_cache,
                    )
                    latest = id_tracker.update(result)
                    detect_count += 1

                union = self._result_to_union(latest, H, W)
                if feather:
                    union = feather_mask(union, feather)
                masks.append(union)
                if progress and total and fi % max(1, total // 20) == 0:
                    progress(fi / max(total, 1), f"mlx frame {fi}/{total} dets={len(getattr(latest,'scores',[]) or [])}")
                fi += 1

        cap.release()
        if not masks:
            masks = [np.zeros((H, W), dtype=np.uint8)]
        masks = temporal_median(masks, window=3)
        write_mask_sequence(masks, output_dir, prefix="mask")
        # write quick preview overlay every Nth frame
        try:
            self._write_preview(video_path, masks, output_dir / "mlx_track_preview.mp4", meta.fps)
        except Exception as e:  # noqa: BLE001
            log.warning("preview_write_failed", error=str(e))

        cov = coverage_mean(masks)
        stab = temporal_stability_score(masks)
        objects = [
            ObjectTrack(
                object_id=1,
                label=primary_prompt,
                confidence_mean=max(conf, 0.5),
                frame_span=(0, max(len(masks) - 1, 0)),
                area_ratio_mean=cov,
            )
        ]
        if progress:
            progress(1.0, f"mlx done frames={len(masks)} cov={cov:.3f}")
        return SegmentationResult(
            prompt=primary_prompt,
            backend=f"{self.name}:{self.model_id}",
            mask_dir=output_dir,
            masks=masks,
            objects=objects,
            fps=meta.fps,
            width=W,
            height=H,
            frame_count=len(masks),
            temporal_stability=stab,
            coverage_mean=cov,
            meta={
                "model_id": self.model_id,
                "detect_every": self.detect_every,
                "resolution": self.resolution,
                "multiplex": True,
                "prompts": prompts,
                "max_frames": max_frames,
            },
        )

    def _result_to_union(self, result: Any, h: int, w: int) -> np.ndarray:
        union = np.zeros((h, w), dtype=np.uint8)
        if result is None:
            return union
        masks = getattr(result, "masks", None)
        if masks is None:
            return union
        arr = np.asarray(masks)
        if arr.size == 0:
            return union
        if arr.ndim == 2:
            m = arr
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            return to_u8_mask(m)
        if arr.ndim >= 3:
            for i in range(arr.shape[0]):
                m = arr[i]
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                union = np.maximum(union, to_u8_mask(m))
        return union

    def _write_preview(self, video_path: Path, masks: list[np.ndarray], out: Path, fps: float) -> None:
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return
        h, w = frame.shape[:2]
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), max(fps / 4, 8), (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        step = max(1, len(masks) // 90)
        for i, m in enumerate(masks):
            if i % step != 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h))
            overlay = frame.copy()
            overlay[m > 127] = (overlay[m > 127] * 0.45 + np.array([0, 0, 255]) * 0.55).astype(np.uint8)
            wr.write(overlay)
        wr.release()
        cap.release()
