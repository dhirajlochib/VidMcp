"""Deterministic mock SAM backend with multi-object / multi-prompt support.

Enables full harness development without GPU / HF weights while preserving
the same SegmentationResult contract as real SAM 3.1 backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.perception.base import ObjectTrack, PerceptionBackend, ProgressFn, SegmentationResult
from vidmcp.perception.mask_ops import (
    coverage_mean,
    feather_mask,
    morph_clean,
    temporal_median,
    temporal_stability_score,
)
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames, probe_video, write_mask_sequence

log = get_logger("vidmcp.perception.mock")


class MockPerceptionBackend(PerceptionBackend):
    name = "mock"

    def is_available(self) -> bool:
        return True

    def segment_video(
        self,
        video_path: Path,
        prompt: str,
        *,
        output_dir: Path,
        conf: float = 0.25,
        progress: ProgressFn | None = None,
        feather: int = 5,
        **kwargs: Any,
    ) -> SegmentationResult:
        prompts = kwargs.get("prompts") or [prompt]
        return self.segment_multi(
            video_path,
            list(prompts) if not isinstance(prompts, str) else [prompts],
            output_dir=output_dir,
            conf=conf,
            progress=progress,
            feather=feather,
            primary_prompt=prompt,
        )

    def segment_multi(
        self,
        video_path: Path,
        prompts: list[str],
        *,
        output_dir: Path,
        conf: float = 0.25,
        progress: ProgressFn | None = None,
        feather: int = 5,
        primary_prompt: str | None = None,
        **kwargs: Any,
    ) -> SegmentationResult:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = probe_video(video_path)
        prompts = [p for p in prompts if p] or [primary_prompt or "person"]
        primary_prompt = primary_prompt or prompts[0]
        log.info("mock_segment_start", video=str(video_path), prompts=prompts, frames=meta.frame_count)

        subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=16, detectShadows=False)
        masks: list[np.ndarray] = []
        # secondary object (simulated) for multi-prompt
        masks_obj2: list[np.ndarray] = []
        total = max(meta.frame_count, 1)

        for idx, frame in iter_frames(video_path):
            h, w = frame.shape[:2]
            fg = subtractor.apply(frame)
            yy, xx = np.mgrid[0:h, 0:w]
            cx, cy = w * 0.5, h * 0.45
            prior = np.exp(-(((xx - cx) ** 2) / (2 * (w * 0.22) ** 2) + ((yy - cy) ** 2) / (2 * (h * 0.32) ** 2)))
            prior = (prior * 255).astype(np.uint8)

            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

            score = (
                0.35 * fg.astype(np.float32)
                + 0.35 * prior.astype(np.float32)
                + 0.20 * skin.astype(np.float32)
                + 0.10 * edges.astype(np.float32)
            )
            p0 = prompts[0].lower()
            if any(k in p0 for k in ("person", "speaker", "human", "man", "woman", "face", "subject")):
                score = score * 1.05 + skin.astype(np.float32) * 0.1
            thr = 40 + (1.0 - conf) * 40
            binary = (score > thr).astype(np.uint8) * 255
            binary = morph_clean(binary, open_k=3, close_k=7)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if n > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest = 1 + int(np.argmax(areas))
                binary = np.where(labels == largest, 255, 0).astype(np.uint8)
            binary = feather_mask(binary, radius=feather)
            masks.append(binary)

            # faux second object (corner blob) if multi-prompt
            if len(prompts) > 1:
                m2 = np.zeros((h, w), dtype=np.uint8)
                ox = int(w * 0.15 + 5 * np.sin(idx / 5))
                oy = int(h * 0.2)
                cv2.circle(m2, (ox, oy), max(12, w // 20), 255, -1)
                m2 = feather_mask(m2, radius=feather)
                masks_obj2.append(m2)
            if progress and idx % max(1, total // 20) == 0:
                progress(idx / total, f"mock_segment frame {idx}/{total}")

        masks = temporal_median(masks, window=3)
        write_mask_sequence(masks, output_dir, prefix="mask")
        objects = [
            ObjectTrack(
                object_id=1,
                label=prompts[0],
                confidence_mean=max(conf, 0.55),
                frame_span=(0, max(len(masks) - 1, 0)),
                area_ratio_mean=coverage_mean(masks),
            )
        ]
        if masks_obj2 and len(prompts) > 1:
            masks_obj2 = temporal_median(masks_obj2, window=3)
            odir = output_dir / "obj_002"
            write_mask_sequence(masks_obj2, odir, prefix="mask")
            write_mask_sequence(masks, output_dir / "obj_001", prefix="mask")
            objects.append(
                ObjectTrack(
                    object_id=2,
                    label=prompts[1],
                    confidence_mean=max(conf, 0.4),
                    frame_span=(0, max(len(masks_obj2) - 1, 0)),
                    area_ratio_mean=coverage_mean(masks_obj2),
                )
            )
            # union already primary; optional merge into main if requested
            if kwargs.get("union_all"):
                masks = [np.maximum(a, b) for a, b in zip(masks, masks_obj2)]
                write_mask_sequence(masks, output_dir, prefix="mask")

        cov = coverage_mean(masks)
        stab = temporal_stability_score(masks)
        if progress:
            progress(1.0, "mock_segment done")
        return SegmentationResult(
            prompt=primary_prompt,
            backend=self.name,
            mask_dir=output_dir,
            masks=masks,
            objects=objects,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            frame_count=len(masks),
            temporal_stability=stab,
            coverage_mean=cov,
            meta={
                "method": "mog2+center_prior+skin",
                "conf": conf,
                "multiplex_sim": len(prompts) > 1,
                "prompts": prompts,
            },
        )
