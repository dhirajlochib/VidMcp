"""Abstract perception backend for video subject segmentation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


ProgressFn = Callable[[float, str], None]


@dataclass
class ObjectTrack:
    object_id: int
    label: str
    confidence_mean: float
    frame_span: tuple[int, int]
    area_ratio_mean: float


@dataclass
class SegmentationResult:
    prompt: str
    backend: str
    mask_dir: Path
    masks: list[np.ndarray]  # may be empty if streamed to disk only
    objects: list[ObjectTrack]
    fps: float
    width: int
    height: int
    frame_count: int
    mask_video: Path | None = None
    temporal_stability: float = 0.0
    coverage_mean: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class PerceptionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def segment_video(
        self,
        video_path: Path,
        prompt: str,
        *,
        output_dir: Path,
        conf: float = 0.25,
        progress: ProgressFn | None = None,
        **kwargs: Any,
    ) -> SegmentationResult:
        ...
