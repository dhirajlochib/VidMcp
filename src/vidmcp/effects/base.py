"""Effect plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vidmcp.models.layers import EffectParams


@dataclass
class EffectContext:
    frame_index: int
    timestamp: float
    fps: float
    width: int
    height: int
    subject_mask: np.ndarray | None = None  # u8
    source_frame: np.ndarray | None = None  # BGR
    project_dir: Path | None = None
    extras: dict[str, Any] | None = None


class Effect(ABC):
    name: str
    kind: str  # background | particles | grade | overlay

    @abstractmethod
    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        """Return BGR or BGRA uint8 frame (same HxW as source)."""
        ...

    def prepare(self, params: EffectParams, ctx_meta: dict[str, Any]) -> None:
        """Optional one-time setup (load plate, etc.)."""
        return None
