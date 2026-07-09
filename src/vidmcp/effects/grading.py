"""Color grading adjustment effects (full-frame or selective)."""

from __future__ import annotations

import cv2
import numpy as np

from vidmcp.effects.base import Effect, EffectContext
from vidmcp.models.layers import EffectParams


class ColorGradeEffect(Effect):
    name = "color_grade"
    kind = "grade"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        assert ctx.source_frame is not None
        img = ctx.source_frame.astype(np.float32)
        contrast = float(params.params.get("contrast", 1.1))
        saturation = float(params.params.get("saturation", 1.2))
        temperature = float(params.params.get("temperature", 0.0))  # -1 cold .. +1 warm
        img = (img - 127.5) * contrast + 127.5
        img[:, :, 2] = np.clip(img[:, :, 2] + temperature * 25 * params.intensity, 0, 255)  # R
        img[:, :, 0] = np.clip(img[:, :, 0] - temperature * 15 * params.intensity, 0, 255)  # B
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation * params.intensity, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
