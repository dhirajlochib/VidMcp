"""Background effects applied outside (or behind) the subject matte."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.effects.base import Effect, EffectContext
from vidmcp.models.layers import EffectParams
from vidmcp.perception.mask_ops import to_u8_mask


def _parse_color(color: str | list | tuple, default=(10, 5, 30)) -> tuple[int, int, int]:
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return int(color[0]), int(color[1]), int(color[2])
    if isinstance(color, str):
        c = color.lstrip("#")
        if len(c) == 6:
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return b, g, r  # BGR
    return default  # type: ignore[return-value]


class BlurBackgroundEffect(Effect):
    name = "blur"
    kind = "background"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        assert ctx.source_frame is not None
        radius = int(params.params.get("blur_radius", 25 * params.intensity))
        radius = max(1, radius | 1)  # odd
        return cv2.GaussianBlur(ctx.source_frame, (radius, radius), 0)


class SolidBackgroundEffect(Effect):
    name = "solid"
    kind = "background"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        color = _parse_color(params.params.get("color", "#0a051e"))
        frame = np.zeros((ctx.height, ctx.width, 3), dtype=np.uint8)
        frame[:] = color
        return frame


class ImagePlateBackgroundEffect(Effect):
    name = "image_plate"
    kind = "background"

    def __init__(self) -> None:
        self._plate: np.ndarray | None = None
        self._path: str | None = None

    def prepare(self, params: EffectParams, ctx_meta: dict[str, Any]) -> None:
        path = params.params.get("plate_path")
        if path and path != self._path:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            self._plate = img
            self._path = str(path)

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        if self._plate is None:
            self.prepare(params, {})
        assert self._plate is not None
        return cv2.resize(self._plate, (ctx.width, ctx.height), interpolation=cv2.INTER_AREA)


class CyberpunkBackgroundEffect(Effect):
    """Stylized neon cyberpunk plate from source (grade + glow + scanlines) for behind-subject."""

    name = "cyberpunk"
    kind = "background"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        assert ctx.source_frame is not None
        img = ctx.source_frame.astype(np.float32)
        # crush shadows, lift neon mids
        img = img * np.array([1.15, 0.85, 1.35], dtype=np.float32)  # BGR boost blue/red
        img = np.clip(img, 0, 255)
        # vignette
        h, w = ctx.height, ctx.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = h / 2, w / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        vig = 1.0 - 0.55 * (dist / (dist.max() + 1e-6)) ** 1.5
        img *= vig[..., None]
        # blur for depth
        radius = int(params.params.get("blur_radius", 21))
        radius = max(1, radius | 1)
        out = cv2.GaussianBlur(img.astype(np.uint8), (radius, radius), 0)
        # scanlines
        if params.params.get("scanlines", True):
            out[::3, :, :] = (out[::3, :, :].astype(np.float32) * 0.75).astype(np.uint8)
        # purple/cyan grade boost
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.5 * params.intensity), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # neon edge glow from inverted mask region edges
        if ctx.subject_mask is not None:
            m = to_u8_mask(ctx.subject_mask)
            edges = cv2.Canny(m, 50, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
            glow = cv2.GaussianBlur(edges, (0, 0), 6)
            neon = np.zeros_like(out)
            neon[:, :, 0] = glow  # blue
            neon[:, :, 2] = glow  # red → magenta-ish
            out = cv2.addWeighted(out, 1.0, neon, 0.45 * params.intensity, 0)
        return out


class GenerativePlaceholderEffect(Effect):
    """Hook for diffusion / video models — currently procedural noise plate + prompt hash color."""

    name = "generative"
    kind = "background"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        prompt = str(params.params.get("prompt", "abstract scene"))
        seed = int(params.params.get("seed", abs(hash(prompt)) % (2**31)))
        rng = np.random.default_rng(seed + ctx.frame_index)
        noise = rng.integers(0, 255, (ctx.height, ctx.width, 3), dtype=np.uint8)
        noise = cv2.GaussianBlur(noise, (0, 0), 8)
        # palette from prompt hash
        hue = abs(hash(prompt)) % 180
        base = np.zeros((ctx.height, ctx.width, 3), dtype=np.uint8)
        base[:] = (hue % 255, 180, 40)
        base = cv2.cvtColor(base, cv2.COLOR_HSV2BGR)
        out = cv2.addWeighted(base, 0.55, noise, 0.45, 0)
        if ctx.source_frame is not None:
            # preserve some structure for temporal feel
            soft = cv2.GaussianBlur(ctx.source_frame, (0, 0), 20)
            out = cv2.addWeighted(out, 0.7, soft, 0.3, 0)
        return out
