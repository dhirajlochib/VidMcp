"""Grading primitives — lift/gamma/gain, filmic curve, vibrance, HSL shift, skin protection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vidmcp.effects.base import Effect, EffectContext
from vidmcp.models.layers import EffectParams


def skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Float [0,1] skin-probability mask via YCrCb locus, feathered."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)
    m = ((cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 5.0)


def lift_gamma_gain(img: np.ndarray, lift=0.0, gamma=1.0, gain=1.0) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = np.clip((x ** (1.0 / max(gamma, 1e-3))) * gain + lift, 0, 1)
    return (x * 255).astype(np.uint8)


def filmic_curve(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Soft S-curve with highlight roll-off (Reinhard-ish shoulder)."""
    x = img.astype(np.float32) / 255.0
    s = x * (1.0 + x / 1.44) / (1.0 + x)  # gentle shoulder
    s = np.clip(s * 1.12, 0, 1)
    out = x * (1 - strength) + s * strength
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def vibrance(img: np.ndarray, amount: float = 0.2) -> np.ndarray:
    """Saturation boost weighted toward under-saturated pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    boost = 1.0 + amount * (1.0 - s)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def hsl_shift(img: np.ndarray, hue_center: float, hue_width: float, d_hue: float = 0.0, d_sat: float = 1.0, d_lum: float = 1.0) -> np.ndarray:
    """Secondary: shift pixels within a hue band (degrees, OpenCV hue 0-180)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0] * 2.0  # → degrees
    d = np.minimum(np.abs(h - hue_center), 360 - np.abs(h - hue_center))
    w = np.clip(1.0 - d / max(hue_width, 1e-3), 0, 1)
    hsv[:, :, 0] = ((hsv[:, :, 0] + w * d_hue / 2.0) % 180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + w * (d_sat - 1)), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + w * (d_lum - 1)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_grade(img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Composite grade from a params dict; protect_skin limits chroma moves on skin."""
    out = img
    if any(k in params for k in ("lift", "gamma", "gain")):
        out = lift_gamma_gain(out, float(params.get("lift", 0.0)), float(params.get("gamma", 1.0)), float(params.get("gain", 1.0)))
    if params.get("filmic"):
        out = filmic_curve(out, float(params.get("filmic", 1.0)))
    if params.get("vibrance"):
        out = vibrance(out, float(params["vibrance"]))
    if params.get("saturation") and float(params["saturation"]) != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(params["saturation"]), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if params.get("temperature"):
        t = float(params["temperature"])
        f = out.astype(np.float32)
        f[:, :, 2] = np.clip(f[:, :, 2] + t * 22, 0, 255)
        f[:, :, 0] = np.clip(f[:, :, 0] - t * 14, 0, 255)
        out = f.astype(np.uint8)
    if params.get("hsl"):
        h = params["hsl"]
        out = hsl_shift(out, float(h.get("hue", 0)), float(h.get("width", 40)), float(h.get("d_hue", 0)), float(h.get("d_sat", 1)), float(h.get("d_lum", 1)))
    if params.get("protect_skin", True) and img is not out:
        w = skin_mask(img)[..., None] * float(params.get("skin_protect_strength", 0.65))
        out = (img.astype(np.float32) * w + out.astype(np.float32) * (1 - w)).astype(np.uint8)
    return out


class GradeV2Effect(Effect):
    name = "grade_v2"
    kind = "grade"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        assert ctx.source_frame is not None
        return apply_grade(ctx.source_frame, dict(params.params))
