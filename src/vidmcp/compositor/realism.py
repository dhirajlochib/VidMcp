"""Compositing realism — light wrap, spill suppression, contact shadow, grain match.

Kills the 'sticker on a plate' look. All functions BGR uint8 in/out, alpha uint8.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def estimate_grain_sigma(img: np.ndarray) -> float:
    """Noise sigma on flat regions (median absolute laplacian estimator)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # exclude strong edges from the estimate
    edges = cv2.dilate(cv2.Canny(gray.astype(np.uint8), 50, 150), np.ones((5, 5), np.uint8)) > 0
    flat = np.abs(lap)[~edges]
    if flat.size == 0:
        return 0.0
    return float(np.median(flat) / 0.6745 / 6.0)  # scaled to ~display-referred sigma


def add_grain(img: np.ndarray, sigma: float, seed: int = 7) -> np.ndarray:
    if sigma <= 0.05:
        return img
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, img.shape[:2]).astype(np.float32)
    out = img.astype(np.float32) + noise[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _edge_band(alpha_u8: np.ndarray, width_px: int) -> np.ndarray:
    """Float [0,1] band around the alpha edge, feathered."""
    edges = cv2.Canny(alpha_u8, 40, 120)
    k = max(3, width_px | 1)
    band = cv2.dilate(edges, np.ones((k, k), np.uint8)).astype(np.float32) / 255.0
    return cv2.GaussianBlur(band, (0, 0), max(1.0, width_px / 3.0))


def light_wrap(fg: np.ndarray, bg: np.ndarray, alpha_u8: np.ndarray, *, width_px: int = 12, strength: float = 0.35) -> np.ndarray:
    """Bleed blurred background light into the subject's edge band (screen blend)."""
    band = _edge_band(alpha_u8, width_px) * (alpha_u8.astype(np.float32) / 255.0)
    bg_blur = cv2.GaussianBlur(bg, (0, 0), max(2.0, width_px / 2.0)).astype(np.float32)
    f = fg.astype(np.float32)
    screened = 255.0 - (255.0 - f) * (255.0 - bg_blur) / 255.0
    w = (band * strength)[..., None]
    return np.clip(f * (1 - w) + screened * w, 0, 255).astype(np.uint8)


def spill_suppress(fg: np.ndarray, alpha_u8: np.ndarray, *, width_px: int = 10, strength: float = 0.5) -> np.ndarray:
    """Pull edge-band chroma toward the subject interior color (removes BG color fringe)."""
    a = alpha_u8.astype(np.float32) / 255.0
    interior = cv2.erode(alpha_u8, np.ones((width_px * 2 + 1,) * 2, np.uint8)) > 200
    if interior.sum() < 50:
        return fg
    band = _edge_band(alpha_u8, width_px) * a
    lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_ab = lab[interior][:, 1:].mean(axis=0)
    w = (band * strength)[..., None]
    lab[:, :, 1:] = lab[:, :, 1:] * (1 - w) + mean_ab[None, None, :] * w
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def contact_shadow(
    bg: np.ndarray,
    alpha_u8: np.ndarray,
    *,
    offset_frac: float = 0.02,
    blur_frac: float = 0.03,
    opacity: float = 0.35,
) -> np.ndarray:
    """Soft shadow of the subject silhouette projected slightly down-right onto the BG."""
    h, w = alpha_u8.shape[:2]
    dy = int(h * offset_frac)
    dx = int(w * offset_frac * 0.4)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    sil = cv2.warpAffine(alpha_u8, M, (w, h))
    sil = cv2.GaussianBlur(sil, (0, 0), max(2.0, h * blur_frac))
    # only where the subject itself is NOT (shadow behind, not on subject)
    sil = np.where(alpha_u8 > 40, 0, sil)
    shade = (sil.astype(np.float32) / 255.0 * opacity)[..., None]
    return np.clip(bg.astype(np.float32) * (1.0 - shade), 0, 255).astype(np.uint8)


DEFAULT_REALISM: dict[str, Any] = {
    "light_wrap": True,
    "decontaminate": True,
    "contact_shadow": True,
    "grain_match": True,
    "wrap_px": 12,
    "wrap_strength": 0.35,
    "shadow_opacity": 0.3,
}


def composite_subject_realistic(
    canvas_bg: np.ndarray,
    frame: np.ndarray,
    alpha_u8: np.ndarray,
    opts: dict[str, Any] | None = None,
    grain_sigma: float | None = None,
) -> np.ndarray:
    """Full realism composite: shadow → grain-matched BG → spill → light wrap → over."""
    o = {**DEFAULT_REALISM, **(opts or {})}
    bg = canvas_bg
    if o.get("contact_shadow"):
        bg = contact_shadow(bg, alpha_u8, opacity=float(o.get("shadow_opacity", 0.3)))
    if o.get("grain_match"):
        sigma = grain_sigma if grain_sigma is not None else estimate_grain_sigma(frame)
        bg = add_grain(bg, sigma)
    fg = frame
    if o.get("decontaminate"):
        fg = spill_suppress(fg, alpha_u8)
    if o.get("light_wrap"):
        fg = light_wrap(fg, bg, alpha_u8, width_px=int(o.get("wrap_px", 12)), strength=float(o.get("wrap_strength", 0.35)))
    a = (alpha_u8.astype(np.float32) / 255.0)[..., None]
    out = fg.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)
