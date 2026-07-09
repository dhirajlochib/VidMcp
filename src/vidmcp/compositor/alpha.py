"""Alpha compositing utilities."""

from __future__ import annotations

import cv2
import numpy as np

from vidmcp.perception.mask_ops import to_u8_mask


def load_mask_for_frame(mask_dir: str | None, frame_index: int, h: int, w: int) -> np.ndarray | None:
    if not mask_dir:
        return None
    from pathlib import Path

    p = Path(mask_dir) / f"mask_{frame_index:06d}.png"
    if not p.exists():
        # try nearest existing
        files = sorted(Path(mask_dir).glob("mask_*.png"))
        if not files:
            return None
        idx = min(frame_index, len(files) - 1)
        p = files[idx]
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return m


def over(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Porter-Duff source-over. alpha is u8 HxW or HxWx1."""
    a = to_u8_mask(alpha).astype(np.float32) / 255.0 * float(opacity)
    if a.ndim == 2:
        a = a[..., None]
    bg_f = bg.astype(np.float32)
    fg_f = fg.astype(np.float32)
    if fg_f.shape[2] == 4:
        a = a * (fg_f[:, :, 3:4] / 255.0)
        fg_f = fg_f[:, :, :3]
    out = fg_f * a + bg_f * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def screen_blend(base: np.ndarray, layer: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    b = base.astype(np.float32) / 255.0
    l = layer.astype(np.float32) / 255.0
    out = 1.0 - (1.0 - b) * (1.0 - l)
    out = b * (1 - opacity) + out * opacity
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def add_blend(base: np.ndarray, layer: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    out = base.astype(np.float32) + layer.astype(np.float32) * opacity
    return np.clip(out, 0, 255).astype(np.uint8)
