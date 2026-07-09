"""Mask post-processing: feather, temporal smooth, stability metrics."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def to_u8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        m = mask
    elif mask.max() <= 1.0:
        m = (np.clip(mask.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    else:
        m = np.clip(mask, 0, 255).astype(np.uint8)
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m


def feather_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    m = to_u8_mask(mask)
    if radius <= 0:
        return m
    k = radius * 2 + 1
    return cv2.GaussianBlur(m, (k, k), 0)


def morph_clean(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    m = to_u8_mask(mask)
    if open_k > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    if close_k > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    return m


def temporal_median(masks: list[np.ndarray], window: int = 3) -> list[np.ndarray]:
    if window <= 1 or len(masks) < 3:
        return [to_u8_mask(m) for m in masks]
    half = window // 2
    out: list[np.ndarray] = []
    u8 = [to_u8_mask(m) for m in masks]
    for i in range(len(u8)):
        lo = max(0, i - half)
        hi = min(len(u8), i + half + 1)
        stack = np.stack(u8[lo:hi], axis=0)
        out.append(np.median(stack, axis=0).astype(np.uint8))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = to_u8_mask(a) > 127
    b_b = to_u8_mask(b) > 127
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def temporal_stability_score(masks: Iterable[np.ndarray]) -> float:
    ms = [to_u8_mask(m) for m in masks]
    if len(ms) < 2:
        return 1.0
    scores = [iou(ms[i], ms[i + 1]) for i in range(len(ms) - 1)]
    return float(np.mean(scores)) if scores else 1.0


def coverage_mean(masks: Iterable[np.ndarray]) -> float:
    vals = []
    for m in masks:
        u = to_u8_mask(m)
        vals.append(float((u > 127).mean()))
    return float(np.mean(vals)) if vals else 0.0


def union_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("no masks")
    acc = np.zeros_like(to_u8_mask(masks[0]), dtype=np.uint8)
    for m in masks:
        acc = np.maximum(acc, to_u8_mask(m))
    return acc
