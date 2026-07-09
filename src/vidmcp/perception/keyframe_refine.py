"""Keyframe refinement — re-prompt SAM at weak frames and re-propagate (surgical matte fix)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vidmcp.perception.mask_ops import (
    coverage_mean,
    feather_mask,
    temporal_median,
    temporal_stability_score,
    to_u8_mask,
)
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import iter_frames, probe_video, write_mask_sequence

log = get_logger("vidmcp.perception.keyframe_refine")
ProgressFn = Callable[[float, str], None]


@dataclass
class KeyframeHint:
    frame_index: int
    prompt: str | None = None
    # optional geometric refine (normalized 0-1 or pixel xyxy)
    box_xyxy: list[float] | None = None
    points: list[list[float]] | None = None  # [[x,y], ...]
    point_labels: list[int] | None = None  # 1=fg 0=bg
    reason: str = ""


@dataclass
class RefineResult:
    mask_dir: Path
    keyframes: list[int]
    temporal_stability_before: float
    temporal_stability_after: float
    coverage_before: float
    coverage_after: float
    backend: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_dir": str(self.mask_dir),
            "keyframes": self.keyframes,
            "temporal_stability_before": self.temporal_stability_before,
            "temporal_stability_after": self.temporal_stability_after,
            "coverage_before": self.coverage_before,
            "coverage_after": self.coverage_after,
            "backend": self.backend,
            "meta": self.meta,
        }


def detect_weak_keyframes(
    mask_dir: Path,
    *,
    max_keyframes: int = 5,
    iou_threshold: float = 0.55,
    coverage_drop: float = 0.45,
) -> list[int]:
    """Pick frames where matte flickers or coverage collapses."""
    files = sorted(Path(mask_dir).glob("mask_*.png"))
    if len(files) < 3:
        return [0]
    masks = []
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append(to_u8_mask(m))
    scores: list[tuple[float, int]] = []  # lower = worse
    mean_cov = float(np.mean([(m > 127).mean() for m in masks]))
    for i in range(1, len(masks)):
        a, b = masks[i - 1] > 127, masks[i] > 127
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        iou = float(inter / union) if union else 1.0
        cov = float(b.mean())
        penalty = 0.0
        if iou < iou_threshold:
            penalty += (iou_threshold - iou) * 2
        if mean_cov > 1e-6 and cov < mean_cov * coverage_drop:
            penalty += 1.0
        if penalty > 0:
            scores.append((penalty, i))
    scores.sort(reverse=True)
    kfs = [0]  # always include start
    for _, idx in scores:
        if idx not in kfs:
            kfs.append(idx)
        if len(kfs) >= max_keyframes:
            break
    return sorted(kfs)


def refine_masks_local(
    video_path: Path,
    mask_dir: Path,
    *,
    output_dir: Path,
    keyframes: list[int] | None = None,
    hints: list[KeyframeHint] | None = None,
    prompt: str = "person",
    feather: int = 5,
    window: int = 8,
    progress: ProgressFn | None = None,
) -> RefineResult:
    """
    Local refine path that works without GPU SAM:

    1) Detect weak keyframes (or use provided)
    2) At each keyframe, rebuild mask with stronger center+motion prior
    3) Blend refined keyframe masks into neighbors (temporal heal)
    4) Temporal median smooth

    When official SAM backend is available, ``refine_with_sam_backend`` is preferred.
    """
    video_path = Path(video_path)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(mask_dir.glob("mask_*.png"))
    before = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    before = [to_u8_mask(m) for m in before if m is not None]
    stab_b = temporal_stability_score(before)
    cov_b = coverage_mean(before)

    if keyframes is None:
        if hints:
            keyframes = sorted({h.frame_index for h in hints})
        else:
            keyframes = detect_weak_keyframes(mask_dir)
    if not keyframes:
        keyframes = [0]

    meta = probe_video(video_path)
    # index frames we need
    needed = set()
    for kf in keyframes:
        for j in range(max(0, kf - window), min(len(before), kf + window + 1)):
            needed.add(j)

    frame_cache: dict[int, np.ndarray] = {}
    for idx, frame in iter_frames(video_path):
        if idx in needed:
            frame_cache[idx] = frame
        if idx > max(needed, default=0):
            break

    refined = [m.copy() for m in before]
    hint_map = {h.frame_index: h for h in (hints or [])}

    for ki, kf in enumerate(keyframes):
        if progress:
            progress(ki / max(len(keyframes), 1), f"refine keyframe {kf}")
        frame = frame_cache.get(kf)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        base = refined[kf] if kf < len(refined) else np.zeros((h, w), dtype=np.uint8)
        improved = _reestimate_mask(frame, base, prompt=prompt, hint=hint_map.get(kf))
        improved = feather_mask(improved, feather)
        refined[kf] = improved
        # heal neighbors toward keyframe
        for j in range(max(0, kf - window), min(len(refined), kf + window + 1)):
            if j == kf:
                continue
            dist = abs(j - kf)
            alpha = max(0.15, 1.0 - dist / max(window, 1)) * 0.65
            a = refined[j].astype(np.float32)
            b = improved.astype(np.float32)
            blended = (1 - alpha) * a + alpha * b
            refined[j] = np.clip(blended, 0, 255).astype(np.uint8)

    refined = temporal_median(refined, window=3)
    write_mask_sequence(refined, output_dir, prefix="mask")
    stab_a = temporal_stability_score(refined)
    cov_a = coverage_mean(refined)
    if progress:
        progress(1.0, "keyframe refine done")
    return RefineResult(
        mask_dir=output_dir,
        keyframes=keyframes,
        temporal_stability_before=stab_b,
        temporal_stability_after=stab_a,
        coverage_before=cov_b,
        coverage_after=cov_a,
        backend="local_keyframe_heal",
        meta={"window": window, "n_masks": len(refined), "video_fps": meta.fps},
    )


def refine_with_sam_backend(
    backend: Any,
    video_path: Path,
    *,
    output_dir: Path,
    prompt: str,
    hints: list[KeyframeHint],
    conf: float = 0.25,
    feather: int = 3,
    progress: ProgressFn | None = None,
    previous_mask_dir: Path | None = None,
) -> RefineResult:
    """Use official/ultralytics backend: multi keyframe text/box prompts + full propagate."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stab_b = cov_b = 0.0
    if previous_mask_dir and Path(previous_mask_dir).exists():
        files = sorted(Path(previous_mask_dir).glob("mask_*.png"))
        ms = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
        ms = [to_u8_mask(m) for m in ms if m is not None]
        if ms:
            stab_b = temporal_stability_score(ms)
            cov_b = coverage_mean(ms)

    # Prefer multi-prompt at frame 0 + resegment; if backend supports keyframe list, pass meta
    prompts = [prompt] + [h.prompt for h in hints if h.prompt and h.prompt != prompt]
    # unique
    seen = set()
    plist = []
    for p in prompts:
        if p and p not in seen:
            seen.add(p)
            plist.append(p)

    if hasattr(backend, "segment_multi"):
        result = backend.segment_multi(
            video_path,
            plist or [prompt],
            output_dir=output_dir,
            conf=conf,
            progress=progress,
            feather=feather,
            primary_prompt=prompt,
            frame_index=hints[0].frame_index if hints else 0,
            keep_per_object=True,
            keyframe_hints=[
                {
                    "frame_index": h.frame_index,
                    "prompt": h.prompt or prompt,
                    "box_xyxy": h.box_xyxy,
                    "points": h.points,
                    "point_labels": h.point_labels,
                }
                for h in hints
            ],
        )
    else:
        result = backend.segment_video(
            video_path,
            prompt,
            output_dir=output_dir,
            conf=conf,
            progress=progress,
            feather=feather,
        )

    # If we still have previous masks, optional blend at non-keyframe regions could go here
    return RefineResult(
        mask_dir=result.mask_dir,
        keyframes=[h.frame_index for h in hints] or [0],
        temporal_stability_before=stab_b,
        temporal_stability_after=result.temporal_stability,
        coverage_before=cov_b,
        coverage_after=result.coverage_mean,
        backend=f"sam_refine:{result.backend}",
        meta={"objects": len(result.objects), "prompts": plist},
    )


def _reestimate_mask(
    frame: np.ndarray,
    base: np.ndarray,
    *,
    prompt: str,
    hint: KeyframeHint | None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    if base.shape[:2] != (h, w):
        base = cv2.resize(base, (w, h))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    yy, xx = np.mgrid[0:h, 0:w]
    # center of mass of base mask as prior center
    ys, xs = np.where(base > 127)
    if len(xs) > 0:
        cx, cy = float(xs.mean()), float(ys.mean())
    else:
        cx, cy = w * 0.5, h * 0.45
    if hint and hint.box_xyxy and len(hint.box_xyxy) == 4:
        x0, y0, x1, y1 = hint.box_xyxy
        # normalized?
        if max(hint.box_xyxy) <= 1.5:
            x0, x1 = x0 * w, x1 * w
            y0, y1 = y0 * h, y1 * h
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(box_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    else:
        box_mask = None

    prior = np.exp(-(((xx - cx) ** 2) / (2 * (w * 0.2) ** 2) + ((yy - cy) ** 2) / (2 * (h * 0.28) ** 2)))
    prior = (prior * 255).astype(np.uint8)
    score = (
        0.40 * base.astype(np.float32)
        + 0.25 * prior.astype(np.float32)
        + 0.20 * skin.astype(np.float32)
        + 0.15 * edges.astype(np.float32)
    )
    if box_mask is not None:
        score = score * 0.5 + box_mask.astype(np.float32) * 0.5
    if hint and hint.points:
        for i, pt in enumerate(hint.points):
            x, y = pt[0], pt[1]
            if x <= 1.5 and y <= 1.5:
                x, y = x * w, y * h
            lab = 1
            if hint.point_labels and i < len(hint.point_labels):
                lab = hint.point_labels[i]
            cv2.circle(score, (int(x), int(y)), 25, 255 if lab else 0, -1)

    binary = (score > 90).astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ker)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        binary = np.where(labels == largest, 255, 0).astype(np.uint8)
    return binary
