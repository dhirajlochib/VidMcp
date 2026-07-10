"""Cross-shot identity lock: appearance signature + masklet re-ID across cuts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from vidmcp.perception.mask_ops import to_u8_mask
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import sample_frames

log = get_logger("vidmcp.identity")


@dataclass
class IdentitySignature:
    global_id: str
    label: str
    color_hist: np.ndarray  # 32-bin HSV H hist
    mean_lab: np.ndarray
    area_ratio: float
    shot_ids: list[str] = field(default_factory=list)

    def distance(self, other: IdentitySignature) -> float:
        h1 = self.color_hist.astype(np.float32) + 1e-6
        h2 = other.color_hist.astype(np.float32) + 1e-6
        h1 = h1 / h1.sum()
        h2 = h2 / h2.sum()
        hist_d = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
        lab_d = float(np.linalg.norm(self.mean_lab - other.mean_lab) / 100.0)
        area_d = abs(self.area_ratio - other.area_ratio)
        return 0.55 * hist_d + 0.35 * lab_d + 0.10 * area_d


def _signature_from_frame(frame: np.ndarray, mask: np.ndarray, label: str, shot_id: str) -> IdentitySignature:
    m = to_u8_mask(mask) > 127
    if m.sum() < 16:
        hist = np.zeros(32, dtype=np.float32)
        lab_mean = np.zeros(3, dtype=np.float32)
        area = 0.0
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        sel_h = hsv[:, :, 0][m]
        hist, _ = np.histogram(sel_h, bins=32, range=(0, 180))
        hist = hist.astype(np.float32)
        lab_mean = lab[m].mean(axis=0).astype(np.float32)
        area = float(m.mean())
    return IdentitySignature(
        global_id=str(uuid4())[:8],
        label=label,
        color_hist=hist,
        mean_lab=lab_mean,
        area_ratio=area,
        shot_ids=[shot_id],
    )


@dataclass
class IdentityLock:
    identities: list[IdentitySignature] = field(default_factory=list)
    threshold: float = 0.45

    def assign(self, sig: IdentitySignature) -> str:
        best_id, best_d = None, 1e9
        for existing in self.identities:
            d = existing.distance(sig)
            if d < best_d:
                best_d, best_id = d, existing.global_id
                best_existing = existing
        if best_id is not None and best_d <= self.threshold:
            best_existing.shot_ids = list(dict.fromkeys(best_existing.shot_ids + sig.shot_ids))
            # EMA update appearance
            best_existing.color_hist = 0.7 * best_existing.color_hist + 0.3 * sig.color_hist
            best_existing.mean_lab = 0.7 * best_existing.mean_lab + 0.3 * sig.mean_lab
            best_existing.area_ratio = 0.7 * best_existing.area_ratio + 0.3 * sig.area_ratio
            return best_id
        self.identities.append(sig)
        return sig.global_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "identities": [
                {
                    "global_id": i.global_id,
                    "label": i.label,
                    "area_ratio": i.area_ratio,
                    "shot_ids": i.shot_ids,
                    "mean_lab": i.mean_lab.tolist(),
                }
                for i in self.identities
            ]
        }


def lock_identity_across_shots(
    shots: list[dict[str, Any]],
    *,
    threshold: float = 0.45,
) -> dict[str, Any]:
    """
    shots: [{shot_id, video_path, mask_dir, label?}, ...]
    Returns global_id mapping per shot.
    """
    lock = IdentityLock(threshold=threshold)
    assignments = []
    for shot in shots:
        shot_id = shot.get("shot_id") or str(uuid4())[:8]
        label = shot.get("label") or "person"
        video = Path(shot["video_path"])
        mask_dir = Path(shot["mask_dir"])
        frames = sample_frames(video, max_frames=3, max_side=480)
        mask_files = sorted(mask_dir.glob("mask_*.png"))
        if not frames or not mask_files:
            continue
        # middle sample
        fi, _, frame = frames[len(frames) // 2]
        mi = min(fi, len(mask_files) - 1)
        mask = cv2.imread(str(mask_files[mi]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        sig = _signature_from_frame(frame, mask, label, shot_id)
        gid = lock.assign(sig)
        assignments.append({"shot_id": shot_id, "global_id": gid, "label": label})
    return {"ok": True, "assignments": assignments, "registry": lock.to_dict()}
