"""Shot-to-shot + reference color matching — regularized Lab statistics transfer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.color.grade import skin_mask
from vidmcp.effects.base import Effect, EffectContext
from vidmcp.models.layers import EffectParams, Layer, LayerKind
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import sample_frames

log = get_logger("vidmcp.color_match")


def lab_stats(img: np.ndarray) -> tuple[list[float], list[float]]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    return [float(x) for x in flat.mean(axis=0)], [float(x) for x in flat.std(axis=0) + 1e-6]


def transfer_lab(img: np.ndarray, src_mean, src_std, dst_mean, dst_std, strength: float = 0.8, max_shift: float = 18.0) -> np.ndarray:
    """Regularized Reinhard transfer: clamp mean shift, protect skin chroma."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = lab.copy()
    for c in range(3):
        scale = np.clip(dst_std[c] / max(src_std[c], 1e-6), 0.6, 1.6)
        shift = np.clip(dst_mean[c] - src_mean[c] * scale, -max_shift, max_shift)
        out[:, :, c] = lab[:, :, c] * scale + shift
    out = np.clip(out, 0, 255)
    blended = lab * (1 - strength) + out * strength
    result = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_LAB2BGR)
    w = skin_mask(img)[..., None] * 0.5
    return (img.astype(np.float32) * w + result.astype(np.float32) * (1 - w)).astype(np.uint8)


def grade_fingerprint(img: np.ndarray, n_points: int = 9) -> dict[str, Any]:
    """Compact look descriptor: per-channel quantile curves + saturation."""
    qs = np.linspace(2, 98, n_points)
    curves = {}
    for i, ch in enumerate("bgr"):
        curves[ch] = [float(np.percentile(img[:, :, i], q)) for q in qs]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return {"quantiles": [float(q) for q in qs], "curves": curves, "saturation": float(hsv[:, :, 1].mean())}


class ColorMatchEffect(Effect):
    """Per-shot Lab transfer toward a reference; params baked at plan time."""

    name = "color_match"
    kind = "grade"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        assert ctx.source_frame is not None
        p = params.params
        shots = p.get("shots") or []
        cur = None
        for s in shots:
            if s["start"] <= ctx.timestamp < s["end"]:
                cur = s
                break
        if cur is None or cur.get("identity"):
            return ctx.source_frame
        return transfer_lab(
            ctx.source_frame,
            cur["src_mean"], cur["src_std"],
            p["dst_mean"], p["dst_std"],
            strength=float(p.get("strength", 0.8)),
        )


def _reference_frame(project: Any, reference: str) -> np.ndarray | None:
    if reference.startswith("shot:"):
        idx = int(reference.split(":", 1)[1])
        shots = project.manifest.analysis.get("shots") or []
        if idx >= len(shots):
            return None
        video = project.abs(project.manifest.source_video)
        mid_t = (shots[idx]["start"] + shots[idx]["end"]) / 2
        from vidmcp.utils.video_io import probe_video

        meta = probe_video(video)
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_t * meta.fps))
        ok, img = cap.read()
        cap.release()
        return img if ok else None
    p = Path(reference).expanduser()
    if not p.exists():
        return None
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return cv2.imread(str(p))
    frames = sample_frames(p, max_frames=1)
    return frames[0][2] if frames else None


def match_color_project(project: Any, reference: str = "shot:0", strength: float = 0.8) -> dict[str, Any]:
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    ref = _reference_frame(project, reference)
    if ref is None:
        return {"ok": False, "message": f"Cannot resolve reference '{reference}'"}
    dst_mean, dst_std = lab_stats(ref)

    video = project.abs(m.source_video)
    shots = m.analysis.get("shots") or []
    if not shots:
        from vidmcp.utils.video_io import probe_video

        shots = [{"start": 0.0, "end": probe_video(video).duration_sec}]
    frames = sample_frames(video, max_frames=60, max_side=480)
    by_ts = [(ts, img) for _, ts, img in frames]

    shot_specs = []
    deltas = []
    for s in shots:
        imgs = [img for ts, img in by_ts if s["start"] <= ts < s["end"]]
        if not imgs:
            shot_specs.append({"start": s["start"], "end": s["end"], "identity": True})
            continue
        src_mean, src_std = lab_stats(imgs[len(imgs) // 2])
        delta = float(np.abs(np.array(src_mean) - np.array(dst_mean)).mean())
        deltas.append(delta)
        shot_specs.append(
            {"start": s["start"], "end": s["end"], "src_mean": src_mean, "src_std": src_std, "delta": round(delta, 2)}
        )

    layer = m.layers.add(
        Layer(
            name="color_match",
            kind=LayerKind.GRADE,
            z_index=39,
            effect=EffectParams(
                effect_type="color_match",
                params={"shots": shot_specs, "dst_mean": dst_mean, "dst_std": dst_std, "strength": strength},
            ),
        )
    )
    m.append_history("match_color", {"reference": reference, "n_shots": len(shot_specs)})
    project.save()
    return {
        "ok": True,
        "layer_id": layer.id,
        "reference": reference,
        "n_shots": len(shot_specs),
        "mean_delta": round(float(np.mean(deltas)) if deltas else 0.0, 2),
        "fingerprint": grade_fingerprint(ref),
    }
