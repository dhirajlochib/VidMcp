"""Auto white balance + exposure — per-shot constants, skin-sane, subject-anchored."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from vidmcp.color.grade import skin_mask
from vidmcp.effects.base import Effect, EffectContext
from vidmcp.models.layers import EffectParams, Layer, LayerKind
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import sample_frames

log = get_logger("vidmcp.auto_color")


def estimate_wb_gains(img: np.ndarray) -> tuple[float, float, float]:
    """Gray-world + white-patch fusion → per-channel gains (B,G,R), clamped."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    gw = means.mean() / means
    # white patch: top 2% brightest pixels should be neutral
    gray = f.mean(axis=2)
    thresh = np.percentile(gray, 98)
    bright = f[gray >= thresh].reshape(-1, 3)
    wp = np.ones(3, np.float32)
    if len(bright) > 20:
        bm = bright.mean(axis=0) + 1e-6
        wp = bm.mean() / bm
    gains = 0.6 * gw + 0.4 * wp
    gains = np.clip(gains, 0.75, 1.35)
    return float(gains[0]), float(gains[1]), float(gains[2])


def estimate_exposure_gamma(img: np.ndarray, subject_mask: np.ndarray | None = None) -> float:
    """Gamma that anchors subject (or frame) median to mid-gray, with roll-off clamp."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    region = gray[subject_mask > 127] if subject_mask is not None and (subject_mask > 127).sum() > 100 else gray
    med = float(np.median(region))
    if med < 0.02:
        return 1.0
    gamma = np.log(max(med, 1e-3)) / np.log(0.42)
    return float(np.clip(gamma, 0.6, 1.6))


def correct_frame(img: np.ndarray, gains: tuple[float, float, float], gamma: float, skin_guard: bool = True) -> np.ndarray:
    f = img.astype(np.float32)
    f *= np.array(gains, np.float32)[None, None, :]
    f = np.clip(f, 0, 255) / 255.0
    f = np.clip(f ** (1.0 / max(gamma, 1e-3)), 0, 1) * 255.0
    out = f.astype(np.uint8)
    if skin_guard:
        w = skin_mask(img)[..., None] * 0.5
        out = (img.astype(np.float32) * w + out.astype(np.float32) * (1 - w)).astype(np.uint8)
    return out


class AutoColorEffect(Effect):
    """Per-shot WB/exposure params baked at plan time; looked up by timestamp."""

    name = "auto_color"
    kind = "grade"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        assert ctx.source_frame is not None
        shots = params.params.get("shots") or []
        cur = None
        for s in shots:
            if s["start"] <= ctx.timestamp < s["end"]:
                cur = s
                break
        if cur is None:
            cur = shots[-1] if shots else {"gains": [1, 1, 1], "gamma": 1.0}
        return correct_frame(
            ctx.source_frame,
            tuple(cur.get("gains", [1, 1, 1])),
            float(cur.get("gamma", 1.0)),
            skin_guard=bool(params.params.get("protect_skin", True)),
        )


def auto_color_project(
    project: Any,
    wb: bool = True,
    exposure: bool = True,
    per_shot: bool = True,
) -> dict[str, Any]:
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    video = project.abs(m.source_video)
    shots = m.analysis.get("shots") or []
    if not shots or not per_shot:
        from vidmcp.utils.video_io import probe_video

        shots = [{"start": 0.0, "end": probe_video(video).duration_sec}]

    shot_params: list[dict[str, Any]] = []
    frames = sample_frames(video, max_frames=60, max_side=480)
    by_ts = [(ts, img) for _, ts, img in frames]
    for s in shots:
        imgs = [img for ts, img in by_ts if s["start"] <= ts < s["end"]]
        if not imgs:
            shot_params.append({"start": s["start"], "end": s["end"], "gains": [1, 1, 1], "gamma": 1.0})
            continue
        mid = imgs[len(imgs) // 2]
        gains = estimate_wb_gains(mid) if wb else (1.0, 1.0, 1.0)
        gamma = estimate_exposure_gamma(mid) if exposure else 1.0
        shot_params.append(
            {"start": s["start"], "end": s["end"], "gains": [round(g, 4) for g in gains], "gamma": round(gamma, 4)}
        )

    layer = m.layers.add(
        Layer(
            name="auto_color",
            kind=LayerKind.GRADE,
            z_index=38,
            effect=EffectParams(effect_type="auto_color", params={"shots": shot_params, "protect_skin": True}),
        )
    )
    m.append_history("auto_color", {"n_shots": len(shot_params)})
    project.save()
    return {"ok": True, "layer_id": layer.id, "n_shots": len(shot_params), "per_shot_params": shot_params[:6]}
