"""Style / look emulation — grade fingerprint + grain + vignette + halation as one layer.

Cheap, always-works path for "make it look like this reference". Neural style transfer
slot reserved via models_registry (onnx), off by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidmcp.effects.base import Effect, EffectContext
from vidmcp.models.layers import EffectParams, Layer, LayerKind
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.style")


def _match_curves(img: np.ndarray, fingerprint: dict[str, Any], strength: float) -> np.ndarray:
    """Quantile-curve remap per channel toward the reference look."""
    qs = np.array(fingerprint["quantiles"], np.float32)
    out = img.astype(np.float32)
    for i, ch in enumerate("bgr"):
        ref_vals = np.array(fingerprint["curves"][ch], np.float32)
        src_vals = np.array([np.percentile(img[:, :, i], q) for q in qs], np.float32)
        lut = np.interp(np.arange(256, dtype=np.float32), src_vals, ref_vals)
        mapped = lut[img[:, :, i]]
        out[:, :, i] = out[:, :, i] * (1 - strength) + mapped * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def vignette(img: np.ndarray, amount: float = 0.25) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    mask = 1.0 - amount * np.clip(d - 0.5, 0, 1) ** 2 * 2
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def halation(img: np.ndarray, amount: float = 0.2) -> np.ndarray:
    """Highlight bloom with warm tint (film halation feel)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hi = np.where(gray > 210, gray, 0).astype(np.float32)
    bloom = cv2.GaussianBlur(hi, (0, 0), 12)[..., None] / 255.0
    tint = np.array([0.4, 0.6, 1.0], np.float32)  # warm (BGR)
    return np.clip(img.astype(np.float32) + bloom * tint * amount * 255, 0, 255).astype(np.uint8)


class StyleLookEffect(Effect):
    name = "style_look"
    kind = "grade"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        assert ctx.source_frame is not None
        p = params.params
        out = ctx.source_frame
        strength = float(np.clip(params.intensity, 0, 1))
        if p.get("fingerprint"):
            out = _match_curves(out, p["fingerprint"], strength)
        if p.get("lut"):
            from vidmcp.color.lut import apply_lut, resolve_table

            out = apply_lut(out, resolve_table(str(p["lut"])), strength)
        if p.get("grain"):
            from vidmcp.compositor.realism import add_grain

            out = add_grain(out, float(p["grain"]), seed=ctx.frame_index % 97)
        if p.get("vignette"):
            out = vignette(out, float(p["vignette"]))
        if p.get("halation"):
            out = halation(out, float(p["halation"]))
        return out


def apply_style_project(
    project: Any,
    reference: str | None = None,
    lut: str | None = None,
    grain: float = 1.2,
    vignette_amount: float = 0.22,
    halation_amount: float = 0.15,
    intensity: float = 0.85,
) -> dict[str, Any]:
    m = project.manifest
    fingerprint = None
    if reference:
        from vidmcp.color.match import grade_fingerprint

        ref_path = Path(reference).expanduser()
        img = None
        if ref_path.exists():
            if ref_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                img = cv2.imread(str(ref_path))
            else:
                from vidmcp.utils.video_io import sample_frames

                frames = sample_frames(ref_path, max_frames=1)
                img = frames[0][2] if frames else None
        if img is None:
            return {"ok": False, "message": f"Cannot read reference '{reference}'"}
        fingerprint = grade_fingerprint(img)
    layer = m.layers.add(
        Layer(
            name=f"style_{Path(reference).stem if reference else lut or 'look'}",
            kind=LayerKind.GRADE,
            z_index=42,
            effect=EffectParams(
                effect_type="style_look",
                intensity=intensity,
                params={
                    "fingerprint": fingerprint,
                    "lut": lut,
                    "grain": grain,
                    "vignette": vignette_amount,
                    "halation": halation_amount,
                },
            ),
        )
    )
    m.append_history("apply_style", {"reference": reference, "lut": lut})
    project.save()
    return {"ok": True, "layer_id": layer.id, "has_fingerprint": fingerprint is not None, "lut": lut}
