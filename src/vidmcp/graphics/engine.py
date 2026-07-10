"""Graphics engine — renders declarative item specs as an RGBA overlay layer.

Item spec: {template, t, duration, fields?, brand?, lock?: "speech:<keyword>"}.
Registered as effect 'graphics' (returns BGRA; compositor alpha-blends onto canvas).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from vidmcp.effects.base import Effect, EffectContext
from vidmcp.graphics.brand import get_brand_kit
from vidmcp.graphics.templates import FIELD_DOCS, TEMPLATES
from vidmcp.models.layers import EffectParams, Layer, LayerKind
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.graphics")


class GraphicsOverlayEffect(Effect):
    name = "graphics"
    kind = "overlay"

    def render_frame(self, ctx: EffectContext, params: EffectParams) -> np.ndarray:
        items = params.params.get("items") or []
        kit = get_brand_kit(str(params.params.get("brand", "default")))
        canvas = Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))
        t = ctx.timestamp
        for item in items:
            t0 = float(item.get("t", 0.0))
            dur = float(item.get("duration", 3.0))
            if not (t0 <= t <= t0 + dur):
                continue
            fn = TEMPLATES.get(str(item.get("template")))
            if fn is None:
                continue
            p = (t - t0) / max(dur, 1e-3)
            fields = dict(item.get("fields") or {})
            if item.get("template") == "progress_bar" and "fraction" not in fields:
                total = float(item.get("total_duration", 0)) or None
                if total:
                    fields["fraction"] = min(t / total, 1.0)
            try:
                layer_img = fn(ctx.width, ctx.height, p, fields, kit)
                canvas = Image.alpha_composite(canvas, layer_img)
            except Exception as e:  # noqa: BLE001
                log.warning("template_render_failed", template=item.get("template"), error=str(e))
        rgba = np.array(canvas)  # RGBA
        bgra = rgba[:, :, [2, 1, 0, 3]].copy()
        return bgra


def _resolve_speech_locks(project: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """lock: 'speech:<keyword>' → set t to first keyword hit in the word timeline."""
    locked = [i for i in items if str(i.get("lock", "")).startswith("speech:")]
    if not locked:
        return items
    try:
        from vidmcp.perception.indexer import load_index

        index = load_index(project) or {}
        words = index.get("words") or []
    except Exception:  # noqa: BLE001
        words = []
    for item in locked:
        kw = str(item["lock"]).split(":", 1)[1].strip().lower()
        for w in words:
            if kw in str(w.get("word", "")).lower():
                item["t"] = max(0.0, float(w["start"]) - 0.15)
                item["locked_word"] = kw
                break
    return items


def add_graphics_project(
    project: Any,
    items: list[dict[str, Any]],
    brand: str = "default",
) -> dict[str, Any]:
    m = project.manifest
    bad = [i.get("template") for i in items if i.get("template") not in TEMPLATES]
    if bad:
        return {"ok": False, "message": f"Unknown templates {bad}. Available: {sorted(TEMPLATES)}"}
    items = _resolve_speech_locks(project, [dict(i) for i in items])
    duration = (m.source_meta or {}).get("duration") or (m.analysis or {}).get("duration_sec")
    for i in items:
        if i.get("template") == "progress_bar" and duration:
            i["total_duration"] = float(duration)
            i.setdefault("duration", float(duration))
    layer = m.layers.add(
        Layer(
            name=f"graphics_{len(items)}items",
            kind=LayerKind.OVERLAY,
            z_index=80,
            effect=EffectParams(effect_type="graphics", params={"items": items, "brand": brand}),
            meta={"graphics": True},
        )
    )
    m.append_history("add_graphics", {"n_items": len(items), "templates": [i["template"] for i in items]})
    project.save()
    return {
        "ok": True,
        "layer_id": layer.id,
        "n_items": len(items),
        "items": [{"template": i["template"], "t": i.get("t"), "duration": i.get("duration")} for i in items],
    }


def list_graphic_templates() -> dict[str, Any]:
    return {
        "ok": True,
        "templates": [{"name": name, "fields": FIELD_DOCS.get(name, "")} for name in sorted(TEMPLATES)],
        "item_spec": "{template, t, duration, fields?, lock?: 'speech:<keyword>'}",
    }
