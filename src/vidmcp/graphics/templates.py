"""Graphic templates — each draws RGBA at progress p∈[0,1] with brand injection.

Template fn signature: fn(w, h, p, fields, kit) -> PIL RGBA Image.
Animation curves handled here (in/out easing on p).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from vidmcp.captions.fonts import resolve_font
from vidmcp.graphics.brand import color


def _font(kit: dict[str, Any], size: int) -> ImageFont.ImageFont:
    path = kit.get("font") or resolve_font()
    if path:
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:  # noqa: BLE001
            pass
    return ImageFont.load_default()


def ease_out_cubic(x: float) -> float:
    return 1 - (1 - min(max(x, 0.0), 1.0)) ** 3


def spring(x: float) -> float:
    import math

    x = min(max(x, 0.0), 1.0)
    return 1 - math.exp(-6 * x) * math.cos(9 * x)


def io_envelope(p: float, in_frac: float = 0.18, out_frac: float = 0.15) -> tuple[float, float]:
    """(enter_progress, alpha) — eased entrance, fade exit."""
    enter = ease_out_cubic(p / in_frac) if p < in_frac else 1.0
    alpha = 1.0 if p < 1 - out_frac else max(0.0, (1 - p) / out_frac)
    return enter, alpha


def _canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _glass_rect(draw: ImageDraw.ImageDraw, box, kit, alpha: int = 190, radius: int = 18) -> None:
    dark = color(kit, "dark")
    draw.rounded_rectangle(box, radius=radius, fill=(*dark, alpha))
    draw.rounded_rectangle(box, radius=radius, outline=(*color(kit, "primary"), min(255, alpha + 40)), width=2)


def lower_third(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    enter, alpha = io_envelope(p)
    title = str(fields.get("title", ""))
    subtitle = str(fields.get("subtitle", ""))
    f1 = _font(kit, max(20, h // 22))
    f2 = _font(kit, max(14, h // 34))
    margin = int(w * kit.get("safe_margin_pct", 5.0) / 100)
    bw = int(w * 0.36)
    bh = int(h * 0.13)
    y = int(h - margin - bh + (1 - enter) * bh * 1.4)
    x = margin
    a = int(255 * alpha)
    _glass_rect(d, (x, y, x + bw, y + bh), kit, alpha=int(190 * alpha))
    d.rectangle((x, y, x + 8, y + bh), fill=(*color(kit, "primary"), a))
    d.text((x + 26, y + int(bh * 0.16)), title, font=f1, fill=(*color(kit, "paper"), a))
    d.text((x + 26, y + int(bh * 0.58)), subtitle, font=f2, fill=(*color(kit, "muted"), a))
    return img


def title_card(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    enter, alpha = io_envelope(p, 0.25, 0.2)
    a = int(255 * alpha)
    title = str(fields.get("title", ""))
    subtitle = str(fields.get("subtitle", ""))
    f1 = _font(kit, max(30, h // 9))
    f2 = _font(kit, max(16, h // 26))
    # dark veil
    d.rectangle((0, 0, w, h), fill=(*color(kit, "dark"), int(140 * alpha)))
    tw = d.textlength(title, font=f1)
    ty = int(h * 0.4 + (1 - spring(enter)) * 40)
    d.text(((w - tw) / 2, ty), title, font=f1, fill=(*color(kit, "paper"), a))
    d.rectangle(((w - tw) / 2, ty + f1.size + 12, (w - tw) / 2 + tw * enter, ty + f1.size + 18),
                fill=(*color(kit, "primary"), a))
    sw = d.textlength(subtitle, font=f2)
    d.text(((w - sw) / 2, ty + f1.size + 34), subtitle, font=f2, fill=(*color(kit, "muted"), a))
    return img


def stat_counter(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    enter, alpha = io_envelope(p)
    a = int(255 * alpha)
    try:
        value = float(fields.get("value", 100))
    except (TypeError, ValueError):
        value = 100.0
    label = str(fields.get("label", ""))
    prefix, suffix = str(fields.get("prefix", "")), str(fields.get("suffix", ""))
    shown = value * ease_out_cubic(min(p / 0.55, 1.0))
    text = f"{prefix}{shown:,.0f}{suffix}" if value >= 10 else f"{prefix}{shown:.1f}{suffix}"
    f1 = _font(kit, max(28, h // 10))
    f2 = _font(kit, max(14, h // 30))
    bw, bh = int(w * 0.28), int(h * 0.2)
    x, y = int(w * 0.66), int(h * 0.12 + (1 - enter) * -30)
    _glass_rect(d, (x, y, x + bw, y + bh), kit, alpha=int(200 * alpha))
    d.text((x + 24, y + int(bh * 0.14)), text, font=f1, fill=(*color(kit, "secondary"), a))
    d.text((x + 24, y + int(bh * 0.66)), label, font=f2, fill=(*color(kit, "muted"), a))
    return img


def callout(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    enter, alpha = io_envelope(p)
    a = int(255 * alpha)
    text = str(fields.get("text", ""))
    tx, ty = float(fields.get("x", 0.68)), float(fields.get("y", 0.3))
    f = _font(kit, max(16, h // 26))
    px, py = int(tx * w), int(ty * h)
    tw = d.textlength(text, font=f)
    bw, bh = int(tw + 44), int(f.size * 2.2)
    _glass_rect(d, (px, py, px + bw * enter, py + bh), kit, alpha=int(200 * alpha), radius=int(bh / 2))
    if enter > 0.9:
        d.text((px + 22, py + int(bh * 0.26)), text, font=f, fill=(*color(kit, "paper"), a))
        d.ellipse((px - 10, py + bh - 6, px + 2, py + bh + 6), outline=(*color(kit, "primary"), a), width=3)
    return img


def progress_bar(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    frac = float(fields.get("fraction", p))
    bh = max(4, h // 160)
    d.rectangle((0, h - bh, w, h), fill=(*color(kit, "dark"), 160))
    d.rectangle((0, h - bh, int(w * frac), h), fill=(*color(kit, "primary"), 235))
    return img


def chapter_card(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    enter, alpha = io_envelope(p, 0.22, 0.22)
    a = int(255 * alpha)
    num = str(fields.get("number", "01"))
    title = str(fields.get("title", ""))
    f0 = _font(kit, max(40, h // 6))
    f1 = _font(kit, max(24, h // 16))
    margin = int(w * 0.07)
    y = int(h * 0.68)
    d.text((margin, y - f0.size + (1 - enter) * 30), num, font=f0,
           fill=(*color(kit, "primary"), int(a * 0.55)))
    d.text((margin + int(f0.size * 1.4), y - f1.size), title, font=f1, fill=(*color(kit, "paper"), a))
    d.rectangle((margin, y + 14, margin + int(w * 0.22 * enter), y + 18), fill=(*color(kit, "secondary"), a))
    return img


def subscribe_reminder(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    enter, alpha = io_envelope(p, 0.3, 0.25)
    a = int(255 * alpha)
    text = str(fields.get("text", "subscribe"))
    f = _font(kit, max(18, h // 28))
    tw = d.textlength(text, font=f)
    bw, bh = int(tw + 60), int(f.size * 2.4)
    x = int((w - bw) / 2)
    y = int(h * 0.84 + (1 - spring(enter)) * 50)
    d.rounded_rectangle((x, y, x + bw, y + bh), radius=int(bh / 2), fill=(*color(kit, "primary"), a))
    d.text((x + 30, y + int(bh * 0.27)), text, font=f, fill=(*color(kit, "dark"), a))
    return img


def quote_card(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    enter, alpha = io_envelope(p, 0.25, 0.2)
    a = int(255 * alpha)
    text = str(fields.get("text", ""))
    author = str(fields.get("author", ""))
    f1 = _font(kit, max(22, h // 18))
    f2 = _font(kit, max(14, h // 32))
    d.rectangle((0, 0, w, h), fill=(*color(kit, "dark"), int(120 * alpha)))
    # word-wrap
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if d.textlength(trial, font=f1) > w * 0.7 and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    y = int(h * 0.36 - len(lines) * f1.size * 0.6)
    d.text((int(w * 0.14), y - f1.size * 2), "“", font=_font(kit, f1.size * 3),
           fill=(*color(kit, "primary"), a))
    n_show = max(1, int(len(lines) * min(enter * 1.3, 1.0)))
    for i, line in enumerate(lines[:n_show]):
        d.text((int(w * 0.15), y + i * int(f1.size * 1.35)), line, font=f1, fill=(*color(kit, "paper"), a))
    d.text((int(w * 0.15), y + len(lines) * int(f1.size * 1.35) + 16), f"— {author}", font=f2,
           fill=(*color(kit, "muted"), a))
    return img


def bar_chart(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    enter, alpha = io_envelope(p, 0.4, 0.15)
    a = int(255 * alpha)
    data = fields.get("data") or [{"label": "A", "value": 3}, {"label": "B", "value": 7}]
    title = str(fields.get("title", ""))
    f1 = _font(kit, max(16, h // 30))
    f2 = _font(kit, max(12, h // 40))
    bw, bh = int(w * 0.34), int(h * 0.42)
    x0, y0 = int(w * 0.62), int(h * 0.12)
    _glass_rect(d, (x0, y0, x0 + bw, y0 + bh), kit, alpha=int(205 * alpha))
    d.text((x0 + 20, y0 + 14), title, font=f1, fill=(*color(kit, "paper"), a))
    vmax = max(float(x.get("value", 0)) for x in data) or 1.0
    n = len(data)
    slot = (bw - 60) / n
    base = y0 + bh - 40
    top = y0 + 60
    cols = ["primary", "secondary", "accent"]
    for i, item in enumerate(data):
        frac = float(item.get("value", 0)) / vmax * ease_out_cubic(min(p / 0.6, 1.0))
        bx = x0 + 30 + int(i * slot)
        bar_h = int((base - top) * frac)
        c = color(kit, cols[i % len(cols)])
        d.rounded_rectangle((bx, base - bar_h, bx + int(slot * 0.55), base), radius=5, fill=(*c, a))
        d.text((bx, base + 8), str(item.get("label", "")), font=f2, fill=(*color(kit, "muted"), a))
    return img


def line_chart(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    enter, alpha = io_envelope(p, 0.5, 0.15)
    a = int(255 * alpha)
    values = [float(v) for v in (fields.get("values") or [1, 3, 2, 5, 4, 7])]
    title = str(fields.get("title", ""))
    f1 = _font(kit, max(16, h // 30))
    bw, bh = int(w * 0.34), int(h * 0.36)
    x0, y0 = int(w * 0.62), int(h * 0.12)
    _glass_rect(d, (x0, y0, x0 + bw, y0 + bh), kit, alpha=int(205 * alpha))
    d.text((x0 + 20, y0 + 12), title, font=f1, fill=(*color(kit, "paper"), a))
    vmin, vmax = min(values), max(values)
    rng = (vmax - vmin) or 1.0
    n_show = max(2, int(len(values) * min(p / 0.7, 1.0)))
    pts = []
    for i, v in enumerate(values[:n_show]):
        px = x0 + 24 + (bw - 48) * i / (len(values) - 1)
        py = y0 + bh - 24 - (bh - 70) * (v - vmin) / rng
        pts.append((px, py))
    if len(pts) >= 2:
        d.line(pts, fill=(*color(kit, "secondary"), a), width=3)
        d.ellipse((pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5),
                  fill=(*color(kit, "primary"), a))
    return img


def animated_list(w, h, p, fields, kit) -> Image.Image:
    img, d = _canvas(w, h)
    _, alpha = io_envelope(p, 0.1, 0.15)
    a = int(255 * alpha)
    items = [str(x) for x in (fields.get("items") or [])]
    title = str(fields.get("title", ""))
    f1 = _font(kit, max(18, h // 26))
    f2 = _font(kit, max(15, h // 30))
    x0, y0 = int(w * 0.06), int(h * 0.14)
    d.text((x0, y0), title, font=f1, fill=(*color(kit, "primary"), a))
    per = 0.7 / max(len(items), 1)
    for i, item in enumerate(items):
        ip = (p - 0.15 - i * per) / per
        if ip <= 0:
            continue
        e = ease_out_cubic(min(ip, 1.0))
        y = y0 + int(f1.size * 1.6) + i * int(f2.size * 1.9)
        d.ellipse((x0 + 2, y + f2.size // 2 - 4, x0 + 10, y + f2.size // 2 + 4),
                  fill=(*color(kit, "secondary"), int(a * e)))
        d.text((x0 + 24 + int((1 - e) * 24), y), item, font=f2, fill=(*color(kit, "paper"), int(a * e)))
    return img


def logo_watermark(w, h, p, fields, kit) -> Image.Image:
    img, _ = _canvas(w, h)
    logo_path = fields.get("logo") or kit.get("logo")
    if not logo_path:
        return img
    try:
        logo = Image.open(logo_path).convert("RGBA")
    except Exception:  # noqa: BLE001
        return img
    lw = int(w * 0.09)
    logo = logo.resize((lw, int(logo.height * lw / logo.width)))
    margin = int(w * kit.get("safe_margin_pct", 5.0) / 100)
    pos = {
        "top_right": (w - lw - margin, margin),
        "top_left": (margin, margin),
        "bottom_right": (w - lw - margin, h - logo.height - margin),
        "bottom_left": (margin, h - logo.height - margin),
    }.get(kit.get("logo_position", "top_right"), (w - lw - margin, margin))
    faded = logo.copy()
    faded.putalpha(faded.getchannel("A").point(lambda v: int(v * kit.get("watermark_opacity", 0.55))))
    img.paste(faded, pos, faded)
    return img


TEMPLATES: dict[str, Callable[..., Image.Image]] = {
    "lower_third": lower_third,
    "title_card": title_card,
    "stat_counter": stat_counter,
    "callout": callout,
    "progress_bar": progress_bar,
    "chapter_card": chapter_card,
    "subscribe_reminder": subscribe_reminder,
    "quote_card": quote_card,
    "bar_chart": bar_chart,
    "line_chart": line_chart,
    "animated_list": animated_list,
    "logo_watermark": logo_watermark,
}

FIELD_DOCS: dict[str, str] = {
    "lower_third": "title, subtitle",
    "title_card": "title, subtitle",
    "stat_counter": "value, label, prefix, suffix",
    "callout": "text, x (0-1), y (0-1)",
    "progress_bar": "fraction (defaults to playback progress)",
    "chapter_card": "number, title",
    "subscribe_reminder": "text",
    "quote_card": "text, author",
    "bar_chart": "title, data: [{label, value}]",
    "line_chart": "title, values: [..]",
    "animated_list": "title, items: [..]",
    "logo_watermark": "logo (path; defaults to brand kit)",
}
