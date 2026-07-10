#!/usr/bin/env python3
"""Time dilation lesson v2 — premium graphics, cinematic BGM, pro vocal chain."""
from __future__ import annotations

import math
import struct
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORK = Path("/tmp/vidmcp_img4447_v2")
SRC_V = Path("/tmp/vidmcp_img4447/video_cut.mp4")
SRC_VOX = Path("/tmp/vidmcp_img4447/vocals_cut.wav")  # hesitation already cut
FALLBACK_VOX = Path("/tmp/vidmcp_img4447/final_audio.wav")
OUT = Path("/Users/dhirajlochib/Developer/VidMcp/final_output/time_dilation_edit.mp4")
DESK = Path("/Users/dhirajlochib/Desktop/time_dilation_edit.mp4")
FONTS = Path("/Users/dhirajlochib/Developer/VidMcp/scripts/fonts")

LIME = (212, 255, 42)
CYAN = (42, 255, 209)
VIOLET = (177, 140, 255)
PEACH = (255, 162, 78)
PAPER = (239, 239, 235)
MUTED = (155, 160, 170)
DARK = (6, 8, 14)
GOLD = (255, 200, 90)

# Timeline (hesitation removed)
CAPTIONS = [
    (1.30, 5.80, "Time isn’t fixed — Einstein said it’s relative."),
    (6.20, 7.55, "So what does this mean?"),
    (7.70, 12.50, "Take GPS satellites — they need time calibration."),
    (12.80, 18.80, "About 38 microseconds of drift — every day."),
    (19.00, 26.50, "Without Earth sync, navigation would fail."),
    (27.00, 34.50, "Twin paradox: travel near light speed…"),
    (34.80, 41.50, "5 years for the traveler ~ 835 years on Earth."),
    (42.00, 49.50, "That’s time dilation — a mind-bending fact."),
    (50.50, 54.20, "The traveler only ages 5 years."),
    (54.60, 57.50, "Like · Share · Subscribe"),
]

BEATS = [
    (0.0, 5.8, "title", {"title": "TIME DILATION", "sub": "Einstein · relativity · reality"}),
    (5.5, 7.8, "quote", {"q": "Time isn’t fixed.", "a": "It’s relative."}),
    (7.8, 14.2, "gps", {}),
    (13.8, 20.5, "micros", {"value": 38, "unit": "us / day", "label": "GPS clock drift"}),
    (20.0, 27.2, "earth_sync", {}),
    (27.0, 35.2, "twins", {}),
    (34.8, 43.5, "big_num", {"value": 835, "unit": "YEARS ON EARTH", "sub": "while the traveler ages only 5"}),
    (42.5, 52.0, "formula", {}),
    (51.5, 57.9, "outro", {}),
]

CHAPTERS = [
    (0.0, 7.5, "01", "Relative time"),
    (7.5, 20.0, "02", "GPS clocks"),
    (20.0, 27.0, "03", "Earth sync"),
    (27.0, 43.0, "04", "Twin paradox"),
    (43.0, 58.0, "05", "The proof"),
]


def font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except Exception:
        return ImageFont.load_default()


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def beat_alpha(t: float, a: float, b: float, fade: float = 0.4) -> float:
    if t < a or t > b:
        return 0.0
    al = 1.0
    if t < a + fade:
        al = ease((t - a) / fade)
    if t > b - fade:
        al = min(al, ease((b - t) / fade))
    return al


def text_size(draw, text, f):
    b = draw.textbbox((0, 0), text, font=f)
    return b[2] - b[0], b[3] - b[1]


def wrap(draw, text, f, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if text_size(draw, trial, f)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def rr(draw, xy, r, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def glass_panel(draw, xy, r, al, accent=CYAN, thick=2):
    """Frosted dark glass card."""
    x0, y0, x1, y1 = xy
    rr(draw, xy, r, (8, 10, 16, int(215 * al)), outline=(*accent, int(140 * al)), width=thick)
    # top highlight line
    draw.line((x0 + r, y0 + 1, x1 - r, y0 + 1), fill=(255, 255, 255, int(40 * al)), width=1)
    # left accent bar
    draw.rectangle((x0, y0 + r, x0 + 3, y1 - r), fill=(*accent, int(220 * al)))


def alpha_composite_bgr(base_bgr: np.ndarray, overlay_rgba: Image.Image) -> np.ndarray:
    ov = np.array(overlay_rgba).astype(np.float32)
    a = ov[:, :, 3:4] / 255.0
    rgb = ov[:, :, :3][:, :, ::-1]
    base = base_bgr.astype(np.float32)
    return (base * (1 - a) + rgb * a).astype(np.uint8)


def grade_frame(frame: np.ndarray, t: float) -> np.ndarray:
    """Cinematic grade: contrast, teal shadows, warm mids, vignette, grain."""
    f = frame.astype(np.float32)
    # contrast + slight lift
    f = (f - 128) * 1.08 + 128 + 3
    # teal-ish shadows / warm mid push
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    r = r * 1.04 + 2
    g = g * 1.01
    b = b * 1.06 + 4
    f = np.stack([b, g, r], axis=-1)
    # vignette
    h, w = frame.shape[:2]
    yy = np.linspace(-1, 1, h)[:, None]
    xx = np.linspace(-1, 1, w)[None, :]
    vig = 1.0 - 0.28 * np.clip((xx * xx + yy * yy * 0.9), 0, 1) ** 1.2
    f *= vig[..., None]
    # subtle film grain
    rng = np.random.default_rng(int(t * 30) % 10000)
    grain = rng.normal(0, 2.2, f.shape).astype(np.float32)
    f = np.clip(f + grain, 0, 255)
    return f.astype(np.uint8)


def draw_constellation(draw, cx, cy, t, al, scale=1.0):
    a = int(255 * al)
    # Earth
    R = int(36 * scale)
    draw.ellipse((cx - R, cy - R, cx + R, cy + R), fill=(18, 45, 90, a), outline=(*CYAN, a), width=2)
    draw.ellipse((cx - int(R * 0.6), cy - int(R * 0.3), cx + int(R * 0.15), cy + int(R * 0.4)), fill=(25, 100, 55, int(a * 0.85)))
    # orbital ring
    oR, oH = int(95 * scale), int(38 * scale)
    draw.ellipse((cx - oR, cy - oH, cx + oR, cy + oH), outline=(*CYAN, int(a * 0.55)), width=2)
    # satellites
    for i, phase in enumerate((0.0, 2.1, 4.2)):
        ang = t * 1.15 + phase
        sx = cx + int(oR * math.cos(ang))
        sy = cy + int(oH * math.sin(ang))
        col = LIME if i == 0 else (CYAN if i == 1 else VIOLET)
        body = [(sx - 8, sy - 5), (sx + 8, sy - 5), (sx + 8, sy + 5), (sx - 8, sy + 5)]
        draw.polygon(body, fill=(*PAPER, a))
        draw.rectangle((sx - 18, sy - 2, sx - 8, sy + 2), fill=(*col, a))
        draw.rectangle((sx + 8, sy - 2, sx + 18, sy + 2), fill=(*col, a))
        # signal to earth
        for j in range(5):
            p = (j + (t * 3 + i) % 1) / 5
            x = int(sx + (cx - sx) * p)
            y = int(sy + (cy - sy) * p)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(*col, int(a * (1 - p) * 0.9)))


def draw_twins_scene(draw, cx, cy, t, al, f_small, f_mono):
    a = int(255 * al)
    # traveler rocket left
    lx, ly = cx - 130, cy
    # ship body
    draw.polygon(
        [(lx, ly - 28), (lx + 22, ly), (lx, ly + 28), (lx - 50, ly + 14), (lx - 50, ly - 14)],
        fill=(*LIME, int(a * 0.9)),
    )
    # exhaust
    for i in range(4):
        ex = lx - 55 - i * 10 - int(6 * math.sin(t * 12 + i))
        ey = ly + int(4 * math.sin(t * 20 + i))
        draw.ellipse((ex - 6, ey - 5, ex + 6, ey + 5), fill=(*PEACH, int(a * (0.8 - i * 0.15))))
    # earth right
    rx, ry = cx + 130, cy
    draw.ellipse((rx - 34, ry - 34, rx + 34, ry + 34), fill=(20, 50, 100, a), outline=(*CYAN, a), width=2)
    # clocks
    for (x, y, speed, col, lab) in (
        (lx - 10, ly - 70, 2.8, LIME, "SHIP"),
        (rx, ry - 70, 0.28, CYAN, "EARTH"),
    ):
        draw.ellipse((x - 22, y - 22, x + 22, y + 22), outline=(*col, a), width=2)
        ang = t * speed
        hx = x + int(14 * math.sin(ang))
        hy = y - int(14 * math.cos(ang))
        draw.line((x, y, hx, hy), fill=(*col, a), width=2)
        tw, _ = text_size(draw, lab, f_mono)
        draw.text((x - tw // 2, y + 28), lab, font=f_mono, fill=(*col, a))
    # warp lines
    for i in range(8):
        y = cy - 50 + i * 14
        x0 = cx - 40
        x1 = cx + 40
        wob = int(8 * math.sin(t * 4 + i))
        draw.line((x0, y + wob, x1, y - wob), fill=(*VIOLET, int(a * 0.45)), width=1)


def draw_light_cone(draw, cx, cy, t, al):
    a = int(120 * al)
    for i in range(6):
        spread = 20 + i * 18 + int(6 * math.sin(t + i))
        draw.polygon(
            [(cx, cy - 40), (cx - spread, cy + 50), (cx + spread, cy + 50)],
            outline=(*CYAN, a // (i + 1)),
        )


def draw_formula_card(draw, px, py, pw, ph, t, al, f_title, f_body, f_mono, f_huge):
    glass_panel(draw, (px, py, px + pw, py + ph), 18, al, VIOLET)
    A = int(255 * al)
    draw.text((px + 28, py + 20), "THE RULE", font=f_mono, fill=(*VIOLET, A))
    # gamma-ish simplified
    draw.text((px + 28, py + 55), "Faster motion", font=f_title, fill=(*PAPER, A))
    draw.text((px + 28, py + 100), "→ slower aging", font=f_title, fill=(*LIME, A))
    # mini bars
    rows = [("0.1 c", 0.15), ("0.5 c", 0.35), ("0.9 c", 0.65), ("0.99 c", 0.95)]
    for i, (lab, frac) in enumerate(rows):
        yy = py + 155 + i * 36
        draw.text((px + 28, yy), lab, font=f_body, fill=(*MUTED, A))
        bar_x = px + 120
        bar_w = pw - 160
        rr(draw, (bar_x, yy + 6, bar_x + bar_w, yy + 20), 4, (30, 35, 45, A))
        anim = ease(min(1.0, max(0.0, (t % 8) / 2 - i * 0.15))) * frac
        rr(draw, (bar_x, yy + 6, bar_x + int(bar_w * anim), yy + 20), 4, (*LIME, A))


def build_overlay(w, h, t, total) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f_hero = font("CrimsonPro-Bold.ttf", 58)
    f_title = font("CrimsonPro-Bold.ttf", 34)
    f_body = font("Outfit-Medium.ttf", 24)
    f_cap = font("Outfit-Medium.ttf", 27)
    f_small = font("Outfit-Light.ttf", 17)
    f_mono = font("JetBrainsMono-Regular.ttf", 15)
    f_huge = font("CrimsonPro-Bold.ttf", 110)
    f_num = font("CrimsonPro-Bold.ttf", 88)

    # cinematic letterbox
    lb = int(h * 0.065)
    for y in range(lb):
        a = int(220 * (1 - y / lb) ** 0.6)
        draw.line([(0, y), (w, y)], fill=(4, 5, 8, a))
        draw.line([(0, h - 1 - y), (w, h - 1 - y)], fill=(4, 5, 8, a))

    # brand
    rr(draw, (24, 18, 148, 50), 8, (8, 10, 16, 200), outline=(*LIME, 170), width=1)
    draw.text((40, 26), "VIDMCP", font=f_mono, fill=(*LIME, 255))
    draw.text((160, 28), "relativity series", font=f_small, fill=(*MUTED, 200))

    # chapter chips top-right
    active_ch = CHAPTERS[0]
    for ch in CHAPTERS:
        if ch[0] <= t <= ch[1]:
            active_ch = ch
            break
    cx0 = w - 280
    rr(draw, (cx0, 18, w - 24, 50), 8, (8, 10, 16, 190), outline=(*CYAN, 120), width=1)
    draw.text((cx0 + 14, 26), f"{active_ch[2]}  {active_ch[3]}", font=f_mono, fill=(*CYAN, 255))

    # progress rail
    prog = t / max(total, 0.01)
    draw.rectangle((0, h - 4, w, h), fill=(18, 20, 28, 220))
    draw.rectangle((0, h - 4, int(w * prog), h), fill=(*LIME, 255))
    # chapter ticks
    for a0, _, _, _ in CHAPTERS:
        x = int(w * (a0 / max(total, 0.01)))
        draw.rectangle((x, h - 8, x + 2, h), fill=(*PAPER, 180))

    # floating ambient particles (right third, subtle)
    rng = np.random.default_rng(42)
    for i in range(28):
        px = int(w * (0.55 + 0.42 * ((i * 37 % 100) / 100)))
        base_y = (i * 97) % h
        py = int((base_y + t * (12 + i % 7) * 8) % h)
        size = 1 + (i % 3)
        col = [LIME, CYAN, VIOLET, PEACH][i % 4]
        aa = int(40 + 50 * (0.5 + 0.5 * math.sin(t * 1.5 + i)))
        draw.ellipse((px, py, px + size, py + size), fill=(*col, aa))

    for a0, b0, kind, payload in BEATS:
        al = beat_alpha(t, a0, b0, 0.45)
        if al < 0.02:
            continue
        A = int(255 * al)

        if kind == "title":
            title = payload["title"]
            sub = payload["sub"]
            tw, th = text_size(draw, title, f_hero)
            # keep title in upper band — never over the face
            x = (w - tw) // 2
            y = int(h * 0.09)
            glass_panel(draw, (x - 36, y - 14, x + tw + 36, y + th + 42), 16, al, LIME, 2)
            draw.text((x, y), title, font=f_hero, fill=(*PAPER, A))
            sw, _ = text_size(draw, sub, f_body)
            draw.text(((w - sw) // 2, y + th + 6), sub, font=f_body, fill=(*CYAN, A))
            # orbit motif on RIGHT only (clear of speaker)
            draw_constellation(draw, int(w * 0.82), int(h * 0.42), t, al * 0.9, scale=0.95)
            draw_light_cone(draw, int(w * 0.82), int(h * 0.58), t, al * 0.4)

        elif kind == "quote":
            panel_w = 540
            px = w - panel_w - 36
            py = int(h * 0.20)
            glass_panel(draw, (px, py, px + panel_w, py + 170), 18, al, VIOLET)
            draw.text((px + 32, py + 28), "“", font=f_huge, fill=(*VIOLET, int(A * 0.35)))
            draw.text((px + 48, py + 40), payload["q"], font=f_title, fill=(*PAPER, A))
            draw.text((px + 48, py + 100), payload["a"], font=f_title, fill=(*LIME, A))

        elif kind == "gps":
            panel_w = 500
            px = w - panel_w - 32
            py = int(h * 0.16)
            glass_panel(draw, (px, py, px + panel_w, py + 360), 18, al, CYAN)
            draw.text((px + 26, py + 18), "01  ·  GPS TIME", font=f_mono, fill=(*LIME, A))
            draw.text((px + 26, py + 48), "Orbit clocks drift", font=f_title, fill=(*PAPER, A))
            draw.text((px + 26, py + 90), "Satellites need constant\ntime calibration", font=f_body, fill=(*MUTED, A))
            draw_constellation(draw, px + panel_w // 2, py + 240, t * 1.15, al, scale=1.0)

        elif kind == "micros":
            panel_w = 520
            px = w - panel_w - 32
            py = int(h * 0.18)
            glass_panel(draw, (px, py, px + panel_w, py + 300), 18, al, LIME)
            draw.text((px + 28, py + 20), "DRIFT RATE", font=f_mono, fill=(*MUTED, A))
            frac = ease(min(1.0, (t - a0) / 1.2))
            val = int(payload["value"] * frac)
            num = str(val)
            draw.text((px + 28, py + 55), num, font=f_huge, fill=(*LIME, A))
            # unit with mu
            draw.text((px + 28, py + 175), payload["unit"], font=f_title, fill=(*CYAN, A))
            draw.text((px + 28, py + 220), payload["label"], font=f_body, fill=(*PAPER, A))
            # EQ-style ticks
            for i in range(16):
                x = px + 28 + i * 28
                hgt = 8 + int(22 * (0.5 + 0.5 * math.sin(t * 4 + i * 0.7)))
                col = LIME if i < int(16 * frac) else MUTED
                draw.rectangle((x, py + 265, x + 10, py + 265 + hgt), fill=(*col, int(A * 0.85)))

        elif kind == "earth_sync":
            panel_w = 500
            px = w - panel_w - 32
            py = int(h * 0.20)
            glass_panel(draw, (px, py, px + panel_w, py + 280), 18, al, PEACH)
            draw.text((px + 26, py + 18), "02  ·  EARTH SYNC", font=f_mono, fill=(*PEACH, A))
            steps = [
                ("01", "Orbit clocks drift"),
                ("02", "Position error grows"),
                ("03", "Calibrate from Earth"),
                ("04", "Navigation stays true"),
            ]
            for i, (n, s) in enumerate(steps):
                yy = py + 65 + i * 50
                on = ease(min(1.0, max(0.0, (t - a0 - i * 0.45) / 0.35)))
                aa = int(A * max(0.3, on))
                col = LIME if on > 0.55 else MUTED
                draw.ellipse((px + 28, yy + 4, px + 48, yy + 24), outline=(*col, aa), width=2)
                tw, th = text_size(draw, n, f_mono)
                draw.text((px + 38 - tw // 2, yy + 6), n[-1], font=f_mono, fill=(*col, aa))
                if i < len(steps) - 1:
                    draw.line((px + 38, yy + 26, px + 38, yy + 50), fill=(*MUTED, int(A * 0.4)), width=2)
                draw.text((px + 60, yy + 2), s, font=f_body, fill=(*PAPER, aa))

        elif kind == "twins":
            panel_w = 540
            px = w - panel_w - 32
            py = int(h * 0.15)
            glass_panel(draw, (px, py, px + panel_w, py + 370), 18, al, VIOLET)
            draw.text((px + 26, py + 16), "03  ·  TWIN PARADOX", font=f_mono, fill=(*VIOLET, A))
            draw.text((px + 26, py + 48), "0.99 × speed of light", font=f_title, fill=(*PAPER, A))
            draw_twins_scene(draw, px + panel_w // 2, py + 200, t, al, f_small, f_mono)
            draw.text((px + 26, py + 320), "Same departure · different aging", font=f_small, fill=(*MUTED, A))

        elif kind == "big_num":
            # lower-center hero card — face stays visible above
            plate_h = 240
            py = int(h * 0.48)
            x0, x1 = int(w * 0.18), int(w * 0.82)
            # more translucent so face still reads
            rr(
                draw,
                (x0, py, x1, py + plate_h),
                22,
                (6, 8, 14, int(175 * al)),
                outline=(*LIME, int(200 * al)),
                width=2,
            )
            frac = ease(min(1.0, (t - a0) / 1.0))
            val = int(payload["value"] * frac)
            num = f"{val:,}"
            tw, th = text_size(draw, num, f_num)
            draw.text(((w - tw) // 2, py + 28), num, font=f_num, fill=(*LIME, A))
            uw, _ = text_size(draw, payload["unit"], f_title)
            draw.text(((w - uw) // 2, py + 130), payload["unit"], font=f_title, fill=(*CYAN, A))
            sw, _ = text_size(draw, payload["sub"], f_body)
            draw.text(((w - sw) // 2, py + 180), payload["sub"], font=f_body, fill=(*PAPER, A))
            ring_r = int(70 + 14 * math.sin(t * 2))
            cx, cy = w // 2, py + 80
            for k in range(3):
                rr0 = ring_r + k * 14
                draw.ellipse(
                    (cx - rr0, cy - rr0 // 3, cx + rr0, cy + rr0 // 3),
                    outline=(*CYAN, int(A * (0.3 - k * 0.07))),
                    width=1,
                )

        elif kind == "formula":
            panel_w = 520
            px = w - panel_w - 32
            py = int(h * 0.18)
            draw_formula_card(draw, px, py, panel_w, 320, t, al, f_title, f_body, f_mono, f_huge)

        elif kind == "outro":
            title = "Time dilation is real"
            sub = "GPS proves it — every single day"
            cta = "Like  ·  Share  ·  Subscribe"
            tw, th = text_size(draw, title, f_title)
            x = (w - tw) // 2
            y = int(h * 0.16)
            glass_panel(draw, (x - 36, y - 18, x + tw + 36, y + th + 100), 18, al, LIME)
            draw.text((x, y), title, font=f_title, fill=(*PAPER, A))
            sw, _ = text_size(draw, sub, f_body)
            draw.text(((w - sw) // 2, y + th + 10), sub, font=f_body, fill=(*CYAN, A))
            cw, _ = text_size(draw, cta, f_mono)
            draw.text(((w - cw) // 2, y + th + 52), cta, font=f_mono, fill=(*LIME, A))
            # soft full fade plate near end
            if t > total - 2.0:
                fade = ease((t - (total - 2.0)) / 2.0)
                draw.rectangle((0, 0, w, h), fill=(5, 5, 7, int(100 * fade * al)))

    # lower third
    if t < 6.2:
        al = beat_alpha(t, 0.25, 6.2, 0.35)
        A = int(255 * al)
        lx, ly = 36, int(h * 0.76)
        draw.rectangle((lx, ly, lx + 5, ly + 60), fill=(*LIME, A))
        rr(draw, (lx + 5, ly, lx + 380, ly + 60), 4, (8, 10, 16, int(220 * al)))
        draw.text((lx + 20, ly + 8), "Dhiraj Lochib", font=f_body, fill=(*PAPER, A))
        draw.text((lx + 20, ly + 34), "vidmcp.com  ·  spacetime", font=f_small, fill=(*MUTED, A))

    # captions
    for a, b, text in CAPTIONS:
        al = beat_alpha(t, a, b, 0.28)
        if al < 0.02:
            continue
        A = int(255 * al)
        max_w = int(w * 0.76)
        lines = wrap(draw, text, f_cap, max_w)
        lh = text_size(draw, "Ay", f_cap)[1] + 8
        box_h = lh * len(lines) + 28
        box_w = max(text_size(draw, ln, f_cap)[0] for ln in lines) + 52
        bx = (w - box_w) // 2
        by = h - box_h - int(h * 0.085)
        rr(draw, (bx, by, bx + box_w, by + box_h), 14, (5, 7, 12, int(230 * al)), outline=(*LIME, int(90 * al)), width=1)
        # accent
        draw.rectangle((bx + 22, by, bx + box_w - 22, by + 3), fill=(*LIME, A))
        yy = by + 16
        for ln in lines:
            lw, _ = text_size(draw, ln, f_cap)
            draw.text(((w - lw) // 2, yy), ln, font=f_cap, fill=(*PAPER, A))
            yy += lh

    # timecode
    mm, ss = divmod(int(t), 60)
    draw.text((w - 88, h - 34), f"{mm:02d}:{ss:02d}", font=f_mono, fill=(*MUTED, 190))
    return img


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def generate_space_bgm(out_wav: Path, duration_sec: float, sr: int = 48000) -> Path:
    """Premium space-cinematic BGM: drones, pads, soft motif, tension lifts."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    rng = np.random.default_rng(11)

    def soft_sin(freq, amp, tt=None):
        tt = t if tt is None else tt
        ph = 2 * np.pi * freq * tt
        return amp * (0.62 * np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph))

    # deep sub + dark pad
    pad = np.zeros(n)
    for f, a in [
        (36.7, 0.11),   # D1
        (55.0, 0.10),
        (73.4, 0.09),
        (110.0, 0.08),
        (146.8, 0.07),
        (220.0, 0.055),
        (329.6, 0.04),
        (440.0, 0.028),
        (554.4, 0.018),
    ]:
        lfo = 1 + 0.06 * np.sin(2 * np.pi * (0.03 + f * 8e-6) * t)
        pad += soft_sin(f, a) * lfo

    # slow harmonic swell cycles
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * t / 22.0)
    swell2 = 0.7 + 0.3 * np.sin(2 * np.pi * t / 9.5 + 1.2)
    pad *= swell * swell2

    # shimmer high layer (air)
    shimmer = np.zeros(n)
    for f, a in [(880, 0.012), (1174, 0.01), (1568, 0.008)]:
        shimmer += soft_sin(f, a) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + f))
    # gentle noise dust
    dust = rng.normal(0, 1, n)
    for _ in range(5):
        dust = np.convolve(dust, np.ones(120) / 120, mode="same")
    dust *= 0.014

    # melodic motif hits at chapter moments (seconds)
    motif_hits = [
        (0.8, 220.0, 0.09, 4.0),
        (5.5, 277.2, 0.08, 3.0),
        (8.0, 329.6, 0.1, 3.5),
        (14.0, 392.0, 0.09, 4.0),
        (20.0, 349.2, 0.08, 3.0),
        (27.0, 196.0, 0.11, 5.0),
        (35.0, 261.6, 0.12, 5.0),
        (42.0, 329.6, 0.1, 4.5),
        (51.0, 440.0, 0.08, 4.0),
    ]
    piano = np.zeros(n)
    for start, f, amp, length in motif_hits:
        i0 = int(start * sr)
        nn = int(length * sr)
        if i0 >= n:
            continue
        nn = min(nn, n - i0)
        tt = np.arange(nn) / sr
        env = np.exp(-tt * 0.55) * (1 - np.exp(-tt * 18))
        tone = soft_sin(f, amp, tt) + 0.35 * soft_sin(f * 2, amp * 0.3, tt)
        tone += 0.15 * soft_sin(f * 3, amp * 0.15, tt)
        piano[i0 : i0 + nn] += tone * env

    # tension riser into twin paradox (~26–28s) and big number (~34s)
    def riser(t0, dur, amp=0.05):
        i0 = int(t0 * sr)
        nn = int(dur * sr)
        if i0 >= n:
            return
        nn = min(nn, n - i0)
        tt = np.arange(nn) / sr
        freq = 80 + 420 * (tt / max(dur, 0.01)) ** 1.4
        phase = 2 * np.pi * np.cumsum(freq) / sr
        env = (tt / dur) ** 1.5 * amp
        noise = rng.normal(0, 1, nn)
        for _ in range(3):
            noise = np.convolve(noise, np.ones(40) / 40, mode="same")
        pad[i0 : i0 + nn] += np.sin(phase) * env + noise * env * 0.35

    riser(25.5, 2.2, 0.045)
    riser(33.5, 1.8, 0.04)
    riser(50.5, 1.5, 0.03)

    # soft pulse under GPS section
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    gps_mask = ((t >= 8) & (t <= 20)).astype(np.float64)
    # smooth mask
    k = int(0.4 * sr)
    if k > 1:
        gps_mask = np.convolve(gps_mask, np.ones(k) / k, mode="same")
    kick = soft_sin(55, 0.04) * pulse * gps_mask * 0.6

    mix = pad + shimmer + dust + piano + kick
    # fades
    fi, fo = int(1.8 * sr), int(3.0 * sr)
    mix[:fi] *= np.linspace(0, 1, fi)
    mix[-fo:] *= np.linspace(1, 0, fo)
    peak = float(np.max(np.abs(mix)) + 1e-9)
    mix = mix / peak * 0.58
    # wide stereo
    left = mix * 0.92 + np.roll(mix, 140) * 0.08
    right = mix * 0.92 + np.roll(mix, -110) * 0.08
    stereo = np.stack([left, right], axis=1)
    pcm = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out_wav


def polish_vocals(src: Path, out: Path) -> Path:
    """Pro vocal chain: denoise, de-ess, presence, warmth, loudnorm."""
    out.parent.mkdir(parents=True, exist_ok=True)
    # strength-forward but still natural
    af = (
        "highpass=f=90,"
        "lowpass=f=12000,"
        "afftdn=nr=16:nf=-28:tn=1,"
        "anlmdn=s=0.0002:p=0.002:r=0.002:m=12,"
        "acompressor=threshold=-20dB:ratio=3.2:attack=8:release=140:makeup=4,"
        "equalizer=f=180:t=q:w=0.9:g=-2.5,"
        "equalizer=f=350:t=q:w=1.0:g=-1.5,"
        "equalizer=f=2800:t=q:w=1.1:g=3.5,"
        "equalizer=f=5200:t=q:w=1.3:g=2.2,"
        "equalizer=f=7500:t=q:w=1.5:g=-3.5,"  # de-ess-ish
        "equalizer=f=120:t=q:w=0.8:g=1.2,"    # chest warmth
        "agate=threshold=0.018:ratio=2.5:attack=4:release=90,"
        "loudnorm=I=-14:TP=-1.5:LRA=10"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-af", af, "-ar", "48000", "-ac", "2", str(out)],
        check=True,
        capture_output=True,
    )
    return out


def mix_audio(vox: Path, bgm: Path, out: Path, duration: float, bgm_vol: float = 0.42) -> Path:
    fade_in, fade_out = 1.4, 2.8
    fade_out_start = max(0.0, duration - fade_out)
    # sidechain duck BGM under voice; keep music present in gaps
    fc = (
        f"[1:a]atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,"
        f"volume={bgm_vol},"
        f"afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_out_start:.3f}:d={fade_out},"
        f"equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3000:t=q:w=1:g=-1.5[bg];"
        f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.08[vox];"
        f"[bg][vox]sidechaincompress=threshold=0.018:ratio=7:attack=40:release=280:level_sc=1[bgd];"
        f"[vox][bgd]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
        f"alimiter=limit=0.95:level=false,"
        f"loudnorm=I=-14:TP=-1.2:LRA=11[a]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(vox),
            "-i", str(bgm),
            "-filter_complex", fc,
            "-map", "[a]",
            "-ar", "48000", "-ac", "2",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert SRC_V.exists(), f"missing {SRC_V}"

    # duration
    duration = float(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(SRC_V)],
            text=True,
        ).strip()
    )
    print(f"duration={duration:.2f}s")

    # --- audio pipeline ---
    vox_src = SRC_VOX if SRC_VOX.exists() else FALLBACK_VOX
    print("polish vocals from", vox_src)
    vox = polish_vocals(vox_src, WORK / "vox_pro.wav")
    print("generate space BGM…")
    bgm = generate_space_bgm(WORK / "bgm_space.wav", duration + 1.5)
    print("mix + duck…")
    final_a = mix_audio(vox, bgm, WORK / "final_audio.wav", duration, bgm_vol=0.44)
    print("audio ready", final_a)

    # --- video ---
    cap = cv2.VideoCapture(str(SRC_V))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"render {w}x{h} @ {fps:.2f}")

    raw = WORK / "with_gfx.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        graded = grade_frame(frame, t)
        ov = build_overlay(w, h, t, duration)
        out = alpha_composite_bgr(graded, ov)
        writer.write(out)
        idx += 1
        if idx % 90 == 0:
            print(f"  frame {idx}  t={t:.1f}s")
    cap.release()
    writer.release()
    print("frames", idx)

    # encode + mux
    print("encode final…")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(raw),
            "-i", str(final_a),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k",
            "-shortest", "-movflags", "+faststart",
            str(OUT),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(["cp", str(OUT), str(DESK)], check=True)
    still = OUT.parent / "time_dilation_still.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "16", "-i", str(OUT), "-frames:v", "1", "-q:v", "2", str(still)],
        check=True, capture_output=True,
    )
    teaser = OUT.parent / "time_dilation_5s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", "14", "-i", str(OUT), "-t", "5",
            "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", str(teaser),
        ],
        check=True, capture_output=True,
    )
    info = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(OUT)],
        text=True,
    )
    print(info)
    print("OUT", OUT)
    print("DESKTOP", DESK)


if __name__ == "__main__":
    main()
