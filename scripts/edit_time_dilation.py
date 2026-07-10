#!/usr/bin/env python3
"""Time dilation lesson: hesitation cut already applied on inputs; rich infographics."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WORK = Path("/tmp/vidmcp_img4447")
SRC_V = WORK / "video_cut.mp4"
SRC_A = WORK / "final_audio.wav"
OUT = Path("/Users/dhirajlochib/Developer/VidMcp/final_output/time_dilation_edit.mp4")
DESK = Path("/Users/dhirajlochib/Desktop/time_dilation_edit.mp4")
FONTS = Path("/Users/dhirajlochib/Developer/VidMcp/scripts/fonts")

LIME = (212, 255, 42)
CYAN = (42, 255, 209)
VIOLET = (177, 140, 255)
PEACH = (255, 162, 78)
PAPER = (239, 239, 235)
MUTED = (150, 155, 165)
DARK = (8, 10, 16)

# Captions on NEW timeline (hesitation removed; times remapped: -3.88 after 7.62)
# orig_gap = 11.50 - 7.62 = 3.88
CAPTIONS = [
    (1.30, 5.80, "Time isn’t fixed — Einstein said it’s relative."),
    (6.20, 7.55, "So what does this mean?"),
    (7.70, 12.50, "Take GPS satellites — they need time calibration."),
    (12.80, 18.80, "About 38 microseconds of drift — every day."),
    (19.00, 26.50, "Without Earth sync, navigation would fail."),
    (27.00, 34.50, "Twin paradox: travel near light speed…"),
    (34.80, 41.50, "5 years for the traveler ≈ 835 years on Earth."),
    (42.00, 49.50, "That’s time dilation — a mind-bending fact."),
    (50.50, 54.20, "The traveler only ages 5 years."),
    (54.60, 57.50, "Like · Share · Subscribe"),
]

# Infographic beats (start, end, kind, payload)
BEATS = [
    (0.0, 5.5, "title", {"title": "TIME DILATION", "sub": "Einstein’s relative time"}),
    (5.5, 7.6, "quote", {"q": "Time isn’t fixed.", "a": "It’s relative."}),
    (8.0, 14.0, "gps", {}),
    (14.0, 20.5, "micros", {"value": "38", "unit": "us / day", "label": "GPS clock drift"}),
    (20.5, 27.0, "earth_sync", {}),
    (27.5, 35.0, "twins", {}),
    (35.0, 43.0, "big_num", {"value": "835", "unit": "YEARS", "sub": "Earth ages while traveler lives 5"}),
    (43.0, 52.0, "compare", {}),
    (52.0, 57.8, "outro", {}),
]


def font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except Exception:
        return ImageFont.load_default()


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def beat_alpha(t: float, a: float, b: float, fade: float = 0.45) -> float:
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


def alpha_composite_bgr(base_bgr: np.ndarray, overlay_rgba: Image.Image) -> np.ndarray:
    ov = np.array(overlay_rgba).astype(np.float32)
    a = ov[:, :, 3:4] / 255.0
    rgb = ov[:, :, :3][:, :, ::-1]  # RGB->BGR
    base = base_bgr.astype(np.float32)
    return (base * (1 - a) + rgb * a).astype(np.uint8)


def draw_orbit_satellite(draw, cx, cy, t, alpha):
    a = int(255 * alpha)
    R = 70
    # orbit
    draw.ellipse((cx - R, cy - R // 2, cx + R, cy + R // 2), outline=(*CYAN, int(a * 0.7)), width=2)
    # earth
    draw.ellipse((cx - 28, cy - 28, cx + 28, cy + 28), fill=(20, 40, 80, a), outline=(*CYAN, a), width=2)
    draw.ellipse((cx - 18, cy - 10, cx + 5, cy + 12), fill=(30, 90, 50, int(a * 0.8)))
    # satellite position
    ang = t * 1.4
    sx = cx + int(R * math.cos(ang))
    sy = cy + int((R // 2) * math.sin(ang))
    body = [(sx - 10, sy - 6), (sx + 10, sy - 6), (sx + 10, sy + 6), (sx - 10, sy + 6)]
    draw.polygon(body, fill=(*PAPER, a))
    draw.rectangle((sx - 22, sy - 3, sx - 10, sy + 3), fill=(*LIME, a))
    draw.rectangle((sx + 10, sy - 3, sx + 22, sy + 3), fill=(*LIME, a))
    # signal dashes
    for i in range(4):
        p = (i + (t * 2) % 1) / 4
        x = int(sx + (cx - sx) * p)
        y = int(sy + (cy - sy) * p)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(*LIME, int(a * (1 - p))))


def draw_twins(draw, cx, cy, t, alpha, f_small, f_body):
    a = int(255 * alpha)
    # left traveler, right earth
    for i, (lab, col, speed) in enumerate(
        [("TRAVELER", LIME, 2.5), ("EARTH", CYAN, 0.35)]
    ):
        x = cx - 110 + i * 220
        y = cy
        # clock face
        draw.ellipse((x - 40, y - 40, x + 40, y + 40), outline=(*col, a), width=3)
        ang = t * speed
        hx = x + int(28 * math.sin(ang))
        hy = y - int(28 * math.cos(ang))
        draw.line((x, y, hx, hy), fill=(*col, a), width=3)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(*col, a))
        tw, _ = text_size(draw, lab, f_small)
        draw.text((x - tw // 2, y + 48), lab, font=f_small, fill=(*col, a))
    # arrow
    draw.line((cx - 50, cy - 70, cx + 50, cy - 70), fill=(*VIOLET, a), width=2)
    draw.polygon(
        [(cx + 50, cy - 70), (cx + 40, cy - 76), (cx + 40, cy - 64)], fill=(*VIOLET, a)
    )
    draw.text((cx - 70, cy - 100), "near light speed", font=f_small, fill=(*MUTED, a))


def build_overlay(w, h, t, total) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f_hero = font("CrimsonPro-Bold.ttf", 54)
    f_title = font("CrimsonPro-Bold.ttf", 36)
    f_body = font("Outfit-Medium.ttf", 26)
    f_cap = font("Outfit-Medium.ttf", 28)
    f_small = font("Outfit-Light.ttf", 18)
    f_mono = font("JetBrainsMono-Regular.ttf", 16)
    f_huge = font("CrimsonPro-Bold.ttf", 96)

    # cinematic letterbox top/bottom
    lb = int(h * 0.07)
    for y in range(lb):
        a = int(200 * (1 - y / lb) ** 0.5)
        draw.line([(0, y), (w, y)], fill=(5, 5, 7, a))
        draw.line([(0, h - 1 - y), (w, h - 1 - y)], fill=(5, 5, 7, a))

    # brand chip
    rr(draw, (28, 22, 140, 52), 8, (10, 12, 18, 180), outline=(*LIME, 160), width=1)
    draw.text((44, 30), "VIDMCP", font=f_mono, fill=(*LIME, 255))

    # progress
    prog = t / max(total, 0.01)
    draw.rectangle((0, h - 5, w, h), fill=(20, 22, 28, 200))
    draw.rectangle((0, h - 5, int(w * prog), h), fill=(*LIME, 255))

    # active beat graphics (right panel / center)
    for a0, b0, kind, payload in BEATS:
        al = beat_alpha(t, a0, b0)
        if al < 0.02:
            continue
        A = int(255 * al)

        if kind == "title":
            title = payload["title"]
            sub = payload["sub"]
            tw, th = text_size(draw, title, f_hero)
            x = (w - tw) // 2
            y = int(h * 0.16)
            rr(
                draw,
                (x - 36, y - 20, x + tw + 36, y + th + 50),
                20,
                (8, 10, 16, int(200 * al)),
                outline=(*LIME, int(200 * al)),
                width=2,
            )
            draw.text((x, y), title, font=f_hero, fill=(*PAPER, A))
            sw, _ = text_size(draw, sub, f_body)
            draw.text(((w - sw) // 2, y + th + 8), sub, font=f_body, fill=(*CYAN, A))
            # orbit motif
            draw_orbit_satellite(draw, w // 2, int(h * 0.55), t, al * 0.9)

        elif kind == "quote":
            panel_w = 520
            px = w - panel_w - 40
            py = int(h * 0.22)
            rr(draw, (px, py, px + panel_w, py + 160), 18, (8, 10, 16, int(210 * al)), outline=(*VIOLET, int(180 * al)), width=2)
            draw.text((px + 28, py + 28), payload["q"], font=f_title, fill=(*PAPER, A))
            draw.text((px + 28, py + 90), payload["a"], font=f_body, fill=(*LIME, A))

        elif kind == "gps":
            panel_w = 480
            px = w - panel_w - 36
            py = int(h * 0.18)
            rr(draw, (px, py, px + panel_w, py + 320), 18, (8, 10, 16, int(200 * al)), outline=(*CYAN, int(160 * al)), width=2)
            draw.text((px + 24, py + 18), "01  ·  GPS TIME", font=f_mono, fill=(*LIME, A))
            draw.text((px + 24, py + 48), "Satellites need\nclock calibration", font=f_title, fill=(*PAPER, A))
            draw_orbit_satellite(draw, px + panel_w // 2, py + 210, t * 1.2, al)
            draw.text((px + 24, py + 280), "Relativity shifts orbit clocks", font=f_small, fill=(*MUTED, A))

        elif kind == "micros":
            panel_w = 500
            px = w - panel_w - 36
            py = int(h * 0.20)
            rr(draw, (px, py, px + panel_w, py + 280), 18, (8, 10, 16, int(210 * al)), outline=(*LIME, int(180 * al)), width=2)
            draw.text((px + 28, py + 22), "DRIFT RATE", font=f_mono, fill=(*MUTED, A))
            # animated count-up
            frac = ease((t - a0) / max(0.01, (b0 - a0) * 0.5))
            val = payload["value"]
            draw.text((px + 28, py + 60), val, font=f_huge, fill=(*LIME, A))
            draw.text((px + 28, py + 170), payload["unit"], font=f_title, fill=(*CYAN, A))
            draw.text((px + 28, py + 220), payload["label"], font=f_body, fill=(*PAPER, A))
            # tick marks
            for i in range(12):
                x = px + 28 + i * 36
                hgt = 10 + int(18 * (0.5 + 0.5 * math.sin(t * 3 + i)))
                draw.rectangle((x, py + 250, x + 8, py + 250 + hgt), fill=(*CYAN, int(A * 0.7)))

        elif kind == "earth_sync":
            panel_w = 480
            px = w - panel_w - 36
            py = int(h * 0.22)
            rr(draw, (px, py, px + panel_w, py + 240), 18, (8, 10, 16, int(205 * al)), outline=(*PEACH, int(160 * al)), width=2)
            draw.text((px + 24, py + 20), "02  ·  EARTH SYNC", font=f_mono, fill=(*PEACH, A))
            steps = ["Orbit clocks drift", "Signal error grows", "Calibrate from Earth", "GPS stays accurate"]
            for i, s in enumerate(steps):
                yy = py + 70 + i * 40
                on = ease((t - a0 - i * 0.35) / 0.4)
                aa = int(A * max(0.25, on))
                draw.ellipse((px + 28, yy + 6, px + 44, yy + 22), fill=(*LIME, aa) if on > 0.5 else (*MUTED, aa))
                if i < len(steps) - 1:
                    draw.line((px + 36, yy + 22, px + 36, yy + 40), fill=(*MUTED, int(A * 0.5)), width=2)
                draw.text((px + 56, yy), s, font=f_body, fill=(*PAPER, aa))

        elif kind == "twins":
            panel_w = 520
            px = w - panel_w - 36
            py = int(h * 0.18)
            rr(draw, (px, py, px + panel_w, py + 340), 18, (8, 10, 16, int(210 * al)), outline=(*VIOLET, int(170 * al)), width=2)
            draw.text((px + 24, py + 18), "03  ·  TWIN PARADOX", font=f_mono, fill=(*VIOLET, A))
            draw.text((px + 24, py + 50), "99% speed of light", font=f_title, fill=(*PAPER, A))
            draw_twins(draw, px + panel_w // 2, py + 180, t, al, f_small, f_body)
            draw.text((px + 24, py + 300), "Same departure · different aging", font=f_small, fill=(*MUTED, A))

        elif kind == "big_num":
            # center dramatic number
            val = payload["value"]
            # scale pop
            pop = 0.92 + 0.08 * ease(min(1.0, (t - a0) / 0.6))
            # draw on translucent plate
            plate_h = 280
            py = (h - plate_h) // 2 - 20
            rr(draw, (int(w * 0.18), py, int(w * 0.82), py + plate_h), 24, (5, 5, 7, int(200 * al)), outline=(*LIME, int(200 * al)), width=2)
            tw, th = text_size(draw, val, f_huge)
            draw.text(((w - tw) // 2, py + 40), val, font=f_huge, fill=(*LIME, A))
            uw, _ = text_size(draw, payload["unit"], f_title)
            draw.text(((w - uw) // 2, py + 150), payload["unit"], font=f_title, fill=(*CYAN, A))
            sw, _ = text_size(draw, payload["sub"], f_body)
            draw.text(((w - sw) // 2, py + 210), payload["sub"], font=f_body, fill=(*PAPER, A))

        elif kind == "compare":
            panel_w = 520
            px = w - panel_w - 36
            py = int(h * 0.22)
            rr(draw, (px, py, px + panel_w, py + 260), 18, (8, 10, 16, int(210 * al)), outline=(*CYAN, int(160 * al)), width=2)
            draw.text((px + 24, py + 18), "COMPARE", font=f_mono, fill=(*CYAN, A))
            rows = [
                ("Traveler ages", "5 years", LIME),
                ("Earth ages", "835 years", PEACH),
                ("Speed", "0.99 c", VIOLET),
            ]
            for i, (lab, val, col) in enumerate(rows):
                yy = py + 70 + i * 55
                draw.text((px + 28, yy), lab, font=f_body, fill=(*MUTED, A))
                vw, _ = text_size(draw, val, f_title)
                draw.text((px + panel_w - 28 - vw, yy - 4), val, font=f_title, fill=(*col, A))
                draw.line((px + 28, yy + 36, px + panel_w - 28, yy + 36), fill=(40, 45, 55, A), width=1)

        elif kind == "outro":
            title = "Time dilation is real"
            sub = "GPS proves it every day"
            tw, th = text_size(draw, title, f_title)
            x = (w - tw) // 2
            y = int(h * 0.18)
            rr(draw, (x - 28, y - 16, x + tw + 28, y + th + 50), 16, (8, 10, 16, int(200 * al)), outline=(*LIME, int(180 * al)), width=2)
            draw.text((x, y), title, font=f_title, fill=(*PAPER, A))
            sw, _ = text_size(draw, sub, f_body)
            draw.text(((w - sw) // 2, y + th + 8), sub, font=f_body, fill=(*CYAN, A))

    # lower third name early
    if t < 6.0:
        al = beat_alpha(t, 0.3, 6.0, 0.4)
        A = int(255 * al)
        lx, ly = 40, int(h * 0.78)
        draw.rectangle((lx, ly, lx + 5, ly + 56), fill=(*LIME, A))
        rr(draw, (lx + 5, ly, lx + 340, ly + 56), 4, (10, 12, 18, int(210 * al)))
        draw.text((lx + 18, ly + 6), "Dhiraj Lochib", font=f_body, fill=(*PAPER, A))
        draw.text((lx + 18, ly + 32), "vidmcp.com  ·  relativity", font=f_small, fill=(*MUTED, A))

    # captions
    for a, b, text in CAPTIONS:
        al = beat_alpha(t, a, b, 0.3)
        if al < 0.02:
            continue
        A = int(255 * al)
        max_w = int(w * 0.78)
        lines = wrap(draw, text, f_cap, max_w)
        lh = text_size(draw, "Ay", f_cap)[1] + 6
        box_h = lh * len(lines) + 26
        box_w = max(text_size(draw, ln, f_cap)[0] for ln in lines) + 48
        bx = (w - box_w) // 2
        by = h - box_h - int(h * 0.09)
        rr(draw, (bx, by, bx + box_w, by + box_h), 14, (6, 8, 14, int(215 * al)), outline=(*LIME, int(80 * al)), width=1)
        draw.rectangle((bx + 20, by, bx + box_w - 20, by + 3), fill=(*LIME, A))
        yy = by + 14
        for ln in lines:
            lw, _ = text_size(draw, ln, f_cap)
            draw.text(((w - lw) // 2, yy), ln, font=f_cap, fill=(*PAPER, A))
            yy += lh

    # timecode
    mm, ss = divmod(int(t), 60)
    draw.text((w - 90, h - 36), f"{mm:02d}:{ss:02d}", font=f_mono, fill=(*MUTED, 200))
    return img


def main():
    cap = cv2.VideoCapture(str(SRC_V))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total = n / fps if n else 57.77
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(SRC_V)],
            capture_output=True,
            text=True,
            check=True,
        )
        total = float(p.stdout.strip())
    except Exception:
        pass
    print(f"render {w}x{h} @ {fps:.2f} fps  total={total:.2f}s")

    raw = WORK / "with_gfx.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        # mild grade
        f = np.clip(frame.astype(np.float32) * 1.03 + 2, 0, 255).astype(np.uint8)
        ov = build_overlay(w, h, t, total)
        out = alpha_composite_bgr(f, ov)
        writer.write(out)
        idx += 1
        if idx % 90 == 0:
            print(f"  {idx} frames  t={t:.1f}s")
    cap.release()
    writer.release()
    print("frames", idx)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw),
        "-i", str(SRC_A),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(OUT),
    ]
    subprocess.run(cmd, check=True)
    subprocess.run(["cp", str(OUT), str(DESK)], check=True)
    # stills
    still = OUT.parent / "time_dilation_still.jpg"
    subprocess.run(["ffmpeg", "-y", "-ss", "16", "-i", str(OUT), "-frames:v", "1", "-q:v", "2", str(still)], check=True, capture_output=True)
    teaser = OUT.parent / "time_dilation_5s.mp4"
    subprocess.run(["ffmpeg", "-y", "-ss", "14", "-i", str(OUT), "-t", "5", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", str(teaser)], check=True, capture_output=True)
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(OUT)], capture_output=True, text=True)
    print(pr.stdout)
    print("OUT", OUT)
    print("DESKTOP", DESK)


if __name__ == "__main__":
    main()
