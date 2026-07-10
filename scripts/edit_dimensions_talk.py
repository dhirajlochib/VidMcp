#!/usr/bin/env python3
"""Polish Photo Booth talk-head: dimensions lesson + captions + brand graphics."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path(
    "/Users/dhirajlochib/Pictures/Photo Booth Library/Pictures/"
    "Movie on 10-07-26 at 12.45 PM.mov"
)
OUT_DIR = Path("/Users/dhirajlochib/Developer/VidMcp/final_output")
DESKTOP = Path("/Users/dhirajlochib/Desktop")
WORK = Path("/tmp/vidmcp_dimensions_edit")
FONTS = Path("/Users/dhirajlochib/Developer/VidMcp/scripts/fonts")

# Brand (dhirajlochib.com / vidmcp)
BG = (5, 5, 7)
LIME = (212, 255, 42)
CYAN = (42, 255, 209)
VIOLET = (177, 140, 255)
PEACH = (255, 162, 78)
PAPER = (239, 239, 235)
MUTED = (160, 165, 175)

# Cleaned caption cards (faithful to speech, readable)
CAPTIONS = [
    (0.46, 7.40, "Hello — today we're going to see what dimensions are."),
    (11.18, 16.94, "A person sees through different dimensions."),
    (16.94, 22.52, "First: 1D — we only have one point."),
    (23.38, 31.32, "In the 2nd dimension you can go forward & backward."),
    (31.78, 34.84, "That freedom is a line — the second dimension."),
    (34.84, 40.50, "The third dimension is the world we live in."),
]

# Chapter badges synced to speech
CHAPTERS = [
    (0.0, 3.2, "LESSON", "What are dimensions?"),
    (16.5, 23.2, "01", "1D · Point"),
    (23.2, 34.5, "02", "2D · Line"),
    (34.5, 40.6, "03", "3D · Space we live in"),
]


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS / name
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def alpha_composite(base_bgr: np.ndarray, overlay_rgba: Image.Image) -> np.ndarray:
    ov = np.array(overlay_rgba)
    if ov.shape[2] == 3:
        a = np.ones(ov.shape[:2], dtype=np.float32)
        rgb = ov.astype(np.float32)
    else:
        a = ov[:, :, 3].astype(np.float32) / 255.0
        rgb = ov[:, :, :3].astype(np.float32)
    # PIL RGB -> OpenCV BGR
    rgb_bgr = rgb[:, :, ::-1]
    base = base_bgr.astype(np.float32)
    a3 = a[..., None]
    out = base * (1 - a3) + rgb_bgr * a3
    return out.astype(np.uint8)


def rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text_size(draw, text, font):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def wrap_text(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        tw, _ = text_size(draw, trial, font)
        if tw <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def caption_for(t: float):
    for a, b, text in CAPTIONS:
        if a <= t <= b:
            # local fade
            fade = 0.35
            alpha = 1.0
            if t < a + fade:
                alpha = ease_out((t - a) / fade)
            if t > b - fade:
                alpha = min(alpha, ease_out((b - t) / fade))
            return text, alpha
    return None, 0.0


def chapter_for(t: float):
    for a, b, code, title in CHAPTERS:
        if a <= t <= b:
            fade = 0.4
            alpha = 1.0
            if t < a + fade:
                alpha = ease_out((t - a) / fade)
            if t > b - 0.5:
                alpha = min(alpha, ease_out((b - t) / 0.5))
            return code, title, alpha
    return None, None, 0.0


def draw_dimension_glyph(draw, kind: str, cx: int, cy: int, scale: float, alpha: float):
    """Tiny schematic: point / line / cube-ish for chapter."""
    a = int(255 * alpha)
    col = (*LIME, a)
    col2 = (*CYAN, a)
    if kind == "01":
        r = int(8 * scale)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
        # faint orbit
        R = int(22 * scale)
        draw.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(*CYAN, int(a * 0.5)), width=2)
    elif kind == "02":
        L = int(36 * scale)
        draw.line((cx - L, cy, cx + L, cy), fill=col, width=max(2, int(3 * scale)))
        # arrow heads
        ah = int(8 * scale)
        draw.polygon(
            [(cx + L, cy), (cx + L - ah, cy - ah // 2), (cx + L - ah, cy + ah // 2)],
            fill=col2,
        )
        draw.polygon(
            [(cx - L, cy), (cx - L + ah, cy - ah // 2), (cx - L + ah, cy + ah // 2)],
            fill=col2,
        )
    else:
        s = int(18 * scale)
        # isometric-ish square stack
        pts = [
            (cx, cy - s),
            (cx + s, cy - s // 3),
            (cx, cy + s // 2),
            (cx - s, cy - s // 3),
        ]
        draw.polygon(pts, outline=col, width=2)
        draw.line((cx, cy - s, cx, cy + s // 2), fill=col2, width=2)


def build_overlay(w: int, h: int, t: float, total: float) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    f_title = load_font("CrimsonPro-Bold.ttf", max(28, int(h * 0.042)))
    f_body = load_font("Outfit-Medium.ttf", max(20, int(h * 0.028)))
    f_cap = load_font("Outfit-Medium.ttf", max(22, int(h * 0.032)))
    f_small = load_font("Outfit-Light.ttf", max(14, int(h * 0.018)))
    f_mono = load_font("JetBrainsMono-Regular.ttf", max(13, int(h * 0.016)))

    # soft top gradient bar (brand strip)
    strip_h = int(h * 0.09)
    for y in range(strip_h):
        a = int(140 * (1 - y / strip_h))
        draw.line([(0, y), (w, y)], fill=(5, 5, 7, a))

    # lower vignette for captions
    vig_h = int(h * 0.28)
    for i in range(vig_h):
        y = h - vig_h + i
        a = int(170 * (i / vig_h) ** 1.2)
        draw.line([(0, y), (w, y)], fill=(5, 5, 7, a))

    # Intro hero title
    if t < 3.4:
        a = ease_out(t / 0.5) if t < 0.5 else (ease_out((3.4 - t) / 0.5) if t > 2.9 else 1.0)
        title = "What are dimensions?"
        sub = "A quick visual lesson"
        tw, th = text_size(draw, title, f_title)
        x = (w - tw) // 2
        y = int(h * 0.14)
        # glass pill behind
        pad_x, pad_y = 28, 18
        rounded_rect(
            draw,
            (x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y + 28),
            18,
            (12, 14, 20, int(200 * a)),
            outline=(*LIME, int(180 * a)),
            width=2,
        )
        draw.text((x, y), title, font=f_title, fill=(*PAPER, int(255 * a)))
        sw, _ = text_size(draw, sub, f_small)
        draw.text(((w - sw) // 2, y + th + 6), sub, font=f_small, fill=(*CYAN, int(240 * a)))

    # Lower third (name) first ~5s then soft brand chip
    if t < 6.5:
        a = 1.0
        if t < 0.4:
            a = ease_out(t / 0.4)
        if t > 5.8:
            a = ease_out((6.5 - t) / 0.7)
        lx, ly = int(w * 0.05), int(h * 0.72)
        # accent bar
        draw.rectangle((lx, ly, lx + 5, ly + 58), fill=(*LIME, int(255 * a)))
        rounded_rect(
            draw,
            (lx + 5, ly, lx + 320, ly + 58),
            4,
            (10, 12, 18, int(210 * a)),
        )
        draw.text((lx + 18, ly + 6), "Dhiraj Lochib", font=f_body, fill=(*PAPER, int(255 * a)))
        draw.text(
            (lx + 18, ly + 32),
            "vidmcp.com  ·  dimensions",
            font=f_small,
            fill=(*MUTED, int(255 * a)),
        )
    else:
        # compact brand chip top-left
        a = min(1.0, (t - 6.5) / 0.4)
        chip = "VIDMCP"
        cw, ch = text_size(draw, chip, f_mono)
        rounded_rect(
            draw,
            (24, 18, 24 + cw + 28, 18 + ch + 14),
            8,
            (10, 12, 18, int(170 * a)),
            outline=(*LIME, int(160 * a)),
            width=1,
        )
        draw.text((38, 24), chip, font=f_mono, fill=(*LIME, int(255 * a)))

    # Chapter badge top-right
    code, title, ca = chapter_for(t)
    if code and ca > 0.02:
        badge_w = int(w * 0.34)
        bx = w - badge_w - 28
        by = 22
        rounded_rect(
            draw,
            (bx, by, bx + badge_w, by + 72),
            14,
            (10, 12, 18, int(200 * ca)),
            outline=(*CYAN, int(160 * ca)),
            width=2,
        )
        draw.text((bx + 18, by + 10), code, font=f_mono, fill=(*LIME, int(255 * ca)))
        draw.text((bx + 18, by + 34), title, font=f_body, fill=(*PAPER, int(255 * ca)))
        # glyph
        draw_dimension_glyph(draw, code if code != "LESSON" else "01", bx + badge_w - 40, by + 36, 1.1, ca)

    # Side dimension ladder (after intro)
    if t > 4.0:
        la = min(1.0, (t - 4.0) / 0.6) * 0.9
        steps = [
            ("0D", "point", 16.9, 23.2),
            ("1D", "line", 23.2, 34.5),
            ("2D", "plane", None, None),  # mentioned conceptually later
            ("3D", "space", 34.5, 40.6),
        ]
        # Re-map to his speech: 1D=point, 2D=line, 3D=space
        steps = [
            ("1D", "point", 16.5, 23.2),
            ("2D", "line", 23.2, 34.5),
            ("3D", "space", 34.5, 40.6),
        ]
        sx = w - 118
        sy0 = int(h * 0.28)
        for i, (lab, sub, a0, a1) in enumerate(steps):
            active = a0 is not None and a0 <= t <= a1
            past = a1 is not None and t > a1
            y = sy0 + i * 70
            col = LIME if active else (CYAN if past else MUTED)
            aa = int(255 * la * (1.0 if active or past else 0.55))
            r = 10 if active else 7
            draw.ellipse((sx + 18 - r, y + 10 - r, sx + 18 + r, y + 10 + r), fill=(*col, aa))
            if i < len(steps) - 1:
                draw.line((sx + 18, y + 20, sx + 18, y + 70), fill=(*MUTED, int(120 * la)), width=2)
            draw.text((sx + 36, y), lab, font=f_body, fill=(*col, aa))
            draw.text((sx + 36, y + 24), sub, font=f_small, fill=(*MUTED, int(200 * la)))

    # Captions
    text, ca = caption_for(t)
    if text and ca > 0.02:
        max_w = int(w * 0.82)
        lines = wrap_text(draw, text, f_cap, max_w)
        line_h = text_size(draw, "Ay", f_cap)[1] + 8
        box_h = line_h * len(lines) + 28
        box_w = 0
        for ln in lines:
            box_w = max(box_w, text_size(draw, ln, f_cap)[0])
        box_w += 48
        bx = (w - box_w) // 2
        by = h - box_h - int(h * 0.06)
        rounded_rect(
            draw,
            (bx, by, bx + box_w, by + box_h),
            16,
            (8, 10, 16, int(210 * ca)),
            outline=(*LIME, int(90 * ca)),
            width=1,
        )
        # accent line on top of caption
        draw.rectangle(
            (bx + 24, by, bx + box_w - 24, by + 3),
            fill=(*LIME, int(220 * ca)),
        )
        ty = by + 16
        for ln in lines:
            lw, _ = text_size(draw, ln, f_cap)
            draw.text(((w - lw) // 2, ty), ln, font=f_cap, fill=(*PAPER, int(255 * ca)))
            ty += line_h

    # Progress bar bottom
    prog = max(0.0, min(1.0, t / max(total, 0.001)))
    bar_y = h - 6
    draw.rectangle((0, bar_y, w, h), fill=(20, 22, 28, 180))
    draw.rectangle((0, bar_y, int(w * prog), h), fill=(*LIME, 255))

    # Timecode
    mm, ss = divmod(int(t), 60)
    tc = f"{mm:02d}:{ss:02d}"
    draw.text((w - 90, h - 36), tc, font=f_mono, fill=(*MUTED, 200))

    # End card flash
    if t > total - 2.2:
        a = ease_in_out((t - (total - 2.2)) / 2.2)
        # darken
        overlay = Image.new("RGBA", (w, h), (5, 5, 7, int(160 * a)))
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
        end = "Dimensions · 1D → 2D → 3D"
        tw, th = text_size(draw, end, f_title)
        draw.text(((w - tw) // 2, h // 2 - 40), end, font=f_title, fill=(*PAPER, int(255 * a)))
        sub = "Dhiraj Lochib  ·  vidmcp.com"
        sw, _ = text_size(draw, sub, f_body)
        draw.text(((w - sw) // 2, h // 2 + 20), sub, font=f_body, fill=(*CYAN, int(255 * a)))

    return img


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(SRC))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {SRC}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # prefer reliable fps from nb_frames/duration if weird
    n_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"input {w}x{h} fps={fps:.3f} frames~{n_declared}")

    raw_path = WORK / "video_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (w, h))

    # total duration from ffprobe-ish: use frame count
    # We'll compute as we go
    frames = []
    idx = 0
    # First pass: we need total duration - use CAP_PROP or probe
    total = n_declared / fps if n_declared > 0 else 40.54
    # better total from audio/video file
    try:
        import json as _json

        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(SRC),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        total = float(p.stdout.strip())
    except Exception:
        pass

    print(f"total duration {total:.2f}s")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        # subtle grade: slight contrast + darken bottom already via overlay
        # mild warm lift
        f = frame.astype(np.float32)
        f = f * 1.04 + 3
        f = np.clip(f, 0, 255).astype(np.uint8)

        overlay = build_overlay(w, h, t, total)
        composed = alpha_composite(f, overlay)
        writer.write(composed)
        idx += 1
        if idx % 60 == 0:
            print(f"  frame {idx} t={t:.1f}s")

    cap.release()
    writer.release()
    print(f"wrote raw frames {idx}")

    # mux original audio
    out_final = OUT_DIR / "dimensions_lesson_captions.mp4"
    out_desktop = DESKTOP / "dimensions_lesson_captions.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-i",
        str(SRC),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_final),
    ]
    print("ffmpeg mux…")
    subprocess.run(cmd, check=True)
    # copy desktop
    subprocess.run(["cp", str(out_final), str(out_desktop)], check=True)

    # still + 5s teaser
    still = OUT_DIR / "dimensions_lesson_still.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "18",
            "-i",
            str(out_final),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(still),
        ],
        check=True,
        capture_output=True,
    )
    teaser = OUT_DIR / "dimensions_lesson_5s.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "16",
            "-i",
            str(out_final),
            "-t",
            "5",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(teaser),
        ],
        check=True,
        capture_output=True,
    )

    # probe result
    pr = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height",
            "-of",
            "json",
            str(out_final),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    print(pr.stdout)
    print("OUT", out_final)
    print("DESKTOP", out_desktop)


if __name__ == "__main__":
    main()
