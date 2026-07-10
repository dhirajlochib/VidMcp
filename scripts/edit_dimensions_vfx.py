#!/usr/bin/env python3
"""BG remove + animated dimension BG + graphics + BGM for dimensions lesson."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

SRC = Path("/Users/dhirajlochib/Developer/VidMcp/final_output/dimensions_lesson_captions.mp4")
OUT = Path("/Users/dhirajlochib/Developer/VidMcp/final_output/dimensions_lesson_vfx.mp4")
DESK = Path("/Users/dhirajlochib/Desktop/dimensions_lesson_vfx.mp4")
WORK = Path("/tmp/vidmcp_dim_vfx")
FONTS = Path("/Users/dhirajlochib/Developer/VidMcp/scripts/fonts")
MODEL = Path("/tmp/mp_models/selfie_segmenter.tflite")
BGM = Path("/tmp/vidmcp_img4447/bgm_cinematic.wav")

LIME = (212, 255, 42)
CYAN = (42, 255, 209)
VIOLET = (177, 140, 255)
PEACH = (255, 162, 78)
PAPER = (239, 239, 235)
MUTED = (150, 155, 165)


def font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except Exception:
        return ImageFont.load_default()


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def text_size(draw, text, f):
    b = draw.textbbox((0, 0), text, font=f)
    return b[2] - b[0], b[3] - b[1]


def rr(draw, xy, r, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def make_bg(h: int, w: int, t: float, total: float) -> np.ndarray:
    """Animated deep-space dimension background (BGR)."""
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    # base gradient
    pulse = 0.5 + 0.5 * math.sin(t * 0.35)
    r = (8 + 18 * y + 10 * pulse) 
    g = (10 + 12 * (1 - y) + 8 * math.sin(t * 0.2))
    b = (22 + 40 * (1 - y) + 20 * math.cos(t * 0.15))
    # radial glow center-left (behind subject)
    cx, cy = 0.45, 0.42
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 * 1.2)
    glow = np.clip(1.0 - dist * 1.8, 0, 1) ** 2
    r = r + glow * (40 + 30 * math.sin(t * 0.5))
    g = g + glow * (90 + 40 * math.sin(t * 0.5 + 1))
    b = b + glow * (60 + 20 * math.cos(t * 0.4))
    # secondary cyan swirl
    swirl = 0.5 + 0.5 * np.sin(8 * (x + 0.1 * math.sin(t)) + 6 * y + t * 0.8)
    g = g + swirl * 12 * glow
    b = b + swirl * 18 * glow
    bg = np.stack([b, g, r], axis=-1)  # BGR
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # stars
    rng = np.random.default_rng(7)
    n_stars = 120
    xs = rng.integers(0, w, n_stars)
    ys = rng.integers(0, h, n_stars)
    for i, (sx, sy) in enumerate(zip(xs, ys)):
        tw = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 2.5 + i))
        c = int(180 * tw)
        cv2.circle(bg, (int(sx), int(sy)), 1, (c, c, min(255, c + 40)), -1)

    # floating dimension glyphs
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    f_big = font("CrimsonPro-Bold.ttf", 42)
    f_sm = font("Outfit-Light.ttf", 16)
    labels = [
        (0.12, 0.22, "0D", "point"),
        (0.82, 0.28, "1D", "line"),
        (0.15, 0.72, "2D", "plane"),
        (0.80, 0.70, "3D", "space"),
    ]
    for i, (fx, fy, lab, sub) in enumerate(labels):
        ox = int(w * fx + 12 * math.sin(t * 0.7 + i))
        oy = int(h * fy + 10 * math.cos(t * 0.5 + i * 1.3))
        a = int(90 + 50 * (0.5 + 0.5 * math.sin(t + i)))
        col = [LIME, CYAN, VIOLET, PEACH][i]
        rr(draw, (ox - 10, oy - 10, ox + 90, oy + 50), 12, (8, 10, 18, a), outline=(*col, min(255, a + 40)), width=1)
        draw.text((ox, oy), lab, font=f_big, fill=(*col, min(255, a + 80)))
        draw.text((ox, oy + 30), sub, font=f_sm, fill=(*MUTED, min(255, a + 40)))

    # grid perspective lines
    for i in range(8):
        yy = int(h * (0.55 + i * 0.06))
        alpha = max(20, 90 - i * 10)
        # draw on numpy
        cv2.line(bg, (0, yy), (w, yy), (30, 40, 50), 1, cv2.LINE_AA)
    for i in range(-6, 7):
        x0 = w // 2 + i * 90
        cv2.line(bg, (x0, int(h * 0.55)), (w // 2 + i * 220, h), (25, 35, 45), 1, cv2.LINE_AA)

    ov = np.array(overlay)
    a = ov[:, :, 3:4].astype(np.float32) / 255.0
    rgb = ov[:, :, :3][:, :, ::-1].astype(np.float32)
    bg_f = bg.astype(np.float32)
    bg = (bg_f * (1 - a) + rgb * a).astype(np.uint8)
    return bg


def make_hud(w: int, h: int, t: float, total: float) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f_title = font("CrimsonPro-Bold.ttf", 36)
    f_body = font("Outfit-Medium.ttf", 22)
    f_mono = font("JetBrainsMono-Regular.ttf", 14)
    f_sm = font("Outfit-Light.ttf", 16)

    # brand
    rr(draw, (24, 20, 130, 50), 8, (10, 12, 18, 180), outline=(*LIME, 160), width=1)
    draw.text((38, 28), "VIDMCP", font=f_mono, fill=(*LIME, 255))

    # top title early
    if t < 5.5:
        a = ease(t / 0.5) if t < 0.5 else (ease((5.5 - t) / 0.5) if t > 5.0 else 1.0)
        A = int(255 * a)
        title = "What are dimensions?"
        tw, th = text_size(draw, title, f_title)
        x = (w - tw) // 2
        y = 70
        rr(draw, (x - 24, y - 12, x + tw + 24, y + th + 36), 14, (8, 10, 16, int(200 * a)), outline=(*LIME, int(180 * a)), width=2)
        draw.text((x, y), title, font=f_title, fill=(*PAPER, A))
        sub = "0D → 1D → 2D → 3D"
        sw, _ = text_size(draw, sub, f_sm)
        draw.text(((w - sw) // 2, y + th + 6), sub, font=f_sm, fill=(*CYAN, A))

    # chapter ladder right
    steps = [
        (0.0, 8.0, "1D", "point"),
        (8.0, 18.0, "1D", "point"),
        (18.0, 28.0, "2D", "line"),
        (28.0, 40.5, "3D", "space"),
    ]
    # simplified active by time
    ladder = [("1D", "point", 12, 22), ("2D", "line", 22, 32), ("3D", "space", 32, 40.5)]
    sx = w - 120
    sy0 = int(h * 0.30)
    for i, (lab, sub, a0, a1) in enumerate(ladder):
        active = a0 <= t <= a1
        past = t > a1
        col = LIME if active else (CYAN if past else MUTED)
        aa = 255 if active or past else 140
        y = sy0 + i * 64
        r = 9 if active else 6
        draw.ellipse((sx + 10 - r, y + 8 - r, sx + 10 + r, y + 8 + r), fill=(*col, aa))
        if i < len(ladder) - 1:
            draw.line((sx + 10, y + 18, sx + 10, y + 64), fill=(*MUTED, 120), width=2)
        draw.text((sx + 28, y), lab, font=f_body, fill=(*col, aa))
        draw.text((sx + 28, y + 24), sub, font=f_sm, fill=(*MUTED, 200))

    # lower vignette for text readability (existing captions stay in source; we add rim)
    # progress
    prog = t / max(total, 0.01)
    draw.rectangle((0, h - 5, w, h), fill=(20, 22, 28, 200))
    draw.rectangle((0, h - 5, int(w * prog), h), fill=(*LIME, 255))

    # name lower third first seconds
    if t < 5.0:
        a = ease(min(1.0, t / 0.4)) * (ease((5.0 - t) / 0.5) if t > 4.5 else 1.0)
        A = int(255 * a)
        lx, ly = 36, int(h * 0.78)
        draw.rectangle((lx, ly, lx + 4, ly + 52), fill=(*LIME, A))
        rr(draw, (lx + 4, ly, lx + 300, ly + 52), 4, (10, 12, 18, int(200 * a)))
        draw.text((lx + 16, ly + 6), "Dhiraj Lochib", font=f_body, fill=(*PAPER, A))
        draw.text((lx + 16, ly + 30), "vidmcp.com · dimensions", font=f_sm, fill=(*MUTED, A))

    # end card
    if t > total - 2.5:
        a = ease((t - (total - 2.5)) / 2.5)
        A = int(255 * a)
        draw.rectangle((0, 0, w, h), fill=(5, 5, 7, int(140 * a)))
        msg = "Dimensions · 1D → 2D → 3D"
        tw, th = text_size(draw, msg, f_title)
        draw.text(((w - tw) // 2, h // 2 - 30), msg, font=f_title, fill=(*PAPER, A))
        sub = "Dhiraj Lochib · vidmcp.com"
        sw, _ = text_size(draw, sub, f_body)
        draw.text(((w - sw) // 2, h // 2 + 20), sub, font=f_body, fill=(*CYAN, A))

    return img


def soft_mask(mask: np.ndarray) -> np.ndarray:
    """mask float 0-1 HxW → soft edge."""
    m = (mask * 255).astype(np.uint8)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    # slight erode then blur for hair-ish edge
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.erode(m, k, iterations=1)
    m = cv2.GaussianBlur(m, (0, 0), 2.5)
    return m.astype(np.float32) / 255.0


def composite(fg_bgr: np.ndarray, bg_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = mask[..., None]
    # rim light
    rim = cv2.GaussianBlur((mask * 255).astype(np.uint8), (0, 0), 8).astype(np.float32) / 255.0
    edge = np.clip(rim - mask, 0, 1)[..., None]
    rim_col = np.array([80, 220, 180], dtype=np.float32)  # cyan-ish BGR-ish
    fg = fg_bgr.astype(np.float32)
    bg = bg_bgr.astype(np.float32)
    out = bg * (1 - a) + fg * a
    out = out + edge * rim_col * 0.45
    # subtle drop shadow under subject
    sh = cv2.GaussianBlur((mask * 255).astype(np.uint8), (0, 0), 20).astype(np.float32) / 255.0
    sh = np.roll(sh, 12, axis=0)
    sh = np.roll(sh, 6, axis=1)
    sh = np.clip(sh - mask, 0, 1)[..., None]
    out = out * (1 - sh * 0.35)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    base = mp_python.BaseOptions(model_asset_path=str(MODEL))
    options = vision.ImageSegmenterOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=True,
    )
    segmenter = vision.ImageSegmenter.create_from_options(options)

    cap = cv2.VideoCapture(str(SRC))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total = n / fps if n else 40.4
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(SRC)],
            capture_output=True, text=True, check=True,
        )
        total = float(p.stdout.strip())
    except Exception:
        pass
    print(f"{w}x{h} @ {fps:.2f}  total={total:.2f}s")

    raw = WORK / "vfx_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    ts_ms = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        # MediaPipe wants RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = segmenter.segment_for_video(mp_image, ts_ms)
        # category mask: person often category 0 or confidence mask
        cat = result.category_mask.numpy_view()
        # selfie model: 0=bg 1=person OR inverted depending on model
        # float16 selfie_segmenter usually gives confidence in category 0 as person
        if cat.ndim == 3:
            cat = cat[:, :, 0]
        # values are 0/1 for category mask
        unique = np.unique(cat)
        # person is typically the non-zero class
        if cat.max() <= 1:
            mask = cat.astype(np.float32)
            # if mostly 1s, person might be 0
            if mask.mean() > 0.55:
                mask = 1.0 - mask
        else:
            mask = (cat > 0).astype(np.float32)

        # MediaPipe selfie category mask: 0=person often; force person-high mask
        # Prefer center mass as person
        mask = mask.astype(np.float32)
        cy0, cy1 = int(h * 0.12), int(h * 0.98)
        cx0, cx1 = int(w * 0.15), int(w * 0.85)
        center_m = mask[cy0:cy1, cx0:cx1].mean()
        edge_m = float(np.concatenate([mask[:40, :].ravel(), mask[-40:, :].ravel()]).mean())
        # If edges more "person" than center, invert
        if edge_m > center_m:
            mask = 1.0 - mask
        # Always ensure center has significant person
        if mask[cy0:cy1, cx0:cx1].mean() < 0.2:
            mask = 1.0 - mask
        mask = soft_mask(mask)

        bg = make_bg(h, w, t, total)
        comp = composite(frame, bg, mask)
        hud = make_hud(w, h, t, total)
        ov = np.array(hud)
        a = ov[:, :, 3:4].astype(np.float32) / 255.0
        rgb_hud = ov[:, :, :3][:, :, ::-1].astype(np.float32)
        comp_f = comp.astype(np.float32)
        final = (comp_f * (1 - a) + rgb_hud * a).astype(np.uint8)
        writer.write(final)

        idx += 1
        ts_ms = int(idx * 1000 / fps)
        if idx % 60 == 0:
            print(f"  frame {idx} t={t:.1f}s mask_mean={mask.mean():.3f}")

    cap.release()
    writer.release()
    segmenter.close()
    print("frames", idx)

    # mix original audio + BGM low
    audio_mix = WORK / "audio_mix.wav"
    if BGM.exists():
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(SRC),
            "-i", str(BGM),
            "-filter_complex",
            f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.0[vox];"
            f"[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.13,atrim=0:{total},asetpts=PTS-STARTPTS[bg];"
            f"[vox][bg]amix=inputs=2:duration=first:dropout_transition=0,alimiter=limit=0.95[a]",
            "-map", "[a]", audio_mix,
        ], check=True, capture_output=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-i", str(SRC), "-vn", audio_mix], check=True, capture_output=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(raw),
        "-i", str(audio_mix),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(OUT),
    ], check=True)
    subprocess.run(["cp", str(OUT), str(DESK)], check=True)

    still = OUT.parent / "dimensions_lesson_vfx_still.jpg"
    subprocess.run(["ffmpeg", "-y", "-ss", "12", "-i", str(OUT), "-frames:v", "1", "-q:v", "2", str(still)], check=True, capture_output=True)
    print("OUT", OUT)
    print("DESKTOP", DESK)
    subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(OUT)])


if __name__ == "__main__":
    main()
