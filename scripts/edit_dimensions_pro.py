#!/usr/bin/env python3
"""Premium dimensions lesson edit from Photo Booth source.

Features: pro vocals + space BGM duck · MediaPipe matte · animated dim BG ·
infographic chapters · captions · brand HUD.
"""
from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

SRC = Path(
    "/Users/dhirajlochib/Pictures/Photo Booth Library/Pictures/"
    "Movie on 10-07-26 at 12.45 PM.mov"
)
WORK = Path("/tmp/vidmcp_dim_pro")
OUT = Path("/Users/dhirajlochib/Developer/VidMcp/final_output/dimensions_lesson.mp4")
DESK = Path("/Users/dhirajlochib/Desktop/dimensions_lesson.mp4")
FONTS = Path("/Users/dhirajlochib/Developer/VidMcp/scripts/fonts")
MODEL = Path("/tmp/mp_models/selfie_segmenter.tflite")

LIME = (212, 255, 42)
CYAN = (42, 255, 209)
VIOLET = (177, 140, 255)
PEACH = (255, 162, 78)
PAPER = (239, 239, 235)
MUTED = (155, 160, 170)

# Clean captions from Whisper (readable, not raw filler)
CAPTIONS = [
    (0.20, 4.90, "Hello — today we look at what dimensions are."),
    (5.60, 11.50, "Dimensions are how we see space at different levels."),
    (11.50, 16.90, "A person experiences the world through dimensions."),
    (17.40, 23.50, "1D — we start with a single point."),
    (23.50, 28.80, "In 2D you can move forward and backward."),
    (28.80, 34.00, "That freedom is a line — the second dimension."),
    (34.00, 40.00, "3D is the world we live in — full space."),
]

CHAPTERS = [
    (0.0, 5.5, "LESSON", "What are dimensions?"),
    (5.5, 17.0, "INTRO", "How we see space"),
    (17.0, 25.5, "01", "1D · Point"),
    (25.5, 34.0, "02", "2D · Line"),
    (34.0, 40.6, "03", "3D · Space"),
]

# Active dimension viz window
DIM_VIZ = [
    (17.0, 25.5, "1d"),
    (25.5, 34.0, "2d"),
    (34.0, 40.6, "3d"),
]


def font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except Exception:
        return ImageFont.load_default()


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


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


def glass(draw, xy, r, al, accent=CYAN, thick=2):
    x0, y0, x1, y1 = xy
    rr(draw, xy, r, (8, 10, 16, int(210 * al)), outline=(*accent, int(150 * al)), width=thick)
    draw.line((x0 + r, y0 + 1, x1 - r, y0 + 1), fill=(255, 255, 255, int(35 * al)), width=1)
    draw.rectangle((x0, y0 + r, x0 + 3, y1 - r), fill=(*accent, int(220 * al)))


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def generate_space_bgm(out_wav: Path, duration_sec: float, sr: int = 48000) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    rng = np.random.default_rng(21)

    def soft(f, amp, tt=None):
        tt = t if tt is None else tt
        ph = 2 * np.pi * f * tt
        return amp * (0.62 * np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph))

    pad = np.zeros(n)
    for f, a in [
        (41.2, 0.11),
        (55.0, 0.10),
        (82.4, 0.09),
        (110.0, 0.08),
        (164.8, 0.065),
        (220.0, 0.05),
        (329.6, 0.035),
        (440.0, 0.022),
    ]:
        pad += soft(f, a) * (1 + 0.05 * np.sin(2 * np.pi * (0.04 + f * 1e-5) * t))
    pad *= 0.55 + 0.45 * np.sin(2 * np.pi * t / 18.0)
    pad *= 0.75 + 0.25 * np.sin(2 * np.pi * t / 7.5 + 0.8)

    shimmer = np.zeros(n)
    for f, a in [(880, 0.012), (1318, 0.008)]:
        shimmer += soft(f, a) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.08 * t + f * 0.01))

    dust = rng.normal(0, 1, n)
    for _ in range(5):
        dust = np.convolve(dust, np.ones(100) / 100, mode="same")
    dust *= 0.013

    # motif at chapter hits
    piano = np.zeros(n)
    for start, f, amp, length in [
        (0.6, 220.0, 0.09, 3.5),
        (5.5, 277.2, 0.08, 3.0),
        (17.0, 329.6, 0.10, 3.5),  # 1D
        (25.5, 392.0, 0.10, 4.0),  # 2D
        (34.0, 261.6, 0.11, 4.5),  # 3D
        (37.5, 440.0, 0.08, 3.0),
    ]:
        i0 = int(start * sr)
        nn = min(int(length * sr), n - i0)
        if nn <= 0 or i0 >= n:
            continue
        tt = np.arange(nn) / sr
        env = np.exp(-tt * 0.5) * (1 - np.exp(-tt * 16))
        tone = soft(f, amp, tt) + 0.3 * soft(f * 2, amp * 0.28, tt)
        piano[i0 : i0 + nn] += tone * env

    # soft risers into each dim chapter
    def riser(t0, dur, amp=0.04):
        i0 = int(t0 * sr)
        nn = min(int(dur * sr), n - i0)
        if nn <= 0 or i0 >= n:
            return
        tt = np.arange(nn) / sr
        freq = 90 + 380 * (tt / max(dur, 0.01)) ** 1.3
        phase = 2 * np.pi * np.cumsum(freq) / sr
        env = (tt / dur) ** 1.4 * amp
        pad[i0 : i0 + nn] += np.sin(phase) * env

    riser(15.5, 1.8, 0.04)
    riser(24.0, 1.6, 0.035)
    riser(32.5, 1.6, 0.04)

    mix = pad + shimmer + dust + piano
    fi, fo = int(1.5 * sr), int(2.5 * sr)
    mix[:fi] *= np.linspace(0, 1, fi)
    mix[-fo:] *= np.linspace(1, 0, fo)
    peak = float(np.max(np.abs(mix)) + 1e-9)
    mix = mix / peak * 0.56
    left = mix * 0.92 + np.roll(mix, 120) * 0.08
    right = mix * 0.92 + np.roll(mix, -100) * 0.08
    stereo = np.stack([left, right], axis=1)
    pcm = np.clip(stereo * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out_wav


def polish_vocals(src_video: Path, out: Path) -> Path:
    raw = WORK / "vox_raw.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_video), "-vn", "-ac", "2", "-ar", "48000", str(raw)],
        check=True, capture_output=True,
    )
    af = (
        "highpass=f=90,"
        "lowpass=f=11500,"
        "afftdn=nr=15:nf=-28:tn=1,"
        "anlmdn=s=0.0002:p=0.002:r=0.002:m=12,"
        "acompressor=threshold=-20dB:ratio=3.2:attack=8:release=140:makeup=4,"
        "equalizer=f=180:t=q:w=0.9:g=-2.2,"
        "equalizer=f=350:t=q:w=1.0:g=-1.2,"
        "equalizer=f=2800:t=q:w=1.1:g=3.2,"
        "equalizer=f=5200:t=q:w=1.3:g=2.0,"
        "equalizer=f=7500:t=q:w=1.5:g=-3.2,"
        "equalizer=f=120:t=q:w=0.8:g=1.0,"
        "agate=threshold=0.016:ratio=2.2:attack=4:release=90,"
        "loudnorm=I=-14:TP=-1.5:LRA=10"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw), "-af", af, "-ar", "48000", "-ac", "2", str(out)],
        check=True, capture_output=True,
    )
    return out


def mix_audio(vox: Path, bgm: Path, out: Path, duration: float, bgm_vol: float = 0.40) -> Path:
    fade_in, fade_out = 1.2, 2.4
    fade_out_start = max(0.0, duration - fade_out)
    fc = (
        f"[1:a]atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,"
        f"volume={bgm_vol},"
        f"afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_out_start:.3f}:d={fade_out},"
        f"equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3000:t=q:w=1:g=-1.5[bg];"
        f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.08[vox];"
        f"[bg][vox]sidechaincompress=threshold=0.018:ratio=7:attack=40:release=260:level_sc=1[bgd];"
        f"[vox][bgd]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
        f"alimiter=limit=0.95:level=false,"
        f"loudnorm=I=-14:TP=-1.2:LRA=11[a]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(vox), "-i", str(bgm),
            "-filter_complex", fc, "-map", "[a]", "-ar", "48000", "-ac", "2", str(out),
        ],
        check=True, capture_output=True,
    )
    return out


# ---------------------------------------------------------------------------
# Background + matte
# ---------------------------------------------------------------------------

def make_bg(h: int, w: int, t: float, total: float, dim_mode: str | None) -> np.ndarray:
    """Deep space education plate (BGR)."""
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    pulse = 0.5 + 0.5 * math.sin(t * 0.4)
    # palette shifts by active dimension
    if dim_mode == "1d":
        r = 10 + 20 * y + 15 * pulse
        g = 14 + 30 * (1 - y) + 20 * math.sin(t * 0.25)
        b = 28 + 50 * (1 - y)
    elif dim_mode == "2d":
        r = 8 + 15 * y
        g = 18 + 40 * (1 - y) + 25 * pulse
        b = 35 + 55 * (1 - y) + 15 * math.cos(t * 0.2)
    elif dim_mode == "3d":
        r = 12 + 25 * y + 20 * pulse
        g = 12 + 20 * (1 - y)
        b = 40 + 60 * (1 - y) + 20 * math.sin(t * 0.3)
    else:
        r = 8 + 18 * y + 12 * pulse
        g = 12 + 18 * (1 - y) + 10 * math.sin(t * 0.2)
        b = 26 + 45 * (1 - y) + 18 * math.cos(t * 0.15)

    cx, cy = 0.48, 0.40
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 * 1.15)
    glow = np.clip(1.0 - dist * 1.7, 0, 1) ** 2
    r = r + glow * (35 + 25 * math.sin(t * 0.5))
    g = g + glow * (80 + 35 * math.sin(t * 0.5 + 1))
    b = b + glow * (55 + 25 * math.cos(t * 0.4))
    swirl = 0.5 + 0.5 * np.sin(7 * (x + 0.08 * math.sin(t)) + 5 * y + t * 0.7)
    g = g + swirl * 10 * glow
    b = b + swirl * 16 * glow
    bg = np.clip(np.stack([b, g, r], axis=-1), 0, 255).astype(np.uint8)

    # stars
    rng = np.random.default_rng(9)
    n_stars = 140
    xs = rng.integers(0, w, n_stars)
    ys = rng.integers(0, h, n_stars)
    for i, (sx, sy) in enumerate(zip(xs, ys)):
        tw = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 2.2 + i))
        c = int(170 * tw)
        cv2.circle(bg, (int(sx), int(sy)), 1, (c, c, min(255, c + 45)), -1)

    # perspective grid lower half
    for i in range(10):
        yy = int(h * (0.58 + i * 0.045))
        cv2.line(bg, (0, yy), (w, yy), (28, 38, 50), 1, cv2.LINE_AA)
    for i in range(-8, 9):
        x0 = w // 2 + i * 80
        cv2.line(bg, (x0, int(h * 0.58)), (w // 2 + i * 200, h), (22, 32, 42), 1, cv2.LINE_AA)

    return bg


def soft_mask(mask: np.ndarray) -> np.ndarray:
    """Tighten person mask: drop thin background islands (AC / wall bits)."""
    m = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    # binary cleanup first
    _, bw = cv2.threshold(m, 120, 255, cv2.THRESH_BINARY)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # open removes speckles, close fills hair holes
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k5, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k7, iterations=2)
    # keep largest connected component (the person)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if nlab > 1:
        # label 0 is background
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        bw = np.where(labels == best, 255, 0).astype(np.uint8)
    # shrink slightly so room edges don't stick, then soft feather
    bw = cv2.erode(bw, k5, iterations=1)
    m = cv2.GaussianBlur(bw, (0, 0), 4.0)
    return m.astype(np.float32) / 255.0


def composite(fg_bgr: np.ndarray, bg_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = mask[..., None]
    rim = cv2.GaussianBlur((mask * 255).astype(np.uint8), (0, 0), 9).astype(np.float32) / 255.0
    edge = np.clip(rim - mask, 0, 1)[..., None]
    rim_col = np.array([90, 230, 190], dtype=np.float32)
    fg = fg_bgr.astype(np.float32)
    bg = bg_bgr.astype(np.float32)
    out = bg * (1 - a) + fg * a
    out = out + edge * rim_col * 0.4
    sh = cv2.GaussianBlur((mask * 255).astype(np.uint8), (0, 0), 22).astype(np.float32) / 255.0
    sh = np.roll(np.roll(sh, 14, axis=0), 6, axis=1)
    sh = np.clip(sh - mask, 0, 1)[..., None]
    out = out * (1 - sh * 0.32)
    return np.clip(out, 0, 255).astype(np.uint8)


def grade(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    f = (f - 128) * 1.06 + 128 + 2
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    r = r * 1.03 + 1
    b = b * 1.05 + 3
    f = np.stack([b, g, r], axis=-1)
    h, w = frame.shape[:2]
    yy = np.linspace(-1, 1, h)[:, None]
    xx = np.linspace(-1, 1, w)[None, :]
    vig = 1.0 - 0.22 * np.clip(xx * xx + yy * yy * 0.85, 0, 1) ** 1.15
    f *= vig[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Graphics
# ---------------------------------------------------------------------------

def draw_dim_viz(draw, kind: str, cx: int, cy: int, t: float, al: float, scale: float = 1.0):
    a = int(255 * al)
    if kind == "1d":
        # pulsing point
        r = int((10 + 4 * math.sin(t * 4)) * scale)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*LIME, a))
        for k in range(3):
            R = int((22 + k * 16 + 6 * math.sin(t * 3 + k)) * scale)
            draw.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(*CYAN, int(a * (0.5 - k * 0.12))), width=2)
        draw.text((cx - 18, cy + int(50 * scale)), "POINT", font=font("JetBrainsMono-Regular.ttf", 14), fill=(*LIME, a))
    elif kind == "2d":
        L = int(70 * scale)
        # animated line with travel dot
        draw.line((cx - L, cy, cx + L, cy), fill=(*CYAN, a), width=max(3, int(4 * scale)))
        # arrowheads
        ah = int(12 * scale)
        draw.polygon([(cx + L, cy), (cx + L - ah, cy - ah // 2), (cx + L - ah, cy + ah // 2)], fill=(*LIME, a))
        draw.polygon([(cx - L, cy), (cx - L + ah, cy - ah // 2), (cx - L + ah, cy + ah // 2)], fill=(*LIME, a))
        # traveler
        p = 0.5 + 0.5 * math.sin(t * 2.2)
        tx = int(cx - L + 2 * L * p)
        draw.ellipse((tx - 7, cy - 7, tx + 7, cy + 7), fill=(*LIME, a))
        # faint plane grid
        for i in range(-2, 3):
            yy = cy + i * int(18 * scale)
            draw.line((cx - L, yy, cx + L, yy), fill=(*MUTED, int(a * 0.25)), width=1)
        draw.text((cx - 14, cy + int(40 * scale)), "LINE", font=font("JetBrainsMono-Regular.ttf", 14), fill=(*CYAN, a))
    else:
        # isometric cube
        s = int(42 * scale)
        ox, oy = cx, cy
        # back face offset
        d = int(22 * scale)
        # front square
        fl = (ox - s, oy - s // 2)
        fr = (ox + s, oy - s // 2)
        bl = (ox - s, oy + s // 2)
        br = (ox + s, oy + s // 2)
        # top diamond
        top = (ox, oy - s - d // 2)
        # animated rotate-ish wobble
        wob = int(4 * math.sin(t * 1.8))
        pts_front = [
            (fl[0] + wob, fl[1]),
            (fr[0] + wob, fr[1]),
            (br[0] - wob, br[1]),
            (bl[0] - wob, bl[1]),
        ]
        draw.polygon(pts_front, outline=(*LIME, a), width=3)
        # top edges
        draw.line((pts_front[0][0], pts_front[0][1], top[0], top[1] + wob // 2), fill=(*CYAN, a), width=2)
        draw.line((pts_front[1][0], pts_front[1][1], top[0] + d, top[1] + wob // 2), fill=(*CYAN, a), width=2)
        draw.line((top[0], top[1] + wob // 2, top[0] + d, top[1] + wob // 2), fill=(*VIOLET, a), width=2)
        # depth edges
        draw.line((pts_front[1][0], pts_front[1][1], pts_front[1][0] + d, pts_front[1][1] - d // 2), fill=(*VIOLET, int(a * 0.8)), width=2)
        draw.line((pts_front[2][0], pts_front[2][1], pts_front[2][0] + d, pts_front[2][1] - d // 2), fill=(*VIOLET, int(a * 0.8)), width=2)
        draw.line((pts_front[1][0] + d, pts_front[1][1] - d // 2, pts_front[2][0] + d, pts_front[2][1] - d // 2), fill=(*PEACH, a), width=2)
        draw.line((top[0] + d, top[1] + wob // 2, pts_front[1][0] + d, pts_front[1][1] - d // 2), fill=(*CYAN, a), width=2)
        draw.text((cx - 22, cy + int(70 * scale)), "SPACE", font=font("JetBrainsMono-Regular.ttf", 14), fill=(*PEACH, a))


def build_overlay(w: int, h: int, t: float, total: float) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    f_hero = font("CrimsonPro-Bold.ttf", 48)
    f_title = font("CrimsonPro-Bold.ttf", 32)
    f_body = font("Outfit-Medium.ttf", 22)
    f_cap = font("Outfit-Medium.ttf", 24)
    f_small = font("Outfit-Light.ttf", 15)
    f_mono = font("JetBrainsMono-Regular.ttf", 14)

    # letterbox
    lb = int(h * 0.06)
    for y in range(lb):
        a = int(210 * (1 - y / lb) ** 0.55)
        draw.line([(0, y), (w, y)], fill=(4, 5, 8, a))
        draw.line([(0, h - 1 - y), (w, h - 1 - y)], fill=(4, 5, 8, a))

    # brand
    rr(draw, (20, 16, 132, 46), 8, (8, 10, 16, 200), outline=(*LIME, 170), width=1)
    draw.text((34, 24), "VIDMCP", font=f_mono, fill=(*LIME, 255))
    draw.text((144, 24), "dimensions", font=f_small, fill=(*MUTED, 200))

    # active chapter
    ch = CHAPTERS[0]
    for c in CHAPTERS:
        if c[0] <= t <= c[1]:
            ch = c
            break
    chip = f"{ch[2]}  {ch[3]}"
    cw, _ = text_size(draw, chip, f_mono)
    cx0 = w - cw - 48
    rr(draw, (cx0, 16, w - 18, 46), 8, (8, 10, 16, 195), outline=(*CYAN, 130), width=1)
    draw.text((cx0 + 14, 24), chip, font=f_mono, fill=(*CYAN, 255))

    # progress
    prog = t / max(total, 0.01)
    draw.rectangle((0, h - 4, w, h), fill=(18, 20, 28, 220))
    draw.rectangle((0, h - 4, int(w * prog), h), fill=(*LIME, 255))
    for a0, _, _, _ in CHAPTERS:
        x = int(w * (a0 / max(total, 0.01)))
        draw.rectangle((x, h - 8, x + 2, h), fill=(*PAPER, 170))

    # intro title (upper band only)
    if t < 5.2:
        al = beat_alpha(t, 0.15, 5.2, 0.4)
        A = int(255 * al)
        title = "What are dimensions?"
        sub = "1D  ·  2D  ·  3D"
        # keep title high in letterbox band so face stays clear
        f_intro = font("CrimsonPro-Bold.ttf", 40)
        tw, th = text_size(draw, title, f_intro)
        x = (w - tw) // 2
        y = int(h * 0.07)
        glass(draw, (x - 24, y - 10, x + tw + 24, y + th + 34), 14, al, LIME)
        draw.text((x, y), title, font=f_intro, fill=(*PAPER, A))
        sw, _ = text_size(draw, sub, f_body)
        draw.text(((w - sw) // 2, y + th + 4), sub, font=f_body, fill=(*CYAN, A))

    # lower third
    if t < 5.8:
        al = beat_alpha(t, 0.2, 5.8, 0.35)
        A = int(255 * al)
        lx, ly = 28, int(h * 0.74)
        draw.rectangle((lx, ly, lx + 4, ly + 54), fill=(*LIME, A))
        rr(draw, (lx + 4, ly, lx + 320, ly + 54), 4, (8, 10, 16, int(215 * al)))
        draw.text((lx + 16, ly + 6), "Dhiraj Lochib", font=f_body, fill=(*PAPER, A))
        draw.text((lx + 16, ly + 30), "vidmcp.com  ·  dimensions", font=f_small, fill=(*MUTED, A))

    # side ladder
    if t > 4.0:
        la = min(1.0, (t - 4.0) / 0.5)
        steps = [
            ("1D", "point", 17.0, 25.5, LIME),
            ("2D", "line", 25.5, 34.0, CYAN),
            ("3D", "space", 34.0, 40.6, PEACH),
        ]
        sx = w - 108
        sy0 = int(h * 0.28)
        for i, (lab, sub, a0, a1, col) in enumerate(steps):
            active = a0 <= t <= a1
            past = t > a1
            y = sy0 + i * 68
            c = col if (active or past) else MUTED
            aa = int(255 * la * (1.0 if active or past else 0.45))
            r = 9 if active else 6
            draw.ellipse((sx + 8 - r, y + 6 - r, sx + 8 + r, y + 6 + r), fill=(*c, aa))
            if i < 2:
                draw.line((sx + 8, y + 16, sx + 8, y + 68), fill=(*MUTED, int(120 * la)), width=2)
            draw.text((sx + 24, y - 2), lab, font=f_body, fill=(*c, aa))
            draw.text((sx + 24, y + 22), sub, font=f_small, fill=(*MUTED, aa))

    # right panel dim viz + card
    for a0, b0, kind in DIM_VIZ:
        al = beat_alpha(t, a0, b0, 0.45)
        if al < 0.02:
            continue
        A = int(255 * al)
        panel_w = 300
        px = w - panel_w - 28
        py = int(h * 0.18)
        labels = {
            "1d": ("01 · POINT", "A single point", "No length · no width"),
            "2d": ("02 · LINE", "A line of freedom", "Forward & backward"),
            "3d": ("03 · SPACE", "The world we live in", "Full spatial freedom"),
        }
        lab, title, sub = labels[kind]
        accent = {"1d": LIME, "2d": CYAN, "3d": PEACH}[kind]
        glass(draw, (px, py, px + panel_w, py + 320), 16, al, accent)
        draw.text((px + 20, py + 16), lab, font=f_mono, fill=(*accent, A))
        draw.text((px + 20, py + 44), title, font=f_title, fill=(*PAPER, A))
        draw.text((px + 20, py + 86), sub, font=f_small, fill=(*MUTED, A))
        draw_dim_viz(draw, kind, px + panel_w // 2, py + 200, t, al, scale=1.15)

    # floating labels on bg left
    if t > 6:
        for i, (lab, col, fy) in enumerate([
            ("0D", MUTED, 0.22),
            ("1D", LIME, 0.38),
            ("2D", CYAN, 0.54),
            ("3D", PEACH, 0.70),
        ]):
            al = 0.35 + 0.25 * (0.5 + 0.5 * math.sin(t + i))
            ox = int(40 + 8 * math.sin(t * 0.6 + i))
            oy = int(h * fy + 6 * math.cos(t * 0.5 + i))
            draw.text((ox, oy), lab, font=f_mono, fill=(*col, int(180 * al)))

    # captions
    for a, b, text in CAPTIONS:
        al = beat_alpha(t, a, b, 0.28)
        if al < 0.02:
            continue
        A = int(255 * al)
        max_w = int(w * 0.72)
        lines = wrap(draw, text, f_cap, max_w)
        lh = text_size(draw, "Ay", f_cap)[1] + 6
        box_h = lh * len(lines) + 24
        box_w = max(text_size(draw, ln, f_cap)[0] for ln in lines) + 44
        bx = (w - box_w) // 2
        by = h - box_h - int(h * 0.09)
        rr(draw, (bx, by, bx + box_w, by + box_h), 12, (5, 7, 12, int(225 * al)), outline=(*LIME, int(80 * al)), width=1)
        draw.rectangle((bx + 18, by, bx + box_w - 18, by + 3), fill=(*LIME, A))
        yy = by + 14
        for ln in lines:
            lw, _ = text_size(draw, ln, f_cap)
            draw.text(((w - lw) // 2, yy), ln, font=f_cap, fill=(*PAPER, A))
            yy += lh

    # end card
    if t > total - 2.8:
        al = ease((t - (total - 2.8)) / 2.8)
        A = int(255 * al)
        draw.rectangle((0, 0, w, h), fill=(5, 5, 7, int(150 * al)))
        msg = "1D  →  2D  →  3D"
        tw, _ = text_size(draw, msg, f_hero)
        draw.text(((w - tw) // 2, h // 2 - 40), msg, font=f_hero, fill=(*PAPER, A))
        sub = "Dhiraj Lochib  ·  vidmcp.com"
        sw, _ = text_size(draw, sub, f_body)
        draw.text(((w - sw) // 2, h // 2 + 20), sub, font=f_body, fill=(*CYAN, A))

    mm, ss = divmod(int(t), 60)
    draw.text((w - 80, h - 32), f"{mm:02d}:{ss:02d}", font=f_mono, fill=(*MUTED, 180))
    return img


def alpha_composite_bgr(base_bgr: np.ndarray, overlay_rgba: Image.Image) -> np.ndarray:
    ov = np.array(overlay_rgba).astype(np.float32)
    a = ov[:, :, 3:4] / 255.0
    rgb = ov[:, :, :3][:, :, ::-1]
    base = base_bgr.astype(np.float32)
    return (base * (1 - a) + rgb * a).astype(np.uint8)


def active_dim(t: float) -> str | None:
    for a, b, k in DIM_VIZ:
        if a <= t <= b:
            return k
    return None


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert SRC.exists(), f"missing source: {SRC}"
    assert MODEL.exists(), f"missing mediapipe model: {MODEL}"

    duration = float(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(SRC)],
            text=True,
        ).strip()
    )
    print(f"source duration={duration:.2f}s")

    final_a = WORK / "final_audio.wav"
    if final_a.exists() and final_a.stat().st_size > 1000:
        print("reuse existing audio mix")
    else:
        print("1/4 polish vocals…")
        vox = polish_vocals(SRC, WORK / "vox_pro.wav")
        print("2/4 generate BGM…")
        bgm = generate_space_bgm(WORK / "bgm_space.wav", duration + 1.5)
        print("3/4 mix audio…")
        final_a = mix_audio(vox, bgm, WORK / "final_audio.wav", duration, bgm_vol=0.42)
        print("audio ready")

    print("render video + matte + graphics…")
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
    print(f"  {w}x{h} @ {fps:.2f}")

    raw = WORK / "vfx_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    ts_ms = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = segmenter.segment_for_video(mp_image, ts_ms)
        cat = result.category_mask.numpy_view()
        if cat.ndim == 3:
            cat = cat[:, :, 0]
        mask = cat.astype(np.float32)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.float32)
        # ensure person is high in center
        cy0, cy1 = int(h * 0.15), int(h * 0.95)
        cx0, cx1 = int(w * 0.2), int(w * 0.8)
        center_m = mask[cy0:cy1, cx0:cx1].mean()
        edge_m = float(np.concatenate([mask[:30, :].ravel(), mask[-30:, :].ravel()]).mean())
        if edge_m > center_m or center_m < 0.2:
            mask = 1.0 - mask
        mask = soft_mask(mask)

        dim = active_dim(t)
        bg = make_bg(h, w, t, duration, dim)
        comp = composite(frame, bg, mask)
        comp = grade(comp)
        ov = build_overlay(w, h, t, duration)
        final = alpha_composite_bgr(comp, ov)
        writer.write(final)

        idx += 1
        ts_ms = int(idx * 1000 / fps)
        if idx % 60 == 0:
            print(f"  frame {idx} t={t:.1f}s mask={mask.mean():.2f}")

    cap.release()
    writer.release()
    segmenter.close()
    print("frames", idx)

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
        check=True, capture_output=True,
    )
    subprocess.run(["cp", str(OUT), str(DESK)], check=True)
    still = OUT.parent / "dimensions_lesson_still.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "20", "-i", str(OUT), "-frames:v", "1", "-q:v", "2", str(still)],
        check=True, capture_output=True,
    )
    teaser = OUT.parent / "dimensions_lesson_5s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", "18", "-i", str(OUT), "-t", "5",
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
