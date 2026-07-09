#!/usr/bin/env python3
"""Complex brand-matched marketing samples (dhirajlochib.com palette + fonts)."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path(__file__).resolve().parent / "fonts"
OUTS = [ROOT / "demos" / "samples", ROOT / "site" / "assets" / "demos"]
W, H, FPS = 1280, 720, 30
DUR = 6.0

# Brand (RGB) from dhirajlochib.com
BG = (5, 5, 7)
TEXT = (239, 239, 235)
SUB = (160, 157, 166)
MUTED = (90, 88, 96)
LIME = (212, 255, 42)
CYAN = (42, 255, 209)
PEACH = (255, 162, 78)
VIOLET = (177, 140, 255)
ROSE = (255, 107, 138)


def fnt(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def ease_io(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def mesh_bg(t: float = 0.0) -> Image.Image:
    """Multi-stop radial mesh like personal site."""
    arr = np.zeros((H, W, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    xf, yf = xx / W, yy / H
    # base
    arr[:] = BG
    # drifting orbs
    orbs = [
        (0.12 + 0.03 * math.sin(t * 0.4), 0.18 + 0.02 * math.cos(t * 0.3), 0.55, LIME, 0.11),
        (0.88 + 0.02 * math.cos(t * 0.35), 0.78 + 0.03 * math.sin(t * 0.25), 0.5, CYAN, 0.09),
        (0.55 + 0.04 * math.sin(t * 0.2), 0.45 + 0.03 * math.cos(t * 0.28), 0.45, VIOLET, 0.07),
        (0.3 + 0.02 * math.cos(t * 0.5), 0.85, 0.35, PEACH, 0.05),
    ]
    for cx, cy, rad, col, strength in orbs:
        d = np.sqrt((xf - cx) ** 2 + (yf - cy) ** 2 * 0.85)
        mask = np.clip(1.0 - d / rad, 0, 1) ** 2
        for i in range(3):
            arr[:, :, i] += mask * col[i] * strength
    # vignette
    vx = (xf - 0.5) ** 2 + (yf - 0.5) ** 2
    arr *= np.clip(1.15 - vx * 1.6, 0.45, 1.0)[..., None]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    # film grain
    rng = np.random.default_rng(int(t * 30) % 10000)
    grain = rng.integers(-3, 4, (H, W, 1), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def glow_layer(size: tuple[int, int], draw_fn, blur: int = 18, color=LIME, alpha: float = 0.55) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    draw_fn(d)
    g = layer.filter(ImageFilter.GaussianBlur(blur))
    # tint
    arr = np.array(g)
    if arr.shape[2] == 4:
        a = arr[:, :, 3:4].astype(np.float32) / 255.0
        out = np.zeros_like(arr)
        out[:, :, 0] = color[0]
        out[:, :, 1] = color[1]
        out[:, :, 2] = color[2]
        out[:, :, 3] = (a[:, :, 0] * alpha * 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA")
    return g


def text(draw, s, xy, font, fill=TEXT, anchor="lt"):
    draw.text(xy, s, font=font, fill=fill, anchor=anchor)


def brand_footer(im: Image.Image, p: float):
    d = ImageDraw.Draw(im)
    mono = fnt("JetBrainsMono-Regular.ttf", 14)
    light = fnt("Outfit-Light.ttf", 14)
    d.line([(48, H - 56), (W - 48, H - 56)], fill=(40, 40, 48), width=1)
    d.line([(48, H - 56), (48 + int((W - 96) * p), H - 56)], fill=LIME, width=2)
    text(d, "VidMCP", (48, H - 36), mono, fill=MUTED)
    text(d, "Dhiraj Lochib", (W - 48, H - 36), light, fill=MUTED, anchor="rt")


def encode(path: Path, frames: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".raw.mp4")
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fr in frames:
        if isinstance(fr, Image.Image):
            bgr = cv2.cvtColor(np.array(fr.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            bgr = fr
        wr.write(bgr)
    wr.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "20", "-preset", "medium", "-movflags", "+faststart", str(path)],
        check=True, capture_output=True,
    )
    raw.unlink(missing_ok=True)
    return path


def to_gif(mp4: Path, gif: Path, scale=480, fps=12, dur=5.0):
    pal = gif.with_suffix(".pal.png")
    subprocess.run(
        ["ffmpeg", "-y", "-t", str(dur), "-i", str(mp4),
         "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos,palettegen=max_colors=96:stats_mode=diff",
         str(pal)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-t", str(dur), "-i", str(mp4), "-i", str(pal),
         "-lavfi", f"fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=sierra2_4a",
         "-loop", "0", str(gif)],
        check=True, capture_output=True,
    )
    pal.unlink(missing_ok=True)


def still(mp4: Path, jpg: Path, ss=2.5):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(ss), "-i", str(mp4), "-frames:v", "1", "-q:v", "2", str(jpg)],
        check=True, capture_output=True,
    )


# ── 1: Flow field forming brand ──────────────────────────────────
def sample_flowfield():
    n = int(DUR * FPS)
    rng = np.random.default_rng(7)
    N = 900
    px = rng.random(N) * W
    py = rng.random(N) * H
    frames = []
    for fi in range(n):
        t = fi / FPS
        p = fi / max(n - 1, 1)
        im = mesh_bg(t).convert("RGBA")
        # flow
        trail = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        td = ImageDraw.Draw(trail)
        for i in range(N):
            x, y = px[i], py[i]
            # multi-octave curl-ish field
            a = (
                math.sin(x * 0.006 + t * 0.7) * math.cos(y * 0.005)
                + 0.5 * math.sin((x + y) * 0.003 + t)
            )
            b = (
                math.cos(y * 0.006 - t * 0.5) * math.sin(x * 0.004)
                + 0.5 * math.cos((x - y) * 0.003 - t * 0.8)
            )
            # attract to center circle over time
            cx, cy = W / 2, H / 2 + 20
            dx, dy = cx - x, cy - y
            dist = math.hypot(dx, dy) + 1e-3
            pull = 0.15 * ease_out(min(1, t / 3))
            a += (dx / dist) * pull
            b += (dy / dist) * pull
            sp = 2.2 + 1.2 * math.sin(i)
            nx, ny = x + a * sp, y + b * sp
            col = LIME if i % 5 == 0 else (CYAN if i % 3 == 0 else VIOLET)
            al = int(40 + 80 * (0.5 + 0.5 * math.sin(t + i)))
            td.line([(x, y), (nx, ny)], fill=(*col, al), width=1)
            px[i], py[i] = nx % W, ny % H
        im = Image.alpha_composite(im, trail)
        # glow title
        title_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        td2 = ImageDraw.Draw(title_layer)
        big = fnt("CrimsonPro-Bold.ttf", 72)
        fade = ease_out(min(1, max(0, (t - 0.8) / 1.2)))
        td2.text((W // 2, H // 2 - 10), "VidMCP", font=big, fill=(*TEXT, int(255 * fade)), anchor="mm")
        g = title_layer.filter(ImageFilter.GaussianBlur(12))
        # recolor glow to lime
        ga = np.array(g)
        if fade > 0:
            mask = ga[:, :, 3] > 0
            ga[mask, 0], ga[mask, 1], ga[mask, 2] = LIME
            ga[mask, 3] = (ga[mask, 3].astype(np.float32) * 0.45).astype(np.uint8)
            im = Image.alpha_composite(im, Image.fromarray(ga, "RGBA"))
        im = Image.alpha_composite(im, title_layer)
        d = ImageDraw.Draw(im)
        sub = fnt("Outfit-Light.ttf", 20)
        d.text((W // 2, H // 2 + 48), "video tools agents can run", font=sub,
               fill=(*SUB, int(220 * fade)), anchor="mm")
        brand_footer(im, p)
        frames.append(im.convert("RGB"))
    return frames


# ── 2: Complex tesseract with trails ─────────────────────────────
def sample_tesseract():
    n = int(DUR * FPS)
    V0 = np.array([[1 if (i & (1 << b)) else -1 for b in range(4)] for i in range(16)], float)
    edges = [(i, i ^ (1 << b)) for i in range(16) for b in range(4) if (i ^ (1 << b)) > i]
    frames = []
    trail_buf = []
    for fi in range(n):
        t = fi / FPS
        p = fi / max(n - 1, 1)
        base = mesh_bg(t).convert("RGBA")
        pts = V0.copy()
        for ang, i, j in ((t * 0.7, 0, 3), (t * 0.55, 1, 3), (t * 0.4, 2, 3), (t * 0.25, 0, 1)):
            c, s = math.cos(ang), math.sin(ang)
            R = np.eye(4)
            R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
            pts = pts @ R.T
        dist = 3.5
        fct = dist / (dist + pts[:, 3] + 1e-6)
        x3 = pts[:, 0] * fct
        y3 = pts[:, 1] * fct
        z3 = pts[:, 2] * fct
        sc = 100 + 8 * math.sin(t)
        px = W / 2 + sc * (x3 - z3 * 0.5)
        py = H / 2 + 30 + sc * (-y3 + z3 * 0.32)
        trail_buf.append((px.copy(), py.copy()))
        if len(trail_buf) > 12:
            trail_buf.pop(0)
        # trails
        trail = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        td = ImageDraw.Draw(trail)
        for ti, (tpx, tpy) in enumerate(trail_buf[:-1]):
            al = int(15 + 25 * (ti / max(len(trail_buf), 1)))
            for a, b in edges:
                col = (*CYAN, al) if ((a ^ b) & 8) else (*VIOLET, al // 2)
                td.line([(tpx[a], tpy[a]), (tpx[b], tpy[b])], fill=col, width=1)
        trail = trail.filter(ImageFilter.GaussianBlur(2))
        base = Image.alpha_composite(base, trail)
        # main edges with glow
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        main = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        md = ImageDraw.Draw(main)
        for a, b in edges:
            fourth = ((a ^ b) & 8) != 0
            col = LIME if fourth else CYAN
            gd.line([(px[a], py[a]), (px[b], py[b])], fill=(*col, 180), width=6)
            md.line([(px[a], py[a]), (px[b], py[b])], fill=(*col, 255), width=2)
        for i in range(16):
            r = 4
            md.ellipse([px[i] - r, py[i] - r, px[i] + r, py[i] + r], fill=(*TEXT, 255))
        base = Image.alpha_composite(base, glow.filter(ImageFilter.GaussianBlur(10)))
        base = Image.alpha_composite(base, main)
        d = ImageDraw.Draw(base)
        title = fnt("CrimsonPro-Bold.ttf", 44)
        d.text((W // 2, 64), "Four dimensions", font=title, fill=TEXT, anchor="mm")
        sub = fnt("Outfit-Light.ttf", 18)
        d.text((W // 2, 108), "tesseract · hypercube in R⁴", font=sub, fill=SUB, anchor="mm")
        brand_footer(base, p)
        frames.append(base.convert("RGB"))
    return frames


# ── 3: Behind subject — layered depth ────────────────────────────
def sample_behind():
    n = int(DUR * FPS)
    rng = np.random.default_rng(3)
    parts = rng.random((280, 6))
    frames = []
    for fi in range(n):
        t = fi / FPS
        p = fi / max(n - 1, 1)
        im = mesh_bg(t).convert("RGBA")
        # parallax particles
        part = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        pd = ImageDraw.Draw(part)
        cx = W / 2 + 25 * math.sin(t * 0.5)
        cy = H / 2 + 35
        # subject mask ellipse region
        for i in range(len(parts)):
            depth = 0.3 + parts[i, 4] * 0.7
            parts[i, 0] = (parts[i, 0] + 0.0008 * depth) % 1
            parts[i, 1] = (parts[i, 1] + 0.0015 * depth + 0.0003 * math.sin(t + i)) % 1
            x = parts[i, 0] * W
            y = parts[i, 1] * H
            # skip if inside subject body (approx)
            dx, dy = (x - cx) / 100, (y - (cy + 10)) / 160
            if dx * dx + dy * dy < 1.0 and y > cy - 200:
                continue
            r = int(1 + parts[i, 2] * 4 * depth)
            cols = [LIME, CYAN, VIOLET, PEACH, ROSE]
            col = cols[i % 5]
            al = int(50 + 120 * depth)
            pd.ellipse([x - r, y - r, x + r, y + r], fill=(*col, al))
        # soft blur far particles
        far = part.filter(ImageFilter.GaussianBlur(1.2))
        im = Image.alpha_composite(im, far)
        # subject with rim light
        subj = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(subj)
        # body gradient-ish solid
        head = [cx - 50, cy - 200, cx + 50, cy - 95]
        body = [cx - 95, cy - 100, cx + 95, cy + 175]
        sd.ellipse(head, fill=(18, 18, 22, 255))
        sd.ellipse(body, fill=(16, 16, 20, 255))
        # rim glow
        rim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        rd = ImageDraw.Draw(rim)
        rd.ellipse(head, outline=(*LIME, 200), width=3)
        rd.ellipse(body, outline=(*CYAN, 160), width=3)
        rim = rim.filter(ImageFilter.GaussianBlur(6))
        im = Image.alpha_composite(im, rim)
        im = Image.alpha_composite(im, subj)
        # sharp rim
        d = ImageDraw.Draw(im)
        d.ellipse(head, outline=(*LIME, 120), width=1)
        d.ellipse(body, outline=(*CYAN, 90), width=1)
        title = fnt("CrimsonPro-Bold.ttf", 40)
        d.text((W // 2, 56), "Behind the subject", font=title, fill=TEXT, anchor="mm")
        sub = fnt("Outfit-Light.ttf", 18)
        d.text((W // 2, 100), "effects under the matte — the product moat", font=sub, fill=SUB, anchor="mm")
        brand_footer(im, p)
        frames.append(im.convert("RGB"))
    return frames


# ── 4: Kinetic editorial type ────────────────────────────────────
def sample_kinetic():
    n = int(DUR * FPS)
    words = [(0.0, "Segment"), (1.3, "Compose"), (2.6, "Narrate"), (3.9, "Render"), (5.2, "Ship")]
    frames = []
    for fi in range(n):
        t = fi / FPS
        p = fi / max(n - 1, 1)
        im = mesh_bg(t).convert("RGBA")
        active, local = words[0][1], 0.0
        for i, (ts, w) in enumerate(words):
            if t >= ts:
                active = w
                nxt = words[i + 1][0] if i + 1 < len(words) else DUR
                local = (t - ts) / max(0.01, nxt - ts)
        fade = ease_out(min(1, local * 2.5))
        slide = (1 - fade) * 40
        # big type with glow
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        big = fnt("CrimsonPro-Bold.ttf", 96 if len(active) < 10 else 72)
        ld.text((W // 2, H // 2 - 10 + slide), active, font=big, fill=(*TEXT, int(255 * fade)), anchor="mm")
        g = layer.filter(ImageFilter.GaussianBlur(16))
        ga = np.array(g)
        mask = ga[:, :, 3] > 0
        # alternate accent glow
        accents = [LIME, CYAN, VIOLET, PEACH, ROSE]
        ac = accents[int(t * 0.75) % 5]
        ga[mask, 0], ga[mask, 1], ga[mask, 2] = ac
        ga[mask, 3] = (ga[mask, 3].astype(np.float32) * 0.5).astype(np.uint8)
        im = Image.alpha_composite(im, Image.fromarray(ga, "RGBA"))
        im = Image.alpha_composite(im, layer)
        d = ImageDraw.Draw(im)
        # thin rules
        d.line([(W * 0.2, H / 2 - 90), (W * 0.8, H / 2 - 90)], fill=(*MUTED, 120), width=1)
        d.line([(W * 0.2, H / 2 + 90), (W * 0.8, H / 2 + 90)], fill=(*MUTED, 120), width=1)
        mono = fnt("JetBrainsMono-Regular.ttf", 14)
        d.text((W // 2, H / 2 + 120), "AGENT EDIT LANGUAGE", font=mono, fill=(*LIME, 200), anchor="mm")
        brand_footer(im, p)
        frames.append(im.convert("RGB"))
    return frames


# ── 5: Unit circle with orbit trail ──────────────────────────────
def sample_circle():
    n = int(DUR * FPS)
    frames = []
    hist = []
    for fi in range(n):
        t = fi / FPS
        p = fi / max(n - 1, 1)
        im = mesh_bg(t).convert("RGBA")
        cx, cy, R = W / 2, H / 2 + 25, 175
        ang = t * 1.25
        x = cx + R * math.cos(ang)
        y = cy - R * math.sin(ang)
        hist.append((x, y))
        if len(hist) > 90:
            hist.pop(0)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        main = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        md = ImageDraw.Draw(main)
        # circle
        gd.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(*CYAN, 100), width=8)
        md.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(*CYAN, 220), width=2)
        md.line([(cx - R - 50, cy), (cx + R + 50, cy)], fill=(*MUTED, 100), width=1)
        md.line([(cx, cy - R - 50), (cx, cy + R + 50)], fill=(*MUTED, 100), width=1)
        # trail
        for i in range(1, len(hist)):
            al = int(20 + 180 * (i / len(hist)))
            md.line([hist[i - 1], hist[i]], fill=(*LIME, al), width=2)
        md.line([(cx, cy), (x, y)], fill=(*TEXT, 200), width=2)
        md.line([(x, y), (x, cy)], fill=(*VIOLET, 180), width=1)
        md.line([(x, y), (cx, y)], fill=(*PEACH, 180), width=1)
        md.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(*LIME, 255))
        im = Image.alpha_composite(im, glow.filter(ImageFilter.GaussianBlur(12)))
        im = Image.alpha_composite(im, main)
        d = ImageDraw.Draw(im)
        d.text((W // 2, 56), "Unit circle", font=fnt("CrimsonPro-Bold.ttf", 42), fill=TEXT, anchor="mm")
        mono = fnt("JetBrainsMono-Regular.ttf", 16)
        d.text((W // 2, H - 90), f"sin {math.sin(ang):+.3f}   cos {math.cos(ang):+.3f}",
               font=mono, fill=SUB, anchor="mm")
        brand_footer(im, p)
        frames.append(im.convert("RGB"))
    return frames


# ── 6: Graph / pipeline nodes ────────────────────────────────────
def sample_pipeline():
    n = int(DUR * FPS)
    nodes = ["Import", "Segment", "Layer", "Composite", "Render"]
    frames = []
    for fi in range(n):
        t = fi / FPS
        p = fi / max(n - 1, 1)
        im = mesh_bg(t).convert("RGBA")
        active = min(len(nodes) - 1, int(p * len(nodes)))
        # draw network
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        xs = [140 + i * 250 for i in range(5)]
        y = H // 2 + 10
        for i in range(4):
            col = LIME if i < active else MUTED
            # animated dash progress
            gd.line([(xs[i] + 55, y), (xs[i + 1] - 55, y)], fill=(*col, 80), width=6)
            ld.line([(xs[i] + 55, y), (xs[i + 1] - 55, y)], fill=(*col, 200), width=2)
        for i, name in enumerate(nodes):
            on = i <= active
            col = LIME if on else MUTED
            r = 36 + (4 if i == active else 0)
            gd.ellipse([xs[i] - r - 6, y - r - 6, xs[i] + r + 6, y + r + 6], fill=(*col, 50))
            ld.ellipse([xs[i] - r, y - r, xs[i] + r, y + r], outline=(*col, 255), width=2)
            if on:
                ld.ellipse([xs[i] - 10, y - 10, xs[i] + 10, y + 10], fill=(*col, 255))
            mono = fnt("Outfit-Medium.ttf", 16)
            fc = TEXT if on else SUB
            ld.text((xs[i], y + r + 28), name, font=mono, fill=(*fc, 255), anchor="mm")
        # pulse on active
        pulse = 0.5 + 0.5 * math.sin(t * 6)
        ar = 40 + 12 * pulse
        if active < len(xs):
            gd.ellipse([xs[active] - ar, y - ar, xs[active] + ar, y + ar], outline=(*CYAN, int(100 * pulse)), width=3)
        im = Image.alpha_composite(im, glow.filter(ImageFilter.GaussianBlur(14)))
        im = Image.alpha_composite(im, layer)
        d = ImageDraw.Draw(im)
        d.text((W // 2, 60), "Agent pipeline", font=fnt("CrimsonPro-Bold.ttf", 42), fill=TEXT, anchor="mm")
        d.text((W // 2, 108), "connect → segment → layer → composite → render",
               font=fnt("Outfit-Light.ttf", 18), fill=SUB, anchor="mm")
        brand_footer(im, p)
        frames.append(im.convert("RGB"))
    return frames


SAMPLES = [
    ("01_flowfield", sample_flowfield),
    ("02_tesseract", sample_tesseract),
    ("03_behind_subject", sample_behind),
    ("04_kinetic", sample_kinetic),
    ("05_unit_circle", sample_circle),
    ("06_pipeline", sample_pipeline),
]


def main():
    for o in OUTS:
        o.mkdir(parents=True, exist_ok=True)
        for old in o.glob("*"):
            if old.suffix in {".mp4", ".gif", ".jpg"}:
                old.unlink(missing_ok=True)
    for slug, fn in SAMPLES:
        print(f"→ {slug}")
        frames = fn()
        mp4 = OUTS[0] / f"{slug}.mp4"
        encode(mp4, frames)
        to_gif(mp4, OUTS[0] / f"{slug}.gif")
        still(mp4, OUTS[0] / f"{slug}.jpg")
        for dest in OUTS[1:]:
            for ext in (".mp4", ".gif", ".jpg"):
                src = OUTS[0] / f"{slug}{ext}"
                (dest / src.name).write_bytes(src.read_bytes())
        print(f"  {mp4.stat().st_size/1e6:.2f}MB mp4  gif {(OUTS[0]/f'{slug}.gif').stat().st_size/1e6:.2f}MB")
    print("done")


if __name__ == "__main__":
    main()
