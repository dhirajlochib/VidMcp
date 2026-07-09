#!/usr/bin/env python3
"""Render a 4D dimensions math explainer with TTS audio (and optional talk-head PiP)."""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "demos" / "longform"
WORK = ROOT / "demos" / "_work_4d"
SRC = Path("/__none__/test-rec.mov")  # disabled: demos never use personal footage
MASK_DIR = Path("/__none__/masks")
W, H, FPS = 1280, 720, 30

# Beat script: (label, short_title, equation, narration_sentence)
BEATS: list[dict] = [
    {
        "dim": 0,
        "title": "0D — A Point",
        "eq": "no length · no area · just position",
        "blurb": "Zero dimensions: a single point. No length, no width — only location.",
        "narr": "Zero dimensions. Just a point. No length, no width, no height — only a location in space.",
    },
    {
        "dim": 1,
        "title": "1D — A Line",
        "eq": "length  ·  R¹",
        "blurb": "One dimension: extend the point into a line. You can move left or right.",
        "narr": "One dimension. Extend that point into a line. Now you have length — you can move left or right.",
    },
    {
        "dim": 2,
        "title": "2D — A Square",
        "eq": "length × width  ·  R²",
        "blurb": "Two dimensions: extrude the line into a square. Flat plane — left-right and up-down.",
        "narr": "Two dimensions. Extrude the line into a square. Now you have length and width — a flat plane.",
    },
    {
        "dim": 3,
        "title": "3D — A Cube",
        "eq": "l × w × h  ·  R³",
        "blurb": "Three dimensions: extrude the square into a cube. Depth arrives — the world we live in.",
        "narr": "Three dimensions. Extrude the square into a cube. Depth appears. This is the world we live in: length, width, and height.",
    },
    {
        "dim": 4,
        "title": "4D — A Tesseract",
        "eq": "l × w × h × w₄  ·  R⁴",
        "blurb": "Four dimensions: extrude the cube into a tesseract — a hypercube. A fourth axis, orthogonal to all three.",
        "narr": "Four dimensions. Extrude the cube into a tesseract — a hypercube. There is a fourth axis, orthogonal to length, width, and height. We cannot see it directly, but we can project it into 3D, and then onto this screen.",
    },
    {
        "dim": 4,
        "title": "Why it matters",
        "eq": "relativity · spacetime · data science",
        "blurb": "4D is not sci-fi only: spacetime uses 3 space + 1 time. High-dim data is everyday machine learning.",
        "narr": "Why it matters: in physics, spacetime is three dimensions of space plus one of time. In data science, features live in high-dimensional space every day. Four is just the next step.",
    },
]


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")


def pick_voice() -> str:
    out = subprocess.check_output(["say", "-v", "?"], text=True)
    for preferred in ("Samantha", "Daniel", "Karen", "Moira", "Rishi", "Reed"):
        if preferred in out:
            # Reed may need full name
            if preferred == "Reed":
                return "Reed"
            return preferred
    return "Samantha"


def synthesize_narration(text: str, wav: Path, voice: str) -> Path:
    wav.parent.mkdir(parents=True, exist_ok=True)
    aiff = wav.with_suffix(".aiff")
    run(["say", "-v", voice, "-r", "165", "-o", str(aiff), text])
    run(
        [
            "ffmpeg", "-y", "-i", str(aiff),
            "-ac", "1", "-ar", "48000",
            str(wav),
        ]
    )
    aiff.unlink(missing_ok=True)
    return wav


def wav_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def rotate4d(points: np.ndarray, angles: tuple[float, float, float]) -> np.ndarray:
    """Rotate 4D points in XW, YW, ZW planes."""
    ax, ay, az = angles
    # XW
    c, s = math.cos(ax), math.sin(ax)
    R = np.eye(4)
    R[0, 0], R[0, 3] = c, -s
    R[3, 0], R[3, 3] = s, c
    pts = points @ R.T
    # YW
    c, s = math.cos(ay), math.sin(ay)
    R = np.eye(4)
    R[1, 1], R[1, 3] = c, -s
    R[3, 1], R[3, 3] = s, c
    pts = pts @ R.T
    # ZW
    c, s = math.cos(az), math.sin(az)
    R = np.eye(4)
    R[2, 2], R[2, 3] = c, -s
    R[3, 2], R[3, 3] = s, c
    pts = pts @ R.T
    return pts


def project_to_2d(pts4: np.ndarray, *, scale: float, cx: float, cy: float, dist: float = 3.5) -> np.ndarray:
    # perspective from 4D → 3D (using w), then 3D → 2D
    w = pts4[:, 3]
    factor = dist / (dist + w + 1e-6)
    x3 = pts4[:, 0] * factor
    y3 = pts4[:, 1] * factor
    z3 = pts4[:, 2] * factor
    # simple isometric-ish 3D→2D
    px = cx + scale * (x3 - z3 * 0.55)
    py = cy + scale * (-y3 + z3 * 0.35)
    return np.stack([px, py], axis=1)


def tesseract_edges() -> list[tuple[int, int]]:
    edges = []
    for i in range(16):
        for b in range(4):
            j = i ^ (1 << b)
            if j > i:
                edges.append((i, j))
    return edges


def tesseract_verts() -> np.ndarray:
    verts = []
    for i in range(16):
        verts.append(
            [
                1.0 if (i & 1) else -1.0,
                1.0 if (i & 2) else -1.0,
                1.0 if (i & 4) else -1.0,
                1.0 if (i & 8) else -1.0,
            ]
        )
    return np.array(verts, dtype=np.float64)


def cube_edges() -> list[tuple[int, int]]:
    edges = []
    for i in range(8):
        for b in range(3):
            j = i ^ (1 << b)
            if j > i:
                edges.append((i, j))
    return edges


def cube_verts() -> np.ndarray:
    verts = []
    for i in range(8):
        verts.append(
            [
                1.0 if (i & 1) else -1.0,
                1.0 if (i & 2) else -1.0,
                1.0 if (i & 4) else -1.0,
            ]
        )
    return np.array(verts, dtype=np.float64)


def project3(pts3: np.ndarray, *, scale: float, cx: float, cy: float, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    x = pts3[:, 0] * c - pts3[:, 2] * s
    z = pts3[:, 0] * s + pts3[:, 2] * c
    y = pts3[:, 1]
    # perspective
    dist = 4.0
    f = dist / (dist + z + 2.0)
    px = cx + scale * x * f
    py = cy - scale * y * f
    return np.stack([px, py], axis=1)


def draw_bg(frame: np.ndarray, t: float) -> None:
    h, w = frame.shape[:2]
    # deep space gradient
    yy = np.linspace(0, 1, h)[:, None]
    base = np.zeros_like(frame)
    base[:, :, 0] = (18 + 10 * yy).astype(np.uint8)  # B
    base[:, :, 1] = (14 + 8 * yy).astype(np.uint8)
    base[:, :, 2] = (28 + 20 * yy).astype(np.uint8)
    frame[:] = base
    # subtle stars / particles
    rng = np.random.default_rng(42)
    for _ in range(80):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        bright = int(80 + 100 * (0.5 + 0.5 * math.sin(t * 2 + x * 0.01)))
        cv2.circle(frame, (x, y), 1, (bright, bright, bright + 20), -1, cv2.LINE_AA)
    # soft vignette-ish top bar
    cv2.rectangle(frame, (0, 0), (w, 90), (20, 16, 28), -1)


def put_center_text(frame, text, y, scale, color, thickness=2):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (w - tw) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_progress(frame, beat_i, n_beats, local_p):
    h, w = frame.shape[:2]
    y0 = h - 36
    margin = 60
    total_w = w - 2 * margin
    seg = total_w / n_beats
    for i in range(n_beats):
        x0 = int(margin + i * seg)
        x1 = int(margin + (i + 1) * seg - 6)
        cv2.rectangle(frame, (x0, y0), (x1, y0 + 10), (50, 45, 60), -1)
        if i < beat_i:
            cv2.rectangle(frame, (x0, y0), (x1, y0 + 10), (90, 200, 255), -1)
        elif i == beat_i:
            fill = int(x0 + (x1 - x0) * local_p)
            cv2.rectangle(frame, (x0, y0), (fill, y0 + 10), (80, 180, 255), -1)
    cv2.putText(
        frame,
        f"step {beat_i + 1}/{n_beats}",
        (margin, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 170, 190),
        1,
        cv2.LINE_AA,
    )


def draw_dim_diagram(frame, dim: int, t: float, phase: float) -> None:
    """phase in [0,1] within beat."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2 + 20
    grow = min(1.0, phase * 2.5)
    col_hi = (120, 220, 255)
    col_mid = (100, 180, 255)
    col_dim = (70, 100, 160)

    if dim == 0:
        r = int(6 + 10 * grow)
        pulse = 1.0 + 0.15 * math.sin(t * 6)
        rr = int(r * pulse)
        cv2.circle(frame, (cx, cy), rr + 18, (40, 60, 90), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), rr, col_hi, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(2, rr // 2), (255, 255, 255), -1, cv2.LINE_AA)
        put_center_text(frame, "•  point", cy + 70, 0.7, (200, 210, 230), 1)

    elif dim == 1:
        half = int(180 * grow)
        x0, x1 = cx - half, cx + half
        cv2.line(frame, (x0, cy), (x1, cy), col_mid, 4, cv2.LINE_AA)
        cv2.circle(frame, (x0, cy), 7, col_hi, -1, cv2.LINE_AA)
        cv2.circle(frame, (x1, cy), 7, col_hi, -1, cv2.LINE_AA)
        # arrow heads
        cv2.arrowedLine(frame, (cx - 40, cy - 40), (cx + 80, cy - 40), (140, 160, 200), 2, tipLength=0.25)
        cv2.putText(frame, "x", (cx + 90, cy - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 230), 1, cv2.LINE_AA)

    elif dim == 2:
        s = int(110 * grow)
        pts = np.array(
            [[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]],
            np.int32,
        )
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (40, 80, 120))
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.polylines(frame, [pts], True, col_hi, 3, cv2.LINE_AA)
        # axes labels
        cv2.arrowedLine(frame, (cx - s - 30, cy + s + 30), (cx + s + 20, cy + s + 30), col_dim, 2, tipLength=0.15)
        cv2.arrowedLine(frame, (cx - s - 30, cy + s + 30), (cx - s - 30, cy - s - 10), col_dim, 2, tipLength=0.15)
        cv2.putText(frame, "x", (cx + s + 25, cy + s + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 180, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, "y", (cx - s - 50, cy - s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 180, 210), 1, cv2.LINE_AA)

    elif dim == 3:
        scale = 90 * grow
        ang = t * 0.9
        verts = cube_verts()
        proj = project3(verts, scale=scale, cx=cx, cy=cy, ang=ang)
        for a, b in cube_edges():
            p1 = tuple(proj[a].astype(int))
            p2 = tuple(proj[b].astype(int))
            cv2.line(frame, p1, p2, col_hi, 2, cv2.LINE_AA)
        for p in proj:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (200, 240, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "x", (cx + int(scale) + 20, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 170, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "y", (cx - 20, cy - int(scale) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 170, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "z", (cx - int(scale) - 40, cy + int(scale) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 170, 200), 1, cv2.LINE_AA)

    else:  # 4D tesseract or recap
        scale = 70 * grow
        ang = (t * 0.7, t * 0.55, t * 0.4)
        verts = rotate4d(tesseract_verts(), ang)
        proj = project_to_2d(verts, scale=scale, cx=cx, cy=cy - 10)
        edges = tesseract_edges()
        # color edges that flip the 4th bit differently
        for a, b in edges:
            p1 = tuple(proj[a].astype(int))
            p2 = tuple(proj[b].astype(int))
            fourth = ((a ^ b) & 8) != 0
            color = (80, 140, 255) if fourth else (120, 230, 255)
            thick = 2 if fourth else 2
            cv2.line(frame, p1, p2, color, thick, cv2.LINE_AA)
        for p in proj:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (220, 240, 255), -1, cv2.LINE_AA)
        # legend
        cv2.line(frame, (cx - 200, cy + 160), (cx - 160, cy + 160), (120, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "3-space edges", (cx - 150, cy + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)
        cv2.line(frame, (cx + 20, cy + 160), (cx + 60, cy + 160), (80, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "4th-axis edges", (cx + 70, cy + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)


def draw_beat_frame(frame, beat: dict, beat_i: int, n_beats: int, t: float, local_p: float) -> None:
    draw_bg(frame, t)
    # top title
    put_center_text(frame, beat["title"], 52, 1.15, (240, 235, 250), 2)
    # equation bar
    put_center_text(frame, beat["eq"], 90, 0.65, (120, 200, 255), 1)
    # diagram
    dim = beat["dim"]
    # for last beat use tesseract still
    draw_dim_diagram(frame, dim if beat_i < 5 else 4, t, local_p)
    # bottom blurb
    blurb = beat["blurb"]
    # wrap roughly
    words = blurb.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if len(test) > 70:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    y = H - 100
    for line in lines[:3]:
        put_center_text(frame, line, y, 0.55, (190, 200, 220), 1)
        y += 26
    draw_progress(frame, beat_i, n_beats, local_p)
    # brand
    cv2.putText(frame, "VidMCP  ·  4D explainer", (24, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 120), 1, cv2.LINE_AA)


def load_talk_head_frame(fi: int, src_fps: float, n_src: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (bgr, mask) for talk-head PiP, looping source if needed."""
    if not SRC.exists() or not MASK_DIR.exists():
        return None
    idx = fi % max(n_src, 1)
    mask_path = MASK_DIR / f"mask_{idx:06d}.png"
    if not mask_path.exists():
        return None
    cap = getattr(load_talk_head_frame, "_cap", None)
    if cap is None:
        cap = cv2.VideoCapture(str(SRC))
        load_talk_head_frame._cap = cap  # type: ignore[attr-defined]
        load_talk_head_frame._last = -1  # type: ignore[attr-defined]
    last = load_talk_head_frame._last  # type: ignore[attr-defined]
    if idx < last:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    elif idx != last + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    load_talk_head_frame._last = idx  # type: ignore[attr-defined]
    if not ok:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return frame, mask


def composite_pip(frame: np.ndarray, head: np.ndarray, mask: np.ndarray) -> None:
    """Bottom-right circular-ish talk-head PiP."""
    h, w = frame.shape[:2]
    # crop to person bbox from mask
    ys, xs = np.where(mask > 40)
    if len(xs) < 50:
        return
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = 20
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(head.shape[1], x1 + pad), min(head.shape[0], y1 + pad)
    crop = head[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    pip_h = int(H * 0.32)
    scale = pip_h / max(crop.shape[0], 1)
    pip_w = max(1, int(crop.shape[1] * scale))
    crop_r = cv2.resize(crop, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
    m_r = cv2.resize(m, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)
    # place bottom-right
    mx, my = 24, 24
    x, y = w - pip_w - mx, h - pip_h - my - 40
    if x < 0 or y < 0:
        return
    alpha = (m_r.astype(np.float32) / 255.0)[..., None]
    # soft edge
    alpha = np.clip(alpha * 1.15, 0, 1)
    roi = frame[y : y + pip_h, x : x + pip_w].astype(np.float32)
    out = crop_r.astype(np.float32) * alpha + roi * (1 - alpha)
    frame[y : y + pip_h, x : x + pip_w] = out.astype(np.uint8)
    # ring
    cv2.rectangle(frame, (x - 2, y - 2), (x + pip_w + 2, y + pip_h + 2), (80, 180, 255), 2, cv2.LINE_AA)


def intro_frame(frame, t: float, dur: float) -> None:
    draw_bg(frame, t)
    p = min(1.0, t / max(dur * 0.6, 0.1))
    put_center_text(frame, "What is the 4th Dimension?", int(H * 0.42), 1.3 * p + 0.01, (245, 240, 255), 3)
    put_center_text(frame, "from a point to a tesseract", int(H * 0.52), 0.75, (120, 200, 255), 1)
    put_center_text(frame, "a visual math lesson", int(H * 0.62), 0.55, (160, 170, 190), 1)


def outro_frame(frame, t: float) -> None:
    draw_bg(frame, t)
    put_center_text(frame, "0D → 1D → 2D → 3D → 4D", int(H * 0.38), 1.0, (240, 235, 255), 2)
    put_center_text(frame, "Each dimension = extrude the previous one", int(H * 0.50), 0.7, (120, 200, 255), 1)
    put_center_text(frame, "Tesseract = hypercube in R⁴", int(H * 0.60), 0.65, (180, 200, 220), 1)
    put_center_text(frame, "VidMCP education", int(H * 0.78), 0.5, (120, 120, 140), 1)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    voice = pick_voice()
    print(f"voice={voice}")

    intro_narr = "What is the fourth dimension? Let's build it step by step — from a point, all the way to a tesseract."
    outro_narr = "So remember: each new dimension is just the previous shape, extruded along a new axis. Point, line, square, cube, tesseract."

    full_text_parts = [intro_narr] + [b["narr"] for b in BEATS] + [outro_narr]
    full_text = " ... ".join(full_text_parts)

    # Per-segment audio for accurate timing
    segs: list[tuple[str, Path, float]] = []
    labels = ["intro"] + [f"beat_{i}" for i in range(len(BEATS))] + ["outro"]
    texts = [intro_narr] + [b["narr"] for b in BEATS] + [outro_narr]
    for lab, tx in zip(labels, texts):
        wav = WORK / f"{lab}.wav"
        synthesize_narration(tx, wav, voice)
        d = wav_duration(wav)
        segs.append((lab, wav, d))
        print(f"  {lab}: {d:.2f}s")

    # Concatenate audio with short gaps
    gap = 0.35
    list_file = WORK / "audio_list.txt"
    with list_file.open("w") as f:
        for i, (lab, wav, d) in enumerate(segs):
            f.write(f"file '{wav.resolve()}'\n")
            if i < len(segs) - 1:
                silence = WORK / f"gap_{i}.wav"
                run(
                    [
                        "ffmpeg", "-y",
                        "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=mono",
                        "-t", str(gap),
                        str(silence),
                    ]
                )
                f.write(f"file '{silence.resolve()}'\n")
    full_wav = WORK / "full_narration.wav"
    run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(full_wav),
        ]
    )
    total_audio = wav_duration(full_wav)
    print(f"total audio: {total_audio:.2f}s")

    # Build timeline with visual durations matching audio + gaps
    timeline: list[tuple[str, float, dict | None]] = []
    for i, (lab, wav, d) in enumerate(segs):
        if lab == "intro":
            timeline.append(("intro", d, None))
        elif lab == "outro":
            timeline.append(("outro", d, None))
        else:
            bi = int(lab.split("_")[1])
            timeline.append(("beat", d, BEATS[bi]))
        if i < len(segs) - 1:
            timeline.append(("gap", gap, None))

    total_vis = sum(d for _, d, _ in timeline)
    n_frames = int(round(total_vis * FPS))
    print(f"rendering {n_frames} frames @ {FPS} fps (~{total_vis:.1f}s)")

    # source probe for PiP
    n_src = 0
    if SRC.exists():
        cap0 = cv2.VideoCapture(str(SRC))
        n_src = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap0.release()

    raw = WORK / "video_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")

    # precompute frame ownership
    beat_count = len(BEATS)
    t_cursor = 0.0
    for fi in range(n_frames):
        t = fi / FPS
        # find segment
        acc = 0.0
        seg_kind, seg_dur, seg_beat, beat_i = "gap", 0.1, None, 0
        bi_count = 0
        for kind, dur, beat in timeline:
            if acc + dur > t + 1e-9 or abs(acc + dur - total_vis) < 1e-6 and t >= acc:
                seg_kind, seg_dur, seg_beat = kind, dur, beat
                if kind == "beat":
                    # find which beat index
                    for j, b in enumerate(BEATS):
                        if beat is b:
                            bi_count = j
                            break
                break
            if kind == "beat":
                bi_count = BEATS.index(beat) if beat in BEATS else bi_count
            acc += dur
        local_t = max(0.0, t - acc)
        local_p = min(1.0, local_t / max(seg_dur, 1e-3))

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        if seg_kind == "intro":
            intro_frame(frame, local_t, seg_dur)
        elif seg_kind == "outro":
            outro_frame(frame, t)
        elif seg_kind == "beat" and seg_beat:
            draw_beat_frame(frame, seg_beat, bi_count, beat_count, t, local_p)
        else:
            # short gap: hold last style dark
            draw_bg(frame, t)

        # talk-head PiP on beats (not intro title)
        if seg_kind in ("beat", "outro") and n_src > 0:
            # map time into source
            src_fi = int((t * 43) % max(n_src, 1))  # ~43 fps source
            th = load_talk_head_frame(src_fi, 43.0, n_src)
            if th is not None:
                composite_pip(frame, th[0], th[1])

        writer.write(frame)
        if fi % 60 == 0:
            print(f"  frame {fi}/{n_frames}")

    writer.release()
    cap = getattr(load_talk_head_frame, "_cap", None)
    if cap is not None:
        cap.release()

    # encode h264 + mux audio
    silent_h264 = WORK / "video_h264.mp4"
    run(
        [
            "ffmpeg", "-y", "-i", str(raw),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
            str(silent_h264),
        ]
    )
    out = OUT_DIR / "four_dimensions.mp4"
    # pad/trim video to audio with -shortest carefully: extend video if needed
    run(
        [
            "ffmpeg", "-y",
            "-i", str(silent_h264),
            "-i", str(full_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out),
        ]
    )

    # still + short social cut
    still = OUT_DIR / "four_dimensions_still.jpg"
    run(["ffmpeg", "-y", "-ss", "12", "-i", str(out), "-frames:v", "1", "-q:v", "2", str(still)])

    # 5s teaser with audio
    teaser = OUT_DIR / "four_dimensions_5s.mp4"
    run(
        [
            "ffmpeg", "-y", "-i", str(out),
            "-t", "5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            str(teaser),
        ]
    )

    # desktop copies
    desk = Path.home() / "Desktop"
    for p in (out, still, teaser):
        if p.exists():
            shutil.copy2(p, desk / p.name)

    # probe final
    info = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames",
            "-of", "json",
            str(out),
        ],
        text=True,
    )
    print("OUTPUT", out)
    print(info)
    print("size_mb", out.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
