"""Thumbnail A/B generation — scored frame candidates + brand title variants + contact sheet."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vidmcp.graphics.brand import color, get_brand_kit
from vidmcp.graphics.templates import _font
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import sample_frames

log = get_logger("vidmcp.thumbs")


def _score_frame(img: np.ndarray, face_area: float, n_faces: int) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = min(float(cv2.Laplacian(gray, cv2.CV_32F).var()) / 400.0, 1.0)
    bright = float(gray.mean())
    exposure = 1.0 - abs(bright - 125) / 125.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colorfulness = min(float(hsv[:, :, 1].mean()) / 120.0, 1.0)
    face = min(face_area * 12.0, 1.0) + (0.15 if n_faces == 1 else 0.0)
    return 0.35 * face + 0.25 * sharp + 0.2 * exposure + 0.2 * colorfulness


def _candidates(project: Any, n: int) -> list[tuple[float, np.ndarray, float]]:
    """(t, frame, score) — top distinct frames from the render or source."""
    m = project.manifest
    video = None
    for r in reversed(m.renders or []):
        p = project.abs(r.get("output_path"))
        if p and p.exists():
            video = p
            break
    if video is None:
        video = project.abs(m.source_video)
    face_by_t: dict[int, tuple[float, int]] = {}
    try:
        from vidmcp.perception.indexer import load_index

        for v in (load_index(project) or {}).get("visual") or []:
            face_by_t[int(v["t"])] = (float(v.get("face_area", 0)), int(v.get("n_faces", 0)))
    except Exception:  # noqa: BLE001
        pass
    scored = []
    for _, ts, img in sample_frames(video, max_frames=40, max_side=720):
        fa, nf = face_by_t.get(int(ts), (0.0, 0))
        if not face_by_t:
            from vidmcp.perception.faces import detect_faces

            faces = detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            nf = len(faces)
            fa = sum(w * h for (_, _, w, h) in faces) / (img.shape[0] * img.shape[1]) if nf else 0.0
        scored.append((ts, img, _score_frame(img, fa, nf)))
    scored.sort(key=lambda x: -x[2])
    picked: list[tuple[float, np.ndarray, float]] = []
    for ts, img, sc in scored:
        if any(abs(ts - p[0]) < 4.0 for p in picked):
            continue
        picked.append((ts, img, sc))
        if len(picked) >= n:
            break
    return picked


def _compose(img: np.ndarray, title: str, kit: dict[str, Any], variant: int) -> np.ndarray:
    h, w = img.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    d = ImageDraw.Draw(pil)
    f = _font(kit, max(28, h // 8))
    if not title:
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    words = title.split()
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if d.textlength(trial, font=f) > w * 0.55 and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    if variant % 2 == 0:  # bottom bar
        y0 = h - len(lines) * int(f.size * 1.25) - 40
        d.rectangle((0, y0 - 16, w, h), fill=(*color(kit, "dark"), 200))
        for i, line in enumerate(lines):
            d.text((int(w * 0.05), y0 + i * int(f.size * 1.25)), line, font=f,
                   fill=(*color(kit, "primary" if i == 0 else "paper"), 255),
                   stroke_width=2, stroke_fill=(*color(kit, "dark"), 255))
    else:  # top-left stacked with outline
        for i, line in enumerate(lines):
            d.text((int(w * 0.05), int(h * 0.08) + i * int(f.size * 1.2)), line, font=f,
                   fill=(*color(kit, "paper"), 255),
                   stroke_width=4, stroke_fill=(*color(kit, "dark"), 255))
        d.rectangle((int(w * 0.05), int(h * 0.08) + len(lines) * int(f.size * 1.2) + 8,
                     int(w * 0.35), int(h * 0.08) + len(lines) * int(f.size * 1.2) + 16),
                    fill=(*color(kit, "primary"), 255))
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def generate_thumbnails_project(
    project: Any,
    n: int = 3,
    title_variants: list[str] | None = None,
    brand: str = "default",
) -> dict[str, Any]:
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    kit = get_brand_kit(brand)
    if not title_variants:
        # derive a title from the transcript hook
        title = None
        try:
            from vidmcp.perception.indexer import load_index

            sents = (load_index(project) or {}).get("sentences") or []
            if sents:
                title = sents[0]["text"][:60]
        except Exception:  # noqa: BLE001
            pass
        title_variants = [title or m.name]
    cands = _candidates(project, n)
    if not cands:
        return {"ok": False, "message": "No candidate frames"}
    out_dir = project.previews_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i, (ts, img, score) in enumerate(cands):
        title = title_variants[i % len(title_variants)]
        composed = _compose(img, title, kit, i)
        p = out_dir / f"thumb_{chr(65 + i)}.jpg"
        cv2.imwrite(str(p), composed, [cv2.IMWRITE_JPEG_QUALITY, 92])
        variants.append({"variant": chr(65 + i), "path": project.rel(p), "t": round(ts, 1),
                         "frame_score": round(score, 3), "title": title})
    # contact sheet
    thumbs = [cv2.resize(cv2.imread(str(project.abs(v["path"]))), (426, 240)) for v in variants]
    sheet = np.concatenate(thumbs, axis=1)
    sheet_path = out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    m.append_history("generate_thumbnails", {"n": len(variants)})
    project.save()
    return {"ok": True, "n": len(variants), "variants": variants, "contact_sheet": project.rel(sheet_path)}
