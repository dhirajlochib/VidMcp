"""Publish metadata pack — chapters, titles, description, tags, SRT/VTT + clip extraction."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.metadata")

_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "i", "you", "we",
    "it", "this", "that", "to", "of", "in", "on", "for", "with", "so", "like", "just",
    "going", "get", "have", "do", "be", "my", "your", "me", "what", "when", "how", "really",
}


def _ts_ch(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _ts_srt(sec: float) -> str:
    ms = int(round((sec % 1) * 1000))
    s = int(sec)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d},{ms:03d}"


def _keywords(text: str, n: int = 12) -> list[str]:
    words = [w.lower().strip(".,!?") for w in re.findall(r"[A-Za-z0-9']+", text)]
    freq: dict[str, int] = {}
    for w in words:
        if len(w) > 3 and w not in _STOP:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:n]]


def _words_to_srt_cues(words: list[dict[str, Any]], max_chars: int = 42, max_dur: float = 4.5) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for w in words:
        cur.append(w)
        text = " ".join(str(x.get("word") or "").strip() for x in cur)
        dur = float(cur[-1]["end"]) - float(cur[0]["start"])
        if len(text) >= max_chars or dur >= max_dur or str(w.get("word", "")).rstrip().endswith((".", "!", "?")):
            cues.append({"start": float(cur[0]["start"]), "end": float(cur[-1]["end"]), "text": text})
            cur = []
    if cur:
        cues.append({"start": float(cur[0]["start"]), "end": float(cur[-1]["end"]),
                     "text": " ".join(str(x.get("word") or "").strip() for x in cur)})
    return cues


def generate_metadata_project(project: Any, platform: str = "youtube") -> dict[str, Any]:
    from vidmcp.perception.indexer import build_index_project, load_index

    m = project.manifest
    index = load_index(project)
    if index is None:
        build_index_project(project, include=["speech", "audio_events", "emotion"])
        index = load_index(project) or {}
    sentences = index.get("sentences") or []
    words = index.get("words") or []
    transcript = index.get("transcript") or ""
    shots = m.analysis.get("shots") or []

    # chapters: scene starts snapped to nearest sentence start
    chapters = [{"t": "0:00", "title": "Intro"}]
    scene_starts: list[float] = []
    seen_scenes = set()
    for s in shots:
        sid = s.get("scene_id", 0)
        if sid not in seen_scenes:
            seen_scenes.add(sid)
            if s["start"] > 8.0:
                scene_starts.append(s["start"])
    for t in scene_starts[:9]:
        sent = min(sentences, key=lambda x: abs(x["start"] - t), default=None) if sentences else None
        title = " ".join(_keywords(sent["text"], 3)).title() if sent else f"Part {len(chapters)}"
        chapters.append({"t": _ts_ch(t), "title": title or f"Part {len(chapters)}"})

    kws = _keywords(transcript)
    hook = sentences[0]["text"][:90] if sentences else m.name
    titles = [
        hook,
        " ".join(kws[:4]).title() if kws else m.name,
        f"How {kws[0].title()} Actually Works" if kws else m.name,
    ]
    description = hook + "\n\n"
    if len(chapters) > 1:
        description += "Chapters:\n" + "\n".join(f"{c['t']} {c['title']}" for c in chapters) + "\n\n"
    description += " ".join(f"#{k}" for k in kws[:6])

    # SRT / VTT sidecars
    srt_rel = vtt_rel = None
    if words:
        cues = _words_to_srt_cues(words)
        cap_dir = project.root / "captions"
        cap_dir.mkdir(parents=True, exist_ok=True)
        srt = "\n".join(
            f"{i + 1}\n{_ts_srt(c['start'])} --> {_ts_srt(c['end'])}\n{c['text']}\n" for i, c in enumerate(cues)
        )
        vtt = "WEBVTT\n\n" + "\n".join(
            f"{_ts_srt(c['start']).replace(',', '.')} --> {_ts_srt(c['end']).replace(',', '.')}\n{c['text']}\n"
            for c in cues
        )
        (cap_dir / "captions.srt").write_text(srt)
        (cap_dir / "captions.vtt").write_text(vtt)
        srt_rel, vtt_rel = project.rel(cap_dir / "captions.srt"), project.rel(cap_dir / "captions.vtt")

    m.append_history("generate_metadata", {"platform": platform, "n_chapters": len(chapters)})
    project.save()
    return {
        "ok": True,
        "platform": platform,
        "chapters": chapters,
        "titles": titles,
        "description": description,
        "tags": kws,
        "srt": srt_rel,
        "vtt": vtt_rel,
    }


def extract_clips_project(project: Any, n: int = 3, max_sec: float = 45.0, min_sec: float = 12.0) -> dict[str, Any]:
    """Rank self-contained high-energy windows (complete sentences) for short-form clips."""
    from vidmcp.perception.indexer import build_index_project, load_index

    index = load_index(project)
    if index is None or not index.get("sentences"):
        build_index_project(project)
        index = load_index(project) or {}
    sentences = index.get("sentences") or []
    energy = index.get("energy") or []
    if not sentences:
        return {"ok": False, "message": "No speech found — clips need a transcript"}

    def window_energy(a: float, b: float) -> float:
        if not energy:
            return 0.5
        lo, hi = int(a), min(int(b) + 1, len(energy))
        return float(np.mean(energy[lo:hi])) if hi > lo else 0.3

    candidates: list[dict[str, Any]] = []
    for i, s in enumerate(sentences):
        start = s["start"]
        end = start
        j = i
        while j < len(sentences) and sentences[j]["end"] - start <= max_sec:
            end = sentences[j]["end"]
            j += 1
        if end - start < min_sec:
            continue
        e = window_energy(start, end)
        hook = window_energy(start, min(start + 5, end))
        text = " ".join(x["text"] for x in sentences[i:j])[:120]
        candidates.append({
            "t_start": round(start, 2), "t_end": round(end, 2),
            "hook_score": round(hook, 3),
            "score": round(0.6 * e + 0.4 * hook, 3),
            "reason": f"complete thought, energy {e:.2f}: \"{text}…\"",
        })
    candidates.sort(key=lambda c: -c["score"])
    picked: list[dict[str, Any]] = []
    for c in candidates:
        if any(not (c["t_end"] < p["t_start"] or c["t_start"] > p["t_end"]) for p in picked):
            continue
        picked.append(c)
        if len(picked) >= n:
            break
    picked.sort(key=lambda c: c["t_start"])
    project.manifest.analysis["clips"] = picked
    project.save()
    return {"ok": True, "n": len(picked), "clips": picked}
