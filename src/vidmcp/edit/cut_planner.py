"""Cut planner v2 — semantic pauses, contextual fillers, retake detection, room-tone fill.

Produces a reviewable cut-plan artifact (keep ranges + reasons) before any render.
"""

from __future__ import annotations

import difflib
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import orjson

from vidmcp.edit.smart_cut import DEFAULT_FILLERS, KeepRange, apply_smart_cuts
from vidmcp.perception.indexer import build_index_project, load_index
from vidmcp.utils.logging import get_logger
from vidmcp.utils.video_io import probe_video

log = get_logger("vidmcp.cut_planner")

PLANS_REL = "plans"

# words that mark emphasis before a dramatic pause
_EMPHASIS = {"never", "always", "everything", "nothing", "why", "how", "imagine", "listen", "because", "but"}
_COMPARATOR_BEFORE = {"feels", "looks", "sounds", "is", "was", "seems", "something", "things", "stuff"}


def _energy_at(index: dict[str, Any], t: float) -> float:
    energy = index.get("energy") or []
    i = int(t)
    return float(energy[i]) if i < len(energy) else 0.3


def classify_pause(
    gap: float,
    prev_word: str,
    prev_ends_sentence: bool,
    energy_before: float,
    aggressiveness: float,
) -> str:
    """dead | breath | dramatic"""
    if gap < 0.35:
        return "breath"
    if gap >= 0.45 and (
        prev_word in _EMPHASIS
        or (prev_ends_sentence and energy_before > 0.55 and gap < 2.0)
    ):
        return "dramatic"
    threshold = 0.45 * (1.25 - 0.6 * aggressiveness)
    if gap >= threshold:
        return "dead"
    return "breath"


def _is_contextual_filler(word: str, prev_word: str, next_gap: float, prev_gap: float) -> bool:
    """'like' as comparator survives; isolated hesitation dies."""
    if word in ("um", "uh", "uhh", "erm", "hmm"):
        return True
    if word in ("like", "basically", "actually", "so"):
        if prev_word in _COMPARATOR_BEFORE:
            return False  # semantic use
        return prev_gap > 0.25 or next_gap > 0.35  # isolated → hesitation
    return word in DEFAULT_FILLERS


def detect_retakes(sentences: list[dict[str, Any]], similarity: float = 0.72) -> list[tuple[float, float]]:
    """Adjacent near-duplicate sentences → remove the EARLIER take."""
    removals: list[tuple[float, float]] = []
    for i in range(len(sentences) - 1):
        a, b = sentences[i], sentences[i + 1]
        if not a["text"].strip() or not b["text"].strip():
            continue
        ratio = difflib.SequenceMatcher(None, a["text"].lower(), b["text"].lower()).ratio()
        if ratio >= similarity and len(a["text"].split()) >= 4:
            removals.append((a["start"], min(a["end"] + 0.05, b["start"])))
    return removals


def plan_cuts_project(
    project: Any,
    target_ratio: float | None = None,
    aggressiveness: float = 0.5,
    keep_dramatic_pauses: bool = True,
) -> dict[str, Any]:
    m = project.manifest
    if not m.source_video:
        return {"ok": False, "message": "No source video"}
    index = load_index(project)
    if index is None or not index.get("words"):
        build_index_project(project, include=["speech", "audio_events", "emotion"])
        index = load_index(project) or {}
    words = index.get("words") or []
    sentences = index.get("sentences") or []
    meta = probe_video(project.abs(m.source_video))
    duration = meta.duration_sec
    aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))

    removals: list[dict[str, Any]] = []

    # 1) retakes
    for a, b in detect_retakes(sentences):
        removals.append({"start": a, "end": b, "reason": "retake (kept later take)"})

    # 2) fillers + pauses
    for i, w in enumerate(words):
        tok = str(w.get("word") or "").strip().lower().strip(".,!?")
        start, end = float(w["start"]), float(w["end"])
        prev_w = str(words[i - 1].get("word") or "").strip().lower().strip(".,!?") if i else ""
        prev_gap = start - float(words[i - 1]["end"]) if i else 0.0
        next_gap = float(words[i + 1]["start"]) - end if i + 1 < len(words) else 0.0
        if _is_contextual_filler(tok, prev_w, next_gap, prev_gap) and (end - start) >= (0.32 - 0.12 * aggressiveness):
            removals.append({"start": start, "end": end, "reason": f"filler '{tok}'"})
        if i + 1 < len(words):
            gap = float(words[i + 1]["start"]) - end
            if gap < 0.3:
                continue
            raw = str(w.get("word") or "")
            kind = classify_pause(
                gap, tok, raw.rstrip().endswith((".", "!", "?")), _energy_at(index, end), aggressiveness
            )
            if kind == "dead" or (kind == "dramatic" and not keep_dramatic_pauses):
                pad = 0.12 if kind == "dramatic" else 0.06
                a, b = end + pad, float(words[i + 1]["start"]) - pad
                if b > a:
                    removals.append({"start": a, "end": b, "reason": f"{kind} pause {gap:.2f}s"})
            elif kind == "dramatic":
                removals.append({"start": end, "end": end, "reason": f"KEPT dramatic pause {gap:.2f}s", "kept": True})

    cuts = [r for r in removals if not r.get("kept")]
    cuts.sort(key=lambda r: r["start"])
    merged: list[dict[str, Any]] = []
    for r in cuts:
        if merged and r["start"] <= merged[-1]["end"] + 0.03:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
            merged[-1]["reason"] += f"; {r['reason']}"
        else:
            merged.append(dict(r))

    keeps: list[dict[str, Any]] = []
    cur = 0.0
    for r in merged:
        if r["start"] > cur + 0.05:
            keeps.append({"start": round(cur, 3), "end": round(r["start"], 3)})
        cur = max(cur, r["end"])
    if cur < duration - 0.05:
        keeps.append({"start": round(cur, 3), "end": round(duration, 3)})
    if not keeps:
        keeps = [{"start": 0.0, "end": round(duration, 3)}]

    kept_sec = sum(k["end"] - k["start"] for k in keeps)
    plan_id = f"cuts_{uuid4().hex[:8]}"
    plan = {
        "plan_id": plan_id,
        "duration_in": round(duration, 3),
        "kept_sec": round(kept_sec, 3),
        "removed_sec": round(duration - kept_sec, 3),
        "keep_ranges": keeps,
        "removals": merged,
        "kept_pauses": [r for r in removals if r.get("kept")],
        "aggressiveness": aggressiveness,
    }
    plans_dir = project.root / PLANS_REL
    plans_dir.mkdir(parents=True, exist_ok=True)
    (plans_dir / f"{plan_id}.json").write_bytes(orjson.dumps(plan, option=orjson.OPT_INDENT_2))
    m.analysis.setdefault("cut_plans", []).append(
        {"plan_id": plan_id, "kept_sec": plan["kept_sec"], "removed_sec": plan["removed_sec"]}
    )
    m.append_history("plan_cuts", {"plan_id": plan_id, "removed_sec": plan["removed_sec"]})
    project.save()
    return {
        "ok": True,
        "plan_id": plan_id,
        "kept_sec": plan["kept_sec"],
        "removed_sec": plan["removed_sec"],
        "n_keep_ranges": len(keeps),
        "n_removals": len(merged),
        "n_kept_dramatic_pauses": len(plan["kept_pauses"]),
        "removals_preview": [
            {"start": r["start"], "end": r["end"], "reason": r["reason"][:70]} for r in merged[:10]
        ],
    }


def _room_tone_fill(project: Any, video_in: Path, video_out: Path) -> bool:
    """Loop the quietest source second under the cut video at low gain (kills dead-cut artifacts)."""
    index = load_index(project) or {}
    events = index.get("audio_events") or []
    quiet = min(events, key=lambda e: e["rms"], default=None) if events else None
    t = float(quiet["t"]) if quiet else 0.0
    tone = project.tmp_dir / "room_tone.wav"
    src = project.abs(project.manifest.source_video)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-t", "0.8", "-i", str(src),
             "-vn", "-ar", "48000", "-ac", "2", str(tone)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_in), "-stream_loop", "-1", "-i", str(tone),
             "-filter_complex",
             "[1:a]volume=0.35[tone];[0:a][tone]amix=inputs=2:duration=first:normalize=0[a]",
             "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             str(video_out)],
            check=True, capture_output=True,
        )
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("room_tone_failed", error=str(e))
        return False


def apply_cut_plan_project(project: Any, plan_id: str, room_tone: bool = True) -> dict[str, Any]:
    m = project.manifest
    plan_path = project.root / PLANS_REL / f"{plan_id}.json"
    if not plan_path.exists():
        return {"ok": False, "message": f"Unknown cut plan {plan_id}"}
    plan = orjson.loads(plan_path.read_bytes())
    ranges = [KeepRange(k["start"], k["end"]) for k in plan["keep_ranges"]]
    src = project.abs(m.source_video)
    out = project.renders_dir / f"cut_{plan_id}.mp4"
    result = apply_smart_cuts(src, out, ranges)
    final = out
    if room_tone and result.get("ok"):
        toned = project.renders_dir / f"cut_{plan_id}_tone.mp4"
        if _room_tone_fill(project, out, toned):
            final = toned
    rel = project.rel(final)
    m.renders.append({"render_id": plan_id, "output_path": rel, "kind": "cut"})
    m.append_history("apply_cut_plan", {"plan_id": plan_id, "output": rel})
    project.save()
    return {
        "ok": True,
        "plan_id": plan_id,
        "output_path": rel,
        "duration_out": result.get("duration_out"),
        "removed_sec": result.get("removed_sec"),
        "room_tone": final != out,
        "edl": plan["keep_ranges"][:12],
    }
