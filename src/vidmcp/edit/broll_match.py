"""B-roll suggestion + insertion — match transcript beats to library footage or plates."""

from __future__ import annotations

import re
from typing import Any

from vidmcp.models.layers import Layer, LayerKind
from vidmcp.perception.indexer import load_index
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.broll_match")

_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "i", "you", "we", "they",
    "it", "this", "that", "to", "of", "in", "on", "for", "with", "so", "like", "just", "really",
    "going", "gonna", "get", "got", "have", "has", "had", "do", "does", "did", "be", "been",
    "my", "your", "our", "me", "us", "them", "he", "she", "his", "her", "what", "when", "how",
}


def _beat_keywords(text: str, n: int = 4) -> list[str]:
    words = [w.strip(".,!?").lower() for w in re.findall(r"[A-Za-z0-9']+", text)]
    cands = [w for w in words if len(w) > 3 and w not in _STOP]
    # prefer longer + numeric tokens
    cands.sort(key=lambda w: (-len(w),))
    seen: list[str] = []
    for w in cands:
        if w not in seen:
            seen.append(w)
        if len(seen) >= n:
            break
    return seen


def suggest_broll_project(
    project: Any,
    beats: list[dict[str, Any]] | None = None,
    top_k: int = 3,
    min_gap_sec: float = 8.0,
) -> dict[str, Any]:
    index = load_index(project)
    if index is None:
        return {"ok": False, "message": "No footage index — run build_footage_index first"}
    sentences = index.get("sentences") or []
    if beats is None:
        # pick spaced-out sentences with strong keywords as insert points
        beats = []
        last_t = -min_gap_sec
        for s in sentences:
            kws = _beat_keywords(s["text"])
            if len(kws) >= 2 and s["start"] - last_t >= min_gap_sec:
                beats.append({"t": s["start"], "text": s["text"], "keywords": kws})
                last_t = s["start"]
        beats = beats[: top_k * 2]

    from vidmcp.core.workspace import Workspace

    ws = Workspace()
    suggestions: list[dict[str, Any]] = []
    for b in beats:
        query = " ".join(b.get("keywords") or _beat_keywords(b.get("text", "")))
        match = None
        try:
            from vidmcp.perception.search import search_library

            lib = search_library(ws, query, top_k=1)
            hits = [h for h in lib.get("results", []) if h.get("project_id") != project.manifest.id]
            match = hits[0] if hits else None
        except Exception as e:  # noqa: BLE001
            log.debug("library_match_failed", error=str(e))
        if match:
            suggestions.append(
                {
                    "t": round(float(b["t"]), 2),
                    "duration": 3.0,
                    "source": "clip",
                    "clip_project": match["project_id"],
                    "clip_t_start": match["t_start"],
                    "clip_video": match.get("source_video"),
                    "why": f"library match for '{query[:40]}': {match['why'][:60]}",
                }
            )
        else:
            suggestions.append(
                {
                    "t": round(float(b["t"]), 2),
                    "duration": 3.0,
                    "source": "generated",
                    "plate_prompt": query,
                    "why": f"no library match — generate plate for '{query[:40]}'",
                }
            )
        if len(suggestions) >= top_k:
            break
    return {"ok": True, "n_suggestions": len(suggestions), "suggestions": suggestions}


def insert_broll_project(
    project: Any,
    suggestions: list[dict[str, Any]],
    transition: str = "cut",
) -> dict[str, Any]:
    """Place accepted suggestions as time-windowed BROLL layers (J-cut: audio keeps rolling)."""
    m = project.manifest
    added: list[str] = []
    for s in suggestions:
        asset = s.get("clip_video") or s.get("asset_path")
        if s.get("source") == "clip" and s.get("clip_project") and not asset:
            try:
                from vidmcp.core.workspace import Workspace

                other = Workspace().load_project(s["clip_project"])
                asset = str(other.abs(other.manifest.source_video))
            except Exception:  # noqa: BLE001
                asset = None
        if not asset:
            continue
        t = float(s.get("t", 0))
        layer = Layer(
            name=f"broll_{int(t)}s",
            kind=LayerKind.BROLL,
            z_index=5,
            asset_path=str(asset),
            meta={
                "t_start": t,
                "t_end": t + float(s.get("duration", 3.0)),
                "clip_offset": float(s.get("clip_t_start", 0.0)),
                "transition": transition,
            },
        )
        m.layers.add(layer)
        added.append(layer.id)
    if added:
        m.append_history("insert_broll", {"n": len(added)})
        project.save()
    return {"ok": True, "layers_added": added, "n": len(added)}
