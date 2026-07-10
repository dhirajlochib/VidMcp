"""Semantic search across footage — transcript + audio events + energy, fused ranking.

Optional sentence-transformers upgrade; keyword/tag scoring always works.
"""

from __future__ import annotations

import re
from typing import Any

from vidmcp.perception.indexer import load_index
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.search")

# query concept → audio event tags / emotion labels
CONCEPT_TAGS: dict[str, list[str]] = {
    "laugh": ["burst"], "laughter": ["burst"], "laughing": ["burst"],
    "applause": ["burst"], "clap": ["burst"],
    "quiet": ["silence"], "silence": ["silence"], "pause": ["silence"],
    "music": ["music"], "song": ["music"],
    "excited": ["excited"], "energetic": ["excited"], "peak": ["excited"],
    "calm": ["calm"],
}

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _embed_scores(query: str, sentences: list[dict[str, Any]]) -> list[float] | None:
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        q = model.encode([query], normalize_embeddings=True)
        s = model.encode([x["text"] for x in sentences], normalize_embeddings=True)
        return [float(v) for v in util.cos_sim(q, s)[0]]
    except Exception:  # noqa: BLE001
        return None


def search_index(index: dict[str, Any], query: str, top_k: int = 5) -> list[dict[str, Any]]:
    q_tokens = _tokens(query)
    sentences = index.get("sentences") or []
    events = index.get("audio_events") or []
    emotions = index.get("emotion") or []
    hits: list[dict[str, Any]] = []

    # transcript scoring
    emb = _embed_scores(query, sentences) if sentences else None
    for i, s in enumerate(sentences):
        overlap = len(q_tokens & _tokens(s["text"]))
        kw_score = overlap / max(1, len(q_tokens))
        score = emb[i] if emb is not None else kw_score
        if emb is not None and kw_score > 0:
            score = 0.6 * score + 0.4 * kw_score
        if score > 0.12:
            hits.append(
                {
                    "t_start": s["start"],
                    "t_end": s["end"],
                    "score": round(float(score), 3),
                    "why": f"transcript: \"{s['text'][:80]}\"",
                }
            )

    # audio-event / emotion concept scoring
    wanted_tags: set[str] = set()
    for tok in q_tokens:
        wanted_tags.update(CONCEPT_TAGS.get(tok, []))
    if wanted_tags:
        run_start = None
        run_tag = None
        for e in events + [{"t": 10**9, "tag": None}]:
            tag = e["tag"]
            emo = emotions[e["t"]] if isinstance(e.get("t"), int) and e["t"] < len(emotions) else None
            match = tag in wanted_tags or emo in wanted_tags
            if match and run_start is None:
                run_start, run_tag = e["t"], tag
            elif not match and run_start is not None:
                hits.append(
                    {
                        "t_start": float(run_start),
                        "t_end": float(e["t"]) if e["t"] < 10**9 else float(run_start + 1),
                        "score": 0.8,
                        "why": f"audio event: {run_tag}",
                    }
                )
                run_start = None

    hits.sort(key=lambda h: -h["score"])
    # dedupe overlapping spans
    out: list[dict[str, Any]] = []
    for h in hits:
        if any(not (h["t_end"] < o["t_start"] or h["t_start"] > o["t_end"]) for o in out):
            continue
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def search_footage_project(project: Any, query: str, top_k: int = 5, modality: str = "auto") -> dict[str, Any]:
    index = load_index(project)
    if index is None:
        return {"ok": False, "message": "No footage index — run build_footage_index first"}
    results = search_index(index, query, top_k=top_k)
    return {"ok": True, "query": query, "n_results": len(results), "results": results}


def search_library(workspace: Any, query: str, top_k: int = 8) -> dict[str, Any]:
    """Search every indexed project in the workspace (B-roll retrieval backbone)."""
    all_hits: list[dict[str, Any]] = []
    for info in workspace.list_projects():
        try:
            project = workspace.load_project(info["id"])
            index = load_index(project)
            if index is None:
                continue
            for h in search_index(index, query, top_k=3):
                h["project_id"] = info["id"]
                h["project_name"] = info.get("name")
                h["source_video"] = info.get("source_video")
                all_hits.append(h)
        except Exception as e:  # noqa: BLE001
            log.debug("library_search_skip", project=info.get("id"), error=str(e))
    all_hits.sort(key=lambda h: -h["score"])
    return {"ok": True, "query": query, "n_results": len(all_hits[:top_k]), "results": all_hits[:top_k]}
