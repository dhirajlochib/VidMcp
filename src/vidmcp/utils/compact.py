"""Compact MCP tool results to protect agent context."""

from __future__ import annotations

import json
from typing import Any

from vidmcp.config import get_settings

# Keys rarely useful in agent loops — drop unless detail=True
_DROP_KEYS = frozenset({
    "raw", "manifest", "layers", "edit_history", "objects", "words",
    "masks", "thumbnail_paths", "side_data", "edit_graph", "telemetry_path",
    "analysis_hints", "prompt_candidates",
})


def compact_result(data: Any, *, force: bool | None = None, max_chars: int | None = None) -> Any:
    """Shrink nested tool results for MCP hosts.

    force=None → use settings.compact
    """
    settings = get_settings()
    if force is False:
        return data
    if force is None and not settings.compact:
        return data
    limit = max_chars if max_chars is not None else int(settings.max_result_chars or 4000)
    slim = _slim(data, depth=0)
    try:
        raw = json.dumps(slim, default=str)
    except Exception:
        return {"ok": True, "note": "unserializable_result"}
    if len(raw) <= limit:
        return slim
    # hard truncate string payload
    return {
        "ok": slim.get("ok", True) if isinstance(slim, dict) else True,
        "compact": True,
        "truncated": True,
        "preview": raw[: max(0, limit - 120)],
        "chars": len(raw),
        "hint": "Pass detail=true on project tools or set VIDMCP_COMPACT=0 for full payloads",
    }


def _slim(obj: Any, *, depth: int) -> Any:
    if depth > 6:
        return "…"
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _DROP_KEYS:
                if k == "words" and isinstance(v, list):
                    out["n_words"] = len(v)
                elif k == "edit_history" and isinstance(v, list):
                    out["history_len"] = len(v)
                elif k == "layers":
                    out["layers_present"] = True
                continue
            # shorten long paths lists of renders
            if k in ("renders", "reviews") and isinstance(v, list) and len(v) > 3:
                out[k] = [_slim(x, depth=depth + 1) for x in v[-3:]]
                out[f"{k}_total"] = len(v)
                continue
            if k == "steps" and isinstance(v, dict):
                # keep only ok/path/metrics per step
                steps = {}
                for sk, sv in v.items():
                    if isinstance(sv, dict):
                        steps[sk] = {
                            kk: sv[kk]
                            for kk in ("ok", "path", "final_path", "duration_sec", "coverage_mean", "backend", "preset", "removed_sec", "lufs_out", "n_cues", "warnings")
                            if kk in sv
                        }
                    else:
                        steps[sk] = sv
                out[k] = steps
                continue
            out[k] = _slim(v, depth=depth + 1)
        return out
    if isinstance(obj, list):
        if len(obj) > 20:
            return [_slim(x, depth=depth + 1) for x in obj[:8]] + [f"…+{len(obj)-8} more"]
        return [_slim(x, depth=depth + 1) for x in obj]
    if isinstance(obj, str) and len(obj) > 500:
        return obj[:400] + "…"
    return obj
