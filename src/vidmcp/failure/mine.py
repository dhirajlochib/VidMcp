"""Failure mining loop: store gate/critic failures, cluster, suggest new heuristics."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson
from filelock import FileLock


class FailureStore:
    def __init__(self, root: Path):
        self.path = Path(root) / ".vidmcp" / "failures.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.path) + ".lock")

    def record(self, event: dict[str, Any]) -> str:
        eid = str(uuid4())[:8]
        row = {
            "id": eid,
            "ts": datetime.now(UTC).isoformat(),
            **event,
        }
        with self._lock, open(self.path, "ab") as f:
            f.write(orjson.dumps(row) + b"\n")
        return eid

    def load(self, limit: int = 500) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(orjson.loads(line))
                except Exception:
                    continue
        return rows[-limit:]


def mine_workspace_failures(workspace_root: Path) -> dict[str, Any]:
    store = FailureStore(workspace_root)
    rows = store.load()
    codes = Counter()
    tools = Counter()
    for r in rows:
        for c in r.get("failed_axes") or r.get("codes") or []:
            codes[c] += 1
        for t in r.get("fix_route") or []:
            tools[t] += 1
        if r.get("code"):
            codes[r["code"]] += 1
    return {
        "ok": True,
        "n_failures": len(rows),
        "top_codes": codes.most_common(10),
        "top_fix_tools": tools.most_common(10),
        "recent": rows[-10:],
    }


def suggest_heuristics(mine_result: dict[str, Any]) -> dict[str, Any]:
    suggestions = []
    for code, count in mine_result.get("top_codes") or []:
        if code in ("matte_flicker", "temporal_instability", "temporal_stability"):
            suggestions.append(
                {
                    "heuristic": "auto_refine_on_stab_below_0.65",
                    "action": "Always chain refine_segment_keyframes + compute_uncertainty_field",
                    "support": count,
                }
            )
        if code in ("lighting_match",):
            suggestions.append(
                {
                    "heuristic": "auto_lighting_match_after_bg_replace",
                    "action": "Call match_subject_lighting after any scene/broll composite",
                    "support": count,
                }
            )
        if code in ("edge_quality", "hard_edges"):
            suggestions.append(
                {
                    "heuristic": "increase_feather_and_uncertainty_roi",
                    "action": "VIDMCP_DEFAULT_MASK_FEATHER+=2; refine with uncertainty boxes",
                    "support": count,
                }
            )
        if code in ("render_complete", "not_rendered"):
            suggestions.append(
                {
                    "heuristic": "enforce_composite_before_sign",
                    "action": "Gate sign_render on renders non-empty",
                    "support": count,
                }
            )
    # dedupe by heuristic name
    seen = set()
    uniq = []
    for s in suggestions:
        if s["heuristic"] not in seen:
            seen.add(s["heuristic"])
            uniq.append(s)
    if not uniq:
        uniq.append(
            {
                "heuristic": "baseline_critic_ensemble",
                "action": "Run run_critic_ensemble at end of every pipeline",
                "support": 0,
            }
        )
    return {"ok": True, "suggestions": uniq}
