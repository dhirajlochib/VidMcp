"""Causal content-addressed edit graph with branches and time travel.

Every mutation is a node keyed by hash(parent + op + args). Agents can:
- branch from any node
- checkout historical project state
- merge branches (layer-stack union with conflict report)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson
from filelock import FileLock
from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def content_hash(payload: dict[str, Any]) -> str:
    raw = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(raw).hexdigest()[:16]


class CausalNode(BaseModel):
    id: str
    parent_id: str | None = None
    op: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    manifest_snapshot: dict[str, Any] | None = None
    branch: str = "main"
    created_at: str = Field(default_factory=_utcnow)
    message: str = ""


class CausalGraph(BaseModel):
    project_id: str
    head: str | None = None  # current node id
    branches: dict[str, str] = Field(default_factory=dict)  # name -> head node id
    nodes: dict[str, CausalNode] = Field(default_factory=dict)
    version: int = 1

    def ensure_root(self) -> CausalNode:
        if self.nodes:
            return self.nodes[self.head] if self.head else next(iter(self.nodes.values()))
        nid = content_hash({"op": "root", "project": self.project_id})
        node = CausalNode(id=nid, op="root", message="project genesis", branch="main")
        self.nodes[nid] = node
        self.head = nid
        self.branches["main"] = nid
        return node

    def commit(
        self,
        op: str,
        args: dict[str, Any] | None = None,
        *,
        result_summary: dict[str, Any] | None = None,
        manifest_snapshot: dict[str, Any] | None = None,
        message: str = "",
        branch: str | None = None,
    ) -> CausalNode:
        self.ensure_root()
        branch = branch or self._current_branch()
        parent = self.branches.get(branch) or self.head
        payload = {
            "parent": parent,
            "op": op,
            "args": args or {},
            "branch": branch,
            "nonce": str(uuid4()),  # allow identical ops as distinct commits
        }
        nid = content_hash(payload)
        # collapse pure content if same parent+op+args without nonce for idempotent ops
        node = CausalNode(
            id=nid,
            parent_id=parent,
            op=op,
            args=args or {},
            result_summary=result_summary or {},
            manifest_snapshot=manifest_snapshot,
            branch=branch,
            message=message or op,
        )
        self.nodes[nid] = node
        self.branches[branch] = nid
        self.head = nid
        self.version += 1
        return node

    def _current_branch(self) -> str:
        if not self.head:
            return "main"
        n = self.nodes.get(self.head)
        return n.branch if n else "main"

    def checkout(self, node_id: str) -> CausalNode:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node: {node_id}")
        self.head = node_id
        br = self.nodes[node_id].branch
        self.branches[br] = node_id
        return self.nodes[node_id]

    def branch(self, name: str, from_node: str | None = None) -> str:
        src = from_node or self.head
        if not src or src not in self.nodes:
            raise RuntimeError("No node to branch from")
        if name in self.branches:
            raise ValueError(f"Branch exists: {name}")
        self.branches[name] = src
        # move head to branch tip (same node, new branch label for future commits)
        self.head = src
        return src

    def log(self, branch: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        tip = self.branches.get(branch or self._current_branch()) or self.head
        out: list[dict[str, Any]] = []
        cur = tip
        while cur and len(out) < limit:
            n = self.nodes.get(cur)
            if not n:
                break
            out.append(n.model_dump(mode="json"))
            cur = n.parent_id
        return out

    def ancestry(self, node_id: str) -> list[str]:
        chain = []
        cur = node_id
        while cur:
            chain.append(cur)
            n = self.nodes.get(cur)
            cur = n.parent_id if n else None
        return chain

    def merge(self, source_branch: str, target_branch: str = "main") -> dict[str, Any]:
        """Layer-aware merge: prefer target head manifest layers, append unique source layers."""
        if source_branch not in self.branches or target_branch not in self.branches:
            raise KeyError("Unknown branch")
        src_id = self.branches[source_branch]
        tgt_id = self.branches[target_branch]
        src_n, tgt_n = self.nodes[src_id], self.nodes[tgt_id]
        src_layers = (src_n.manifest_snapshot or {}).get("layers", {}).get("layers", [])
        tgt_layers = (tgt_n.manifest_snapshot or {}).get("layers", {}).get("layers", [])
        tgt_ids = {L.get("id") for L in tgt_layers}
        conflicts = []
        merged = list(tgt_layers)
        for L in src_layers:
            if L.get("id") in tgt_ids:
                # same id different effect?
                existing = next(x for x in tgt_layers if x.get("id") == L.get("id"))
                if existing.get("effect") != L.get("effect"):
                    conflicts.append({"layer_id": L.get("id"), "source": L, "target": existing})
            else:
                merged.append(L)
        merge_node = self.commit(
            "merge",
            {"source": source_branch, "target": target_branch, "src_node": src_id, "tgt_node": tgt_id},
            result_summary={"conflicts": len(conflicts), "layers": len(merged)},
            manifest_snapshot={
                **(tgt_n.manifest_snapshot or {}),
                "layers": {"layers": merged, "version": (tgt_n.manifest_snapshot or {}).get("layers", {}).get("version", 0) + 1},
            },
            message=f"merge {source_branch} → {target_branch}",
            branch=target_branch,
        )
        return {
            "merge_node": merge_node.id,
            "conflicts": conflicts,
            "layer_count": len(merged),
            "head": self.head,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path) + ".lock")
        with lock:
            path.write_bytes(orjson.dumps(self.model_dump(mode="json"), option=orjson.OPT_INDENT_2))

    @classmethod
    def load(cls, path: Path, project_id: str) -> CausalGraph:
        path = Path(path)
        if not path.exists():
            g = cls(project_id=project_id)
            g.ensure_root()
            return g
        data = orjson.loads(path.read_bytes())
        return cls.model_validate(data)


class CausalGraphStore:
    def __init__(self, project_root: Path, project_id: str):
        self.path = Path(project_root) / "causal_graph.json"
        self.project_id = project_id

    def load(self) -> CausalGraph:
        return CausalGraph.load(self.path, self.project_id)

    def save(self, graph: CausalGraph) -> None:
        graph.save(self.path)
