"""Declarative edit graph (DAG) — serializable, resumable, market-MCP differentiator."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    op: str  # tool / service name
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: str = "pending"  # pending|running|done|failed|skipped
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EditGraph(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str | None = None
    intent: str = ""
    nodes: list[GraphNode] = Field(default_factory=list)
    version: int = 1

    def add(self, op: str, args: dict[str, Any] | None = None, depends_on: list[str] | None = None) -> GraphNode:
        n = GraphNode(op=op, args=args or {}, depends_on=depends_on or [])
        self.nodes.append(n)
        return n

    def ready_nodes(self) -> list[GraphNode]:
        done = {n.id for n in self.nodes if n.status == "done"}
        out = []
        for n in self.nodes:
            if n.status != "pending":
                continue
            if all(d in done for d in n.depends_on):
                out.append(n)
        return out

    def topological_wave(self) -> list[list[GraphNode]]:
        """Return execution waves (parallelizable sets)."""
        remaining = {n.id: n for n in self.nodes}
        done: set[str] = set()
        waves: list[list[GraphNode]] = []
        while remaining:
            wave = [
                n
                for n in remaining.values()
                if all(d in done or d not in {x.id for x in self.nodes} for d in n.depends_on)
            ]
            if not wave:
                # cycle or unmet — break
                waves.append(list(remaining.values()))
                break
            waves.append(wave)
            for n in wave:
                done.add(n.id)
                remaining.pop(n.id, None)
        return waves

    @staticmethod
    def from_intent_plan(intent: str, steps: list[dict[str, Any]], project_id: str | None = None) -> EditGraph:
        g = EditGraph(intent=intent, project_id=project_id)
        prev: str | None = None
        for s in steps:
            deps = [prev] if prev else []
            node = g.add(s.get("tool") or s.get("op"), s.get("args") or {}, deps)
            prev = node.id
        return g
