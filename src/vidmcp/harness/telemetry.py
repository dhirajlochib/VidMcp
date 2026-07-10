"""Run telemetry for harness experiments."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import orjson


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class TelemetryRun:
    def __init__(self, project_id: str, kind: str = "pipeline"):
        self.id = str(uuid4())
        self.project_id = project_id
        self.kind = kind
        self.started_at = _utcnow()
        self.events: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.finished_at: str | None = None

    def event(self, name: str, **payload: Any) -> None:
        self.events.append({"ts": _utcnow(), "name": name, **payload})

    def set_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def finish(self) -> dict[str, Any]:
        self.finished_at = _utcnow()
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "project_id": self.project_id,
            "kind": self.kind,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": self.metrics,
            "events": self.events,
        }

    def save(self, path) -> None:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(orjson.dumps(self.to_dict(), option=orjson.OPT_INDENT_2))
