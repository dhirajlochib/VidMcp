"""Progress reporting helpers for long-running MCP tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


ProgressCallback = Callable[[float, str, dict[str, Any] | None], Awaitable[None] | None]


@dataclass
class ProgressTracker:
    """Tracks fractional progress [0,1] with optional MCP context bridge."""

    total_units: float = 100.0
    completed: float = 0.0
    stage: str = "init"
    callback: ProgressCallback | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def set_stage(self, stage: str, completed_units: float | None = None) -> None:
        self.stage = stage
        if completed_units is not None:
            self.completed = completed_units

    def advance(self, units: float = 1.0, message: str | None = None) -> None:
        self.completed = min(self.total_units, self.completed + units)
        if self.callback:
            msg = message or self.stage
            result = self.callback(self.fraction, msg, {**self.meta, "stage": self.stage})
            # fire-and-forget if awaitable — caller may await report()
            if hasattr(result, "__await__"):
                self._pending = result  # type: ignore[attr-defined]

    @property
    def fraction(self) -> float:
        if self.total_units <= 0:
            return 0.0
        return max(0.0, min(1.0, self.completed / self.total_units))

    async def report(self, message: str, fraction: float | None = None) -> None:
        if fraction is not None:
            self.completed = fraction * self.total_units
        if self.callback:
            result = self.callback(self.fraction, message, {**self.meta, "stage": self.stage})
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]
