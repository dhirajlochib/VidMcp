"""Async job records for long-running SAM / render work."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class JobType(str, Enum):
    ANALYZE = "analyze"
    SEGMENT = "segment"
    EFFECTS = "effects"
    BROLL = "broll"
    COMPOSITE = "composite"
    REVIEW = "review"
    PIPELINE = "pipeline"


class JobRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str
    job_type: JobType
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    stage: str = "queued"
    message: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)

    def mark_running(self, stage: str = "running") -> None:
        self.status = JobStatus.RUNNING
        self.stage = stage
        self.started_at = _utcnow()

    def mark_progress(self, progress: float, stage: str | None = None, message: str = "") -> None:
        self.progress = max(0.0, min(1.0, progress))
        if stage:
            self.stage = stage
        if message:
            self.message = message

    def mark_succeeded(self, output: dict[str, Any] | None = None) -> None:
        self.status = JobStatus.SUCCEEDED
        self.progress = 1.0
        self.stage = "done"
        self.finished_at = _utcnow()
        if output:
            self.output = output

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.error = error
        self.stage = "failed"
        self.finished_at = _utcnow()
