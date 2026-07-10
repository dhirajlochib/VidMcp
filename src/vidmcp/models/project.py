"""Project manifest — source of truth for non-destructive edits."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from vidmcp.models.layers import LayerStack


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ProjectStatus(str, Enum):
    CREATED = "created"
    ANALYZED = "analyzed"
    SEGMENTED = "segmented"
    EFFECTS_APPLIED = "effects_applied"
    RENDERED = "rendered"
    REVIEWED = "reviewed"
    FAILED = "failed"


class SegmentObject(BaseModel):
    object_id: int
    label: str
    confidence_mean: float = 0.0
    frame_span: tuple[int, int] = (0, 0)
    area_ratio_mean: float = 0.0
    is_primary: bool = False


class SegmentTrack(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    prompt: str
    backend: str
    created_at: datetime = Field(default_factory=_utcnow)
    mask_dir: str  # relative
    mask_video: str | None = None
    objects: list[SegmentObject] = Field(default_factory=list)
    conf_threshold: float = 0.25
    frame_count: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class ReviewNote(BaseModel):
    severity: str  # info | warning | error
    code: str
    message: str
    frame_hint: int | None = None
    suggestion: str | None = None


class ProjectManifest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = "untitled"
    status: ProjectStatus = ProjectStatus.CREATED
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    # paths relative to project root unless absolute
    source_video: str | None = None
    source_meta: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] = Field(default_factory=dict)
    segments: list[SegmentTrack] = Field(default_factory=list)
    primary_segment_id: str | None = None
    layers: LayerStack = Field(default_factory=LayerStack)
    renders: list[dict[str, Any]] = Field(default_factory=list)
    reviews: list[dict[str, Any]] = Field(default_factory=list)
    edit_history: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    version: int = 1

    def touch(self) -> None:
        self.updated_at = _utcnow()
        self.version += 1

    def append_history(self, action: str, detail: dict[str, Any] | None = None) -> None:
        self.edit_history.append(
            {
                "ts": _utcnow().isoformat(),
                "action": action,
                "detail": detail or {},
                "version": self.version,
            }
        )
        self.touch()

    def primary_segment(self) -> SegmentTrack | None:
        if self.primary_segment_id:
            for s in self.segments:
                if s.id == self.primary_segment_id:
                    return s
        return self.segments[-1] if self.segments else None
