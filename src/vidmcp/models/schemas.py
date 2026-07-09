"""Public tool response schemas (stable contract for LLM consumers)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    ok: bool = True
    project_id: str | None = None
    job_id: str | None = None
    message: str = ""
    warnings: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class AnalyzeVideoResult(ToolResponse):
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    has_audio: bool = False
    scene_hints: list[str] = Field(default_factory=list)
    suggested_prompts: list[str] = Field(default_factory=list)
    thumbnail_paths: list[str] = Field(default_factory=list)
    talking_head_score: float = 0.0


class SegmentSubjectResult(ToolResponse):
    segment_id: str | None = None
    prompt: str = ""
    backend: str = ""
    mask_dir: str | None = None
    mask_video: str | None = None
    object_count: int = 0
    objects: list[dict[str, Any]] = Field(default_factory=list)
    coverage_mean: float = 0.0
    temporal_stability: float = 0.0


class ApplyEffectsResult(ToolResponse):
    layer_ids: list[str] = Field(default_factory=list)
    effect_types: list[str] = Field(default_factory=list)
    stack_version: int = 0


class GenerateBrollResult(ToolResponse):
    broll_path: str | None = None
    layer_id: str | None = None
    mode: str = "procedural"
    duration_sec: float = 0.0


class CompositeResult(ToolResponse):
    output_path: str | None = None
    preview_path: str | None = None
    codec: str = "h264"
    duration_sec: float = 0.0
    render_id: str | None = None


class ReviewResult(ToolResponse):
    score: float = 0.0
    passed: bool = False
    notes: list[dict[str, Any]] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
