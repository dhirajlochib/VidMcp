"""Central configuration via environment + pydantic-settings."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SamBackend(str, Enum):
    AUTO = "auto"
    MLX = "mlx"  # Apple Silicon SAM 3.1 (best on M-series)
    ULTRALYTICS = "ultralytics"
    OFFICIAL = "official"
    MOCK = "mock"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIDMCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    workspace_root: Path = Field(default=Path("./workspaces"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    sam_backend: SamBackend = SamBackend.AUTO
    sam_weights: Path | None = None
    device: str = "auto"
    # SAM 3.1 Object Multiplex
    sam_use_multiplex: bool = True
    sam_use_fa3: bool = False
    mlx_model_id: str = "mlx-community/sam3.1-bf16"
    mlx_detect_every: int = 8  # re-detect every N frames (speed)
    mlx_max_side: int | None = 768  # downscale long side for speed; None = full
    # Fast education harness knobs
    harness_fast_mode: bool = True
    harness_preview_frames: int = 48
    harness_skip_broll_if_scene: bool = True
    harness_parallel_critics: bool = True
    education_default_steps: int = 5
    max_concurrent_jobs: int = 2
    frame_sample_fps: float = 2.0
    default_mask_feather: int = 3
    enable_generative: bool = False
    max_video_duration_sec: float = 600.0
    max_resolution: int = 4096
    job_ttl_hours: int = 72
    preview_max_side: int = 960
    conf_threshold: float = 0.25
    # Harness / quality gates
    harness_max_passes: int = 3
    harness_min_review_score: float = 0.7
    harness_min_temporal_stability: float = 0.55
    harness_auto_refine: bool = True
    harness_variant_count: int = 3
    allow_import_outside_workspace: bool = True
    import_copy_into_workspace: bool = True
    # Agent context control (see harness/packs.py)
    # talking_head | education | vfx | admin | all
    tool_pack: str = "talking_head"
    compact: bool = True
    max_result_chars: int = 4000
    matte_quality: Literal["preview", "final"] = "preview"
    # v2.0 — proxy pipeline / cognition / quality gates (see UPGRADE_ROADMAP.md)
    proxy_max_side: int = 540
    max_repair_passes: int = 2
    max_edge_flicker_dtssd: float = 0.05
    max_color_cast_delta_e: float = 8.0
    max_skin_delta_e: float = 5.0
    lufs_tolerance: float = 1.5
    max_true_peak_db: float = -0.7
    default_brand_kit: str = "default"

    @field_validator("workspace_root", "sam_weights", mode="before")
    @classmethod
    def _expand_path(cls, v: object) -> Path | None:
        if v is None or v == "":
            return None
        return Path(str(v)).expanduser().resolve()

    @field_validator("tool_pack", mode="before")
    @classmethod
    def _normalize_pack(cls, v: object) -> str:
        s = str(v or "talking_head").strip().lower()
        aliases = {
            "vfx_matte": "vfx",
            "creator": "talking_head",
            "default": "talking_head",
            "full": "all",
        }
        return aliases.get(s, s)

    def ensure_dirs(self) -> None:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        (self.workspace_root / ".vidmcp").mkdir(exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
