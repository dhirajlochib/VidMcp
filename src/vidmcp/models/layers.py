"""Non-destructive layer model for compositing."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class LayerKind(str, Enum):
    SOURCE = "source"
    SUBJECT = "subject"  # cutout of tracked subject
    BACKGROUND = "background"  # plate / generative / solid / blurred original
    PARTICLES = "particles"
    GRADE = "grade"
    OVERLAY = "overlay"
    BROLL = "broll"
    ADJUSTMENT = "adjustment"


class BlendMode(str, Enum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    ADD = "add"


class EffectParams(BaseModel):
    """Parametric effect definition — re-renderable, serializable."""

    effect_type: str
    # background: blur | solid | image_plate | video_plate | cyberpunk_grade | generative
    # particles: sparks | neon_dust | rain | custom
    intensity: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)

    # common keys in params:
    # blur_radius, color, plate_path, prompt, seed, density, velocity, palette


class Layer(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    kind: LayerKind
    enabled: bool = True
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    blend_mode: BlendMode = BlendMode.NORMAL
    # z-order: lower draws first (behind)
    z_index: int = 0
    # asset paths relative to project root
    asset_path: str | None = None
    mask_path: str | None = None  # for subject / selective layers
    mask_invert: bool = False
    effect: EffectParams | None = None
    # track linkage
    segment_track_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class LayerStack(BaseModel):
    """Ordered stack; render bottom→top by z_index then insertion order."""

    layers: list[Layer] = Field(default_factory=list)
    version: int = 1

    def add(self, layer: Layer) -> Layer:
        self.layers.append(layer)
        self.version += 1
        return layer

    def get(self, layer_id: str) -> Layer | None:
        for L in self.layers:
            if L.id == layer_id:
                return L
        return None

    def remove(self, layer_id: str) -> bool:
        before = len(self.layers)
        self.layers = [L for L in self.layers if L.id != layer_id]
        if len(self.layers) != before:
            self.version += 1
            return True
        return False

    def sorted_layers(self) -> list[Layer]:
        return sorted(
            [L for L in self.layers if L.enabled],
            key=lambda L: (L.z_index, L.name),
        )

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
