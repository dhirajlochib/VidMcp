"""Effect registry — extensible plugin map."""

from __future__ import annotations

from vidmcp.effects.background import (
    BlurBackgroundEffect,
    CyberpunkBackgroundEffect,
    GenerativePlaceholderEffect,
    ImagePlateBackgroundEffect,
    SolidBackgroundEffect,
)
from vidmcp.effects.base import Effect
from vidmcp.effects.grading import ColorGradeEffect
from vidmcp.effects.particles import ParticleSystemEffect


class EffectRegistry:
    def __init__(self) -> None:
        self._effects: dict[str, Effect] = {}
        self.register(BlurBackgroundEffect())
        self.register(SolidBackgroundEffect())
        self.register(ImagePlateBackgroundEffect())
        self.register(CyberpunkBackgroundEffect())
        self.register(GenerativePlaceholderEffect())
        self.register(ParticleSystemEffect())
        self.register(ColorGradeEffect())
        # aliases
        self._effects["neon_dust"] = self._effects["particles"]
        self._effects["sparks"] = self._effects["particles"]
        self._effects["background_blur"] = self._effects["blur"]
        self._effects["cyberpunk_bg"] = self._effects["cyberpunk"]

    def register(self, effect: Effect) -> None:
        self._effects[effect.name] = effect

    def get(self, name: str) -> Effect:
        if name not in self._effects:
            raise KeyError(f"Unknown effect: {name}. Available: {sorted(self._effects)}")
        return self._effects[name]

    def list_effects(self) -> list[dict[str, str]]:
        seen: set[str] = set()
        out = []
        for name, eff in self._effects.items():
            if id(eff) in seen:
                continue
            seen.add(id(eff))
            out.append({"name": eff.name, "kind": eff.kind})
        return out


_registry: EffectRegistry | None = None


def get_effect_registry() -> EffectRegistry:
    global _registry
    if _registry is None:
        _registry = EffectRegistry()
    return _registry
