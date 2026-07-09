"""A/B edit variants — generate multiple stylistic alternatives for comparison (VideoDiff-style)."""

from __future__ import annotations

from typing import Any


VARIANT_PRESETS: list[dict[str, Any]] = [
    {
        "id": "v_cyberpunk",
        "label": "Cyberpunk Neon",
        "style_tags": ["cyberpunk", "particles"],
        "broll": "cyberpunk_city",
    },
    {
        "id": "v_bokeh",
        "label": "Cinematic Bokeh",
        "style_tags": ["blur"],
        "broll": None,
    },
    {
        "id": "v_noir",
        "label": "Rain Noir",
        "style_tags": ["blur", "rain"],
        "broll": None,
    },
    {
        "id": "v_solid_sparks",
        "label": "Studio Sparks",
        "style_tags": ["solid", "particles"],
        "effects": [
            {"effect_type": "solid", "kind": "background", "params": {"color": "#08060f"}, "name": "studio"},
            {
                "effect_type": "particles",
                "kind": "particles",
                "params": {"style": "sparks", "density": 0.4},
                "blend_mode": "screen",
                "name": "sparks",
            },
        ],
        "broll": None,
    },
]


def pick_variants(n: int = 3) -> list[dict[str, Any]]:
    return VARIANT_PRESETS[: max(1, min(n, len(VARIANT_PRESETS)))]
