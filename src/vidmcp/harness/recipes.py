"""Named production recipes — higher-level than raw tools (what market MCPs lack)."""

from __future__ import annotations

from typing import Any


RECIPES: dict[str, dict[str, Any]] = {
    "talking_head_math": {
        "name": "talking_head_math",
        "description": "Talking-head over animated math/education plate (Manim or procedural)",
        "subject_prompt": "person",
        "alternate_prompts": ["speaker", "human"],
        "style_tags": ["blur"],
        "effects": [
            {
                "effect_type": "color_grade",
                "kind": "grade",
                "params": {"contrast": 1.08, "saturation": 1.05, "temperature": 0.05, "background_only": True},
                "name": "soft_grade",
            }
        ],
        "generate_broll": False,
        "math_scene": True,
        "conf": 0.25,
        "max_passes": 2,
    },

    "cyberpunk_talking_head": {
        "name": "cyberpunk_talking_head",
        "description": "Neon cyberpunk world behind speaker with motion-reactive particles",
        "subject_prompt": "person",
        "alternate_prompts": ["speaker", "human", "person talking"],
        "style_tags": ["cyberpunk", "particles"],
        "effects": None,  # derived by planner
        "broll_style": "cyberpunk_city",
        "generate_broll": True,
        "conf": 0.25,
        "max_passes": 3,
    },
    "cinematic_bokeh": {
        "name": "cinematic_bokeh",
        "description": "Shallow DOF blur behind subject with subtle grade",
        "subject_prompt": "person",
        "alternate_prompts": ["subject", "face"],
        "style_tags": ["blur"],
        "effects": [
            {"effect_type": "blur", "kind": "background", "params": {"blur_radius": 45}, "name": "bokeh"},
            {
                "effect_type": "color_grade",
                "kind": "grade",
                "params": {"contrast": 1.15, "saturation": 1.05, "temperature": 0.1, "background_only": True},
                "name": "warm_grade",
            },
        ],
        "generate_broll": False,
        "conf": 0.3,
        "max_passes": 2,
    },
    "product_spotlight": {
        "name": "product_spotlight",
        "description": "Isolate product, dark solid BG, spark particles",
        "subject_prompt": "product",
        "alternate_prompts": ["object", "item", "bottle", "box"],
        "style_tags": ["solid", "particles"],
        "effects": [
            {"effect_type": "solid", "kind": "background", "params": {"color": "#0a0a12"}, "name": "studio"},
            {
                "effect_type": "particles",
                "kind": "particles",
                "params": {"style": "sparks", "density": 0.35},
                "blend_mode": "screen",
                "name": "sparks",
            },
        ],
        "generate_broll": False,
        "conf": 0.2,
        "max_passes": 3,
    },
    "green_screen_replace": {
        "name": "green_screen_replace",
        "description": "Classic subject cutout over generative/procedural plate",
        "subject_prompt": "person",
        "alternate_prompts": ["speaker"],
        "style_tags": ["background"],
        "effects": [
            {
                "effect_type": "generative",
                "kind": "background",
                "params": {"prompt": "modern office interior soft light"},
                "name": "gen_plate",
            }
        ],
        "broll_style": "abstract",
        "generate_broll": True,
        "conf": 0.25,
        "max_passes": 2,
    },
    "multi_subject_vfx": {
        "name": "multi_subject_vfx",
        "description": "Track multiple concepts with SAM multiplex, style BG only",
        "subject_prompt": "person",
        "multi_prompts": ["person", "microphone"],
        "style_tags": ["cyberpunk", "particles"],
        "generate_broll": False,
        "conf": 0.22,
        "max_passes": 3,
    },
    "rain_noir": {
        "name": "rain_noir",
        "description": "Dark grade + rain particles behind subject",
        "subject_prompt": "person",
        "style_tags": ["blur", "rain"],
        "effects": [
            {"effect_type": "blur", "kind": "background", "params": {"blur_radius": 25}, "name": "soft_bg"},
            {
                "effect_type": "particles",
                "kind": "particles",
                "params": {"style": "rain", "density": 0.7},
                "blend_mode": "screen",
                "name": "rain",
            },
            {
                "effect_type": "color_grade",
                "kind": "grade",
                "params": {"contrast": 1.2, "saturation": 0.7, "temperature": -0.3, "background_only": True},
                "name": "noir",
            },
        ],
        "generate_broll": False,
        "conf": 0.25,
        "max_passes": 2,
    },
}


def list_recipes() -> list[dict[str, Any]]:
    return [
        {"name": r["name"], "description": r["description"], "subject_prompt": r.get("subject_prompt")}
        for r in RECIPES.values()
    ]


def get_recipe(name: str) -> dict[str, Any]:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in RECIPES:
        raise KeyError(f"Unknown recipe '{name}'. Available: {sorted(RECIPES)}")
    return dict(RECIPES[key])
