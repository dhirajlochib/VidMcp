"""Recipe schema v2 — typed, composable, adaptive. Executes through the op batcher.

v1 recipes (harness/recipes.py dicts) keep working via the runtime; v2 adds:
steps referencing ops, `extends` inheritance, `compose` chaining, `when` conditions,
per-content-type `adapt` overrides, brand injection, `requires` validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.recipe_v2")


class RecipeStep(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None
    when: str | None = None  # expression over prior step results
    on_fail: Literal["retry", "skip", "abort"] = "skip"


class RecipeV2(BaseModel):
    name: str
    description: str = ""
    version: str = "2.0"
    extends: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)  # {name: {default, doc}}
    steps: list[RecipeStep] = Field(default_factory=list)
    adapt: dict[str, dict[str, Any]] = Field(default_factory=dict)  # content_type → arg overrides by step id
    brand: str | None = "default"
    requires: list[str] = Field(default_factory=list)  # op names / model names


RECIPES_V2: dict[str, RecipeV2] = {}


def register_recipe(recipe: RecipeV2) -> None:
    RECIPES_V2[recipe.name] = recipe


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_recipe(name_or_dict: str | dict[str, Any]) -> RecipeV2:
    """Resolve by name (with extends chain) or validate an inline dict."""
    if isinstance(name_or_dict, dict):
        recipe = RecipeV2.model_validate(name_or_dict)
    else:
        key = str(name_or_dict).strip().lower().replace("-", "_").replace(" ", "_")
        if key not in RECIPES_V2:
            raise KeyError(f"Unknown v2 recipe '{name_or_dict}'. Available: {sorted(RECIPES_V2)}")
        recipe = RECIPES_V2[key]
    if recipe.extends:
        parent = resolve_recipe(recipe.extends)
        merged = _deep_merge(parent.model_dump(), recipe.model_dump(exclude_unset=True, exclude={"extends"}))
        # child steps replace parent steps only if child defined any
        if not recipe.steps:
            merged["steps"] = parent.model_dump()["steps"]
        merged["extends"] = None
        recipe = RecipeV2.model_validate(merged)
    return recipe


def compose_recipes(names: list[str], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sequential composition: steps concatenated, duplicate render steps deduped (last wins)."""
    try:
        resolved = [resolve_recipe(n) for n in names]
    except KeyError as e:
        return {"ok": False, "message": str(e)}
    steps: list[RecipeStep] = []
    conflicts: list[str] = []
    seen_render = False
    for r in reversed(resolved):  # walk backwards so last recipe's render wins
        for s in reversed(r.steps):
            if s.tool in ("composite_and_render", "export_render", "export_multi"):
                if seen_render:
                    conflicts.append(f"dropped duplicate {s.tool} from {r.name}")
                    continue
                seen_render = True
            steps.insert(0, s)
    composed = RecipeV2(
        name="+".join(r.name for r in resolved),
        description=" then ".join(r.description or r.name for r in resolved),
        steps=steps,
        requires=sorted({req for r in resolved for req in r.requires}),
        brand=resolved[-1].brand,
    )
    data = composed.model_dump()
    if overrides:
        data = _deep_merge(data, overrides)
    return {"ok": True, "recipe": data, "conflicts_resolved": conflicts}


def validate_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    from vidmcp.harness.ops import OP_TABLE

    try:
        r = resolve_recipe(recipe) if isinstance(recipe, dict) else resolve_recipe(str(recipe))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "errors": [str(e)]}
    errors = [f"unknown op '{s.tool}'" for s in r.steps if s.tool not in OP_TABLE]
    missing_models = []
    for req in r.requires:
        if req in OP_TABLE:
            continue
        try:
            from vidmcp.models_registry import ensure_model

            st = ensure_model(req)
            if not st.get("found"):
                missing_models.append({"model": req, "hint": st.get("hint")})
        except Exception:  # noqa: BLE001
            pass
    return {"ok": not errors, "errors": errors, "required_models_missing": missing_models,
            "n_steps": len(r.steps), "name": r.name}


def run_recipe_v2(project: Any, recipe: str | dict[str, Any], content_type: str | None = None) -> dict[str, Any]:
    """Execute a v2 recipe on an existing project via run_ops (adapt overrides applied)."""
    from vidmcp.harness.ops import run_ops

    r = resolve_recipe(recipe)
    if content_type is None:
        ct = (project.manifest.analysis.get("content_type") or {}).get("type")
        content_type = ct
    overrides = r.adapt.get(content_type or "", {})
    ops = []
    for s in r.steps:
        args = dict(s.args)
        if s.id and s.id in overrides:
            args = _deep_merge(args, overrides[s.id])
        ops.append({"tool": s.tool, "args": args, "id": s.id or s.tool, "if": s.when})
    result = run_ops(project, ops, stop_on_error=False)
    result["recipe"] = r.name
    result["adapted_for"] = content_type
    return result


# ---------------------------------------------------------------------------
# Shipped v2 recipes (built on the new op surface)
# ---------------------------------------------------------------------------

register_recipe(RecipeV2(
    name="god_mode_talking_head",
    description="Full pipeline: index → cuts → alpha matte → auto color → graphics → music → mixdown → moves",
    steps=[
        RecipeStep(tool="build_footage_index", id="index"),
        RecipeStep(tool="detect_scenes", id="scenes"),
        RecipeStep(tool="plan_cuts", id="cuts", args={"aggressiveness": 0.5}),
        RecipeStep(tool="apply_cut_plan", id="cut_apply", args={"plan_id": "$cuts.plan_id"},
                   when="cuts.removed_sec > 1.0"),
        RecipeStep(tool="segment_subject", id="seg", args={"prompt": "person"}),
        RecipeStep(tool="refine_alpha", id="alpha", when="seg.ok"),
        RecipeStep(tool="stabilize_matte", id="stab", when="seg.temporal_stability < 0.75"),
        RecipeStep(tool="auto_color", id="color"),
        RecipeStep(tool="apply_lut", id="lut", args={"lut": "filmic_soft", "intensity": 0.7}),
        RecipeStep(tool="generate_music", id="music", args={"style": "cinematic"}),
        RecipeStep(tool="add_sfx", id="sfx"),
        RecipeStep(tool="mixdown_audio", id="mix", args={"target": "youtube"}),
        RecipeStep(tool="composite_and_render", id="render"),
        RecipeStep(tool="add_camera_moves", id="moves", args={"style": "emphasis_punch"}),
        RecipeStep(tool="rewatch_render", id="qc"),
    ],
))

register_recipe(RecipeV2(
    name="podcast_clip_factory",
    description="Find peak moments → vertical clips with captions, punch-ins, loudness",
    steps=[
        RecipeStep(tool="build_footage_index", id="index"),
        RecipeStep(tool="extract_clips", id="clips", args={"n": 3, "max_sec": 45}),
        RecipeStep(tool="smart_reframe", id="reframe", args={"target": "9:16"}),
        RecipeStep(tool="transcribe_and_caption", id="captions", args={"style": "karaoke"}),
        RecipeStep(tool="export_multi", id="export", args={"targets": ["reels_9x16"]}),
    ],
))

register_recipe(RecipeV2(
    name="ad_15s",
    description="Punchy 15s cut: tightest cuts, beat music, stat graphics, square+vertical export",
    steps=[
        RecipeStep(tool="build_footage_index", id="index"),
        RecipeStep(tool="plan_cuts", id="cuts", args={"aggressiveness": 0.85, "keep_dramatic_pauses": False}),
        RecipeStep(tool="apply_cut_plan", id="cut_apply", args={"plan_id": "$cuts.plan_id"}),
        RecipeStep(tool="generate_music", id="music", args={"style": "uplifting", "bpm": 108}),
        RecipeStep(tool="add_sfx", id="sfx"),
        RecipeStep(tool="mixdown_audio", id="mix", args={"target": "reels"}),
        RecipeStep(tool="export_multi", id="export", args={"targets": ["reels_9x16", "square_1x1"]}),
    ],
))

register_recipe(RecipeV2(
    name="cinematic_vlog",
    description="Scene detect → auto color match → teal-orange look → drift moves → music",
    steps=[
        RecipeStep(tool="build_footage_index", id="index"),
        RecipeStep(tool="detect_scenes", id="scenes"),
        RecipeStep(tool="auto_color", id="color"),
        RecipeStep(tool="match_color", id="match", args={"reference": "shot:0"}, when="scenes.n_shots > 1"),
        RecipeStep(tool="apply_lut", id="lut", args={"lut": "teal_orange", "intensity": 0.65}),
        RecipeStep(tool="generate_music", id="music", args={"style": "cinematic"}),
        RecipeStep(tool="mixdown_audio", id="mix", args={"target": "youtube"}),
        RecipeStep(tool="composite_and_render", id="render"),
        RecipeStep(tool="add_camera_moves", id="moves", args={"style": "slow_drift", "max_zoom": 1.06}),
    ],
))

register_recipe(RecipeV2(
    name="lecture_to_shorts",
    description="Long lecture → top clips → vertical + captions + chapters metadata",
    steps=[
        RecipeStep(tool="build_footage_index", id="index"),
        RecipeStep(tool="detect_scenes", id="scenes"),
        RecipeStep(tool="extract_clips", id="clips", args={"n": 5, "max_sec": 58}),
        RecipeStep(tool="smart_reframe", id="reframe", args={"target": "9:16"}),
        RecipeStep(tool="generate_metadata", id="meta", args={"platform": "youtube"}),
    ],
))

register_recipe(RecipeV2(
    name="multilang_release",
    description="Dub to target languages + per-platform export matrix",
    params={"languages": {"default": ["es", "fr"], "doc": "ISO codes for dub tracks"}},
    steps=[
        RecipeStep(tool="build_footage_index", id="index"),
        RecipeStep(tool="dub_video", id="dub_es", args={"language": "es"}),
        RecipeStep(tool="dub_video", id="dub_fr", args={"language": "fr"}),
        RecipeStep(tool="export_multi", id="export", args={"targets": ["youtube_16x9", "reels_9x16"]}),
    ],
))

register_recipe(RecipeV2(
    name="music_video_beatcut",
    description="Beat-grid the BGM, cut on beat, vibrant look, riser SFX",
    steps=[
        RecipeStep(tool="build_footage_index", id="index"),
        RecipeStep(tool="generate_music", id="music", args={"style": "uplifting", "bpm": 100}),
        RecipeStep(tool="apply_lut", id="lut", args={"lut": "vibrant_punch", "intensity": 0.8}),
        RecipeStep(tool="add_sfx", id="sfx"),
        RecipeStep(tool="mixdown_audio", id="mix", args={"target": "youtube"}),
        RecipeStep(tool="composite_and_render", id="render"),
    ],
))


def list_recipes_v2() -> list[dict[str, Any]]:
    return [
        {"name": r.name, "description": r.description, "n_steps": len(r.steps), "requires": r.requires}
        for r in RECIPES_V2.values()
    ]
