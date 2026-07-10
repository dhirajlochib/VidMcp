"""Agent tool packs — preferred tool subsets for common intents."""

from __future__ import annotations

PACKS: dict[str, dict] = {
    "talking_head": {
        "description": "Polish a talk-head clip: audio, captions, optional BG, export",
        "tools": [
            "create_project",
            "import_video",
            "process_audio",
            "mix_bgm",
            "transcribe_and_caption",
            "replace_background",
            "smart_cut_hesitations",
            "export_render",
            "run_talking_head_polish",
            "apply_recipe",
            "get_backend_info",
            "platform_health",
        ],
    },
    "education": {
        "description": "Math / education plates + speech-locked scenes",
        "tools": [
            "create_project",
            "import_video",
            "render_math_scene",
            "talking_head_math_lesson",
            "transcribe_and_caption",
            "add_speech_infographics",
            "run_education_lesson",
            "apply_recipe",
        ],
    },
    "vfx_matte": {
        "description": "Subject matte + behind-subject effects",
        "tools": [
            "segment_subject",
            "refine_segment_keyframes",
            "replace_background",
            "apply_background_effects",
            "composite_and_render",
            "matte_diagnostics",
        ],
    },
}


def list_packs() -> list[dict]:
    return [{"name": k, **v} for k, v in PACKS.items()]


def get_pack(name: str) -> dict:
    if name not in PACKS:
        raise KeyError(f"Unknown pack {name}. Available: {sorted(PACKS)}")
    return {"name": name, **PACKS[name]}
