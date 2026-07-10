"""Agent tool packs — limit MCP tool surface to protect context."""

from __future__ import annotations

from typing import Any

# Always keep these even when a pack is active
ALWAYS_TOOLS: frozenset[str] = frozenset(
    {
        "list_tool_packs",
        "set_tool_pack",
        "project_brief",
        "run_intent",
        "list_projects",
        "get_project",
        "create_project",
        "get_backend_info",
        "platform_health",
        "list_recipes",
    }
)

PACKS: dict[str, dict[str, Any]] = {
    "talking_head": {
        "description": "Talk-head polish (default). Prefer run_talking_head_polish / run_intent.",
        "tools": [
            "import_video",
            "process_audio",
            "mix_bgm",
            "transcribe_and_caption",
            "replace_background",
            "smart_cut_hesitations",
            "export_render",
            "run_talking_head_polish",
            "apply_recipe",
            "generate_thumbnail",
            "export_edl",
            "add_speech_infographics",
        ],
    },
    "education": {
        "description": "Math / education plates + speech-locked lessons",
        "tools": [
            "import_video",
            "analyze_video",
            "render_math_scene",
            "talking_head_math_lesson",
            "transcribe_and_caption",
            "add_speech_infographics",
            "run_education_lesson",
            "run_fast_education_harness",
            "apply_recipe",
            "export_render",
            "attach_narration",
        ],
    },
    "vfx": {
        "description": "Subject matte + behind-subject effects + composite",
        "tools": [
            "import_video",
            "analyze_video",
            "segment_subject",
            "segment_multi_objects",
            "refine_segment_keyframes",
            "replace_background",
            "apply_background_effects",
            "composite_and_render",
            "matte_diagnostics",
            "evaluate_quality_gates",
            "list_effects",
            "export_render",
            "apply_recipe",
        ],
    },
    "admin": {
        "description": "Health, weights, queue, marketplace",
        "tools": [
            "ensure_sam_weights",
            "enqueue_job",
            "queue_status",
            "start_queue_worker",
            "list_marketplace_recipes",
            "plan_harness",
        ],
    },
    "all": {
        "description": "Full tool surface (high context cost — avoid for agents)",
        "tools": [],  # empty means no filter
    },
}


def list_packs() -> list[dict[str, Any]]:
    return [
        {
            "name": k,
            "description": v["description"],
            "n_tools": len(v["tools"]) if k != "all" else "all",
            "tools": v["tools"] if k != "all" else ["*"],
        }
        for k, v in PACKS.items()
    ]


def get_pack(name: str) -> dict[str, Any]:
    key = (name or "talking_head").strip().lower()
    if key == "vfx_matte":
        key = "vfx"
    if key not in PACKS:
        raise KeyError(f"Unknown pack {name}. Available: {sorted(PACKS)}")
    return {"name": key, **PACKS[key]}


def allowed_tool_names(pack: str | None = None) -> set[str] | None:
    """None means allow all tools."""
    from vidmcp.config import get_settings

    key = (pack or get_settings().tool_pack or "talking_head").strip().lower()
    if key in ("all", "full", "*"):
        return None
    if key == "vfx_matte":
        key = "vfx"
    if key not in PACKS or key == "all":
        return None
    names = set(PACKS[key]["tools"]) | set(ALWAYS_TOOLS)
    return names


def apply_tool_pack_filter(mcp: Any, pack: str | None = None) -> dict[str, Any]:
    """Remove tools not in the active pack from a FastMCP server."""
    from vidmcp.config import get_settings

    settings = get_settings()
    key = (pack or settings.tool_pack or "talking_head").strip().lower()
    allowed = allowed_tool_names(key)
    removed: list[str] = []
    kept: list[str] = []

    # list registered tool names via sync helper if possible
    try:
        import asyncio

        async def _names() -> list[str]:
            tools = await mcp.list_tools()
            return [t.name for t in tools]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # fallback: try local provider components
                names = _tool_names_from_provider(mcp)
            else:
                names = loop.run_until_complete(_names())
        except RuntimeError:
            names = asyncio.run(_names())
    except Exception:
        names = _tool_names_from_provider(mcp)

    if allowed is None:
        return {
            "ok": True,
            "pack": "all",
            "kept": len(names),
            "removed": 0,
            "tools": sorted(names),
        }

    provider = getattr(mcp, "local_provider", None) or getattr(mcp, "_local_provider", None)
    for name in names:
        if name in allowed:
            kept.append(name)
            continue
        try:
            if provider is not None and hasattr(provider, "remove_tool"):
                provider.remove_tool(name)
            else:
                mcp.remove_tool(name)
            removed.append(name)
        except Exception:
            # some tools may be versioned / protected
            kept.append(name)

    return {
        "ok": True,
        "pack": key,
        "kept": len(kept),
        "removed": len(removed),
        "tools": sorted(set(kept)),
        "removed_tools": sorted(removed)[:40],
    }


def _tool_names_from_provider(mcp: Any) -> list[str]:
    names: list[str] = []
    provider = getattr(mcp, "local_provider", None) or getattr(mcp, "_local_provider", None)
    if provider is None:
        return names
    # try common attribute layouts
    for attr in ("_tools", "tools", "_components", "components"):
        bag = getattr(provider, attr, None)
        if isinstance(bag, dict):
            for k in bag:
                # keys may be "tool:name@" or similar
                s = str(k)
                if s.startswith("tool:"):
                    s = s[5:]
                s = s.split("@")[0]
                names.append(s)
            if names:
                return names
    return names
