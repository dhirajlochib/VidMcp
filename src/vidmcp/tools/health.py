"""Platform health / readiness report."""

from __future__ import annotations

import importlib.util
import shutil
from typing import Any

from vidmcp.config import get_settings
from vidmcp.perception.weights import describe_weights_status
from vidmcp.scenes.blender_backend import blender_available
from vidmcp.scenes.manim_backend import manim_available


def platform_health() -> dict[str, Any]:
    s = get_settings()
    checks = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
        "openai_whisper": importlib.util.find_spec("whisper") is not None,
        "mediapipe": importlib.util.find_spec("mediapipe") is not None,
        "mlx": importlib.util.find_spec("mlx") is not None,
        "manim": manim_available(),
        "blender": blender_available(),
        "ultralytics": importlib.util.find_spec("ultralytics") is not None,
        "sam3_official": importlib.util.find_spec("sam3") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "opentimelineio": importlib.util.find_spec("opentimelineio") is not None,
        "macos_say": shutil.which("say") is not None,
    }
    weights = describe_weights_status(s.sam_weights)
    ready_education = checks["ffmpeg"] and (checks["faster_whisper"] or checks["macos_say"])
    ready_creator = checks["ffmpeg"]  # denoise/bgm/export always; ASR/matte optional
    ready_sam = checks["sam3_official"] or checks["ultralytics"] or checks["mlx"]
    ready_sam_full = ready_sam and (bool(weights.get("found")) or checks["mlx"])
    from vidmcp import __version__

    return {
        "ok": True,
        "version": __version__,
        "checks": checks,
        "sam_weights": weights,
        "config": {
            "sam_backend": s.sam_backend.value,
            "sam_use_multiplex": s.sam_use_multiplex,
            "workspace": str(s.workspace_root),
            "device": s.device,
            "tool_pack": s.tool_pack,
            "compact": s.compact,
            "max_result_chars": s.max_result_chars,
            "matte_quality": s.matte_quality,
        },
        "readiness": {
            "education_path": ready_education,
            "creator_path": ready_creator,
            "creator_asr": checks["faster_whisper"] or checks["openai_whisper"],
            "creator_matte_fast": checks["mediapipe"],
            "sam_package": ready_sam,
            "sam_with_weights": ready_sam_full,
            "live_mock": checks["ffmpeg"],
            "depth_neural": checks["transformers"] and checks["torch"],
        },
        "next_steps": _next_steps(checks, weights, ready_sam_full, ready_education),
    }


def _next_steps(checks: dict, weights: dict, sam_full: bool, edu: bool) -> list[str]:
    steps = []
    if not checks["ffmpeg"]:
        steps.append("Install ffmpeg (brew install ffmpeg)")
    if not checks["faster_whisper"]:
        steps.append("pip install faster-whisper  # better ASR for captions")
    if not checks.get("mediapipe"):
        steps.append("pip install mediapipe  # fast talking-head BG replace")
    if not sam_full:
        if not checks["sam3_official"] and not checks["ultralytics"]:
            steps.append("Optional: pip install ultralytics OR install facebookresearch/sam3")
        if not weights.get("found"):
            steps.append("Request HF access + set VIDMCP_SAM_WEIGHTS or ensure_sam_weights(download=True)")
    if edu:
        steps.append("Run: python examples/education_lesson_demo.py")
    if not checks["manim"]:
        steps.append("Optional: pip install manim for higher-end math scenes")
    return steps
