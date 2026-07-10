"""Unified optional-model registry — lazy resolve/download for every heavy backend.

Every entry has a dependency-free fallback elsewhere in the codebase; this registry
only manages the *optional* quality-upgrade paths.
"""

from __future__ import annotations

import importlib.util
import urllib.request
from pathlib import Path
from typing import Any

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.models_registry")

CACHE_ROOT = Path.home() / ".cache" / "vidmcp" / "models"

# kind: pip = python package only; hf = HF weights (+optional pip); url = direct file
REGISTRY: dict[str, dict[str, Any]] = {
    "sam3": {
        "kind": "hf", "pip": ["torch"], "hf_repo": "facebook/sam3.1",
        "license": "sam-license", "purpose": "video segmentation (primary matte)",
    },
    "matanyone": {
        "kind": "hf", "pip": ["torch"], "hf_repo": "PeiqingYang/MatAnyone",
        "license": "non-commercial — review", "purpose": "hair-level video matting",
    },
    "rvm": {
        "kind": "url", "pip": ["onnxruntime"],
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx",
        "filename": "rvm_mobilenetv3_fp32.onnx", "license": "GPL-3.0",
        "purpose": "fast talking-head alpha matting",
    },
    "transnetv2": {
        "kind": "pip", "pip": ["scenedetect"], "license": "BSD",
        "purpose": "shot boundary detection (PySceneDetect adaptive)",
    },
    "pyannote": {
        "kind": "pip", "pip": ["pyannote.audio"], "license": "MIT (model gated)",
        "purpose": "speaker diarization",
    },
    "whisper_turbo": {
        "kind": "pip", "pip": ["faster_whisper"], "license": "MIT",
        "purpose": "ASR large-v3-turbo word timeline",
    },
    "audio_tagging": {
        "kind": "pip", "pip": ["panns_inference"], "license": "MIT",
        "purpose": "laughter/applause/music audio events",
    },
    "clip_embed": {
        "kind": "pip", "pip": ["sentence_transformers"], "license": "Apache-2.0",
        "purpose": "visual+text embeddings for semantic search",
    },
    "demucs": {
        "kind": "pip", "pip": ["demucs"], "license": "MIT",
        "purpose": "voice/music stem separation",
    },
    "silero_vad": {
        "kind": "url", "pip": ["onnxruntime"],
        "url": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        "filename": "silero_vad.onnx", "license": "MIT",
        "purpose": "speech probability for ducking v2",
    },
    "depth_anything_v2": {
        "kind": "pip", "pip": ["transformers", "torch"], "license": "Apache-2.0",
        "purpose": "monocular depth for parallax / fog",
    },
    "rife": {
        "kind": "pip", "pip": ["torch"], "license": "MIT",
        "purpose": "frame interpolation for smooth slow-mo",
    },
    "xtts": {
        "kind": "pip", "pip": ["TTS"], "license": "Coqui CPML — non-commercial, review",
        "purpose": "voice cloning TTS for dubbing (consent-gated)",
    },
    "musicgen": {
        "kind": "pip", "pip": ["audiocraft"], "license": "MIT (weights CC-BY-NC)",
        "purpose": "generative BGM",
    },
    "u2net_saliency": {
        "kind": "url", "pip": ["onnxruntime"],
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "filename": "u2netp.onnx", "license": "Apache-2.0",
        "purpose": "saliency for non-person reframe",
    },
    "mediapipe": {
        "kind": "pip", "pip": ["mediapipe"], "license": "Apache-2.0",
        "purpose": "selfie matte, face mesh (parts, thumbnails)",
    },
    "rubberband": {
        "kind": "binary", "binary": "rubberband", "license": "GPL-2.0",
        "purpose": "pitch-preserving speech time-stretch",
    },
}


def _pip_ok(mods: list[str]) -> bool:
    return all(importlib.util.find_spec(m.replace("-", "_")) is not None for m in mods)


def _binary_ok(name: str) -> bool:
    import shutil

    return shutil.which(name) is not None


def model_path(name: str) -> Path | None:
    entry = REGISTRY.get(name)
    if not entry or "filename" not in entry:
        return None
    p = CACHE_ROOT / name / entry["filename"]
    return p if p.exists() and p.stat().st_size > 1024 else None


def ensure_model(name: str, download: bool = False) -> dict[str, Any]:
    """Resolve one registry entry: report availability, optionally download url/hf weights."""
    entry = REGISTRY.get(name)
    if entry is None:
        return {"ok": False, "message": f"Unknown model '{name}'. Available: {sorted(REGISTRY)}"}
    kind = entry["kind"]
    pip_needed = entry.get("pip", [])
    pip_ok = _pip_ok(pip_needed) if pip_needed else True
    out: dict[str, Any] = {
        "ok": True,
        "name": name,
        "kind": kind,
        "purpose": entry.get("purpose"),
        "license": entry.get("license"),
        "pip_required": pip_needed,
        "pip_ok": pip_ok,
    }
    if kind == "binary":
        out["found"] = _binary_ok(entry["binary"])
        out["hint"] = None if out["found"] else f"Install '{entry['binary']}' binary (e.g. brew install {entry['binary']})"
        return out
    if kind == "pip":
        out["found"] = pip_ok
        out["hint"] = None if pip_ok else f"pip install {' '.join(pip_needed)}"
        return out
    # weight-backed kinds
    path = model_path(name)
    if path is None and download:
        try:
            dest = CACHE_ROOT / name / entry.get("filename", "weights.bin")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if kind == "url":
                log.info("model_download", name=name, url=entry["url"])
                urllib.request.urlretrieve(entry["url"], dest)
                path = dest
            elif kind == "hf":
                from huggingface_hub import snapshot_download

                local = snapshot_download(repo_id=entry["hf_repo"], cache_dir=str(CACHE_ROOT / name))
                path = Path(local)
        except Exception as e:  # noqa: BLE001
            out["download_error"] = str(e)
    out["found"] = path is not None and pip_ok
    out["path"] = str(path) if path else None
    if not out["found"]:
        hints = []
        if not pip_ok:
            hints.append(f"pip install {' '.join(pip_needed)}")
        if path is None:
            hints.append(f"ensure_model('{name}', download=True)")
        out["hint"] = "; ".join(hints)
    return out


def list_models() -> dict[str, Any]:
    items = []
    for name in sorted(REGISTRY):
        st = ensure_model(name)
        items.append(
            {
                "name": name,
                "found": st.get("found"),
                "purpose": st.get("purpose"),
                "license": st.get("license"),
                "hint": st.get("hint"),
            }
        )
    return {"ok": True, "cache_root": str(CACHE_ROOT), "models": items}
