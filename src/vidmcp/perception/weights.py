"""SAM 3 / 3.1 checkpoint discovery, validation, and optional HF download."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.perception.weights")

# Common checkpoint filenames for SAM 3 / 3.1 multiplex
CANDIDATE_NAMES = (
    "sam3.1_multiplex.pt",
    "sam3.1_multiplex_fp16.safetensors",
    "sam3.1.pt",
    "sam3_multiplex.pt",
    "sam3.pt",
    "sam3.1_multiplex.pth",
)


def default_search_dirs(explicit: Path | None = None) -> list[Path]:
    dirs: list[Path] = []
    if explicit:
        dirs.append(Path(explicit).expanduser().resolve())
        if explicit.is_file():
            dirs.append(explicit.parent)
    env = os.environ.get("VIDMCP_SAM_WEIGHTS")
    if env:
        p = Path(env).expanduser().resolve()
        dirs.append(p if p.is_dir() else p.parent)
        if p.is_file():
            dirs.insert(0, p)
    for key in ("VIDMCP_WEIGHTS_DIR", "SAM3_CHECKPOINT_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(key)
        if v:
            dirs.append(Path(v).expanduser().resolve())
    # project-local
    cwd = Path.cwd()
    dirs.extend(
        [
            cwd / "weights",
            cwd / "models",
            cwd / "checkpoints",
            Path.home() / ".cache" / "vidmcp" / "weights",
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )
    # dedupe preserve order
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        s = str(d)
        if s not in seen:
            seen.add(s)
            out.append(d)
    return out


def resolve_sam_weights(explicit: Path | str | None = None) -> Path | None:
    """Return path to a usable checkpoint file, or None."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.is_file() and p.stat().st_size > 1_000_000:
            return p
        if p.is_dir():
            hit = _find_in_dir(p)
            if hit:
                return hit

    for d in default_search_dirs(Path(explicit) if explicit else None):
        if d.is_file() and d.stat().st_size > 1_000_000:
            return d
        if d.is_dir():
            hit = _find_in_dir(d)
            if hit:
                return hit
    # HF hub snapshot layout: models--facebook--sam3.1/...
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hub.is_dir():
        for pattern in ("models--facebook--sam3*", "models--facebook--sam3.1*"):
            for root in hub.glob(pattern):
                hit = _find_in_dir(root)
                if hit:
                    return hit
    return None


def _find_in_dir(root: Path) -> Path | None:
    for name in CANDIDATE_NAMES:
        direct = root / name
        if direct.is_file() and direct.stat().st_size > 1_000_000:
            return direct
    # recursive limited
    try:
        for name in CANDIDATE_NAMES:
            matches = list(root.rglob(name))
            matches = [m for m in matches if m.is_file() and m.stat().st_size > 1_000_000]
            if matches:
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return matches[0]
    except OSError:
        pass
    return None


def describe_weights_status(explicit: Path | str | None = None) -> dict[str, Any]:
    found = resolve_sam_weights(explicit)
    return {
        "found": found is not None,
        "path": str(found) if found else None,
        "size_mb": round(found.stat().st_size / 1e6, 1) if found else None,
        "search_hints": [str(p) for p in default_search_dirs(Path(explicit) if explicit else None)[:8]],
        "candidate_names": list(CANDIDATE_NAMES),
        "hf_repos": ["facebook/sam3", "facebook/sam3.1"],
        "install_hint": (
            "Request access on Hugging Face, then: "
            "huggingface-cli download facebook/sam3.1 --local-dir ./weights/sam3.1 "
            "or set VIDMCP_SAM_WEIGHTS=/path/to/sam3.1_multiplex.pt"
        ),
    }


def try_download_sam_weights(
    *,
    repo_id: str = "facebook/sam3.1",
    local_dir: Path | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Best-effort download via huggingface_hub (requires access approval + token)."""
    local_dir = Path(local_dir or (Path.home() / ".cache" / "vidmcp" / "weights" / "sam3.1"))
    local_dir.mkdir(parents=True, exist_ok=True)
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {
            "ok": False,
            "message": "huggingface_hub not installed. pip install huggingface_hub",
            "local_dir": str(local_dir),
        }
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            resume_download=True,
        )
        found = resolve_sam_weights(Path(path))
        return {
            "ok": True,
            "repo_id": repo_id,
            "local_dir": path,
            "weights": str(found) if found else None,
            "message": "Download complete" if found else "Downloaded repo but no matching checkpoint filename found",
        }
    except Exception as e:  # noqa: BLE001
        log.warning("hf_download_failed", error=str(e), repo=repo_id)
        return {
            "ok": False,
            "repo_id": repo_id,
            "local_dir": str(local_dir),
            "message": str(e),
            "hint": "Ensure HF access is approved for the gated model and HF_TOKEN is set",
        }
