"""Path sandboxing and input validation."""

from __future__ import annotations

import re
from pathlib import Path


class SecurityError(Exception):
    pass


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")


def validate_project_name(name: str) -> str:
    name = name.strip()
    if not name or not _SAFE_NAME.match(name.replace(" ", "_")):
        # allow spaces by normalizing
        normalized = re.sub(r"[^A-Za-z0-9._\-]+", "_", name).strip("_")
        if not normalized or len(normalized) > 128:
            raise SecurityError(f"Invalid project name: {name!r}")
        return normalized
    return name.replace(" ", "_")


def resolve_under(root: Path, user_path: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve path and ensure it stays under root (no path traversal)."""
    root = root.resolve()
    p = Path(user_path)
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError as e:
        raise SecurityError(f"Path escapes workspace root: {p}") from e
    if must_exist and not p.exists():
        raise FileNotFoundError(str(p))
    return p


def assert_video_extension(path: Path) -> None:
    allowed = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    if path.suffix.lower() not in allowed:
        raise SecurityError(f"Unsupported video extension: {path.suffix}")
