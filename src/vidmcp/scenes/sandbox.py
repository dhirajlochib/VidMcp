"""Restricted code execution for agent-authored scene scripts."""

from __future__ import annotations

import ast
import re

# Deny dangerous constructs in untrusted agent code
_FORBIDDEN_CALLS = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "exit",
    "quit",
}
_FORBIDDEN_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "http",
    "urllib",
    "requests",
    "ctypes",
    "multiprocessing",
    "pickle",
    "importlib",
    "builtins",
}


class SandboxError(ValueError):
    pass


def validate_scene_source(source: str, *, max_chars: int = 40_000) -> None:
    if not source or not source.strip():
        raise SandboxError("Empty scene source")
    if len(source) > max_chars:
        raise SandboxError(f"Scene source too large ({len(source)} > {max_chars})")
    # quick string bans
    lowered = source.lower()
    for bad in ("__import__", "subprocess", "socket.socket", "eval(", "exec("):
        if bad in lowered:
            raise SandboxError(f"Forbidden pattern: {bad}")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SandboxError(f"Syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    raise SandboxError(f"Import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    raise SandboxError(f"Import not allowed: {node.module}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in _FORBIDDEN_CALLS:
                raise SandboxError(f"Call not allowed: {name}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                if node.attr not in {"__init__", "__name__", "__class__"}:
                    # block dunder probing
                    if node.attr in {"__subclasses__", "__globals__", "__code__", "__builtins__"}:
                        raise SandboxError(f"Attribute not allowed: {node.attr}")


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def ensure_safe_prompt_slug(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text.strip().lower())[:max_len].strip("_")
    return slug or "scene"
