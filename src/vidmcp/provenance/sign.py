"""Cryptographic render provenance manifests (HMAC-SHA256)."""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from vidmcp.core.workspace import ProjectStore


def _secret() -> bytes:
    return (os.environ.get("VIDMCP_PROVENANCE_SECRET") or "vidmcp-dev-secret-change-me").encode()


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(project: ProjectStore) -> dict[str, Any]:
    m = project.manifest
    render = m.renders[-1] if m.renders else None
    render_hash = None
    if render:
        render_hash = _file_sha256(project.abs(render["output_path"]))
    payload = {
        "vidmcp_version": "0.4.0",
        "project_id": m.id,
        "project_name": m.name,
        "created_at": datetime.now(UTC).isoformat(),
        "source": m.source_video,
        "source_sha256": _file_sha256(project.abs(m.source_video)) if m.source_video else None,
        "primary_segment": m.primary_segment_id,
        "segments": [
            {"id": s.id, "prompt": s.prompt, "backend": s.backend, "frame_count": s.frame_count}
            for s in m.segments
        ],
        "layers": m.layers.model_dump(mode="json"),
        "render": render,
        "render_sha256": render_hash,
        "edit_history_tail": m.edit_history[-20:],
        "analysis_hints": (m.analysis or {}).get("scene_hints"),
    }
    return payload


def sign_payload(payload: dict[str, Any]) -> dict[str, Any]:
    body = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    sig = hmac.new(_secret(), body, hashlib.sha256).hexdigest()
    return {
        **payload,
        "signature": {"alg": "HMAC-SHA256", "value": sig},
    }


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    sig = (manifest.get("signature") or {}).get("value")
    if not sig:
        return {"ok": False, "valid": False, "message": "no signature"}
    body = {k: v for k, v in manifest.items() if k != "signature"}
    expect = hmac.new(_secret(), orjson.dumps(body, option=orjson.OPT_SORT_KEYS), hashlib.sha256).hexdigest()
    valid = hmac.compare_digest(sig, expect)
    return {"ok": True, "valid": valid, "alg": "HMAC-SHA256"}


def sign_project_render(project: ProjectStore) -> dict[str, Any]:
    payload = build_manifest(project)
    signed = sign_payload(payload)
    out = project.root / "provenance" / f"manifest_{(project.manifest.renders[-1]['render_id'][:8] if project.manifest.renders else 'none')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(signed, option=orjson.OPT_INDENT_2))
    project.manifest.append_history("sign_render", {"path": project.rel(out), "render_sha256": signed.get("render_sha256")})
    project.save()
    return {
        "ok": True,
        "manifest_path": project.rel(out),
        "absolute_path": str(out),
        "render_sha256": signed.get("render_sha256"),
        "signature": signed["signature"]["value"][:16] + "...",
        "verify": verify_manifest(signed),
    }
