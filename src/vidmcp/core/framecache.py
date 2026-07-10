"""Content-addressed op cache — skip recompute when inputs+params unchanged."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import orjson

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.framecache")


def file_fingerprint(path: Path | str) -> str:
    """Fast content fingerprint: size + mtime + head/tail bytes."""
    p = Path(path)
    if not p.exists():
        return f"missing:{p}"
    st = p.stat()
    h = hashlib.sha1()
    h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    if p.is_file():
        with p.open("rb") as f:
            h.update(f.read(65536))
            if st.st_size > 131072:
                f.seek(-65536, 2)
                h.update(f.read(65536))
    return h.hexdigest()


def op_key(op: str, params: dict[str, Any] | None, inputs: list[Path | str] | None = None) -> str:
    payload = {
        "op": op,
        "params": params or {},
        "inputs": [file_fingerprint(p) for p in (inputs or [])],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


class OpCache:
    """Per-project op result cache under project/cache/<key>/ (result.json + artifacts)."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root) / "cache"

    def _dir(self, key: str) -> Path:
        return self.root / key

    def get(self, key: str) -> dict[str, Any] | None:
        meta = self._dir(key) / "result.json"
        if not meta.exists():
            return None
        try:
            data = orjson.loads(meta.read_bytes())
            data["cache_hit"] = True
            return data
        except Exception:  # noqa: BLE001
            return None

    def put(self, key: str, result: dict[str, Any], artifacts: list[Path] | None = None) -> None:
        d = self._dir(key)
        d.mkdir(parents=True, exist_ok=True)
        stored = dict(result)
        if artifacts:
            copied = []
            for a in artifacts:
                a = Path(a)
                if a.exists():
                    dest = d / a.name
                    if a.is_dir():
                        if not dest.exists():
                            shutil.copytree(a, dest)
                    else:
                        shutil.copy2(a, dest)
                    copied.append(str(dest))
            stored["cached_artifacts"] = copied
        (d / "result.json").write_bytes(orjson.dumps(stored, option=orjson.OPT_INDENT_2))
        log.debug("op_cached", key=key)

    def stats(self) -> dict[str, Any]:
        if not self.root.exists():
            return {"entries": 0, "bytes": 0}
        entries = [p for p in self.root.iterdir() if p.is_dir()]
        size = sum(f.stat().st_size for p in entries for f in p.rglob("*") if f.is_file())
        return {"entries": len(entries), "bytes": size}

    def clear(self) -> int:
        if not self.root.exists():
            return 0
        n = len(list(self.root.iterdir()))
        shutil.rmtree(self.root)
        return n
