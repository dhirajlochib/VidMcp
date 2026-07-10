"""Disk-backed project workspace with atomic manifest writes."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson
from filelock import FileLock

from vidmcp.config import Settings, get_settings
from vidmcp.core.security import (
    SecurityError,
    assert_video_extension,
    resolve_under,
    validate_project_name,
)
from vidmcp.models.project import ProjectManifest, ProjectStatus
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.workspace")

MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".manifest.lock"


class ProjectStore:
    """One editing project on disk."""

    def __init__(self, root: Path, manifest: ProjectManifest):
        self.root = root.resolve()
        self.manifest = manifest
        self._lock = FileLock(str(self.root / LOCK_NAME))

    # --- paths ---
    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def layers_dir(self) -> Path:
        return self.root / "layers"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    @property
    def previews_dir(self) -> Path:
        return self.root / "previews"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    def rel(self, path: Path | str) -> str:
        p = Path(path).resolve()
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def abs(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return (self.root / p).resolve()

    def ensure_layout(self) -> None:
        for d in (
            self.source_dir,
            self.masks_dir,
            self.layers_dir,
            self.renders_dir,
            self.previews_dir,
            self.jobs_dir,
            self.tmp_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.ensure_layout()
        path = self.root / MANIFEST_NAME
        payload = self.manifest.model_dump(mode="json")
        data = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
        with self._lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        log.debug("manifest_saved", project_id=self.manifest.id, version=self.manifest.version)

    def import_video(self, video_path: Path | str, *, copy: bool = True) -> Path:
        src = Path(video_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Video not found: {src}")
        assert_video_extension(src)
        self.ensure_layout()
        dest = self.source_dir / f"source{src.suffix.lower()}"
        if copy:
            shutil.copy2(src, dest)
        else:
            # symlink when allowed
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src)
        self.manifest.source_video = self.rel(dest)
        self.manifest.status = ProjectStatus.CREATED
        self.manifest.append_history("import_video", {"path": str(src), "stored": self.manifest.source_video})
        self.save()
        return dest

    def write_job(self, job_id: str, payload: dict[str, Any]) -> Path:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        path = self.jobs_dir / f"{job_id}.json"
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        return path


class Workspace:
    """Root workspace containing many projects."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.root = self.settings.workspace_root
        self.root.mkdir(parents=True, exist_ok=True)

    def project_path(self, project_id: str) -> Path:
        # projects live at workspaces/{id}/
        safe = project_id.replace("/", "_").replace("..", "_")
        return self.root / safe

    def create_project(self, name: str = "untitled", project_id: str | None = None) -> ProjectStore:
        name = validate_project_name(name) if name else "untitled"
        pid = project_id or str(uuid4())
        root = self.project_path(pid)
        if root.exists() and (root / MANIFEST_NAME).exists():
            raise SecurityError(f"Project already exists: {pid}")
        root.mkdir(parents=True, exist_ok=True)
        manifest = ProjectManifest(id=pid, name=name)
        store = ProjectStore(root, manifest)
        store.ensure_layout()
        store.save()
        log.info("project_created", project_id=pid, name=name, root=str(root))
        return store

    def load_project(self, project_id: str) -> ProjectStore:
        root = self.project_path(project_id)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        data = orjson.loads(manifest_path.read_bytes())
        manifest = ProjectManifest.model_validate(data)
        store = ProjectStore(root, manifest)
        store.ensure_layout()
        return store

    def list_projects(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for child in sorted(self.root.iterdir()):
            mp = child / MANIFEST_NAME
            if not mp.exists():
                continue
            try:
                data = orjson.loads(mp.read_bytes())
                items.append(
                    {
                        "id": data.get("id", child.name),
                        "name": data.get("name"),
                        "status": data.get("status"),
                        "updated_at": data.get("updated_at"),
                        "source_video": data.get("source_video"),
                    }
                )
            except Exception as e:  # noqa: BLE001
                log.warning("skip_corrupt_project", path=str(child), error=str(e))
        return items

    def resolve_import_path(self, path: str) -> Path:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            # try relative to workspace
            alt = resolve_under(self.root, path, must_exist=False)
            if alt.exists():
                return alt
            raise FileNotFoundError(path)
        return p
