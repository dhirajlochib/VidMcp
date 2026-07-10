"""File-based durable job queue with worker loop (no Redis required).

Jobs land in workspace/.vidmcp/queue/{pending,running,done,failed}/
Supports enqueue of named handlers; worker can run in-process or subprocess.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson
from filelock import FileLock

from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.queue")

Handler = Callable[[dict[str, Any]], dict[str, Any]]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class JobQueue:
    def __init__(self, root: Path):
        self.root = Path(root) / ".vidmcp" / "queue"
        for name in ("pending", "running", "done", "failed"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self._handlers: dict[str, Handler] = {}
        self._lock = FileLock(str(self.root / ".queue.lock"))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, name: str, fn: Handler) -> None:
        self._handlers[name] = fn

    def enqueue(
        self,
        handler: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 100,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        job = {
            "id": str(uuid4()),
            "handler": handler,
            "payload": payload or {},
            "priority": priority,
            "project_id": project_id,
            "status": "pending",
            "created_at": _utcnow(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "attempts": 0,
        }
        path = self.root / "pending" / f"{priority:04d}_{job['id']}.json"
        path.write_bytes(orjson.dumps(job, option=orjson.OPT_INDENT_2))
        log.info("job_enqueued", job_id=job["id"], handler=handler)
        return {"ok": True, "job_id": job["id"], "status": "pending", "path": str(path)}

    def get(self, job_id: str) -> dict[str, Any] | None:
        for folder in ("pending", "running", "done", "failed"):
            for p in (self.root / folder).glob(f"*_{job_id}.json"):
                return orjson.loads(p.read_bytes())
            # also exact id filename for done/failed
            p2 = self.root / folder / f"{job_id}.json"
            if p2.exists():
                return orjson.loads(p2.read_bytes())
        return None

    def list_jobs(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        folders = [status] if status else ["pending", "running", "done", "failed"]
        rows = []
        for folder in folders:
            d = self.root / folder
            if not d.exists():
                continue
            for p in sorted(d.glob("*.json"), reverse=True):
                try:
                    rows.append(orjson.loads(p.read_bytes()))
                except Exception:
                    continue
                if len(rows) >= limit:
                    return rows
        return rows[:limit]

    def _claim_next(self) -> Path | None:
        with self._lock:
            pending = sorted((self.root / "pending").glob("*.json"))
            if not pending:
                return None
            src = pending[0]
            data = orjson.loads(src.read_bytes())
            data["status"] = "running"
            data["started_at"] = _utcnow()
            data["attempts"] = int(data.get("attempts") or 0) + 1
            dest = self.root / "running" / f"{data['id']}.json"
            dest.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
            src.unlink(missing_ok=True)
            return dest

    def process_one(self) -> dict[str, Any] | None:
        claimed = self._claim_next()
        if not claimed:
            return None
        job = orjson.loads(claimed.read_bytes())
        handler_name = job.get("handler")
        fn = self._handlers.get(handler_name)
        try:
            if not fn:
                raise RuntimeError(f"Unknown handler: {handler_name}")
            result = fn(job.get("payload") or {})
            job["status"] = "done"
            job["result"] = result
            job["finished_at"] = _utcnow()
            dest = self.root / "done" / f"{job['id']}.json"
            dest.write_bytes(orjson.dumps(job, option=orjson.OPT_INDENT_2))
            claimed.unlink(missing_ok=True)
            log.info("job_done", job_id=job["id"], handler=handler_name)
            return job
        except Exception as e:  # noqa: BLE001
            job["status"] = "failed"
            job["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
            job["finished_at"] = _utcnow()
            dest = self.root / "failed" / f"{job['id']}.json"
            dest.write_bytes(orjson.dumps(job, option=orjson.OPT_INDENT_2))
            claimed.unlink(missing_ok=True)
            log.warning("job_failed", job_id=job["id"], error=str(e))
            return job

    def run_worker(self, *, poll_sec: float = 0.5, max_jobs: int | None = None) -> int:
        n = 0
        while not self._stop.is_set():
            job = self.process_one()
            if job is None:
                time.sleep(poll_sec)
                continue
            n += 1
            if max_jobs is not None and n >= max_jobs:
                break
        return n

    def start_background(self, poll_sec: float = 0.5) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            self.run_worker(poll_sec=poll_sec)

        self._thread = threading.Thread(target=_loop, name="vidmcp-queue-worker", daemon=True)
        self._thread.start()
        log.info("queue_worker_started")

    def stop_background(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


_queues: dict[str, JobQueue] = {}


def get_job_queue(root: Path | None = None) -> JobQueue:
    from vidmcp.config import get_settings

    root = Path(root or get_settings().workspace_root)
    key = str(root.resolve())
    if key not in _queues:
        q = JobQueue(root)
        _register_default_handlers(q)
        _queues[key] = q
    return _queues[key]


def _register_default_handlers(q: JobQueue) -> None:
    def _composite(payload: dict[str, Any]) -> dict[str, Any]:
        from vidmcp.config import get_settings
        from vidmcp.core.workspace import Workspace
        from vidmcp.tools import service

        ws = Workspace(get_settings())
        project = ws.load_project(payload["project_id"])
        return service.composite(project, max_frames=payload.get("max_frames"))

    def _segment(payload: dict[str, Any]) -> dict[str, Any]:
        from vidmcp.config import get_settings
        from vidmcp.core.workspace import Workspace
        from vidmcp.tools import service

        ws = Workspace(get_settings())
        project = ws.load_project(payload["project_id"])
        return service.segment(project, prompt=payload.get("prompt") or "person", conf=payload.get("conf"))

    def _enhance(payload: dict[str, Any]) -> dict[str, Any]:
        from vidmcp.config import get_settings
        from vidmcp.core.workspace import Workspace
        from vidmcp.tools import advanced_service as adv
        from vidmcp.tools import service

        ws = Workspace(get_settings())
        project = ws.load_project(payload["project_id"])
        # mini enhance
        try:
            adv.uncertainty_guided_refine(project, service)
        except Exception:
            pass
        return service.composite(project, max_frames=payload.get("max_frames"))

    def _education(payload: dict[str, Any]) -> dict[str, Any]:
        from vidmcp.education.pipeline import run_education_lesson

        return run_education_lesson(
            video_path=payload.get("video_path"),
            lesson_topic=payload.get("lesson_topic")
            or payload.get("lesson_prompt")
            or payload.get("intent")
            or "lesson",
            project_name=payload.get("project_name") or "queued_lesson",
            max_render_frames=payload.get("max_frames"),
        )

    q.register("composite", _composite)
    q.register("segment", _segment)
    q.register("enhance", _enhance)
    q.register("education_lesson", _education)
