"""In-process job registry with disk checkpoints."""

from __future__ import annotations

import threading
from typing import Any

from vidmcp.core.workspace import ProjectStore
from vidmcp.models.jobs import JobRecord, JobStatus, JobType
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.jobs")


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    def create(
        self,
        project: ProjectStore,
        job_type: JobType,
        input_data: dict[str, Any] | None = None,
    ) -> JobRecord:
        job = JobRecord(project_id=project.manifest.id, job_type=job_type, input=input_data or {})
        with self._lock:
            self._jobs[job.id] = job
        self._persist(project, job)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, project: ProjectStore, job: JobRecord) -> None:
        with self._lock:
            self._jobs[job.id] = job
        self._persist(project, job)

    def _persist(self, project: ProjectStore, job: JobRecord) -> None:
        try:
            project.write_job(job.id, job.model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            log.warning("job_persist_failed", job_id=job.id, error=str(e))

    def list_for_project(self, project_id: str) -> list[JobRecord]:
        with self._lock:
            return [j for j in self._jobs.values() if j.project_id == project_id]


_global_jobs: JobManager | None = None


def get_job_manager() -> JobManager:
    global _global_jobs
    if _global_jobs is None:
        _global_jobs = JobManager()
    return _global_jobs
