"""EchoProof AI - Local in-memory job manager for async processing."""
from __future__ import annotations
import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Coroutine, Optional

from app.config import MAX_CONCURRENT_JOBS
from app.schemas import JobStatus, JobInfo

logger = logging.getLogger(__name__)

# Maximum jobs to keep in memory
MAX_JOB_HISTORY = 100


class JobManager:
    """Simple bounded in-memory job queue."""

    def __init__(self):
        self._jobs: OrderedDict[str, JobInfo] = OrderedDict()
        self._results: dict[str, Any] = {}
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        self._events: dict[str, list[dict]] = {}

    def create_job(self, job_type: str) -> str:
        """Create a new job and return its ID."""
        job_id = uuid.uuid4().hex[:12]
        job = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            job_type=job_type,
            created_at=time.time(),
        )
        self._jobs[job_id] = job
        self._events[job_id] = []

        # Evict old jobs
        while len(self._jobs) > MAX_JOB_HISTORY:
            old_id, _ = self._jobs.popitem(last=False)
            self._results.pop(old_id, None)
            self._events.pop(old_id, None)

        return job_id

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        return self._jobs.get(job_id)

    def update_job(
        self, job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[Any] = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if message is not None:
            job.message = message
        if result is not None:
            job.result = result
            self._results[job_id] = result

        # Add event
        self._events.setdefault(job_id, []).append({
            "time": time.time(),
            "status": job.status.value,
            "progress": job.progress,
            "message": job.message,
        })

    def get_result(self, job_id: str) -> Optional[Any]:
        return self._results.get(job_id)

    def get_events(self, job_id: str) -> list[dict]:
        return self._events.get(job_id, [])

    async def run_job(
        self, job_id: str,
        func: Callable[..., Coroutine],
        *args, **kwargs
    ) -> None:
        """Run a job with concurrency limiting."""
        async with self._semaphore:
            self.update_job(job_id, status=JobStatus.PROCESSING, message="Processing...")
            try:
                result = await func(job_id, *args, **kwargs)
                self.update_job(
                    job_id, status=JobStatus.COMPLETED,
                    progress=1.0, message="Complete", result=result
                )
            except asyncio.CancelledError:
                self.update_job(
                    job_id, status=JobStatus.CANCELLED, message="Cancelled"
                )
            except Exception as e:
                logger.exception(f"Job {job_id} failed")
                self.update_job(
                    job_id, status=JobStatus.FAILED, message=str(e)
                )


# Singleton
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
