"""In-memory async task store for docling-serve compatibility.

The official ``DoclingServiceClient`` always uses the async task flow:
  1. POST ``/v1/convert/file/async`` (or ``/v1/convert/source/async``) → submit
  2. GET  ``/v1/status/poll/{task_id}?wait=N``                → poll status
  3. GET  ``/v1/result/{task_id}``                             → fetch result

PaddleX HPS performs **synchronous** conversion, but to be wire-compatible
with the official client SDK we implement the async protocol as a thin
wrapper: the conversion runs synchronously during submission, the result is
cached in-memory, and the poll/result endpoints return immediately.

This is acceptable because:
  - PaddleX HPS processes one document at a time (no real queue).
  - The conversion completes in <1s for typical images.
  - The client polls with ``wait=5s`` and gets the terminal status on the
    first poll.

The store is a simple ``dict`` protected by an ``asyncio.Lock``.  Tasks are
auto-evicted after ``TASK_TTL_SECONDS`` to prevent unbounded memory growth.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.service.responses import (
    PublicFailureInfo,
    TaskStatusResponse,
)
from docling.datamodel.service.tasks import TaskProcessingMeta, TaskType

from .schema import ConvertDocumentResponse

logger = logging.getLogger("hps_api")

# Tasks auto-expire after 5 minutes to prevent unbounded memory growth.
TASK_TTL_SECONDS = 300


@dataclass
class _TaskEntry:
    """Internal task record stored in the task store."""

    task_id: str
    status: ConversionStatus
    response: Optional[ConvertDocumentResponse] = None
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


class TaskStore:
    """Thread-safe in-memory store for async conversion tasks.

    A single global instance (``task_store``) is shared across all requests.
    Access is serialized via an ``asyncio.Lock`` to prevent races between
    concurrent poll/result requests for the same task.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, _TaskEntry] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        status: ConversionStatus,
        response: Optional[ConvertDocumentResponse] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """Register a completed (or failed) conversion and return its task ID."""
        task_id = uuid.uuid4().hex
        entry = _TaskEntry(
            task_id=task_id,
            status=status,
            response=response,
            error_message=error_message,
        )
        async with self._lock:
            self._tasks[task_id] = entry
            self._evict_expired()
        logger.debug("Created task %s with status %s", task_id, status)
        return task_id

    async def get_status(self, task_id: str) -> TaskStatusResponse:
        """Return the current status of a task.

        Raises ``KeyError`` if the task does not exist (caller maps to 404).
        """
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise KeyError(task_id)
            return self._build_status_response(entry)

    async def get_result(self, task_id: str) -> ConvertDocumentResponse:
        """Return the conversion result for a completed task.

        Raises ``KeyError`` if the task does not exist (caller maps to 404).
        Raises ``RuntimeError`` if the task has no result (shouldn't happen
        for successful tasks).
        """
        async with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise KeyError(task_id)
            if entry.response is None:
                raise RuntimeError(
                    f"Task {task_id} has no result (status={entry.status})"
                )
            return entry.response

    def _evict_expired(self) -> None:
        """Remove tasks older than ``TASK_TTL_SECONDS``.  Caller holds lock."""
        now = time.monotonic()
        expired = [
            tid
            for tid, entry in self._tasks.items()
            if now - entry.created_at > TASK_TTL_SECONDS
        ]
        for tid in expired:
            del self._tasks[tid]
        if expired:
            logger.debug("Evicted %d expired tasks", len(expired))

    @staticmethod
    def _build_status_response(entry: _TaskEntry) -> TaskStatusResponse:
        """Build a ``TaskStatusResponse`` from an internal task entry."""
        failure: Optional[PublicFailureInfo] = None
        if entry.status == ConversionStatus.FAILURE and entry.error_message:
            failure = PublicFailureInfo(
                category="INFERENCE_FAILURE",
                message=entry.error_message,
                retryable=False,
            )

        return TaskStatusResponse(
            task_id=entry.task_id,
            task_type=TaskType.CONVERT,
            task_status=entry.status,
            task_position=0,
            task_meta=TaskProcessingMeta(
                num_docs=1,
                num_processed=1,
                num_succeeded=1 if entry.status == ConversionStatus.SUCCESS else 0,
                num_partially_succeeded=1
                if entry.status == ConversionStatus.PARTIAL_SUCCESS
                else 0,
                num_failed=1 if entry.status == ConversionStatus.FAILURE else 0,
            ),
            error_message=entry.error_message,
            failure=failure,
        )


# Global singleton — shared across all async requests.
task_store = TaskStore()
