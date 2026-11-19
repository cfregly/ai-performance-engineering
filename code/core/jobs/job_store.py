"""Shared async job store for MCP and Engine."""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.utils.cooperative_subprocess import (
    CooperativeCleanupIncomplete,
    CooperativeProcessInterrupted,
    cleanup_incomplete_receipt_from_error,
    operator_interrupt_from_error,
    validate_cleanup_incomplete_receipt,
)


JobRecord = Dict[str, Any]


def _exception_attributes(error: BaseException) -> Dict[str, Any]:
    """Read stored exception fields without invoking custom properties."""
    try:
        values = object.__getattribute__(error, "__dict__")
    except BaseException:
        return {}
    return values if isinstance(values, dict) else {}


def _exception_link(error: BaseException, name: str) -> Optional[BaseException]:
    """Read a built-in exception link without invoking an override."""
    try:
        descriptor = vars(BaseException)[name]
        linked = descriptor.__get__(error, type(error))
    except BaseException:
        return None
    return linked if isinstance(linked, BaseException) else None


def retained_process_result_from_error(error: BaseException) -> Optional[Dict[str, Any]]:
    """Find the original process receipt through exception context and cause links."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        attributes = _exception_attributes(current)
        for name in ("outer_process_result", "result", "cooperative_process_result"):
            candidate = attributes.get(name)
            if isinstance(candidate, dict):
                return candidate
        for previous in (
            _exception_link(current, "__context__"),
            _exception_link(current, "__cause__"),
        ):
            if previous is not None:
                pending.append(previous)
    return None


def process_failure_result_from_error(error: BaseException) -> Dict[str, Any]:
    """Retain outer process output and nested cleanup state at JSON boundaries."""
    captured = retained_process_result_from_error(error)
    protocol_error = None
    try:
        incomplete = cleanup_incomplete_receipt_from_error(error)
    except ValueError as exc:
        incomplete = None
        protocol_error = str(exc)
    result = dict(captured or incomplete or {})
    result.update({"success": False, "error": str(error) or type(error).__name__, "error_type": type(error).__name__})
    if incomplete is not None:
        result["cleanup_incomplete"] = incomplete
    if protocol_error is not None:
        result["cleanup_protocol_error"] = protocol_error
    if operator_interrupt_from_error(error):
        result["cancelled"] = True
    if hasattr(error, "owned_work_admission"):
        result["owned_work_admission"] = error.owned_work_admission
    return result


@dataclass
class JobTicket:
    """Lightweight ticket for clients polling job status."""

    job_id: str
    status: str
    tool: str
    submitted_at: float
    note: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "tool": self.tool,
            "submitted_at": self.submitted_at,
            "note": self.note,
        }


class JobStore:
    """Thread-safe background job registry with TTL cleanup."""

    _instance: Optional["JobStore"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        max_workers: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
        max_entries: Optional[int] = None,
        cleanup_interval_seconds: Optional[float] = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers or os.environ.get("AISP_MCP_JOB_WORKERS", "4") or "4")
        )
        self._store: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = int(ttl_seconds or os.environ.get("AISP_MCP_JOB_TTL_SECONDS", "3600") or "3600")
        self._max_entries = int(max_entries or os.environ.get("AISP_MCP_JOB_MAX_ENTRIES", "1000") or "1000")
        self._cleanup_interval_seconds = float(
            cleanup_interval_seconds or os.environ.get("AISP_MCP_JOB_CLEANUP_INTERVAL_SECONDS", "30") or "30"
        )
        self._last_cleanup_ts = 0.0
        self._incomplete_cleanup_receipt: Optional[Dict[str, Any]] = None

    @classmethod
    def get(cls) -> "JobStore":
        """Return the singleton job store."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_incomplete_cleanup(self, receipt: Dict[str, Any]) -> None:
        """Keep admission blocked independently of completed-job expiry."""
        validate_cleanup_incomplete_receipt(receipt)
        with self._lock:
            if self._incomplete_cleanup_receipt is None:
                self._incomplete_cleanup_receipt = receipt

    def incomplete_cleanup_receipt(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._incomplete_cleanup_receipt

    def cleanup(self, now: Optional[float] = None) -> None:
        """Evict old/completed jobs to keep the store bounded."""
        now = time.time() if now is None else now
        if now - self._last_cleanup_ts < self._cleanup_interval_seconds:
            return
        with self._lock:
            if now - self._last_cleanup_ts < self._cleanup_interval_seconds:
                return
            self._last_cleanup_ts = now

            expired: List[str] = []
            for job_id, record in list(self._store.items()):
                status = record.get("status")
                if status in {"running", "queued"}:
                    continue
                ts = record.get("finished_at") or record.get("submitted_at") or 0.0
                try:
                    age = now - float(ts)
                except (TypeError, ValueError):
                    age = 0.0
                if age > self._ttl_seconds:
                    expired.append(job_id)
            for job_id in expired:
                self._store.pop(job_id, None)

            if len(self._store) <= self._max_entries:
                return
            completed: List[Tuple[float, str]] = []
            for job_id, record in self._store.items():
                status = record.get("status")
                if status in {"running", "queued"}:
                    continue
                ts = record.get("finished_at") or record.get("submitted_at") or 0.0
                try:
                    completed.append((float(ts), job_id))
                except (TypeError, ValueError):
                    completed.append((0.0, job_id))
            completed.sort(key=lambda item: item[0])
            while len(self._store) > self._max_entries and completed:
                _, job_id = completed.pop(0)
                self._store.pop(job_id, None)

    def queue_job(
        self,
        tool_name: str,
        runner: Callable[[], Any],
        *,
        arguments: Optional[Dict[str, Any]] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a task in the background and return a polling ticket."""
        self.cleanup()
        job_id = job_id or f"{tool_name}-{uuid.uuid4().hex[:10]}"
        submitted_at = time.time()
        record: JobRecord = {
            "job_id": job_id,
            "tool": tool_name,
            "status": "queued",
            "submitted_at": submitted_at,
            "arguments": arguments or {},
        }
        if run_metadata:
            for key, value in run_metadata.items():
                record[key] = str(value) if isinstance(value, Path) else value
        with self._lock:
            if len(self._store) >= self._max_entries:
                raise RuntimeError(
                    f"Job queue is full ({len(self._store)} >= {self._max_entries}). "
                    "Poll existing jobs or increase AISP_MCP_JOB_MAX_ENTRIES."
                )
            if job_id in self._store:
                raise RuntimeError(f"Job id already exists: {job_id}")
            self._store[job_id] = record

        def _runner():
            started_at = time.time()
            with self._lock:
                record.update(
                    {
                        "status": "running",
                        "started_at": started_at,
                    }
                )
            try:
                incomplete = self.incomplete_cleanup_receipt()
                if incomplete is not None:
                    raise CooperativeCleanupIncomplete(incomplete)
                result = runner()
                result_is_error = (
                    isinstance(result, dict)
                    and (bool(result.get("error")) or result.get("success") is False)
                )
                status = "error" if result_is_error else "completed"
                error = None
            except (CooperativeCleanupIncomplete, CooperativeProcessInterrupted) as exc:
                try:
                    incomplete = cleanup_incomplete_receipt_from_error(exc)
                except ValueError:
                    incomplete = None
                if incomplete is not None:
                    self.record_incomplete_cleanup(incomplete)
                status = "error"
                error = None
                result = process_failure_result_from_error(exc)
            except Exception as exc:  # pragma: no cover - defensive
                status = "error"
                error = {"error": str(exc)}
                result = None
            finished_at = time.time()
            with self._lock:
                record.update(
                    {
                        "status": status,
                        "result": result if result is not None else error,
                        "finished_at": finished_at,
                        "duration_ms": int((finished_at - submitted_at) * 1000),
                    }
                )

        self._executor.submit(_runner)
        ticket = JobTicket(
            job_id=job_id,
            status="queued",
            tool=tool_name,
            submitted_at=submitted_at,
            note="Poll job status to track completion.",
        ).as_dict()
        if run_metadata:
            for key, value in run_metadata.items():
                ticket[key] = str(value) if isinstance(value, Path) else value
        return ticket

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update an existing job record in-place."""
        with self._lock:
            record = self._store.get(job_id)
            if not record:
                return
            record.update(fields)

    def get_status(self, job_id: str) -> Optional[JobRecord]:
        """Return the stored job record, if present."""
        with self._lock:
            record = self._store.get(job_id)
            return dict(record) if record else None

    def list_jobs(self, tool: Optional[str] = None) -> List[JobRecord]:
        """List all jobs, optionally filtered by tool name."""
        with self._lock:
            records = list(self._store.values())
        if tool:
            records = [record for record in records if record.get("tool") == tool]
        return [dict(record) for record in records]
