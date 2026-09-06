"""Unified HTTP response envelope aligned with MCP tooling."""

from __future__ import annotations

import time
from typing import Any

from core.api.redaction import redact_sensitive_data
from core.engine import get_engine

_CONTEXT_CACHE: dict[str, Any] = {"summary": None, "full": None}
_CONTEXT_TS: dict[str, float] = {"summary": 0.0, "full": 0.0}
_CONTEXT_TTL_SECONDS = 60.0


def _looks_like_error(result: Any, had_exception: bool = False) -> bool:
    if had_exception:
        return True
    if isinstance(result, dict):
        if result.get("error"):
            return True
        if result.get("success") is False:
            return True
    return False


def _build_context(level: str) -> dict[str, Any]:
    engine = get_engine()
    if level == "summary":
        return {
            "gpu": engine.gpu.info(),
            "software": engine.system.software(),
            "dependencies": engine.system.dependencies(),
        }
    return engine.system.context()


def get_cached_context(level: str) -> Any:
    now = time.time()
    level = "full" if level == "full" else "summary"
    if (
        _CONTEXT_CACHE.get(level) is None
        or (now - _CONTEXT_TS.get(level, 0.0)) > _CONTEXT_TTL_SECONDS
    ):
        _CONTEXT_CACHE[level] = _build_context(level)
        _CONTEXT_TS[level] = now
    return _CONTEXT_CACHE[level]


def build_response(
    tool: str,
    arguments: dict[str, Any] | None,
    result: Any,
    duration_ms: int,
    *,
    had_exception: bool = False,
    include_context: bool = False,
    context_level: str = "summary",
) -> dict[str, Any]:
    """Build a response envelope mirroring MCP-style metadata."""
    status_is_error = _looks_like_error(result, had_exception)
    if (
        status_is_error
        and isinstance(result, dict)
        and result.get("error")
        and "error_type" not in result
    ):
        result = dict(result)
        result["error_type"] = "unhandled_exception" if had_exception else "unknown_error"
    safe_arguments = redact_sensitive_data(arguments or {})
    safe_result = redact_sensitive_data(result, source=arguments)
    payload: dict[str, Any] = {
        "tool": tool,
        "status": "error" if status_is_error else "ok",
        "success": not status_is_error,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_ms": duration_ms,
        "arguments": safe_arguments,
        "result": safe_result,
        "context_summary": get_cached_context("summary"),
    }
    if status_is_error and isinstance(safe_result, dict):
        if safe_result.get("error") is not None:
            payload["error"] = safe_result["error"]
        if safe_result.get("error_type") is not None:
            payload["error_type"] = safe_result["error_type"]
    if include_context:
        payload["context"] = get_cached_context(context_level)
        payload["context_level"] = "full" if context_level == "full" else "summary"
    return payload
