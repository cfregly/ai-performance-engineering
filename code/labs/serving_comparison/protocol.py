"""OpenAI-compatible streaming replay with observed token-id event timestamps."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .schema import ArmProfile, ComparisonProfile, ConfigError, RequestSpec

try:
    import httpx
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment gate
    httpx = None  # type: ignore[assignment]
    _HTTPX_IMPORT_ERROR = exc
else:
    _HTTPX_IMPORT_ERROR = None


def require_httpx() -> Any:
    if httpx is None:
        raise ConfigError(
            "serving comparison requires httpx. Install the repository's diagnostics dependencies"
        ) from _HTTPX_IMPORT_ERROR
    return httpx


@dataclass
class RequestObservation:
    request_id: str
    expected_status: str
    scheduled_arrival_s: float
    admitted_s: float
    finish_s: float
    status: str
    prompt_token_ids: list[int]
    echoed_prompt_token_ids: list[int] | None
    output_token_ids: list[int]
    token_timestamps_s: list[float]
    finish_reason: str | None
    usage: dict[str, int] | None
    http_status: int | None
    error: str | None
    cancellation: dict[str, Any] | None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_serving_trace_record(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arrival_s": self.scheduled_arrival_s,
            "token_timestamps_s": list(self.token_timestamps_s),
            "finish_s": self.finish_s,
            "status": self.status,
        }

    def to_dict(self, wall_offset_s: float) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "expected_status": self.expected_status,
            "scheduled_arrival_monotonic_s": self.scheduled_arrival_s,
            "scheduled_arrival_unix_s": self.scheduled_arrival_s + wall_offset_s,
            "admitted_monotonic_s": self.admitted_s,
            "admitted_unix_s": self.admitted_s + wall_offset_s,
            "finish_monotonic_s": self.finish_s,
            "finish_unix_s": self.finish_s + wall_offset_s,
            "status": self.status,
            "prompt_token_ids": list(self.prompt_token_ids),
            "echoed_prompt_token_ids": self.echoed_prompt_token_ids,
            "output_token_ids": list(self.output_token_ids),
            "token_timestamps_monotonic_s": list(self.token_timestamps_s),
            "token_timestamps_unix_s": [value + wall_offset_s for value in self.token_timestamps_s],
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "http_status": self.http_status,
            "error": self.error,
            "cancellation": self.cancellation,
            "events": self.events,
            "timing_semantics": {
                "timestamp_source": "client_observed_sse_event",
                "token_ids_are_explicit": True,
                "same_event_token_ids_share_timestamp": True,
                "text_chunks_are_not_split_into_synthetic_tokens": True,
            },
        }


@dataclass(frozen=True)
class ReplayResult:
    started_monotonic_s: float
    started_unix_s: float
    finished_monotonic_s: float
    observations: tuple[RequestObservation, ...]

    @property
    def wall_offset_s(self) -> float:
        return self.started_unix_s - self.started_monotonic_s


def _headers(arm: ArmProfile) -> dict[str, str]:
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    if arm.api_key_env:
        value = os.environ.get(arm.api_key_env)
        if not value:
            raise ConfigError(f"required API key environment variable is unset: {arm.api_key_env}")
        headers["Authorization"] = f"Bearer {value}"
    return headers


def _extract_token_ids(choice: dict[str, Any], document: dict[str, Any]) -> list[int]:
    raw = choice.get("token_ids", document.get("token_ids", []))
    if raw is None:
        return []
    if not isinstance(raw, list) or any(type(token) is not int or token < 0 for token in raw):
        raise ConfigError("stream token_ids must be a list of nonnegative integers")
    return raw


def _extract_prompt_token_ids(document: dict[str, Any], choice: dict[str, Any]) -> list[int] | None:
    raw = document.get("prompt_token_ids", choice.get("prompt_token_ids"))
    if raw is None:
        return None
    if not isinstance(raw, list) or any(type(token) is not int or token < 0 for token in raw):
        raise ConfigError("stream prompt_token_ids must be a list of nonnegative integers")
    return raw


def _usage(document: dict[str, Any]) -> dict[str, int] | None:
    raw = document.get("usage")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("stream usage must be an object")
    required = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(type(raw.get(key)) is not int or raw[key] < 0 for key in required):
        raise ConfigError("stream usage token counts must be nonnegative integers")
    return {key: raw[key] for key in required}


async def _stream_one(
    client: Any,
    arm: ArmProfile,
    profile: ComparisonProfile,
    request: RequestSpec,
    scheduled_arrival_s: float,
    admitted_s: float,
    progress: dict[str, Any],
) -> RequestObservation:
    output_ids: list[int] = progress["output_token_ids"]
    token_timestamps: list[float] = progress["token_timestamps_s"]
    echoed_prompt: list[int] | None = None
    usage: dict[str, int] | None = None
    finish_reason: str | None = None
    events: list[dict[str, Any]] = progress["events"]
    saw_done = False
    http_status: int | None = None
    payload = {
        "model": profile.model,
        "prompt": list(request.prompt_token_ids),
        "max_tokens": request.max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": profile.seed,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    try:
        async with client.stream(
            "POST", arm.endpoint, headers=_headers(arm), json=payload
        ) as response:
            http_status = response.status_code
            progress["http_status"] = http_status
            if response.status_code >= 400:
                body = await response.aread()
                finish = time.monotonic()
                return RequestObservation(
                    request_id=request.request_id,
                    expected_status=request.expected_status,
                    scheduled_arrival_s=scheduled_arrival_s,
                    admitted_s=admitted_s,
                    finish_s=finish,
                    status="failed",
                    prompt_token_ids=list(request.prompt_token_ids),
                    echoed_prompt_token_ids=None,
                    output_token_ids=[],
                    token_timestamps_s=[],
                    finish_reason=None,
                    usage=None,
                    http_status=http_status,
                    error=f"http_{response.status_code}",
                    cancellation=None,
                    events=[
                        {
                            "observed_monotonic_s": finish,
                            "error_body_bytes": len(body),
                            "error_body_sha256": hashlib.sha256(body).hexdigest(),
                        }
                    ],
                )
            async for line in response.aiter_lines():
                observed = time.monotonic()
                if not line or line.startswith(":") or line.startswith("event:"):
                    continue
                if not line.startswith("data:"):
                    raise ConfigError("stream contained a non-SSE data line")
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    continue
                try:
                    document = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ConfigError("stream contained invalid JSON") from exc
                if not isinstance(document, dict):
                    raise ConfigError("stream event JSON must be an object")
                choices = document.get("choices", [])
                if not isinstance(choices, list):
                    raise ConfigError("stream choices must be a list")
                event_token_ids: list[int] = []
                text_parts: list[str] = []
                for raw_choice in choices:
                    if not isinstance(raw_choice, dict):
                        raise ConfigError("stream choice must be an object")
                    choice_index = raw_choice.get("index", 0)
                    if choice_index != 0:
                        raise ConfigError("stream returned an unexpected completion choice")
                    prompt_ids = _extract_prompt_token_ids(document, raw_choice)
                    if prompt_ids is not None:
                        if echoed_prompt is not None and echoed_prompt != prompt_ids:
                            raise ConfigError("stream changed prompt_token_ids between events")
                        echoed_prompt = prompt_ids
                        progress["echoed_prompt_token_ids"] = prompt_ids
                    ids = _extract_token_ids(raw_choice, document)
                    event_token_ids.extend(ids)
                    output_ids.extend(ids)
                    token_timestamps.extend([observed] * len(ids))
                    text = raw_choice.get("text", "")
                    if text is not None and not isinstance(text, str):
                        raise ConfigError("stream choice text must be a string")
                    if text:
                        text_parts.append(text)
                    if raw_choice.get("finish_reason") is not None:
                        finish_reason = str(raw_choice["finish_reason"])
                        progress["finish_reason"] = finish_reason
                event_usage = _usage(document)
                if event_usage is not None:
                    usage = event_usage
                    progress["usage"] = usage
                text_bytes = "".join(text_parts).encode("utf-8")
                events.append(
                    {
                        "observed_monotonic_s": observed,
                        "token_ids": event_token_ids,
                        "token_id_count": len(event_token_ids),
                        "text_bytes": len(text_bytes),
                        "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
                    }
                )
                progress["events"] = events
    except ConfigError:
        raise
    except require_httpx().HTTPError as exc:
        finish = time.monotonic()
        return RequestObservation(
            request_id=request.request_id,
            expected_status=request.expected_status,
            scheduled_arrival_s=scheduled_arrival_s,
            admitted_s=admitted_s,
            finish_s=finish,
            status="failed",
            prompt_token_ids=list(request.prompt_token_ids),
            echoed_prompt_token_ids=echoed_prompt,
            output_token_ids=output_ids,
            token_timestamps_s=token_timestamps,
            finish_reason=finish_reason,
            usage=usage,
            http_status=http_status,
            error=f"http_transport_error:{type(exc).__name__}",
            cancellation=None,
            events=events,
        )

    finish = time.monotonic()
    error: str | None = None
    if not saw_done:
        error = "stream_missing_done"
    elif finish_reason is None:
        error = "stream_missing_finish_reason"
    elif echoed_prompt != list(request.prompt_token_ids):
        error = "prompt_token_ids_missing_or_mismatched"
    elif usage is None:
        error = "stream_missing_usage"
    elif usage["prompt_tokens"] != len(request.prompt_token_ids):
        error = "usage_prompt_tokens_mismatched"
    elif usage["completion_tokens"] != len(output_ids):
        error = "usage_completion_tokens_mismatched"
    elif usage["total_tokens"] != len(request.prompt_token_ids) + len(output_ids):
        error = "usage_total_tokens_mismatched"
    elif not output_ids:
        error = "text_chunk_missing_explicit_token_ids"
    return RequestObservation(
        request_id=request.request_id,
        expected_status=request.expected_status,
        scheduled_arrival_s=scheduled_arrival_s,
        admitted_s=admitted_s,
        finish_s=finish,
        status="completed" if error is None else "failed",
        prompt_token_ids=list(request.prompt_token_ids),
        echoed_prompt_token_ids=echoed_prompt,
        output_token_ids=output_ids,
        token_timestamps_s=token_timestamps,
        finish_reason=finish_reason,
        usage=usage,
        http_status=http_status,
        error=error,
        cancellation=None,
        events=events,
    )


async def _run_scheduled(
    client: Any,
    arm: ArmProfile,
    profile: ComparisonProfile,
    request: RequestSpec,
    replay_start_s: float,
    semaphore: asyncio.Semaphore,
) -> RequestObservation:
    scheduled = replay_start_s + request.arrival_ms / 1000.0
    delay = scheduled - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    async with semaphore:
        admitted = time.monotonic()
        timeout_ms = request.cancel_after_ms or request.deadline_ms
        progress: dict[str, Any] = {
            "output_token_ids": [],
            "token_timestamps_s": [],
            "events": [],
            "echoed_prompt_token_ids": None,
            "finish_reason": None,
            "usage": None,
            "http_status": None,
        }
        try:
            return await asyncio.wait_for(
                _stream_one(client, arm, profile, request, scheduled, admitted, progress),
                timeout=timeout_ms / 1000.0,
            )
        except TimeoutError:
            finish = time.monotonic()
            cancelled = request.cancel_after_ms is not None
            return RequestObservation(
                request_id=request.request_id,
                expected_status=request.expected_status,
                scheduled_arrival_s=scheduled,
                admitted_s=admitted,
                finish_s=finish,
                status="cancelled" if cancelled else "failed",
                prompt_token_ids=list(request.prompt_token_ids),
                echoed_prompt_token_ids=progress["echoed_prompt_token_ids"],
                output_token_ids=list(progress["output_token_ids"]),
                token_timestamps_s=list(progress["token_timestamps_s"]),
                finish_reason=progress["finish_reason"],
                usage=progress["usage"],
                http_status=progress["http_status"],
                error="client_cancelled_stream" if cancelled else "deadline_exceeded",
                cancellation={
                    "mechanism": "client_stream_close",
                    "server_cancel_confirmation": False,
                    "trigger_ms": timeout_ms,
                }
                if cancelled
                else None,
                events=list(progress["events"]),
            )


async def replay_trace(
    client: Any,
    arm: ArmProfile,
    profile: ComparisonProfile,
    requests: tuple[RequestSpec, ...],
) -> ReplayResult:
    """Replay one deterministic open-loop trace with a bounded client admission queue."""
    start_monotonic = time.monotonic()
    start_unix = time.time()
    semaphore = asyncio.Semaphore(profile.admission_max_concurrency)
    observations = await asyncio.gather(
        *(
            _run_scheduled(client, arm, profile, request, start_monotonic, semaphore)
            for request in requests
        )
    )
    observations.sort(key=lambda item: item.scheduled_arrival_s)
    return ReplayResult(
        started_monotonic_s=start_monotonic,
        started_unix_s=start_unix,
        finished_monotonic_s=time.monotonic(),
        observations=tuple(observations),
    )
