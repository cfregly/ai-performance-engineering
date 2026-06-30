#!/usr/bin/env python3
"""All-in-one Python concurrency reference implementation.

This single file intentionally combines the most important concurrency patterns:
- bounded producer/consumer queues,
- fixed worker pools,
- semaphore + rate limiter controls,
- per-attempt timeout + retry + backoff/jitter,
- idempotency dedupe with in-progress ownership,
- poison routing for exhausted retryable failures,
- hybrid async I/O + process-pool CPU stage,
- global deadline cancellation,
- deterministic ordered terminal output,
- terminal-state and counter invariants.

The code is heavily documented by section so it can be explained quickly while coding.

Quick talk track:
1) Bound pressure first: bounded queues + fixed workers + semaphore + rate limiter.
2) Protect correctness: timeout/retry with jitter, dedupe ownership, poison routing.
3) Split by workload shape: async stage A, process-pool CPU stage B, async stage C.
4) Close cleanly: sentinel ordering + global-timeout cancellation + queue draining.
5) Prove integrity: one terminal record per input index + strict invariant checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Section 1: Status vocabulary and typed records
# ---------------------------------------------------------------------------
# Terminal statuses are immutable final outcomes for an item.
SUCCESS = "success"
DEDUPED = "deduped"
POISON = "poison"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"
FAILED_STAGE_A = "failed_stage_a"
FAILED_STAGE_B = "failed_stage_b"
FAILED_STAGE_C = "failed_stage_c"


# This set is used by invariant checks before emitting final output.
TERMINAL_STATUSES = {
    SUCCESS,
    DEDUPED,
    POISON,
    TIMED_OUT,
    CANCELLED,
    FAILED_STAGE_A,
    FAILED_STAGE_B,
    FAILED_STAGE_C,
}


def _latency_percentiles(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0

    values.sort()
    midpoint = len(values) // 2
    if len(values) % 2:
        p50 = values[midpoint]
    else:
        p50 = (values[midpoint - 1] + values[midpoint]) / 2.0
    p95 = values[max(0, int(0.95 * len(values)) - 1)]
    return round(p50, 2), round(p95, 2)


def _count_whitespace_separated_tokens(text: str) -> int:
    count = 0
    in_token = False
    for char in text:
        if char.isspace():
            in_token = False
        elif not in_token:
            count += 1
            in_token = True
    return count


class TransientStageError(RuntimeError):
    """Retryable stage error (safe to retry by policy)."""


class PermanentStageError(RuntimeError):
    """Non-retryable stage error (should fail fast)."""


@dataclass(slots=True)
class ResultRecord:
    """Final per-item terminal record.

    attempts_fetch:
        Number of stage A attempts.
    attempts_write:
        Number of stage C attempts.
    """

    idx: int
    item_id: str
    idempotency_key: str
    status: str
    attempts_fetch: int
    attempts_write: int
    fetch_ms: float
    cpu_ms: float
    write_ms: float
    total_ms: float
    error: str | None
    output: str | None


@dataclass(slots=True)
class SharedState:
    """Shared mutable state coordinated by async locks.

    completed_keys:
        Keys that already committed side effects.
    in_progress_keys:
        Keys currently owned by an active item.
    poison_items:
        Terminal records routed to poison handling.
    queued/started/completed:
        Lifecycle counters for observability.
    retried_fetch/retried_write:
        Retry counters by stage.
    """

    completed_keys: set[str]
    in_progress_keys: set[str]
    dedupe_lock: asyncio.Lock
    poison_items: list[ResultRecord]
    queued: int
    started: int
    completed: int
    retried_fetch: int
    retried_write: int


# ---------------------------------------------------------------------------
# Section 2: Control primitives (rate limiting and utility helpers)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Simple global starts/sec limiter.

    This controls *start rate*; it is complementary to semaphores, which control
    *in-flight concurrency*. Using both prevents short requests from violating RPS.
    """

    def __init__(self, rps: float) -> None:
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_ts = 0.0

    async def acquire(self) -> None:
        if self.interval <= 0:
            return

        async with self._lock:
            now = time.perf_counter()
            if now < self._next_ts:
                await asyncio.sleep(self._next_ts - now)
                now = time.perf_counter()
            self._next_ts = max(self._next_ts, now) + self.interval


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _jittered_backoff_seconds(attempt_index: int, jitter_rng: random.Random) -> float:
    """Exponential backoff with bounded jitter."""

    base = min(0.5, 0.05 * (2**attempt_index))
    return jitter_rng.uniform(base * 0.5, base)


def _terminal_record(
    *,
    idx: int,
    item: dict[str, Any],
    status: str,
    started_at: float,
    attempts_fetch: int,
    attempts_write: int,
    fetch_ms: float,
    cpu_ms: float,
    write_ms: float,
    error: str | None,
    output: str | None,
) -> ResultRecord:
    """Create a terminal record with consistent total latency calculation."""

    total_ms = (time.perf_counter() - started_at) * 1000.0
    return ResultRecord(
        idx=idx,
        item_id=str(item["id"]),
        idempotency_key=str(item["idempotency_key"]),
        status=status,
        attempts_fetch=attempts_fetch,
        attempts_write=attempts_write,
        fetch_ms=fetch_ms,
        cpu_ms=cpu_ms,
        write_ms=write_ms,
        total_ms=total_ms,
        error=error,
        output=output,
    )


async def _release_key(shared: SharedState, idempotency_key: str, *, mark_completed: bool) -> None:
    """Release key ownership and optionally mark completed atomically."""

    async with shared.dedupe_lock:
        shared.in_progress_keys.discard(idempotency_key)
        if mark_completed:
            shared.completed_keys.add(idempotency_key)


# ---------------------------------------------------------------------------
# Section 3: Stage simulation functions
# ---------------------------------------------------------------------------
async def simulated_fetch(item: dict[str, Any], attempt_number: int, jitter_rng: random.Random) -> str:
    """Stage A (I/O-like) simulation.

    Supported deterministic failure controls:
    - fetch_fail_mode: none|transient_once|transient_always|permanent
    - fetch_fail_rate: probabilistic transient failure rate [0, 1]
    """

    await asyncio.sleep(_safe_float(item.get("fetch_delay_ms", 20), 20) / 1000.0)

    mode = str(item.get("fetch_fail_mode", "none"))
    if mode == "permanent":
        raise PermanentStageError("stage A permanent failure")
    if mode == "transient_always":
        raise TransientStageError("stage A transient failure (always)")
    if mode == "transient_once" and attempt_number == 1:
        raise TransientStageError("stage A transient failure (first attempt)")

    fail_rate = max(0.0, min(1.0, _safe_float(item.get("fetch_fail_rate", 0.0), 0.0)))
    if fail_rate > 0.0 and jitter_rng.random() < fail_rate:
        raise TransientStageError("stage A transient failure (probabilistic)")

    return str(item["payload"])


def cpu_transform(payload: str, cpu_rounds: int, force_fail: bool) -> dict[str, Any]:
    """Stage B (CPU-heavy pure Python) transform.

    This loop intentionally stays in Python bytecode to model CPU-bound work where
    process-based parallelism is typically preferred.
    """

    if force_fail:
        raise ValueError("stage B forced parse failure")

    checksum = 0
    bounded_rounds = max(1, int(cpu_rounds))
    for i in range(1, bounded_rounds + 1):
        checksum = (checksum + ((i * i) % 97)) ^ ((i << 1) & 0xFFFF)

    transformed = payload.strip().upper()
    return {
        "text": transformed,
        "checksum": checksum,
        "token_count": _count_whitespace_separated_tokens(transformed),
    }


async def simulated_write(
    item: dict[str, Any],
    parsed_payload: dict[str, Any],
    attempt_number: int,
    jitter_rng: random.Random,
) -> str:
    """Stage C (I/O-like) simulation.

    Supported deterministic failure controls:
    - write_fail_mode: none|transient_once|transient_always|permanent
    - write_fail_rate: probabilistic transient failure rate [0, 1]
    """

    await asyncio.sleep(_safe_float(item.get("write_delay_ms", 20), 20) / 1000.0)

    mode = str(item.get("write_fail_mode", "none"))
    if mode == "permanent":
        raise PermanentStageError("stage C permanent failure")
    if mode == "transient_always":
        raise TransientStageError("stage C transient failure (always)")
    if mode == "transient_once" and attempt_number == 1:
        raise TransientStageError("stage C transient failure (first attempt)")

    fail_rate = max(0.0, min(1.0, _safe_float(item.get("write_fail_rate", 0.0), 0.0)))
    if fail_rate > 0.0 and jitter_rng.random() < fail_rate:
        raise TransientStageError("stage C transient failure (probabilistic)")

    return (
        f"stored:id={item['id']}:key={item['idempotency_key']}:"
        f"text={parsed_payload['text']}:tokens={parsed_payload['token_count']}:"
        f"checksum={parsed_payload['checksum']}"
    )


# ---------------------------------------------------------------------------
# Section 4: Producer and worker stages
# ---------------------------------------------------------------------------
async def producer(
    *,
    items: list[dict[str, Any]],
    input_queue: asyncio.Queue[tuple[int, dict[str, Any]] | None],
    stage_a_workers: int,
    shared: SharedState,
    stop_event: asyncio.Event,
) -> None:
    """Stream inputs into a bounded queue to enforce producer-side backpressure."""

    for idx, item in enumerate(items):
        if stop_event.is_set():
            break
        await input_queue.put((idx, item))
        shared.queued += 1

    # One sentinel per stage-A worker guarantees deterministic worker shutdown.
    for _ in range(stage_a_workers):
        await input_queue.put(None)


async def stage_a_worker(
    *,
    input_queue: asyncio.Queue[tuple[int, dict[str, Any]] | None],
    queue_a_to_b: asyncio.Queue[dict[str, Any] | None],
    results: list[ResultRecord | None],
    shared: SharedState,
    stop_event: asyncio.Event,
    fetch_retries: int,
    fetch_timeout_ms: int,
    fetch_semaphore: asyncio.Semaphore,
    start_rate_limiter: RateLimiter,
    jitter_rng: random.Random,
) -> None:
    """Stage A: dedupe + bounded fetch with timeout/retry/rate limiting."""

    while True:
        queued = await input_queue.get()

        if queued is None:
            input_queue.task_done()
            return

        idx, item = queued
        item_started_at = time.perf_counter()
        idem_key = str(item["idempotency_key"])

        if stop_event.is_set():
            results[idx] = _terminal_record(
                idx=idx,
                item=item,
                status=CANCELLED,
                started_at=item_started_at,
                attempts_fetch=0,
                attempts_write=0,
                fetch_ms=0.0,
                cpu_ms=0.0,
                write_ms=0.0,
                error="global_timeout_before_start",
                output=None,
            )
            shared.completed += 1
            input_queue.task_done()
            continue

        # Dedupe ownership contract:
        # - completed key => already committed side effect
        # - in-progress key => another owner currently executing
        # This demo suppresses duplicates immediately instead of waiting for owner.
        owns_key = False
        async with shared.dedupe_lock:
            if idem_key in shared.completed_keys or idem_key in shared.in_progress_keys:
                results[idx] = _terminal_record(
                    idx=idx,
                    item=item,
                    status=DEDUPED,
                    started_at=item_started_at,
                    attempts_fetch=0,
                    attempts_write=0,
                    fetch_ms=0.0,
                    cpu_ms=0.0,
                    write_ms=0.0,
                    error=None,
                    output=None,
                )
                shared.completed += 1
                input_queue.task_done()
                continue

            shared.in_progress_keys.add(idem_key)
            shared.started += 1
            owns_key = True

        attempts_fetch = 0
        fetch_ms = 0.0
        terminal_status = FAILED_STAGE_A
        terminal_error: str | None = None
        fetched_payload: str | None = None

        try:
            for attempt_index in range(fetch_retries + 1):
                if stop_event.is_set():
                    terminal_status = CANCELLED
                    terminal_error = "global_timeout_during_stage_a"
                    break

                attempts_fetch = attempt_index + 1

                try:
                    # Rate limit controls starts/sec globally.
                    await start_rate_limiter.acquire()

                    # Semaphore controls in-flight stage-A concurrency.
                    async with fetch_semaphore:
                        fetch_started_at = time.perf_counter()
                        fetched_payload = await asyncio.wait_for(
                            simulated_fetch(item, attempts_fetch, jitter_rng),
                            timeout=float(fetch_timeout_ms) / 1000.0,
                        )
                        fetch_ms = (time.perf_counter() - fetch_started_at) * 1000.0

                    terminal_error = None
                    break

                except asyncio.TimeoutError:
                    terminal_status = TIMED_OUT
                    terminal_error = f"stage_a_timeout_after_{fetch_timeout_ms}ms"

                    if attempt_index < fetch_retries:
                        shared.retried_fetch += 1
                        await asyncio.sleep(_jittered_backoff_seconds(attempt_index, jitter_rng))
                        continue
                    break

                except TransientStageError as exc:
                    terminal_status = POISON
                    terminal_error = f"TransientStageError: {exc}"

                    if attempt_index < fetch_retries:
                        shared.retried_fetch += 1
                        await asyncio.sleep(_jittered_backoff_seconds(attempt_index, jitter_rng))
                        continue
                    break

                except PermanentStageError as exc:
                    terminal_status = FAILED_STAGE_A
                    terminal_error = f"PermanentStageError: {exc}"
                    break

                except Exception as exc:  # noqa: BLE001
                    terminal_status = FAILED_STAGE_A
                    terminal_error = f"{type(exc).__name__}: {exc}"
                    break

            if fetched_payload is not None:
                # Forward only successful stage-A outputs to stage B.
                await queue_a_to_b.put(
                    {
                        "idx": idx,
                        "item": item,
                        "idempotency_key": idem_key,
                        "item_started_at": item_started_at,
                        "attempts_fetch": attempts_fetch,
                        "fetch_ms": fetch_ms,
                        "fetched_payload": fetched_payload,
                    }
                )
            else:
                # Stage-A terminalization path.
                record = _terminal_record(
                    idx=idx,
                    item=item,
                    status=terminal_status,
                    started_at=item_started_at,
                    attempts_fetch=attempts_fetch,
                    attempts_write=0,
                    fetch_ms=fetch_ms,
                    cpu_ms=0.0,
                    write_ms=0.0,
                    error=terminal_error,
                    output=None,
                )
                results[idx] = record
                shared.completed += 1

                if record.status == POISON:
                    shared.poison_items.append(record)

                if owns_key:
                    await _release_key(shared, idem_key, mark_completed=False)

        except asyncio.CancelledError:
            # If cancelled mid-item, ensure the item still reaches terminal status.
            if results[idx] is None:
                record = _terminal_record(
                    idx=idx,
                    item=item,
                    status=CANCELLED,
                    started_at=item_started_at,
                    attempts_fetch=attempts_fetch,
                    attempts_write=0,
                    fetch_ms=fetch_ms,
                    cpu_ms=0.0,
                    write_ms=0.0,
                    error="stage_a_cancelled",
                    output=None,
                )
                results[idx] = record
                shared.completed += 1
            if owns_key:
                await _release_key(shared, idem_key, mark_completed=False)
            raise

        finally:
            input_queue.task_done()


async def stage_b_worker(
    *,
    queue_a_to_b: asyncio.Queue[dict[str, Any] | None],
    queue_b_to_c: asyncio.Queue[dict[str, Any] | None],
    results: list[ResultRecord | None],
    shared: SharedState,
    stop_event: asyncio.Event,
    process_pool: ProcessPoolExecutor,
    cpu_rounds: int,
    cpu_timeout_ms: int,
) -> None:
    """Stage B: process-pool CPU transform with explicit terminalization."""

    loop = asyncio.get_running_loop()

    while True:
        envelope = await queue_a_to_b.get()

        if envelope is None:
            queue_a_to_b.task_done()
            return

        idx = int(envelope["idx"])
        item = envelope["item"]
        idem_key = str(envelope["idempotency_key"])
        item_started_at = float(envelope["item_started_at"])
        attempts_fetch = int(envelope["attempts_fetch"])
        fetch_ms = float(envelope["fetch_ms"])

        cpu_started_at = time.perf_counter()

        try:
            if stop_event.is_set():
                record = _terminal_record(
                    idx=idx,
                    item=item,
                    status=CANCELLED,
                    started_at=item_started_at,
                    attempts_fetch=attempts_fetch,
                    attempts_write=0,
                    fetch_ms=fetch_ms,
                    cpu_ms=0.0,
                    write_ms=0.0,
                    error="global_timeout_before_stage_b",
                    output=None,
                )
                results[idx] = record
                shared.completed += 1
                await _release_key(shared, idem_key, mark_completed=False)
                continue

            parsed_payload = await asyncio.wait_for(
                loop.run_in_executor(
                    process_pool,
                    cpu_transform,
                    str(envelope["fetched_payload"]),
                    int(cpu_rounds),
                    bool(item.get("cpu_fail", False)),
                ),
                timeout=float(cpu_timeout_ms) / 1000.0,
            )

            cpu_ms = (time.perf_counter() - cpu_started_at) * 1000.0

            await queue_b_to_c.put(
                {
                    "idx": idx,
                    "item": item,
                    "idempotency_key": idem_key,
                    "item_started_at": item_started_at,
                    "attempts_fetch": attempts_fetch,
                    "fetch_ms": fetch_ms,
                    "cpu_ms": cpu_ms,
                    "parsed_payload": parsed_payload,
                }
            )

        except asyncio.TimeoutError:
            cpu_ms = (time.perf_counter() - cpu_started_at) * 1000.0
            record = _terminal_record(
                idx=idx,
                item=item,
                status=TIMED_OUT,
                started_at=item_started_at,
                attempts_fetch=attempts_fetch,
                attempts_write=0,
                fetch_ms=fetch_ms,
                cpu_ms=cpu_ms,
                write_ms=0.0,
                error=f"stage_b_timeout_after_{cpu_timeout_ms}ms",
                output=None,
            )
            results[idx] = record
            shared.completed += 1
            await _release_key(shared, idem_key, mark_completed=False)

        except Exception as exc:  # noqa: BLE001
            cpu_ms = (time.perf_counter() - cpu_started_at) * 1000.0
            record = _terminal_record(
                idx=idx,
                item=item,
                status=FAILED_STAGE_B,
                started_at=item_started_at,
                attempts_fetch=attempts_fetch,
                attempts_write=0,
                fetch_ms=fetch_ms,
                cpu_ms=cpu_ms,
                write_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
                output=None,
            )
            results[idx] = record
            shared.completed += 1
            await _release_key(shared, idem_key, mark_completed=False)

        finally:
            queue_a_to_b.task_done()


async def stage_c_worker(
    *,
    queue_b_to_c: asyncio.Queue[dict[str, Any] | None],
    results: list[ResultRecord | None],
    shared: SharedState,
    stop_event: asyncio.Event,
    write_retries: int,
    write_timeout_ms: int,
    write_semaphore: asyncio.Semaphore,
    jitter_rng: random.Random,
) -> None:
    """Stage C: bounded write with timeout/retry and poison routing."""

    while True:
        envelope = await queue_b_to_c.get()

        if envelope is None:
            queue_b_to_c.task_done()
            return

        idx = int(envelope["idx"])
        item = envelope["item"]
        idem_key = str(envelope["idempotency_key"])
        item_started_at = float(envelope["item_started_at"])
        attempts_fetch = int(envelope["attempts_fetch"])
        fetch_ms = float(envelope["fetch_ms"])
        cpu_ms = float(envelope["cpu_ms"])

        attempts_write = 0
        write_ms = 0.0
        final_status = FAILED_STAGE_C
        final_error: str | None = None
        output: str | None = None

        try:
            for attempt_index in range(write_retries + 1):
                if stop_event.is_set():
                    final_status = CANCELLED
                    final_error = "global_timeout_during_stage_c"
                    break

                attempts_write = attempt_index + 1

                try:
                    async with write_semaphore:
                        write_started_at = time.perf_counter()
                        output = await asyncio.wait_for(
                            simulated_write(item, envelope["parsed_payload"], attempts_write, jitter_rng),
                            timeout=float(write_timeout_ms) / 1000.0,
                        )
                        write_ms = (time.perf_counter() - write_started_at) * 1000.0

                    final_status = SUCCESS
                    final_error = None
                    break

                except asyncio.TimeoutError:
                    final_status = TIMED_OUT
                    final_error = f"stage_c_timeout_after_{write_timeout_ms}ms"

                    if attempt_index < write_retries:
                        shared.retried_write += 1
                        await asyncio.sleep(_jittered_backoff_seconds(attempt_index, jitter_rng))
                        continue
                    break

                except TransientStageError as exc:
                    final_status = POISON
                    final_error = f"TransientStageError: {exc}"

                    if attempt_index < write_retries:
                        shared.retried_write += 1
                        await asyncio.sleep(_jittered_backoff_seconds(attempt_index, jitter_rng))
                        continue
                    break

                except PermanentStageError as exc:
                    final_status = FAILED_STAGE_C
                    final_error = f"PermanentStageError: {exc}"
                    break

                except Exception as exc:  # noqa: BLE001
                    final_status = FAILED_STAGE_C
                    final_error = f"{type(exc).__name__}: {exc}"
                    break

            record = _terminal_record(
                idx=idx,
                item=item,
                status=final_status,
                started_at=item_started_at,
                attempts_fetch=attempts_fetch,
                attempts_write=attempts_write,
                fetch_ms=fetch_ms,
                cpu_ms=cpu_ms,
                write_ms=write_ms,
                error=final_error,
                output=output,
            )
            results[idx] = record
            shared.completed += 1

            if record.status == POISON:
                shared.poison_items.append(record)

            await _release_key(shared, idem_key, mark_completed=(record.status == SUCCESS))

        finally:
            queue_b_to_c.task_done()


# ---------------------------------------------------------------------------
# Section 5: Orchestration, cancellation handling, and invariants
# ---------------------------------------------------------------------------
def _drain_input_queue(
    *,
    input_queue: asyncio.Queue[tuple[int, dict[str, Any]] | None],
    results: list[ResultRecord | None],
    reason: str,
) -> None:
    """Drain queued stage-A items and terminalize as cancelled."""

    while True:
        try:
            queued = input_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        if queued is None:
            input_queue.task_done()
            continue

        idx, item = queued
        if results[idx] is None:
            now = time.perf_counter()
            results[idx] = _terminal_record(
                idx=idx,
                item=item,
                status=CANCELLED,
                started_at=now,
                attempts_fetch=0,
                attempts_write=0,
                fetch_ms=0.0,
                cpu_ms=0.0,
                write_ms=0.0,
                error=reason,
                output=None,
            )
        input_queue.task_done()


def _drain_envelope_queue(
    *,
    queue_obj: asyncio.Queue[dict[str, Any] | None],
    results: list[ResultRecord | None],
    reason: str,
) -> list[str]:
    """Drain stage envelopes and terminalize as cancelled.

    Returns idempotency keys that still require ownership release.
    """

    keys_to_release: list[str] = []

    while True:
        try:
            envelope = queue_obj.get_nowait()
        except asyncio.QueueEmpty:
            break

        if envelope is None:
            queue_obj.task_done()
            continue

        idx = int(envelope["idx"])
        item = envelope["item"]
        idem_key = str(envelope["idempotency_key"])
        item_started_at = float(envelope.get("item_started_at", time.perf_counter()))
        attempts_fetch = int(envelope.get("attempts_fetch", 0))
        fetch_ms = _safe_float(envelope.get("fetch_ms", 0.0), 0.0)
        cpu_ms = _safe_float(envelope.get("cpu_ms", 0.0), 0.0)

        if results[idx] is None:
            results[idx] = _terminal_record(
                idx=idx,
                item=item,
                status=CANCELLED,
                started_at=item_started_at,
                attempts_fetch=attempts_fetch,
                attempts_write=0,
                fetch_ms=fetch_ms,
                cpu_ms=cpu_ms,
                write_ms=0.0,
                error=reason,
                output=None,
            )
            keys_to_release.append(idem_key)

        queue_obj.task_done()

    return keys_to_release


async def run_pipeline(
    *,
    items: list[dict[str, Any]],
    stage_a_workers: int,
    stage_b_workers: int,
    stage_c_workers: int,
    queue_size: int,
    rps: float,
    fetch_inflight: int,
    write_inflight: int,
    fetch_timeout_ms: int,
    write_timeout_ms: int,
    cpu_timeout_ms: int,
    fetch_retries: int,
    write_retries: int,
    cpu_rounds: int,
    global_timeout_ms: int,
    seed: int,
) -> tuple[list[ResultRecord], SharedState]:
    """Run the full all-in-one pipeline and return ordered results plus shared stats."""

    a_workers = max(1, int(stage_a_workers))
    b_workers = max(1, int(stage_b_workers))
    c_workers = max(1, int(stage_c_workers))
    bounded_queue_size = max(1, int(queue_size))

    # Separate deterministic RNGs for stable failure behavior and jitter behavior.
    # This keeps retries and probabilistic failures reproducible for walkthroughs.
    base_rng = random.Random(int(seed))
    jitter_rng_a = random.Random(base_rng.randint(0, 10_000_000))
    jitter_rng_c = random.Random(base_rng.randint(0, 10_000_000))

    # Bounded queues are explicit backpressure points between stages.
    input_queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue(maxsize=bounded_queue_size)
    queue_a_to_b: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=bounded_queue_size)
    queue_b_to_c: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=bounded_queue_size)

    results: list[ResultRecord | None] = [None] * len(items)

    shared = SharedState(
        completed_keys=set(),
        in_progress_keys=set(),
        dedupe_lock=asyncio.Lock(),
        poison_items=[],
        queued=0,
        started=0,
        completed=0,
        retried_fetch=0,
        retried_write=0,
    )

    stop_event = asyncio.Event()

    start_rate_limiter = RateLimiter(float(rps))
    fetch_semaphore = asyncio.Semaphore(max(1, int(fetch_inflight)))
    write_semaphore = asyncio.Semaphore(max(1, int(write_inflight)))

    process_pool = ProcessPoolExecutor(max_workers=b_workers)

    producer_task = asyncio.create_task(
        producer(
            items=items,
            input_queue=input_queue,
            stage_a_workers=a_workers,
            shared=shared,
            stop_event=stop_event,
        ),
        name="producer",
    )

    stage_a_tasks = [
        asyncio.create_task(
            stage_a_worker(
                input_queue=input_queue,
                queue_a_to_b=queue_a_to_b,
                results=results,
                shared=shared,
                stop_event=stop_event,
                fetch_retries=max(0, int(fetch_retries)),
                fetch_timeout_ms=max(1, int(fetch_timeout_ms)),
                fetch_semaphore=fetch_semaphore,
                start_rate_limiter=start_rate_limiter,
                jitter_rng=jitter_rng_a,
            ),
            name=f"stage-a-{i}",
        )
        for i in range(a_workers)
    ]

    stage_b_tasks = [
        asyncio.create_task(
            stage_b_worker(
                queue_a_to_b=queue_a_to_b,
                queue_b_to_c=queue_b_to_c,
                results=results,
                shared=shared,
                stop_event=stop_event,
                process_pool=process_pool,
                cpu_rounds=max(1, int(cpu_rounds)),
                cpu_timeout_ms=max(1, int(cpu_timeout_ms)),
            ),
            name=f"stage-b-{i}",
        )
        for i in range(b_workers)
    ]

    stage_c_tasks = [
        asyncio.create_task(
            stage_c_worker(
                queue_b_to_c=queue_b_to_c,
                results=results,
                shared=shared,
                stop_event=stop_event,
                write_retries=max(0, int(write_retries)),
                write_timeout_ms=max(1, int(write_timeout_ms)),
                write_semaphore=write_semaphore,
                jitter_rng=jitter_rng_c,
            ),
            name=f"stage-c-{i}",
        )
        for i in range(c_workers)
    ]

    async def _normal_shutdown() -> None:
        # Producer completes then stage A drains the input queue.
        await producer_task
        await input_queue.join()
        await asyncio.gather(*stage_a_tasks)

        # Stage B receives sentinels only after stage A has fully stopped.
        for _ in range(b_workers):
            await queue_a_to_b.put(None)
        await queue_a_to_b.join()
        await asyncio.gather(*stage_b_tasks)

        # Stage C receives sentinels only after stage B has fully stopped.
        for _ in range(c_workers):
            await queue_b_to_c.put(None)
        await queue_b_to_c.join()
        await asyncio.gather(*stage_c_tasks)

    all_worker_tasks = [*stage_a_tasks, *stage_b_tasks, *stage_c_tasks]

    try:
        if int(global_timeout_ms) > 0:
            await asyncio.wait_for(_normal_shutdown(), timeout=float(global_timeout_ms) / 1000.0)
        else:
            await _normal_shutdown()

    except asyncio.TimeoutError:
        # Global deadline reached: stop intake, cancel workers, and terminalize leftovers.
        stop_event.set()

        producer_task.cancel()
        for task in all_worker_tasks:
            task.cancel()

        await asyncio.gather(producer_task, return_exceptions=True)
        await asyncio.gather(*all_worker_tasks, return_exceptions=True)

        _drain_input_queue(
            input_queue=input_queue,
            results=results,
            reason="global_timeout_before_stage_a",
        )

        keys_ab = _drain_envelope_queue(
            queue_obj=queue_a_to_b,
            results=results,
            reason="global_timeout_before_stage_b",
        )
        keys_bc = _drain_envelope_queue(
            queue_obj=queue_b_to_c,
            results=results,
            reason="global_timeout_before_stage_c",
        )

        # Release ownership for drained envelopes.
        for key in keys_ab + keys_bc:
            await _release_key(shared, key, mark_completed=False)

    finally:
        process_pool.shutdown(wait=True, cancel_futures=True)

    # Final reconciliation: every item must have exactly one terminal record.
    for idx, item in enumerate(items):
        if results[idx] is None:
            now = time.perf_counter()
            results[idx] = _terminal_record(
                idx=idx,
                item=item,
                status=CANCELLED,
                started_at=now,
                attempts_fetch=0,
                attempts_write=0,
                fetch_ms=0.0,
                cpu_ms=0.0,
                write_ms=0.0,
                error="reconciliation_filled_missing_terminal_state",
                output=None,
            )

    # Ensure no stale ownership remains after terminalization.
    async with shared.dedupe_lock:
        shared.in_progress_keys.clear()

    finalized = [record for record in results if record is not None]

    # Recompute completed from finalized records for strict consistency.
    shared.completed = len(finalized)

    # Strict invariant checks: fail fast if terminal-state contract is broken.
    if len(finalized) != len(items):
        raise RuntimeError("terminal-state invariant failed: count mismatch")
    if any(record.status not in TERMINAL_STATUSES for record in finalized):
        raise RuntimeError("terminal-state invariant failed: unknown status")

    return finalized, shared


# ---------------------------------------------------------------------------
# Section 6: Summary, CLI parsing, and entrypoint
# ---------------------------------------------------------------------------
def summarize(results: list[ResultRecord], shared: SharedState) -> dict[str, Any]:
    """Aggregate metrics and invariants for quick operational reading."""

    status_counts = {
        SUCCESS: 0,
        DEDUPED: 0,
        POISON: 0,
        TIMED_OUT: 0,
        CANCELLED: 0,
        FAILED_STAGE_A: 0,
        FAILED_STAGE_B: 0,
        FAILED_STAGE_C: 0,
    }

    totals: list[float] = []
    success_totals: list[float] = []
    for record in results:
        status_counts[record.status] += 1
        totals.append(record.total_ms)
        if record.status == SUCCESS:
            success_totals.append(record.total_ms)

    p50_total_ms, p95_total_ms = _latency_percentiles(totals)
    p50_success_ms, _ = _latency_percentiles(success_totals)

    summary = {
        "total": len(results),
        "queued": shared.queued,
        "started": shared.started,
        "completed": shared.completed,
        "unique_completed_keys": len(shared.completed_keys),
        "retried_fetch": shared.retried_fetch,
        "retried_write": shared.retried_write,
        "p50_total_ms": p50_total_ms,
        "p95_total_ms": p95_total_ms,
        "p50_success_ms": p50_success_ms,
    }

    summary.update(status_counts)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI args with practical defaults."""

    parser = argparse.ArgumentParser(description="All-in-one Python concurrency reference pipeline")

    parser.add_argument("--input", type=Path, required=True)

    # Worker/queue controls
    parser.add_argument("--stage-a-workers", type=int, default=4)
    parser.add_argument("--stage-b-workers", type=int, default=4)
    parser.add_argument("--stage-c-workers", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=32)

    # External pressure controls
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument("--fetch-inflight", type=int, default=8)
    parser.add_argument("--write-inflight", type=int, default=8)

    # Timeout/retry controls
    parser.add_argument("--fetch-timeout-ms", type=int, default=600)
    parser.add_argument("--write-timeout-ms", type=int, default=600)
    parser.add_argument("--cpu-timeout-ms", type=int, default=1500)
    parser.add_argument("--fetch-retries", type=int, default=1)
    parser.add_argument("--write-retries", type=int, default=1)

    # CPU simulation and global cancellation control
    parser.add_argument("--cpu-rounds", type=int, default=30_000)
    parser.add_argument("--global-timeout-ms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate input schema.

    Required fields:
    - id
    - idempotency_key
    - payload
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input must be a JSON array of objects.")

    items: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not an object.")
        for field in ("id", "idempotency_key", "payload"):
            if field not in item:
                raise ValueError(f"Item at index {idx} missing required field '{field}'.")
        items.append(item)

    return items


async def main_async(args: argparse.Namespace) -> int:
    """Async program entrypoint."""

    items = load_items(args.input)

    results, shared = await run_pipeline(
        items=items,
        stage_a_workers=args.stage_a_workers,
        stage_b_workers=args.stage_b_workers,
        stage_c_workers=args.stage_c_workers,
        queue_size=args.queue_size,
        rps=args.rps,
        fetch_inflight=args.fetch_inflight,
        write_inflight=args.write_inflight,
        fetch_timeout_ms=args.fetch_timeout_ms,
        write_timeout_ms=args.write_timeout_ms,
        cpu_timeout_ms=args.cpu_timeout_ms,
        fetch_retries=args.fetch_retries,
        write_retries=args.write_retries,
        cpu_rounds=args.cpu_rounds,
        global_timeout_ms=args.global_timeout_ms,
        seed=args.seed,
    )

    for result in results:
        print(json.dumps(asdict(result), sort_keys=True))

    summary = summarize(results, shared)
    print(json.dumps(summary, sort_keys=True))

    if shared.poison_items:
        print(json.dumps({"poison_items": [asdict(item) for item in shared.poison_items]}, sort_keys=True))

    has_non_success = any(result.status != SUCCESS for result in results)
    return 1 if has_non_success else 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
