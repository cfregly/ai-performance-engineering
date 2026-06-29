#!/usr/bin/env python3
"""Round 3 reliability pattern: idempotency + retries + poison queue.

This script demonstrates at-least-once style processing where duplicates may exist.
Core guarantees in this example:
- duplicate logical keys are deduplicated,
- transient failures are retried with bounded backoff,
- exhausted failures are routed to a poison list,
- every input item receives an explicit terminal status.

Quick talk track:
1) Claim key ownership with `in_progress` before side effects.
2) Retry only transient/timeout paths with bounded backoff.
3) Route exhausted retryable failures to `poison`.
4) Mark successes in `completed_keys` to make replay safe.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


WorkItem = dict[str, Any]

SUCCESS = "success"
DEDUPED = "deduped"
POISON = "poison"
FAILED = "failed"
CANCELLED = "cancelled"


class TransientError(RuntimeError):
    """Retryable downstream failure."""


class PermanentError(RuntimeError):
    """Non-retryable downstream failure."""


@dataclass(slots=True)
class ItemResult:
    """Terminal output record for one input item."""

    idx: int
    item_id: str
    idempotency_key: str
    status: str
    attempts: int
    latency_ms: float
    error: str | None
    output: str | None


@dataclass(slots=True)
class SharedState:
    """Mutable state shared by workers.

    completed_keys:
        Set of keys that already produced side effects.
    in_progress_keys:
        Set of keys currently owned by active workers.
    dedupe_lock:
        Lock guarding check-and-set operations on key ownership state.
    poison_items:
        List of terminal failures routed to poison handling.
    retried_attempts:
        Counter for additional attempts beyond first try.
    """

    completed_keys: set[str]
    in_progress_keys: set[str]
    dedupe_lock: asyncio.Lock
    poison_items: list[ItemResult]
    retried_attempts: int


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""

    parser = argparse.ArgumentParser(description="Round 3 idempotency + poison queue demo")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON array: [{id, idempotency_key, payload, ...}, ...]",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Fixed worker count.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=700,
        help="Per-attempt timeout in milliseconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count for transient failures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic transient failures.",
    )
    parser.add_argument(
        "--global-timeout-ms",
        type=int,
        default=0,
        help="Optional global pipeline timeout in milliseconds (0 disables global timeout).",
    )
    return parser.parse_args()


def load_items(path: Path) -> list[WorkItem]:
    """Load and validate input items."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of objects.")

    items: list[WorkItem] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {index} is not an object.")
        required = {"id", "idempotency_key", "payload"}
        missing = sorted(required - set(item.keys()))
        if missing:
            raise ValueError(f"Item at index {index} is missing fields: {missing}")
        items.append(item)
    return items


async def downstream_side_effect(item: WorkItem) -> str:
    """Simulate downstream side-effecting call.

    Optional fields:
    - delay_ms: artificial latency.
    - transient_fail_rate: probability of transient failure per attempt.
    - permanent_fail: always fail with non-retryable error.
    """

    await asyncio.sleep(float(item.get("delay_ms", 120.0)) / 1000.0)

    if bool(item.get("permanent_fail", False)):
        raise PermanentError("simulated permanent failure")

    if random.random() < float(item.get("transient_fail_rate", 0.0)):
        raise TransientError("simulated transient failure")

    return f"applied:{item['payload']}"


async def run_one(item: WorkItem, timeout_ms: int) -> str:
    """Execute one attempt with timeout wrapper."""

    return await asyncio.wait_for(
        downstream_side_effect(item),
        timeout=float(timeout_ms) / 1000.0,
    )


async def worker(
    queue: asyncio.Queue[tuple[int, WorkItem] | None],
    results: list[ItemResult | None],
    shared: SharedState,
    timeout_ms: int,
    retries: int,
    stop_event: asyncio.Event,
) -> None:
    """Worker loop with idempotency and poison routing."""

    while True:
        queued = await queue.get()

        if queued is None:
            queue.task_done()
            return

        idx, item = queued
        item_id = str(item["id"])
        idem_key = str(item["idempotency_key"])
        started_at = time.perf_counter()

        if stop_event.is_set():
            results[idx] = ItemResult(
                idx=idx,
                item_id=item_id,
                idempotency_key=idem_key,
                status=CANCELLED,
                attempts=0,
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                error="global_timeout_before_start",
                output=None,
            )
            queue.task_done()
            continue

        # First dedupe check:
        # - key in completed => side effects already applied.
        # - key in progress => another worker currently owns execution.
        async with shared.dedupe_lock:
            if idem_key in shared.completed_keys or idem_key in shared.in_progress_keys:
                results[idx] = ItemResult(
                    idx=idx,
                    item_id=item_id,
                    idempotency_key=idem_key,
                    status=DEDUPED,
                    attempts=0,
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                    error=None,
                    output=None,
                )
                queue.task_done()
                continue
            shared.in_progress_keys.add(idem_key)

        attempts = 0
        final_error: str | None = None
        output: str | None = None
        terminal_status = FAILED

        for attempt_index in range(retries + 1):
            if stop_event.is_set():
                final_error = "global_timeout_during_execution"
                terminal_status = CANCELLED
                break

            attempts = attempt_index + 1

            try:
                output = await run_one(item, timeout_ms=timeout_ms)
                terminal_status = SUCCESS
                final_error = None
                break

            except TransientError as exc:
                final_error = f"TransientError: {exc}"
                terminal_status = POISON

                if attempt_index < retries:
                    shared.retried_attempts += 1
                    backoff_seconds = min(0.5, 0.05 * (2**attempt_index))
                    await asyncio.sleep(backoff_seconds)
                    continue
                break

            except asyncio.TimeoutError:
                final_error = f"timeout_after_{timeout_ms}ms"
                terminal_status = POISON

                if attempt_index < retries:
                    shared.retried_attempts += 1
                    backoff_seconds = min(0.5, 0.05 * (2**attempt_index))
                    await asyncio.sleep(backoff_seconds)
                    continue
                break

            except PermanentError as exc:
                final_error = f"PermanentError: {exc}"
                terminal_status = FAILED
                break

            except Exception as exc:  # noqa: BLE001 - explicit status classification.
                final_error = f"{type(exc).__name__}: {exc}"
                terminal_status = FAILED
                break

        latency_ms = (time.perf_counter() - started_at) * 1000.0

        # Release key ownership and mark completion atomically.
        async with shared.dedupe_lock:
            shared.in_progress_keys.discard(idem_key)
            if terminal_status == SUCCESS:
                shared.completed_keys.add(idem_key)

        result = ItemResult(
            idx=idx,
            item_id=item_id,
            idempotency_key=idem_key,
            status=terminal_status,
            attempts=attempts,
            latency_ms=latency_ms,
            error=final_error,
            output=output,
        )

        results[idx] = result

        if terminal_status == POISON:
            shared.poison_items.append(result)

        queue.task_done()


async def run_pipeline(
    items: list[WorkItem],
    max_workers: int,
    timeout_ms: int,
    retries: int,
    global_timeout_ms: int | None = None,
) -> tuple[list[ItemResult], SharedState]:
    """Execute queue/worker pipeline and return ordered outputs + shared stats."""

    worker_count = max(1, int(max_workers))
    bounded_retries = max(0, int(retries))

    queue: asyncio.Queue[tuple[int, WorkItem] | None] = asyncio.Queue()
    for idx, item in enumerate(items):
        queue.put_nowait((idx, item))

    results: list[ItemResult | None] = [None] * len(items)

    shared = SharedState(
        completed_keys=set(),
        in_progress_keys=set(),
        dedupe_lock=asyncio.Lock(),
        poison_items=[],
        retried_attempts=0,
    )
    stop_event = asyncio.Event()

    workers = [
        asyncio.create_task(
            worker(
                queue=queue,
                results=results,
                shared=shared,
                timeout_ms=timeout_ms,
                retries=bounded_retries,
                stop_event=stop_event,
            ),
            name=f"worker-{i}",
        )
        for i in range(worker_count)
    ]

    if global_timeout_ms is not None and int(global_timeout_ms) > 0:
        try:
            await asyncio.wait_for(queue.join(), timeout=float(global_timeout_ms) / 1000.0)
        except asyncio.TimeoutError:
            stop_event.set()

            # Mark all not-yet-started queue entries as cancelled.
            while True:
                try:
                    queued = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if queued is None:
                    queue.task_done()
                    continue

                idx, item = queued
                if results[idx] is None:
                    results[idx] = ItemResult(
                        idx=idx,
                        item_id=str(item["id"]),
                        idempotency_key=str(item["idempotency_key"]),
                        status=CANCELLED,
                        attempts=0,
                        latency_ms=0.0,
                        error="global_timeout_before_start",
                        output=None,
                    )
                queue.task_done()

            # Wait for already-started items to reach terminal state.
            await queue.join()
    else:
        await queue.join()

    for _ in workers:
        queue.put_nowait(None)
    await asyncio.gather(*workers)

    finalized: list[ItemResult] = []
    for idx, result in enumerate(results):
        if result is None:
            raise RuntimeError(f"Missing terminal result at index {idx}")
        finalized.append(result)

    return finalized, shared


def summarize(results: list[ItemResult], shared: SharedState) -> dict[str, Any]:
    """Produce aggregate metrics for reliability behavior."""

    success = 0
    deduped = 0
    poison = 0
    failed = 0
    cancelled = 0
    for result in results:
        if result.status == SUCCESS:
            success += 1
        elif result.status == DEDUPED:
            deduped += 1
        elif result.status == POISON:
            poison += 1
        elif result.status == FAILED:
            failed += 1
        elif result.status == CANCELLED:
            cancelled += 1

    return {
        "total": len(results),
        "success": success,
        "deduped": deduped,
        "poison": poison,
        "failed": failed,
        "cancelled": cancelled,
        "retried_attempts": shared.retried_attempts,
        "unique_completed_keys": len(shared.completed_keys),
    }


async def main_async(args: argparse.Namespace) -> int:
    """Async entrypoint."""

    random.seed(args.seed)
    items = load_items(args.input)

    results, shared = await run_pipeline(
        items=items,
        max_workers=args.max_workers,
        timeout_ms=args.timeout_ms,
        retries=args.retries,
        global_timeout_ms=(args.global_timeout_ms if args.global_timeout_ms > 0 else None),
    )

    for result in results:
        print(json.dumps(asdict(result), sort_keys=True))

    summary = summarize(results, shared)
    print(json.dumps(summary, sort_keys=True))

    if shared.poison_items:
        print(json.dumps({"poison_items": [asdict(r) for r in shared.poison_items]}, sort_keys=True))

    has_failure = any(r.status in {POISON, FAILED, CANCELLED} for r in results)
    return 1 if has_failure else 0


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
