#!/usr/bin/env python3
"""Round 2 asyncio pipeline with additional control-plane features.

This script extends the basic worker-pool design with:
- global request-start rate limiting,
- global deadline enforcement,
- graceful cancellation and partial-result emission,
- detailed status accounting.

The comments are intentionally dense so each mechanism is easy to trace.

Quick talk track:
1) Start-rate limiter controls starts/sec globally.
2) Worker pool still caps in-flight concurrency.
3) Global deadline stops intake and drives explicit cancellation.
4) Ordered terminal output plus lifecycle counters verify closure.
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
FAILED = "failed"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"
NOT_STARTED = "not_started"


@dataclass(slots=True)
class Result:
    """Terminal record for a single item."""

    idx: int
    item_id: str
    status: str
    attempts: int
    latency_ms: float
    error: str | None


class RateLimiter:
    """Simple lock-protected start-rate limiter.

    The limiter enforces a minimum interval between *start events* globally.
    This is enough for stable request pacing in queue/worker systems.
    """

    def __init__(self, rps: float) -> None:
        # `rps <= 0` means "unlimited".
        self.interval = 1.0 / rps if rps > 0.0 else 0.0

        # The lock serializes updates to `_next_allowed_start`.
        self._lock = asyncio.Lock()

        # Monotonic timestamp for the next permitted start.
        self._next_allowed_start = 0.0

    async def acquire(self) -> None:
        """Wait until starting a new attempt is allowed."""

        if self.interval <= 0.0:
            return

        async with self._lock:
            now = time.perf_counter()

            # Sleep when we are ahead of schedule.
            if now < self._next_allowed_start:
                await asyncio.sleep(self._next_allowed_start - now)
                now = time.perf_counter()

            # Reserve the next time slot.
            self._next_allowed_start = max(now, self._next_allowed_start) + self.interval


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Round 2 asyncio controls demo")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON array: [{id, delay_ms, fail_rate}, ...]",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum active workers and in-flight operations.",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=20.0,
        help="Global max attempt starts per second. Set 0 to disable.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=800,
        help="Per-attempt timeout in milliseconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retries per item after the first attempt.",
    )
    parser.add_argument(
        "--global-timeout-ms",
        type=int,
        default=5_000,
        help="Hard deadline for the full run in milliseconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by failure simulation.",
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
        if "id" not in item:
            raise ValueError(f"Item at index {index} is missing 'id'.")
        if "delay_ms" not in item:
            raise ValueError(f"Item at index {index} is missing 'delay_ms'.")
        if "fail_rate" not in item:
            raise ValueError(f"Item at index {index} is missing 'fail_rate'.")
        items.append(item)
    return items


def _success_latency_percentiles(values: list[float]) -> tuple[float, float]:
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


async def simulated_fetch(item: WorkItem) -> str:
    """Simulate I/O latency and probabilistic failure."""

    await asyncio.sleep(float(item["delay_ms"]) / 1000.0)
    if random.random() < float(item.get("fail_rate", 0.0)):
        raise RuntimeError("simulated fetch failure")
    return "ok"


async def run_one(item: WorkItem, timeout_ms: int) -> str:
    """Run one attempt under a strict timeout."""

    return await asyncio.wait_for(
        simulated_fetch(item),
        timeout=float(timeout_ms) / 1000.0,
    )


async def worker(
    queue: asyncio.Queue[tuple[int, WorkItem] | None],
    results: list[Result],
    semaphore: asyncio.Semaphore,
    limiter: RateLimiter,
    timeout_ms: int,
    retries: int,
    stop_event: asyncio.Event,
) -> None:
    """Process items until sentinel or cancellation occurs."""

    while True:
        item = await queue.get()

        # Sentinel path for clean shutdown when all work completes normally.
        if item is None:
            queue.task_done()
            return

        idx, work_item = item

        # If global stop already fired, do not launch more work.
        if stop_event.is_set():
            results[idx] = Result(
                idx=idx,
                item_id=str(work_item["id"]),
                status=CANCELLED,
                attempts=0,
                latency_ms=0.0,
                error="global timeout reached before start",
            )
            queue.task_done()
            continue

        started_at = time.perf_counter()
        attempts = 0
        final_status = FAILED
        final_error: str | None = "unknown_failure"

        for attempt_index in range(retries + 1):
            # Stop launching new attempts once global timeout has fired.
            if stop_event.is_set():
                final_status = CANCELLED
                final_error = "global timeout reached during retries"
                break

            attempts = attempt_index + 1

            try:
                # Global request-start shaping happens before external call.
                await limiter.acquire()

                # Bound true in-flight work against downstream systems.
                async with semaphore:
                    await run_one(work_item, timeout_ms=timeout_ms)

                final_status = SUCCESS
                final_error = None
                break

            except asyncio.TimeoutError:
                final_status = TIMED_OUT
                final_error = f"timeout_after_{timeout_ms}ms"

            except asyncio.CancelledError:
                # Worker task cancellation should map to explicit item status.
                final_status = CANCELLED
                final_error = "worker cancelled"
                break

            except Exception as exc:  # noqa: BLE001 - status classification is explicit.
                final_status = FAILED
                final_error = f"{type(exc).__name__}: {exc}"

            if attempt_index < retries:
                # Exponential backoff with a max cap to avoid runaway waits.
                backoff_seconds = min(0.5, 0.05 * (2**attempt_index))
                await asyncio.sleep(backoff_seconds)

        latency_ms = (time.perf_counter() - started_at) * 1000.0

        results[idx] = Result(
            idx=idx,
            item_id=str(work_item["id"]),
            status=final_status,
            attempts=attempts,
            latency_ms=latency_ms,
            error=final_error,
        )

        queue.task_done()


def summarize(results: list[Result]) -> dict[str, Any]:
    """Aggregate status counts and latency distribution."""

    success_latencies: list[float] = []
    started = 0
    success = 0
    failed = 0
    timed_out = 0
    cancelled = 0
    for result in results:
        if result.attempts > 0:
            started += 1
        if result.status == SUCCESS:
            success += 1
            success_latencies.append(result.latency_ms)
        elif result.status == FAILED:
            failed += 1
        elif result.status == TIMED_OUT:
            timed_out += 1
        elif result.status == CANCELLED:
            cancelled += 1
    p50_success_ms, p95_success_ms = _success_latency_percentiles(success_latencies)

    summary = {
        "queued": len(results),
        "started": started,
        "completed": len(results),
        "success": success,
        "failed": failed,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "p50_success_ms": p50_success_ms,
        "p95_success_ms": p95_success_ms,
    }
    return summary


async def run_pipeline(args: argparse.Namespace, items: list[WorkItem]) -> list[Result]:
    """Run workers with deadline handling and partial-result guarantees."""

    worker_count = max(1, int(args.max_workers))
    retries = max(0, int(args.retries))

    queue: asyncio.Queue[tuple[int, WorkItem] | None] = asyncio.Queue()
    for idx, work_item in enumerate(items):
        queue.put_nowait((idx, work_item))

    # Pre-fill with explicit NOT_STARTED status for deterministic fallback output.
    results: list[Result] = [
        Result(
            idx=idx,
            item_id=str(work_item["id"]),
            status=NOT_STARTED,
            attempts=0,
            latency_ms=0.0,
            error=None,
        )
        for idx, work_item in enumerate(items)
    ]

    semaphore = asyncio.Semaphore(worker_count)
    limiter = RateLimiter(float(args.rps))
    stop_event = asyncio.Event()

    workers = [
        asyncio.create_task(
            worker(
                queue=queue,
                results=results,
                semaphore=semaphore,
                limiter=limiter,
                timeout_ms=int(args.timeout_ms),
                retries=retries,
                stop_event=stop_event,
            ),
            name=f"worker-{i}",
        )
        for i in range(worker_count)
    ]

    try:
        # Global deadline controls total runtime for queue completion.
        await asyncio.wait_for(
            queue.join(),
            timeout=float(args.global_timeout_ms) / 1000.0,
        )

        # Normal path: all items processed before deadline.
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers, return_exceptions=True)

    except asyncio.TimeoutError:
        # Deadline path: stop launching new attempts and preserve partial output.
        stop_event.set()

        # Drain any unstarted queue items and mark them cancelled.
        while True:
            try:
                queued_item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if queued_item is None:
                queue.task_done()
                continue

            idx, work_item = queued_item
            if results[idx].status == NOT_STARTED:
                results[idx] = Result(
                    idx=idx,
                    item_id=str(work_item["id"]),
                    status=CANCELLED,
                    attempts=0,
                    latency_ms=0.0,
                    error="global timeout reached before dispatch",
                )
            queue.task_done()

        # After draining backlog, stop workers cleanly with explicit sentinels.
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers, return_exceptions=True)

    # Convert any remaining NOT_STARTED entries to cancelled for explicit terminal states.
    for idx, result in enumerate(results):
        if result.status == NOT_STARTED:
            results[idx] = Result(
                idx=result.idx,
                item_id=result.item_id,
                status=CANCELLED,
                attempts=0,
                latency_ms=0.0,
                error="pipeline terminated before dispatch",
            )

    return results


async def main_async(args: argparse.Namespace) -> int:
    """Async entrypoint."""

    random.seed(args.seed)
    items = load_items(args.input)

    results = await run_pipeline(args=args, items=items)

    # Emit ordered per-item result rows.
    for result in results:
        print(json.dumps(asdict(result), sort_keys=True))

    # Emit summary as final line.
    summary = summarize(results)
    print(json.dumps(summary, sort_keys=True))

    # Any non-success status yields non-zero exit code.
    has_non_success = any(result.status != SUCCESS for result in results)
    return 1 if has_non_success else 0


def main() -> int:
    """Sync wrapper."""

    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
