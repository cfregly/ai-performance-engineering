#!/usr/bin/env python3
"""Round 1 asyncio worker-pool example.

This module demonstrates a production-style pattern for I/O-bound batch work:
- bounded worker count,
- explicit queue-based backpressure,
- per-item timeout,
- bounded retry policy,
- deterministic output ordering,
- summary metrics and exit code.

The script intentionally includes detailed comments so each control-flow decision
is easy to discuss and reason about.

Quick talk track:
1) Queue + fixed workers for bounded concurrency.
2) Per-item timeout and bounded retries for reliability.
3) Results stored by input index for deterministic output order.
4) Summary counters and exit code make behavior easy to validate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# A narrow alias makes type hints shorter and keeps function signatures readable.
WorkItem = dict[str, Any]


@dataclass(slots=True)
class ItemResult:
    """Terminal state for one input item.

    Attributes:
        idx: Original index from the input list. Used to preserve output order.
        item_id: Stable human-readable identifier from input JSON.
        ok: True when work completed successfully.
        timed_out: True when the final failure mode was timeout.
        attempts: Number of attempts performed (initial try + retries).
        latency_ms: End-to-end latency for this item across all attempts.
        error: Error string for failed items, otherwise None.
        value: Successful transformed output, otherwise None.
    """

    idx: int
    item_id: str
    ok: bool
    timed_out: bool
    attempts: int
    latency_ms: float
    error: str | None
    value: str | None


def parse_args() -> argparse.Namespace:
    """Parse CLI flags.

    Defaults are intentionally conservative and safe:
    - moderate worker count,
    - strict timeout,
    - one retry for transient issues.
    """

    parser = argparse.ArgumentParser(description="Round 1 asyncio worker-pool demo")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON array of items: [{id, payload, duration_ms, should_fail?}, ...]",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of workers and in-flight operations.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=500,
        help="Per-attempt timeout in milliseconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of retries after the first failed attempt.",
    )
    return parser.parse_args()


def load_items(path: Path) -> list[WorkItem]:
    """Load and validate input items.

    We fail fast on malformed inputs so failures happen at startup instead of in
    worker tasks, which simplifies runtime behavior.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of objects.")

    items: list[WorkItem] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {index} is not an object.")
        if "id" not in item:
            raise ValueError(f"Item at index {index} is missing required field 'id'.")
        if "payload" not in item:
            raise ValueError(f"Item at index {index} is missing required field 'payload'.")
        if "duration_ms" not in item:
            raise ValueError(f"Item at index {index} is missing required field 'duration_ms'.")
        items.append(item)
    return items


async def run_one(item: WorkItem) -> str:
    """Simulate one I/O operation.

    The sleep models network or disk latency.
    The optional `should_fail` flag forces deterministic failure injection.
    """

    # Convert milliseconds to seconds for asyncio.sleep.
    delay_seconds = float(item["duration_ms"]) / 1000.0
    await asyncio.sleep(delay_seconds)

    # Force an error path when requested by the input item.
    if bool(item.get("should_fail", False)):
        raise RuntimeError("simulated failure")

    # Return a transformed output payload on success.
    return f"processed:{item['payload']}"


async def worker(
    queue: asyncio.Queue[tuple[int, WorkItem] | None],
    results: list[ItemResult | None],
    semaphore: asyncio.Semaphore,
    timeout_ms: int,
    retries: int,
) -> None:
    """Consume queue items until sentinel is received.

    Worker responsibilities:
    - pull one item from queue,
    - execute retry loop with timeout,
    - record terminal result,
    - always call queue.task_done().
    """

    # Loop forever until we receive a sentinel value.
    while True:
        item = await queue.get()

        # Sentinel design keeps shutdown explicit and deterministic.
        if item is None:
            queue.task_done()
            return

        idx, work_item = item
        started_at = time.perf_counter()

        # Initialize terminal-state variables before attempts begin.
        attempts = 0
        ok = False
        timed_out = False
        error: str | None = None
        value: str | None = None

        # `retries=1` means: first attempt + one retry => 2 total attempts.
        for attempt_index in range(retries + 1):
            attempts = attempt_index + 1

            try:
                # The semaphore is a second safety net that bounds in-flight
                # operations even if more workers exist.
                async with semaphore:
                    value = await asyncio.wait_for(
                        run_one(work_item),
                        timeout=float(timeout_ms) / 1000.0,
                    )

                # Success ends the retry loop immediately.
                ok = True
                timed_out = False
                error = None
                break

            except asyncio.TimeoutError:
                # Timeout is tracked separately from generic failure.
                ok = False
                timed_out = True
                error = f"timeout_after_{timeout_ms}ms"

            except Exception as exc:  # noqa: BLE001 - explicit classification below.
                # Non-timeout failures stay non-timeout even after retries.
                ok = False
                timed_out = False
                error = f"{type(exc).__name__}: {exc}"

            # Apply exponential backoff before next attempt.
            if attempt_index < retries:
                backoff_seconds = min(0.5, 0.05 * (2**attempt_index))
                await asyncio.sleep(backoff_seconds)

        # Measure full item latency across all attempts.
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        # Store result by original index to preserve deterministic output order.
        results[idx] = ItemResult(
            idx=idx,
            item_id=str(work_item["id"]),
            ok=ok,
            timed_out=timed_out,
            attempts=attempts,
            latency_ms=latency_ms,
            error=error if not ok else None,
            value=value,
        )

        # Mark queue item as fully processed.
        queue.task_done()


def percentile(values: list[float], p: float) -> float:
    """Simple percentile helper using nearest-rank style indexing."""

    if not values:
        return 0.0
    values.sort()
    return _percentile_from_ordered(values, p)


def _percentile_from_ordered(ordered: list[float], p: float) -> float:
    rank = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[rank]


def _latency_percentiles(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0

    values.sort()
    midpoint = len(values) // 2
    if len(values) % 2:
        p50 = values[midpoint]
    else:
        p50 = (values[midpoint - 1] + values[midpoint]) / 2.0
    return round(p50, 2), round(_percentile_from_ordered(values, 95.0), 2)


async def run_pipeline(items: list[WorkItem], max_workers: int, timeout_ms: int, retries: int) -> list[ItemResult]:
    """Run full worker-pool pipeline and return ordered terminal results."""

    # Defensive bounds avoid invalid runtime configuration.
    bounded_workers = max(1, int(max_workers))
    bounded_retries = max(0, int(retries))

    # Queue items as (index, payload) tuples.
    queue: asyncio.Queue[tuple[int, WorkItem] | None] = asyncio.Queue()
    for idx, work_item in enumerate(items):
        queue.put_nowait((idx, work_item))

    # Pre-size result buffer so workers can write by index.
    results: list[ItemResult | None] = [None] * len(items)

    # Semaphore and worker count both use max_workers in this baseline pattern.
    semaphore = asyncio.Semaphore(bounded_workers)

    workers = [
        asyncio.create_task(
            worker(
                queue=queue,
                results=results,
                semaphore=semaphore,
                timeout_ms=timeout_ms,
                retries=bounded_retries,
            ),
            name=f"worker-{i}",
        )
        for i in range(bounded_workers)
    ]

    # Wait until all queued work has been processed.
    await queue.join()

    # Send one sentinel per worker so each task exits cleanly.
    for _ in workers:
        queue.put_nowait(None)

    # Ensure all workers actually terminate.
    await asyncio.gather(*workers)

    # Convert Optional list into concrete list after validating no gaps remain.
    finalized: list[ItemResult] = []
    for idx, result in enumerate(results):
        if result is None:
            raise RuntimeError(f"Missing result for index {idx}; worker exited unexpectedly.")
        finalized.append(result)
    return finalized


def summarize(results: list[ItemResult]) -> dict[str, Any]:
    """Build aggregate metrics for quick health/throughput review."""

    latencies: list[float] = []
    success_count = 0
    timed_out_count = 0
    for result in results:
        latencies.append(result.latency_ms)
        if result.ok:
            success_count += 1
        elif result.timed_out:
            timed_out_count += 1
    fail_count = len(results) - success_count
    p50_ms, p95_ms = _latency_percentiles(latencies)

    return {
        "total": len(results),
        "success": success_count,
        "failed": fail_count,
        "timed_out": timed_out_count,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
    }


async def main_async(args: argparse.Namespace) -> int:
    """Top-level async entrypoint."""

    items = load_items(args.input)

    results = await run_pipeline(
        items=items,
        max_workers=args.max_workers,
        timeout_ms=args.timeout_ms,
        retries=args.retries,
    )

    # Emit per-item output in deterministic input order.
    for result in results:
        print(json.dumps(asdict(result), sort_keys=True))

    # Emit aggregate summary as final line.
    summary = summarize(results)
    print(json.dumps(summary, sort_keys=True))

    # Non-zero exit code when any item failed.
    return 0 if summary["failed"] == 0 else 1


def main() -> int:
    """Synchronous wrapper for argparse + asyncio.run."""

    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
