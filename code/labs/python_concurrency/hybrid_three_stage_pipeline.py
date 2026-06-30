#!/usr/bin/env python3
"""Hybrid 3-stage pipeline example.

Pipeline shape:
- Stage A (async I/O): fetch-like asynchronous work.
- Stage B (CPU parse): pure-Python computation in ProcessPoolExecutor.
- Stage C (async I/O): write-like asynchronous work.

Design goals:
- bounded handoff queues for backpressure,
- explicit stage terminal states,
- deterministic final output ordering,
- clear per-stage latency attribution.

Quick talk track:
1) Stage A async fetch handles wait-heavy work.
2) Stage B process pool handles CPU-heavy parsing.
3) Stage C async write finishes side effects.
4) Bounded A->B and B->C queues prevent memory blowups.
5) Sentinel shutdown ordering guarantees clean stage drain.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


WorkItem = dict[str, Any]

SUCCESS = "success"
FAILED_STAGE_A = "failed_stage_a"
FAILED_STAGE_B = "failed_stage_b"
FAILED_STAGE_C = "failed_stage_c"


def _total_latency_percentiles(values: list[float]) -> tuple[float, float]:
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


@dataclass(slots=True)
class PipelineResult:
    """Terminal record for one item across all three stages."""

    idx: int
    item_id: str
    status: str
    fetch_ms: float
    cpu_ms: float
    write_ms: float
    total_ms: float
    output: str | None
    error: str | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Hybrid 3-stage asyncio/process-pool pipeline demo")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to JSON array: "
            "[{id, payload, fetch_delay_ms, write_delay_ms, fetch_fail?, write_fail?}, ...]"
        ),
    )
    parser.add_argument("--stage-a-workers", type=int, default=4, help="Worker count for async fetch stage.")
    parser.add_argument("--stage-b-workers", type=int, default=4, help="Worker/process count for CPU parse stage.")
    parser.add_argument("--stage-c-workers", type=int, default=4, help="Worker count for async write stage.")
    parser.add_argument("--queue-size", type=int, default=16, help="Max size for A->B and B->C bounded queues.")
    parser.add_argument("--fetch-timeout-ms", type=int, default=1_000, help="Stage A timeout per item.")
    parser.add_argument("--write-timeout-ms", type=int, default=1_000, help="Stage C timeout per item.")
    parser.add_argument(
        "--cpu-rounds",
        type=int,
        default=35_000,
        help="CPU loop size used in stage B parser (pure Python).",
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
        for field in ("id", "payload", "fetch_delay_ms", "write_delay_ms"):
            if field not in item:
                raise ValueError(f"Item at index {index} is missing required field '{field}'.")
        items.append(item)
    return items


async def simulated_fetch(item: WorkItem) -> str:
    """Stage A: asynchronous fetch-like operation."""

    await asyncio.sleep(float(item["fetch_delay_ms"]) / 1000.0)
    if bool(item.get("fetch_fail", False)):
        raise RuntimeError("simulated stage A fetch failure")
    return str(item["payload"])


def cpu_parse_payload(payload: str, cpu_rounds: int) -> dict[str, Any]:
    """Stage B: pure-Python CPU transform for process-pool execution.

    The loop intentionally stays in Python bytecode to represent CPU-heavy
    parsing/transform logic that benefits from process-based parallelism.
    """

    checksum = 0
    for i in range(1, cpu_rounds + 1):
        checksum += (i * i) % 97
        checksum ^= (i << 1) & 0xFFFF

    transformed = payload.strip().upper()
    token_count = _count_whitespace_separated_tokens(transformed)

    return {
        "text": transformed,
        "checksum": checksum,
        "token_count": token_count,
    }


async def simulated_write(item: WorkItem, parsed: dict[str, Any]) -> str:
    """Stage C: asynchronous write-like operation."""

    await asyncio.sleep(float(item["write_delay_ms"]) / 1000.0)
    if bool(item.get("write_fail", False)):
        raise RuntimeError("simulated stage C write failure")

    # Return a compact synthesized output for demonstration.
    return (
        f"stored:id={item['id']}:text={parsed['text']}:"
        f"tokens={parsed['token_count']}:checksum={parsed['checksum']}"
    )


async def stage_a_worker(
    input_queue: asyncio.Queue[tuple[int, WorkItem] | None],
    queue_a_to_b: asyncio.Queue[dict[str, Any] | None],
    results: list[PipelineResult | None],
    fetch_timeout_ms: int,
) -> None:
    """Stage A worker: fetch input and push envelope to stage B."""

    while True:
        queued = await input_queue.get()

        if queued is None:
            input_queue.task_done()
            return

        idx, item = queued
        item_started_at = time.perf_counter()
        fetch_started_at = item_started_at

        try:
            fetched_payload = await asyncio.wait_for(
                simulated_fetch(item),
                timeout=float(fetch_timeout_ms) / 1000.0,
            )
            fetch_ms = (time.perf_counter() - fetch_started_at) * 1000.0

            envelope = {
                "idx": idx,
                "item": item,
                "item_started_at": item_started_at,
                "fetch_ms": fetch_ms,
                "fetched_payload": fetched_payload,
            }

            # Bounded queue introduces backpressure when stage B is saturated.
            await queue_a_to_b.put(envelope)

        except Exception as exc:  # noqa: BLE001 - terminal status captures stage-specific failure.
            total_ms = (time.perf_counter() - item_started_at) * 1000.0
            results[idx] = PipelineResult(
                idx=idx,
                item_id=str(item["id"]),
                status=FAILED_STAGE_A,
                fetch_ms=total_ms,
                cpu_ms=0.0,
                write_ms=0.0,
                total_ms=total_ms,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        finally:
            input_queue.task_done()


async def stage_b_worker(
    queue_a_to_b: asyncio.Queue[dict[str, Any] | None],
    queue_b_to_c: asyncio.Queue[dict[str, Any] | None],
    results: list[PipelineResult | None],
    process_pool: ProcessPoolExecutor,
    cpu_rounds: int,
) -> None:
    """Stage B worker: run CPU parse in process pool and push to stage C."""

    loop = asyncio.get_running_loop()

    while True:
        envelope = await queue_a_to_b.get()

        if envelope is None:
            queue_a_to_b.task_done()
            return

        idx = int(envelope["idx"])
        item = envelope["item"]
        item_started_at = float(envelope["item_started_at"])
        fetch_ms = float(envelope["fetch_ms"])

        cpu_started_at = time.perf_counter()

        try:
            parsed_payload = await loop.run_in_executor(
                process_pool,
                cpu_parse_payload,
                str(envelope["fetched_payload"]),
                int(cpu_rounds),
            )
            cpu_ms = (time.perf_counter() - cpu_started_at) * 1000.0

            next_envelope = {
                "idx": idx,
                "item": item,
                "item_started_at": item_started_at,
                "fetch_ms": fetch_ms,
                "cpu_ms": cpu_ms,
                "parsed_payload": parsed_payload,
            }
            # Bounded queue introduces backpressure when stage C is saturated.
            await queue_b_to_c.put(next_envelope)

        except Exception as exc:  # noqa: BLE001
            total_ms = (time.perf_counter() - item_started_at) * 1000.0
            cpu_ms = (time.perf_counter() - cpu_started_at) * 1000.0
            results[idx] = PipelineResult(
                idx=idx,
                item_id=str(item["id"]),
                status=FAILED_STAGE_B,
                fetch_ms=fetch_ms,
                cpu_ms=cpu_ms,
                write_ms=0.0,
                total_ms=total_ms,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        finally:
            queue_a_to_b.task_done()


async def stage_c_worker(
    queue_b_to_c: asyncio.Queue[dict[str, Any] | None],
    results: list[PipelineResult | None],
    write_timeout_ms: int,
) -> None:
    """Stage C worker: perform final async write and record terminal result."""

    while True:
        envelope = await queue_b_to_c.get()

        if envelope is None:
            queue_b_to_c.task_done()
            return

        idx = int(envelope["idx"])
        item = envelope["item"]
        item_started_at = float(envelope["item_started_at"])
        fetch_ms = float(envelope["fetch_ms"])
        cpu_ms = float(envelope["cpu_ms"])

        write_started_at = time.perf_counter()

        try:
            output = await asyncio.wait_for(
                simulated_write(item, envelope["parsed_payload"]),
                timeout=float(write_timeout_ms) / 1000.0,
            )
            write_ms = (time.perf_counter() - write_started_at) * 1000.0
            total_ms = (time.perf_counter() - item_started_at) * 1000.0

            results[idx] = PipelineResult(
                idx=idx,
                item_id=str(item["id"]),
                status=SUCCESS,
                fetch_ms=fetch_ms,
                cpu_ms=cpu_ms,
                write_ms=write_ms,
                total_ms=total_ms,
                output=output,
                error=None,
            )

        except Exception as exc:  # noqa: BLE001
            write_ms = (time.perf_counter() - write_started_at) * 1000.0
            total_ms = (time.perf_counter() - item_started_at) * 1000.0

            results[idx] = PipelineResult(
                idx=idx,
                item_id=str(item["id"]),
                status=FAILED_STAGE_C,
                fetch_ms=fetch_ms,
                cpu_ms=cpu_ms,
                write_ms=write_ms,
                total_ms=total_ms,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        finally:
            queue_b_to_c.task_done()


async def run_pipeline(args: argparse.Namespace, items: list[WorkItem]) -> list[PipelineResult]:
    """Run full 3-stage pipeline with bounded handoff queues."""

    stage_a_workers = max(1, int(args.stage_a_workers))
    stage_b_workers = max(1, int(args.stage_b_workers))
    stage_c_workers = max(1, int(args.stage_c_workers))
    queue_size = max(1, int(args.queue_size))

    input_queue: asyncio.Queue[tuple[int, WorkItem] | None] = asyncio.Queue()
    queue_a_to_b: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=queue_size)
    queue_b_to_c: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=queue_size)

    for idx, item in enumerate(items):
        input_queue.put_nowait((idx, item))

    for _ in range(stage_a_workers):
        input_queue.put_nowait(None)

    results: list[PipelineResult | None] = [None] * len(items)

    # Use spawn so test runners and other multi-threaded hosts do not inherit a
    # forked event-loop state into the CPU stage workers.
    process_pool = ProcessPoolExecutor(
        max_workers=stage_b_workers,
        mp_context=mp.get_context("spawn"),
    )

    try:
        stage_a_tasks = [
            asyncio.create_task(
                stage_a_worker(
                    input_queue=input_queue,
                    queue_a_to_b=queue_a_to_b,
                    results=results,
                    fetch_timeout_ms=int(args.fetch_timeout_ms),
                ),
                name=f"stage-a-{i}",
            )
            for i in range(stage_a_workers)
        ]

        stage_b_tasks = [
            asyncio.create_task(
                stage_b_worker(
                    queue_a_to_b=queue_a_to_b,
                    queue_b_to_c=queue_b_to_c,
                    results=results,
                    process_pool=process_pool,
                    cpu_rounds=int(args.cpu_rounds),
                ),
                name=f"stage-b-{i}",
            )
            for i in range(stage_b_workers)
        ]

        stage_c_tasks = [
            asyncio.create_task(
                stage_c_worker(
                    queue_b_to_c=queue_b_to_c,
                    results=results,
                    write_timeout_ms=int(args.write_timeout_ms),
                ),
                name=f"stage-c-{i}",
            )
            for i in range(stage_c_workers)
        ]

        # Stage A drain and shutdown.
        await input_queue.join()
        await asyncio.gather(*stage_a_tasks)

        # Stage B receives explicit sentinels after stage A has fully stopped.
        for _ in range(stage_b_workers):
            await queue_a_to_b.put(None)
        await queue_a_to_b.join()
        await asyncio.gather(*stage_b_tasks)

        # Stage C receives explicit sentinels after stage B has fully stopped.
        for _ in range(stage_c_workers):
            await queue_b_to_c.put(None)
        await queue_b_to_c.join()
        await asyncio.gather(*stage_c_tasks)

    finally:
        process_pool.shutdown(wait=True, cancel_futures=True)

    finalized: list[PipelineResult] = []
    for idx, result in enumerate(results):
        if result is None:
            item = items[idx]
            finalized.append(
                PipelineResult(
                    idx=idx,
                    item_id=str(item["id"]),
                    status=FAILED_STAGE_C,
                    fetch_ms=0.0,
                    cpu_ms=0.0,
                    write_ms=0.0,
                    total_ms=0.0,
                    output=None,
                    error="missing_terminal_state",
                )
            )
        else:
            finalized.append(result)

    return finalized


def summarize(results: list[PipelineResult]) -> dict[str, Any]:
    """Create aggregate pipeline metrics."""

    total = len(results)
    success = 0
    failed_a = 0
    failed_b = 0
    failed_c = 0
    success_total_ms = 0.0
    totals: list[float] = []
    for result in results:
        totals.append(result.total_ms)
        if result.status == SUCCESS:
            success += 1
            success_total_ms += result.total_ms
        elif result.status == FAILED_STAGE_A:
            failed_a += 1
        elif result.status == FAILED_STAGE_B:
            failed_b += 1
        elif result.status == FAILED_STAGE_C:
            failed_c += 1
    p50_total_ms, p95_total_ms = _total_latency_percentiles(totals)
    mean_success_ms = round(success_total_ms / success, 2) if success else math.nan

    return {
        "total": total,
        "success": success,
        "failed_stage_a": failed_a,
        "failed_stage_b": failed_b,
        "failed_stage_c": failed_c,
        "p50_total_ms": p50_total_ms,
        "p95_total_ms": p95_total_ms,
        "mean_success_ms": mean_success_ms,
    }


async def main_async(args: argparse.Namespace) -> int:
    """Async entrypoint for CLI invocation."""

    items = load_items(args.input)
    results = await run_pipeline(args=args, items=items)

    for result in results:
        print(json.dumps(asdict(result), sort_keys=True))

    summary = summarize(results)
    print(json.dumps(summary, sort_keys=True))

    has_failure = any(result.status != SUCCESS for result in results)
    return 1 if has_failure else 0


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
