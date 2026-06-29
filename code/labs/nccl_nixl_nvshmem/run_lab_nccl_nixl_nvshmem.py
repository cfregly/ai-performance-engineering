#!/usr/bin/env python
"""Standalone runner for the NCCL/NIXL/NVSHMEM lab."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from core.benchmark.utils import scalar_tensor_to_float
from core.harness.benchmark_harness import lock_gpu_clocks
from labs.nccl_nixl_nvshmem.baseline_tier_handoff import get_benchmark as get_baseline_benchmark
from labs.nccl_nixl_nvshmem.comm_stack_common import (
    TierHandoffWorkload,
    apply_cli_overrides,
    default_workload,
    probe_communication_stack,
    require_stack,
)
from labs.nccl_nixl_nvshmem.optimized_tier_handoff import get_benchmark as get_optimized_benchmark


def _measure(bench: Any, *, warmup: int, iterations: int) -> float:
    for _ in range(max(warmup, 0)):
        bench.benchmark_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream()
        start.record(current_stream)
        for _ in range(max(iterations, 1)):
            bench.benchmark_fn()
        end.record(current_stream)
        end.synchronize()
        return float(start.elapsed_time(end) / max(iterations, 1))

    sample_count = max(iterations, 1)
    total_ms = 0.0
    for _ in range(sample_count):
        t0 = time.perf_counter()
        bench.benchmark_fn()
        total_ms += (time.perf_counter() - t0) * 1000.0
    return float(total_ms / sample_count)


def _run_compare(workload: TierHandoffWorkload, *, warmup: int, iterations: int) -> dict[str, Any]:
    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()
    baseline.set_workload(workload)
    optimized.set_workload(workload)
    baseline.setup()
    optimized.setup()
    try:
        baseline_ms = _measure(baseline, warmup=warmup, iterations=iterations)
        optimized_ms = _measure(optimized, warmup=warmup, iterations=iterations)
        baseline_error = baseline.validate_result()
        optimized_error = optimized.validate_result()
        max_abs_diff = scalar_tensor_to_float(
            (baseline.output - optimized.output).abs().max()
        )
        return {
            "workload": {
                "total_blocks": workload.total_blocks,
                "selected_blocks": workload.selected_blocks,
                "block_kib": workload.block_kib,
                "inner_iterations": workload.inner_iterations,
                "seed": workload.seed,
            },
            "baseline_latency_ms": baseline_ms,
            "optimized_latency_ms": optimized_ms,
            "speedup": baseline_ms / optimized_ms if optimized_ms > 0 else float("inf"),
            "max_abs_diff": max_abs_diff,
            "baseline_metrics": baseline.get_custom_metrics(),
            "optimized_metrics": optimized.get_custom_metrics(),
            "baseline_validation_error": baseline_error,
            "optimized_validation_error": optimized_error,
        }
    finally:
        baseline.teardown()
        optimized.teardown()


def _run_sweep(
    workload: TierHandoffWorkload,
    *,
    selected_blocks_values: list[int],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    rows = []
    for selected_blocks in selected_blocks_values:
        sweep_workload = TierHandoffWorkload(
            total_blocks=max(workload.total_blocks, selected_blocks),
            selected_blocks=selected_blocks,
            block_kib=workload.block_kib,
            inner_iterations=workload.inner_iterations,
            seed=workload.seed,
        )
        row = _run_compare(sweep_workload, warmup=warmup, iterations=iterations)
        rows.append(
            {
                "selected_blocks": selected_blocks,
                "baseline_latency_ms": row["baseline_latency_ms"],
                "optimized_latency_ms": row["optimized_latency_ms"],
                "speedup": row["speedup"],
                "max_abs_diff": row["max_abs_diff"],
            }
        )
    return {
        "base_workload": {
            "total_blocks": workload.total_blocks,
            "block_kib": workload.block_kib,
            "inner_iterations": workload.inner_iterations,
            "seed": workload.seed,
        },
        "rows": rows,
    }


def _parse_selected_blocks_csv(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("--sweep-selected-blocks must contain at least one integer")
    return values


def _render_probe_summary(probe: dict[str, Any]) -> str:
    lines = [
        "NCCL/NIXL/NVSHMEM stack probe",
        f"cuda_available: {probe['cuda_available']}",
        f"gpu_count: {probe['gpu_count']}",
        f"gpu_name: {probe['gpu_name']}",
        f"torch_nccl_available: {probe['torch_nccl_available']}",
        f"nccl_tests_binary: {probe['nccl_tests_binary']}",
        f"nixl_import_available: {probe['nixl_import_available']}",
        f"nvshmem_import_available: {probe['nvshmem_import_available']}",
        f"nvshmem_launcher: {probe['nvshmem_launcher']}",
        f"symmetric_memory_available: {probe['symmetric_memory_available']}",
        f"can_run_multi_gpu_nccl: {probe['can_run_multi_gpu_nccl']}",
        f"can_run_nvshmem_one_sided: {probe['can_run_nvshmem_one_sided']}",
        f"recommended_local_path: {probe['recommended_local_path']}",
    ]
    if probe["blockers"]:
        lines.append("blockers:")
        lines.extend(f"  - {item}" for item in probe["blockers"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "compare", "sweep"), default="probe")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per path")
    parser.add_argument("--iterations", type=int, default=20, help="Timed iterations per path")
    parser.add_argument(
        "--require-stack",
        action="append",
        choices=("nccl", "nixl", "nvshmem", "symmetric-memory"),
        default=[],
        help="Fail if the requested stack is unavailable on this host",
    )
    parser.add_argument(
        "--sweep-selected-blocks",
        default="32,64,96,128",
        help="Comma-separated selected-block counts for --mode sweep",
    )
    parser.add_argument(
        "--no-lock-gpu-clocks",
        dest="lock_gpu_clocks",
        action="store_false",
        help="Skip harness clock locking for local experimentation",
    )
    parser.add_argument("--sm-clock-mhz", type=int, default=None, help="Optional SM application clock")
    parser.add_argument("--mem-clock-mhz", type=int, default=None, help="Optional memory application clock")
    parser.add_argument("--json", action="store_true", help="Print JSON payload")
    parser.add_argument("--json-out", type=Path, default=None, help="Write JSON payload to file")
    parser.set_defaults(lock_gpu_clocks=True)
    args, workload_argv = parser.parse_known_args()

    workload = apply_cli_overrides(default_workload(), workload_argv)
    probe = probe_communication_stack()
    for stack in args.require_stack:
        require_stack(probe, stack)

    if args.mode == "probe":
        payload: dict[str, Any] = probe
    else:
        lock_ctx = (
            lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
            if args.lock_gpu_clocks
            else nullcontext()
        )
        with lock_ctx:
            if args.mode == "compare":
                payload = _run_compare(workload, warmup=args.warmup, iterations=args.iterations)
            else:
                payload = _run_sweep(
                    workload,
                    selected_blocks_values=_parse_selected_blocks_csv(args.sweep_selected_blocks),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
        payload["lock_gpu_clocks"] = args.lock_gpu_clocks
        payload["sm_clock_mhz"] = args.sm_clock_mhz
        payload["mem_clock_mhz"] = args.mem_clock_mhz
        payload["probe"] = probe

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.mode == "probe":
        print(_render_probe_summary(payload))
    elif args.mode == "compare":
        print("NCCL/NIXL/NVSHMEM tier handoff direct compare")
        print(
            f"shape=blocks{payload['workload']['total_blocks']} "
            f"selected{payload['workload']['selected_blocks']} "
            f"block_kib{payload['workload']['block_kib']} "
            f"inner{payload['workload']['inner_iterations']}"
        )
        print(f"baseline:  {payload['baseline_latency_ms']:.6f} ms")
        print(f"optimized: {payload['optimized_latency_ms']:.6f} ms")
        print(f"speedup:   {payload['speedup']:.3f}x")
        print(f"max_abs_diff: {payload['max_abs_diff']:.8f}")
        if args.json_out is not None:
            print(f"wrote: {args.json_out}")
    else:
        print("NCCL/NIXL/NVSHMEM tier handoff sweep")
        for row in payload["rows"]:
            print(
                f"selected_blocks={row['selected_blocks']:>4} "
                f"baseline={row['baseline_latency_ms']:.6f} ms "
                f"optimized={row['optimized_latency_ms']:.6f} ms "
                f"speedup={row['speedup']:.3f}x "
                f"max_abs_diff={row['max_abs_diff']:.8f}"
            )
        if args.json_out is not None:
            print(f"wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
