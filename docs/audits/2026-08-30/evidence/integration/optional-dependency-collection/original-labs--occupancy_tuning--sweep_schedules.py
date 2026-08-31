#!/usr/bin/env python3
"""Benchmark every Triton schedule and emit a CSV for the Occupancy lab."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

from labs.occupancy_tuning import triton_matmul
from labs.occupancy_tuning.triton_matmul_schedules import MatmulSchedule, SCHEDULES


@dataclass
class SweepResult:
    """Per-schedule measurement recorded by the sweep."""

    name: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    size_m: int
    size_n: int
    size_k: int
    dtype: str
    warmup: int
    iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    tflops: float
    notes: str
    error: str | None = None


SUPPORTED_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for labs/occupancy_tuning sweep.")


def resolve_schedules(names: Sequence[str] | None) -> List[MatmulSchedule]:
    """Map CLI names to MatmulSchedule entries."""

    if not names:
        return list(SCHEDULES)

    name_to_schedule = {schedule.name: schedule for schedule in SCHEDULES}
    resolved: List[MatmulSchedule] = []
    for name in names:
        if name.lower() == "all":
            return list(SCHEDULES)
        if name not in name_to_schedule:
            known = ", ".join(name_to_schedule)
            raise ValueError(f"Unknown schedule '{name}'. Known schedules: {known}")
        resolved.append(name_to_schedule[name])
    return resolved


def _allocate_inputs(
    size_m: int,
    size_n: int,
    size_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randn((size_m, size_k), dtype=dtype, device=device)
    b = torch.randn((size_k, size_n), dtype=dtype, device=device)
    c = torch.empty((size_m, size_n), dtype=dtype, device=device)
    return a, b, c


def benchmark_schedule(
    schedule: MatmulSchedule,
    *,
    size_m: int,
    size_n: int,
    size_k: int,
    dtype: torch.dtype,
    iterations: int,
    warmup: int,
    use_compile: bool,
) -> SweepResult:
    """Benchmark a single preset and return timing statistics."""

    _ensure_cuda()
    device = torch.device("cuda")
    a, b, c = _allocate_inputs(size_m, size_n, size_k, dtype, device)

    def _run_once() -> torch.Tensor:
        return triton_matmul.run_one(
            M=size_m,
            N=size_n,
            K=size_k,
            bm=schedule.block_m,
            bn=schedule.block_n,
            bk=schedule.block_k,
            nw=schedule.num_warps,
            dtype=dtype,
            device=device,
            a=a,
            b=b,
            c=c,
        )

    runner = _run_once
    if use_compile and hasattr(torch, "compile"):  # pragma: no cover - torch.compile optional
        try:
            runner = torch.compile(_run_once, fullgraph=True)  # type: ignore[arg-type]
        except Exception:
            runner = _run_once

    for _ in range(max(0, warmup)):
        runner()
    torch.cuda.synchronize()

    times_ms: List[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    for _ in range(max(1, iterations)):
        start_event.record(current_stream)
        runner()
        end_event.record(current_stream)
        end_event.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    sample_count = len(times_ms)
    total_ms = 0.0
    for time_ms in times_ms:
        total_ms += time_ms

    times_ms.sort()
    midpoint = sample_count // 2
    if sample_count % 2:
        median_ms = times_ms[midpoint]
    else:
        median_ms = (times_ms[midpoint - 1] + times_ms[midpoint]) / 2.0
    mean_ms = total_ms / sample_count
    min_ms = times_ms[0]
    max_ms = times_ms[-1]

    total_flops = 2.0 * size_m * size_n * size_k
    tflops = (total_flops / (mean_ms / 1e3)) / 1e12

    return SweepResult(
        name=schedule.name,
        block_m=schedule.block_m,
        block_n=schedule.block_n,
        block_k=schedule.block_k,
        num_warps=schedule.num_warps,
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        dtype=str(dtype).replace("torch.", ""),
        warmup=warmup,
        iterations=iterations,
        mean_ms=mean_ms,
        median_ms=median_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        tflops=tflops,
        notes=schedule.notes,
    )


def run_sweep(
    schedules: Iterable[MatmulSchedule],
    *,
    size: int,
    iterations: int,
    warmup: int,
    dtype: torch.dtype,
    use_compile: bool,
) -> List[SweepResult]:
    """Run all schedules and return their measurements."""

    results: List[SweepResult] = []
    for schedule in schedules:
        try:
            result = benchmark_schedule(
                schedule,
                size_m=size,
                size_n=size,
                size_k=size,
                dtype=dtype,
                iterations=iterations,
                warmup=warmup,
                use_compile=use_compile,
            )
        except Exception as exc:  # pragma: no cover - forwarded to CLI user
            result = SweepResult(
                name=schedule.name,
                block_m=schedule.block_m,
                block_n=schedule.block_n,
                block_k=schedule.block_k,
                num_warps=schedule.num_warps,
                size_m=size,
                size_n=size,
                size_k=size,
                dtype=str(dtype).replace("torch.", ""),
                warmup=warmup,
                iterations=iterations,
                mean_ms=0.0,
                median_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                tflops=0.0,
                notes=schedule.notes,
                error=str(exc),
            )
        results.append(result)
    return results


def _write_csv(results: Sequence[SweepResult], path: Path) -> None:
    fields = [
        "name",
        "block_m",
        "block_n",
        "block_k",
        "num_warps",
        "size_m",
        "size_n",
        "size_k",
        "dtype",
        "warmup",
        "iterations",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "tflops",
        "notes",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Triton Proton schedules and log timing/TFLOP numbers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=2048, help="Square matmul size (M=N=K).")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations.")
    parser.add_argument(
        "--dtype",
        choices=sorted(SUPPORTED_DTYPES),
        default="fp16",
        help="Tensor dtype for the sweep.",
    )
    parser.add_argument(
        "--schedule",
        action="append",
        dest="schedules",
        help="Restrict the sweep to one or more schedule names (use --schedule all to keep every preset).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile wrapping of the Triton runner.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known schedules and exit.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/occupancy_tuning/sweep_results.csv"),
        help="Whereto store the CSV output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.list:
        print("Available schedules:")
        for schedule in SCHEDULES:
            print(f"  - {schedule.name}: {schedule.notes}")
        return 0

    try:
        schedules = resolve_schedules(args.schedules)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    dtype = SUPPORTED_DTYPES[args.dtype]
    print(
        f"Sweeping {len(schedules)} schedules (size={args.size}, dtype={args.dtype}, "
        f"iterations={args.iterations}, warmup={args.warmup})"
    )
    results = run_sweep(
        schedules,
        size=args.size,
        iterations=args.iterations,
        warmup=args.warmup,
        dtype=dtype,
        use_compile=not args.no_compile,
    )

    for result in results:
        if result.error:
            status = f"ERROR: {result.error}"
        else:
            status = (
                f"{result.mean_ms:.2f} ms avg (median {result.median_ms:.2f} ms, "
                f"{result.tflops:.2f} TFLOP/s)"
            )
        print(f"{result.name:>30}: {status}")

    _write_csv(results, args.csv)
    print(f"\nCSV saved to {args.csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
