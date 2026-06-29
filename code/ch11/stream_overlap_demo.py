"""stream_overlap_demo.py - Chapter 11 CUDA stream overlap demo (tool).

Runs a small end-to-end overlap experiment using the same benchmark classes as
the Chapter 11 stream overlap examples, but without participating in benchmark
discovery (this file is not named baseline_/optimized_).
"""

from __future__ import annotations

import argparse

import torch

from ch11.stream_overlap_base import ConcurrentStreamOptimized, StridedStreamBaseline


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Chapter 11 stream overlap demo.")
    return torch.device("cuda")


def _time_ms(fn, *, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    start.record(current_stream)
    for _ in range(iters):
        fn()
    end.record(current_stream)
    end.synchronize()
    return float(start.elapsed_time(end)) / max(iters, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 11 stream overlap demo")
    parser.add_argument("--num-elements", type=int, default=24_000_000, help="1D tensor length to stream H2D/D2H.")
    parser.add_argument("--num-segments", type=int, default=16, help="Number of chunks to split the workload into.")
    parser.add_argument("--num-streams", type=int, default=2, help="Number of CUDA streams for the overlapped path.")
    parser.add_argument("--iterations", type=int, default=20, help="Measurement iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    args = parser.parse_args()

    _require_cuda()

    baseline = StridedStreamBaseline(
        "stream_overlap_sequential",
        num_elements=int(args.num_elements),
        num_segments=int(args.num_segments),
    )
    optimized = ConcurrentStreamOptimized(
        "stream_overlap_overlapped",
        num_elements=int(args.num_elements),
        num_segments=int(args.num_segments),
        num_streams=int(args.num_streams),
    )

    baseline.setup()
    optimized.setup()
    try:
        for _ in range(int(args.warmup)):
            baseline.benchmark_fn()
            optimized.benchmark_fn()
        torch.cuda.synchronize()

        base_ms = _time_ms(baseline.benchmark_fn, iters=int(args.iterations))
        opt_ms = _time_ms(optimized.benchmark_fn, iters=int(args.iterations))
    finally:
        baseline.teardown()
        optimized.teardown()

    speedup = (base_ms / opt_ms) if opt_ms > 0 else float("inf")

    print(f"baseline mean:  {base_ms:.3f} ms/iter")
    print(f"optimized mean: {opt_ms:.3f} ms/iter")
    print(f"speedup:        {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
