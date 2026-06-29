#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

import torch

from core.benchmark.utils import scalar_tensor_to_float
from labs.fullstack_cluster.capstone_extension import load_capstone_module


def _time_kernel(fn, iters: int, timeout_s: float, warmup: int = 5) -> float:
    """Return average runtime (ms) for fn across iters with a wall-clock timeout."""
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    start_evt.record(current_stream)
    wall_start = time.perf_counter()
    for _ in range(max(1, iters)):
        fn()
        if timeout_s and (time.perf_counter() - wall_start) > timeout_s:
            raise TimeoutError("Kernel timing exceeded timeout")
    end_evt.record(current_stream)
    end_evt.synchronize()
    return start_evt.elapsed_time(end_evt) / max(1, iters)


def run_benchmark(size: int, iters: int, baseline_iters: int,
                  skip_baseline: bool, timeout_s: float) -> None:
    torch.manual_seed(0)
    module = load_capstone_module()
    device = torch.device("cuda")
    dtype = torch.float16

    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)

    # Correctness check on a manageable slice to avoid long runtimes.
    check_dim = min(size, 256)
    check_a = torch.randn(check_dim, check_dim, dtype=dtype, device=device)
    check_b = torch.randn(check_dim, check_dim, dtype=dtype, device=device)
    ref = module.baseline_matmul(check_a, check_b)
    opt = module.optimized_matmul(check_a, check_b)
    max_diff = scalar_tensor_to_float((ref - opt).abs().max())

    optim_ms = _time_kernel(lambda: module.optimized_matmul(a, b),
                            iters, timeout_s)
    flop = 2.0 * size * size * size
    opt_tflops = flop / (optim_ms * 1e9)

    base_tflops = float("nan")
    if not skip_baseline:
        base_ms = _time_kernel(lambda: module.baseline_matmul(a, b),
                               baseline_iters, timeout_s)
        base_tflops = flop / (base_ms * 1e9)
        speedup = base_ms / optim_ms
    else:
        base_ms = float("nan")
        speedup = float("nan")

    header = (
        f"Blackwell GB10 benchmark @ {size}x{size} matmul "
        f"(it={iters}, baseline_it={baseline_iters})"
    )
    print(header)
    print("-" * len(header))
    print(f"Max |Δ| between kernels on {check_dim}² slice: {max_diff:.3e}")
    print()
    print("Kernel              Avg ms    TFLOP/s")
    print("--------------------------------------")
    if not skip_baseline:
        print(f"baseline_naive   {base_ms:8.3f}   {base_tflops:7.3f}")
    else:
        print("baseline_naive      skipped       skipped")
    print(f"optimized_cluster {optim_ms:8.3f}   {opt_tflops:7.3f}")
    if not skip_baseline:
        print(f"\nSpeedup (baseline/optimized): {speedup:.2f}x")
    else:
        print("\nSpeedup: (baseline skipped)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the lab baseline vs optimized kernel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=2048,
                        help="Matrix dimension (square GEMM)")
    parser.add_argument("--iters", type=int, default=3,
                        help="Timing iterations for optimized kernel")
    parser.add_argument("--baseline-iters", type=int, default=1,
                        help="Timing iterations for the baseline kernel")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip timing the slow baseline kernel")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Wall clock timeout (seconds) per timing loop")
    args = parser.parse_args()

    run_benchmark(args.size, args.iters, args.baseline_iters,
                  args.skip_baseline, args.timeout)


if __name__ == "__main__":
    main()
