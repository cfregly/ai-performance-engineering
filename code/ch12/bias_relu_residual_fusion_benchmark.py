"""Benchmark bias+ReLU and residual-add as separate kernels vs fused kernel.

This script compiles a small CUDA extension on first run via the repo's
extension loader template and then:
1) validates numeric correctness (baseline vs fused), and
2) reports average latency (ms) and speedup.

Outputs:
- Standard console output
- A raw artifact: benchmark_output.txt in this directory
"""

from __future__ import annotations

from pathlib import Path

import json
from datetime import datetime

import torch

from ch12.cuda_extensions import load_bias_relu_residual_extension


def time_kernel(fn, iters: int, warmup: int = 5) -> float:
    """Return average runtime in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def run_benchmark(
    n_elements: int = 16_777_216,
    iterations: int = 100,
    warmup: int = 5,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    extension = load_bias_relu_residual_extension()

    # Flat tensors keeps kernel simple and deterministic.
    x = torch.randn((n_elements,), device="cuda", dtype=torch.float32)
    bias = torch.randn((n_elements,), device="cuda", dtype=torch.float32)
    residual = torch.randn((n_elements,), device="cuda", dtype=torch.float32)
    tmp = torch.empty_like(x)
    out_baseline = torch.empty_like(x)
    out_fused = torch.empty_like(x)

    # Correctness check (single pass each implementation)
    # Keep a fresh copy for each call.
    extension.separate_kernels(x, bias, residual, tmp, out_baseline, 1)
    extension.fused_kernel(x, bias, residual, out_fused, 1)

    expected = torch.relu(x + bias) + residual
    baseline_error = out_baseline - expected
    fused_error = out_fused - expected
    error_stats = torch.empty(4, device=x.device, dtype=torch.float32)
    error_stats[0].copy_(baseline_error.abs().amax())
    error_stats[1].copy_(fused_error.abs().amax())
    error_stats[2].copy_(torch.linalg.vector_norm(baseline_error))
    error_stats[3].copy_(torch.linalg.vector_norm(fused_error))
    error_stats_host = error_stats.detach().cpu()
    max_abs_baseline = float(error_stats_host[0])
    max_abs_fused = float(error_stats_host[1])
    l2_baseline = float(error_stats_host[2])
    l2_fused = float(error_stats_host[3])

    baseline_ms = time_kernel(
        lambda: extension.separate_kernels(x, bias, residual, tmp, out_baseline, iterations),
        iters=1,
        warmup=warmup,
    )
    fused_ms = time_kernel(
        lambda: extension.fused_kernel(x, bias, residual, out_fused, iterations),
        iters=1,
        warmup=warmup,
    )

    speedup = baseline_ms / fused_ms if fused_ms > 0 else float("inf")

    result = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "n_elements": n_elements,
        "iterations_per_timing": iterations,
        "warmup": warmup,
        "dtype": str(x.dtype),
        "device": torch.cuda.get_device_name(0),
        "baseline_ms": baseline_ms,
        "fused_ms": fused_ms,
        "speedup": speedup,
        "max_abs_error": {
            "baseline_vs_reference": max_abs_baseline,
            "fused_vs_reference": max_abs_fused,
        },
        "l2_error": {
            "baseline_vs_reference": l2_baseline,
            "fused_vs_reference": l2_fused,
        },
        "correctness_pass": bool(
            max_abs_baseline < 1e-4 and max_abs_fused < 1e-4
        ),
    }

    out = Path(__file__).parent / "benchmark_output.txt"
    out.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    result = run_benchmark()
    print("Bias+ReLU + residual add benchmark")
    print(f"n_elements      : {result['n_elements']:,}")
    print(f"baseline (ms)   : {result['baseline_ms']:.6f}")
    print(f"fused (ms)      : {result['fused_ms']:.6f}")
    print(f"speedup         : {result['speedup']:.3f}x")
    print(f"correctness     : {'PASS' if result['correctness_pass'] else 'FAIL'}")
    print("  max_abs_error baseline", result['max_abs_error']['baseline_vs_reference'])
    print("  max_abs_error fused   ", result['max_abs_error']['fused_vs_reference'])
    print("artifact        : ch12/benchmark_output.txt")


if __name__ == "__main__":
    main()
