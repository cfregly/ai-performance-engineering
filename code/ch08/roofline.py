#!/usr/bin/env python3
"""Roofline analysis examples for an explicitly identified, reviewed GPU SKU.

Analyzes kernel performance to determine if compute-bound or memory-bound.
Provides actionable insights for optimization.

Roofline Model:
- Y-axis: Compute performance (TFLOPS)
- X-axis: Arithmetic intensity (FLOPs/byte)
- Roofline: min(Peak TFLOPS, Peak Bandwidth * Arithmetic Intensity)

The ceilings come from :mod:`core.benchmark.metrics`.  This chapter does not
infer a product from compute capability alone and does not model GB10 as B200.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from core.analysis.kernel_roofline import (
    ArchitectureSpecs,
    RooflineAnalyzer,
    get_architecture_specs,
)

__all__ = ["ArchitectureSpecs", "RooflineAnalyzer", "get_architecture_specs"]


@contextmanager
def _ieee_fp32_matmul():
    """Force scalar IEEE FP32 policy while measuring the FP32 matmul example."""
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32


def benchmark_example_kernels() -> None:
    """Benchmark example kernels on a supported CUDA GPU and analyze them."""
    print("=" * 80)
    print("GPU Roofline Analysis - Example Kernels")
    print("=" * 80)
    print()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the GPU roofline examples; CPU timing is not "
            "a reviewed GPU benchmark"
        )
    torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    device_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_index)
    print(f"Using GPU: {gpu_name}")
    device = torch.device("cuda")

    def _sync() -> None:
        torch.cuda.synchronize()

    analyzer = RooflineAnalyzer()
    print(f"Architecture: {analyzer.specs.name}")
    print(f"Profile source: {analyzer.specs.profile_source}")
    print(f"Peak FP32: {analyzer.specs.peak_fp32_tflops} TFLOPS")
    print(f"Peak FP16: {analyzer.specs.peak_fp16_tflops} TFLOPS")
    print(f"HBM: {analyzer.specs.memory_bandwidth_gbs} GB/s")
    print()

    copy_elems = 64 * 1024 * 1024
    mat_dims = (4096, 4096, 4096)
    relu_elems = 128 * 1024 * 1024
    copy_iters = 100
    relu_iters = 100
    mat_iters = 50

    # Example 1: Memory-bound copy kernel
    print("\n" + "=" * 80)
    print("Example 1: Memory Copy (highly memory-bound)")
    print("=" * 80)

    N = copy_elems
    x = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.empty_like(x)

    # Warmup
    for _ in range(10):
        y.copy_(x)
    _sync()

    # Benchmark
    start = time.perf_counter()
    for _ in range(copy_iters):
        y.copy_(x)
    _sync()
    elapsed_ms = (time.perf_counter() - start) * (1000.0 / copy_iters)

    # Analysis
    flops = 0  # A device-to-device copy performs no floating-point arithmetic.
    bytes_transferred = 2 * N * 4  # Read + write

    results = analyzer.analyze_kernel(elapsed_ms, flops, bytes_transferred, "fp32")
    analyzer.print_analysis(results, "Memory Copy")

    # Example 2: Compute-bound matmul
    print("\n" + "=" * 80)
    print("Example 2: Large Matrix Multiplication (compute-bound)")
    print("=" * 80)

    M, K, N_dim = mat_dims
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N_dim, device=device, dtype=torch.float32)

    with _ieee_fp32_matmul():
        # Warmup
        for _ in range(10):
            torch.matmul(A, B)
        _sync()

        # Benchmark
        start = time.perf_counter()
        for _ in range(mat_iters):
            torch.matmul(A, B)
        _sync()
        elapsed_ms = (time.perf_counter() - start) * (1000.0 / mat_iters)

    # Analysis
    flops = 2 * M * K * N_dim  # MAD operations
    bytes_transferred = (M * K + K * N_dim + M * N_dim) * 4

    results = analyzer.analyze_kernel(elapsed_ms, flops, bytes_transferred, "fp32")
    analyzer.print_analysis(results, "Matrix Multiplication (FP32)")

    # Example 3: Element-wise operation
    print("\n" + "=" * 80)
    print("Example 3: Element-wise Activation (ReLU - memory-bound)")
    print("=" * 80)

    N = relu_elems
    x = torch.randn(N, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        y = torch.relu(x)
    _sync()

    # Benchmark
    start = time.perf_counter()
    for _ in range(relu_iters):
        y = torch.relu(x)
    _sync()
    elapsed_ms = (time.perf_counter() - start) * (1000.0 / relu_iters)

    # Analysis
    # ReLU performs a comparison/select rather than a floating-point arithmetic
    # operation, so do not present its element count as measured FLOPs.
    flops = 0
    bytes_transferred = 2 * N * 4  # Read + write

    results = analyzer.analyze_kernel(elapsed_ms, flops, bytes_transferred, "fp32")
    analyzer.print_analysis(results, "ReLU Activation")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("Memory-bound kernels benefit from:")
    print("  • Operation fusion (reduce intermediate memory)")
    print("  • Vectorized loads (float4)")
    print("  • Bulk-transfer features supported by the selected GPU SKU")
    print()
    print("Compute-bound kernels benefit from:")
    print("  • Tensor cores (use FP16/TF32)")
    print("  • Higher occupancy")
    print("  • Lower precision supported by the selected GPU SKU")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_example_kernels()
