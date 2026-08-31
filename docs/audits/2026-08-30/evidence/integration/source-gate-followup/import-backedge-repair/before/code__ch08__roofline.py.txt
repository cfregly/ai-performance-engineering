#!/usr/bin/env python3
"""Roofline analysis tool for Blackwell B200 and Grace-Blackwell GB10.

Analyzes kernel performance to determine if compute-bound or memory-bound.
Provides actionable insights for optimization.

Roofline Model:
- Y-axis: Compute performance (TFLOPS)
- X-axis: Arithmetic intensity (FLOPs/byte)
- Roofline: min(Peak TFLOPS, Peak Bandwidth * Arithmetic Intensity)

Architecture specs:
B200:
- Peak FP32: 225 TFLOPS
- Peak TF32: 450 TFLOPS
- Peak FP16: 450 TFLOPS
- Peak FP8: 450 TFLOPS
- HBM3e BW: 7800 GB/s

GB10:
- Same GPU as B200
- Additional: NVLink-C2C 900 GB/s CPU-GPU
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ArchitectureSpecs:
    """Hardware specifications for roofline analysis."""

    name: str
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_fp8_tflops: float
    peak_tf32_tflops: float
    memory_bandwidth_gbs: float
    cpu_gpu_bandwidth_gbs: Optional[float] = None  # For GB10


def get_architecture_specs() -> ArchitectureSpecs:
    """Detect architecture and return specs."""
    if not torch.cuda.is_available():
        return ArchitectureSpecs(
            name="CPU",
            peak_fp32_tflops=0.1,
            peak_fp16_tflops=0.0,
            peak_fp8_tflops=0.0,
            peak_tf32_tflops=0.0,
            memory_bandwidth_gbs=100.0,
        )

    props = torch.cuda.get_device_properties(0)

    if props.major == 12:  # Grace-Blackwell GB10
        return ArchitectureSpecs(
            name="Grace-Blackwell GB10",
            peak_fp32_tflops=225.0,
            peak_fp16_tflops=450.0,
            peak_fp8_tflops=450.0,
            peak_tf32_tflops=450.0,
            memory_bandwidth_gbs=7800.0,
            cpu_gpu_bandwidth_gbs=900.0,
        )
    elif props.major == 10 and props.minor == 3:  # Blackwell Ultra B300
        return ArchitectureSpecs(
            name="Blackwell Ultra B300",
            peak_fp32_tflops=225.0,
            peak_fp16_tflops=450.0,
            peak_fp8_tflops=450.0,
            peak_tf32_tflops=450.0,
            memory_bandwidth_gbs=9000.0,
        )
    elif props.major == 10 and props.minor == 0:  # Blackwell B200
        return ArchitectureSpecs(
            name="Blackwell B200",
            peak_fp32_tflops=225.0,
            peak_fp16_tflops=450.0,
            peak_fp8_tflops=450.0,
            peak_tf32_tflops=450.0,
            memory_bandwidth_gbs=7800.0,
        )
    elif props.major == 9:  # Hopper H100
        return ArchitectureSpecs(
            name="Hopper H100",
            peak_fp32_tflops=67.0,
            peak_fp16_tflops=1979.0,
            peak_fp8_tflops=3958.0,
            peak_tf32_tflops=989.0,
            memory_bandwidth_gbs=3350.0,
        )
    elif props.major == 8:  # Ampere A100
        return ArchitectureSpecs(
            name="Ampere A100",
            peak_fp32_tflops=19.5,
            peak_fp16_tflops=312.0,
            peak_fp8_tflops=0.0,
            peak_tf32_tflops=156.0,
            memory_bandwidth_gbs=2039.0,
        )
    else:
        return ArchitectureSpecs(
            name=f"Unknown (SM {props.major}.{props.minor})",
            peak_fp32_tflops=10.0,
            peak_fp16_tflops=20.0,
            peak_fp8_tflops=0.0,
            peak_tf32_tflops=20.0,
            memory_bandwidth_gbs=1000.0,
        )


class RooflineAnalyzer:
    """Analyze kernel performance using roofline model."""

    def __init__(self) -> None:
        self.specs = get_architecture_specs()

    def analyze_kernel(
        self,
        kernel_time_ms: float,
        flops: float,
        bytes_transferred: float,
        precision: str = "fp32",
    ) -> Dict[str, float]:
        """Analyze kernel performance.

        Args:
            kernel_time_ms: Kernel execution time in milliseconds
            flops: Total floating point operations
            bytes_transferred: Total bytes read + written
            precision: "fp32", "fp16", "fp8", or "tf32"

        Returns:
            Dictionary with analysis results
        """
        # Compute achieved metrics
        achieved_tflops = (flops / 1e12) / (kernel_time_ms / 1000.0)
        achieved_bandwidth_gbs = (bytes_transferred / 1e9) / (kernel_time_ms / 1000.0)
        arithmetic_intensity = flops / bytes_transferred if bytes_transferred > 0 else 0

        # Get peak performance for precision
        peak_tflops = {
            "fp32": self.specs.peak_fp32_tflops,
            "fp16": self.specs.peak_fp16_tflops,
            "fp8": self.specs.peak_fp8_tflops,
            "tf32": self.specs.peak_tf32_tflops,
        }.get(precision, self.specs.peak_fp32_tflops)

        # Roofline analysis
        memory_bound_tflops = (achieved_bandwidth_gbs / 1000.0) * arithmetic_intensity

        # Determine bottleneck
        is_memory_bound = achieved_tflops < memory_bound_tflops * 0.8
        is_compute_bound = achieved_tflops >= memory_bound_tflops * 0.8

        # Compute utilization
        compute_utilization = (achieved_tflops / peak_tflops) * 100.0
        memory_utilization = (achieved_bandwidth_gbs / self.specs.memory_bandwidth_gbs) * 100.0

        # Ridge point (where compute and memory rooflines meet)
        ridge_point = (peak_tflops * 1000.0) / self.specs.memory_bandwidth_gbs

        return {
            "achieved_tflops": achieved_tflops,
            "achieved_bandwidth_gbs": achieved_bandwidth_gbs,
            "arithmetic_intensity": arithmetic_intensity,
            "peak_tflops": peak_tflops,
            "peak_bandwidth_gbs": self.specs.memory_bandwidth_gbs,
            "compute_utilization_pct": compute_utilization,
            "memory_utilization_pct": memory_utilization,
            "is_memory_bound": is_memory_bound,
            "is_compute_bound": is_compute_bound,
            "memory_bound_tflops": memory_bound_tflops,
            "ridge_point": ridge_point,
        }

    def print_analysis(self, results: Dict[str, float], kernel_name: str = "Kernel") -> None:
        """Pretty print analysis results."""
        print("=" * 80)
        print(f"Roofline Analysis: {kernel_name}")
        print("=" * 80)
        print(f"Architecture: {self.specs.name}")
        print()
        print("Achieved Performance:")
        print(
            f"  Compute: {results['achieved_tflops']:8.2f} TFLOPS "
            f"({results['compute_utilization_pct']:5.1f}% of "
            f"{results['peak_tflops']:.0f} TFLOPS peak)"
        )
        print(
            f"  Bandwidth: {results['achieved_bandwidth_gbs']:8.2f} GB/s "
            f"({results['memory_utilization_pct']:5.1f}% of "
            f"{results['peak_bandwidth_gbs']:.0f} GB/s peak)"
        )
        print(f"  Arithmetic Intensity: {results['arithmetic_intensity']:.2f} FLOPs/byte")
        print()

        # Bottleneck analysis
        print("Bottleneck Analysis:")
        if results["is_memory_bound"]:
            print("  Status: ❗ MEMORY-BOUND")
            print(f"  Memory roofline limit: {results['memory_bound_tflops']:.2f} TFLOPS")
            print()
            print("  Recommendations:")
            print("    1. Increase arithmetic intensity (more compute per byte)")
            print("    2. Reduce memory traffic: use shared memory / cache blocking")
            print("    3. Use TMA for bulk transfers (Blackwell)")
        else:
            print("  Status: ✓ COMPUTE-BOUND")
            print(f"  Compute utilization: {results['compute_utilization_pct']:.1f}%")
            print()
            print("  Recommendations:")
            print("    1. Better occupancy (more warps)")
            print("    2. Use tensor cores for matmul")
            print("    3. Consider lower precision (FP16/FP8)")

        print("=" * 80)


def benchmark_example_kernels() -> None:
    """Benchmark example kernels and analyze."""
    print("=" * 80)
    print("Blackwell Roofline Analysis - Example Kernels")
    print("=" * 80)
    print()

    device: torch.device
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available - running CPU fallback benchmarks.\n")
        device = torch.device("cpu")
    else:
        torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        device = torch.device("cuda")

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    analyzer = RooflineAnalyzer()
    print(f"Architecture: {analyzer.specs.name}")
    print(f"Peak FP32: {analyzer.specs.peak_fp32_tflops} TFLOPS")
    print(f"Peak FP16: {analyzer.specs.peak_fp16_tflops} TFLOPS")
    print(f"HBM: {analyzer.specs.memory_bandwidth_gbs} GB/s")
    print()

    # Configure problem sizes based on device
    if device.type == "cuda":
        copy_elems = 64 * 1024 * 1024
        mat_dims = (4096, 4096, 4096)
        relu_elems = 128 * 1024 * 1024
        copy_iters = 100
        relu_iters = 100
        mat_iters = 50
    else:
        copy_elems = 8 * 1024 * 1024
        mat_dims = (1024, 1024, 1024)
        relu_elems = 16 * 1024 * 1024
        copy_iters = 25
        relu_iters = 25
        mat_iters = 10

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
    flops = N  # Minimal compute
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

    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)
    _sync()

    # Benchmark
    start = time.perf_counter()
    for _ in range(mat_iters):
        C = torch.matmul(A, B)
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
    flops = N  # One comparison per element
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
    print("  • TMA on Blackwell")
    print()
    print("Compute-bound kernels benefit from:")
    print("  • Tensor cores (use FP16/TF32)")
    print("  • Higher occupancy")
    print("  • Lower precision (FP8 on Blackwell)")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_example_kernels()
