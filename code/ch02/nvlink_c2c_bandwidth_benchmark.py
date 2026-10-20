#!/usr/bin/env python3

"""
NVLink-C2C Bandwidth Benchmark for Grace-Blackwell GB10

Measures host-to-device, device-to-host, and simultaneous copy bandwidth, plus
GPU kernel reads of pinned host memory with no copy (zero-copy). On GB10 (DGX
Spark) the CPU and GPU share one LPDDR5X memory with a published 273 GB/s
peak, so that peak bounds every rate measured here. On other Blackwell
systems the results measure the GPU's host link and are compared against
that link's spec peak.

Architecture detection:
- GB10 (SM 12.1): Grace-Blackwell with unified LPDDR5X memory
- GB200 (SM 10.0) and GB300 (SM 10.3): Blackwell GPU attached to a Grace CPU
  over NVLink-C2C, 450 GB/s per direction
- B200 (SM 10.0) and B300 (SM 10.3): Blackwell GPU attached to the host over
  PCIe 5.0 x16, 64 GB/s per direction
- Other: Generic GPU, including SM 12.0 workstation GPUs, with no reference peak

SM 10.0 and 10.3 GPUs ship in both kinds of system, so the CUDA driver's
host-link attributes decide between them, not the compute capability. The run
stops with an error when the driver cannot answer.

Usage:
    python -m ch02.nvlink_c2c_bandwidth_benchmark
"""

import os
import time

import torch

from core.common.grace_platform import host_link_flags
from core.utils.pinned_host_view import pinned_host_cuda_view

BENCHMARK_QUICK = os.environ.get("BENCHMARK_QUICK", "0") == "1"

GB10_COMPUTE_CAPABILITY = (12, 1)
# DGX Spark has no separate GPU memory. This published peak covers all CPU and
# GPU reads and writes of its shared LPDDR5X.
GB10_UNIFIED_MEMORY_PEAK_GBPS = 273.0
PCIE5_X16_PER_DIRECTION_GBPS = 64.0
# NVIDIA specifies NVLink-C2C at 900 GB/s bidirectional, 450 GB/s per direction.
NVLINK_C2C_PER_DIRECTION_GBPS = 450.0

# Each SM 10.x capability names a PCIe-attached GPU or a Grace superchip GPU.
_SM10_PLATFORMS = {
    (10, 0): {"pcie": "b200", "nvlink_c2c": "gb200"},
    (10, 3): {"pcie": "b300", "nvlink_c2c": "gb300"},
}
_HOST_LINK_PEAKS = {
    "b200": ("PCIe 5.0 x16", PCIE5_X16_PER_DIRECTION_GBPS),
    "b300": ("PCIe 5.0 x16", PCIE5_X16_PER_DIRECTION_GBPS),
    "gb200": ("NVLink-C2C", NVLINK_C2C_PER_DIRECTION_GBPS),
    "gb300": ("NVLink-C2C", NVLINK_C2C_PER_DIRECTION_GBPS),
}


def _decimal_gb_per_second(
    *, size_mb: int, iterations: int, elapsed_seconds: float, directions: int = 1
) -> float:
    """Return payload bandwidth in decimal GB/s from a MiB workload size."""
    if size_mb <= 0 or iterations <= 0 or directions <= 0 or elapsed_seconds <= 0:
        raise ValueError("size, iterations, directions, and elapsed time must be positive")
    total_bytes = size_mb * 1024 * 1024 * iterations * directions
    return total_bytes / elapsed_seconds / 1e9


def _classify_platform(props, device_index: int) -> str:
    """Map a CUDA device to the platform whose reference peaks apply."""
    capability = (props.major, props.minor)
    if capability == GB10_COMPUTE_CAPABILITY:
        return "gb10"
    if capability not in _SM10_PLATFORMS:
        return "other"
    try:
        nvlink_c2c, _ = host_link_flags(props, device_index)
    except RuntimeError as exc:
        raise RuntimeError(
            f"cannot tell whether the SM {props.major}.{props.minor} GPU reaches host "
            "memory over NVLink-C2C or PCIe, so no reference peak applies"
        ) from exc
    return _SM10_PLATFORMS[capability]["nvlink_c2c" if nvlink_c2c else "pcie"]


def detect_architecture():
    """Return the platform class and compute capability of the current CUDA device."""
    if not torch.cuda.is_available():
        return "cpu", 0, 0

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    return _classify_platform(props, device_index), props.major, props.minor


def measure_h2d_bandwidth(size_mb=1024, iterations=100):
    """Measure Host-to-Device bandwidth"""
    size = size_mb * 1024 * 1024 // 4  # Convert MB to float elements
    
    # CPU tensor (pinned memory for fast transfers)
    cpu_tensor = torch.randn(size, device='cpu', pin_memory=True)
    gpu_tensor = torch.empty(size, device='cuda')
    
    # Warmup
    warmup_iters = 3 if BENCHMARK_QUICK else 10
    for _ in range(warmup_iters):
        gpu_tensor.copy_(cpu_tensor, non_blocking=False)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        gpu_tensor.copy_(cpu_tensor, non_blocking=False)
    end.record()
    
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    bandwidth_gbs = _decimal_gb_per_second(
        size_mb=size_mb,
        iterations=iterations,
        elapsed_seconds=elapsed_ms / 1000.0,
    )
    return bandwidth_gbs, elapsed_ms / iterations


def measure_d2h_bandwidth(size_mb=1024, iterations=100):
    """Measure Device-to-Host bandwidth"""
    size = size_mb * 1024 * 1024 // 4
    
    gpu_tensor = torch.randn(size, device='cuda')
    cpu_tensor = torch.empty(size, device='cpu', pin_memory=True)
    
    # Warmup
    warmup_iters = 3 if BENCHMARK_QUICK else 10
    for _ in range(warmup_iters):
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        cpu_tensor.copy_(gpu_tensor, non_blocking=False)
    end.record()
    
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    
    bandwidth_gbs = _decimal_gb_per_second(
        size_mb=size_mb,
        iterations=iterations,
        elapsed_seconds=elapsed_ms / 1000.0,
    )
    return bandwidth_gbs, elapsed_ms / iterations


def measure_bidirectional_bandwidth(size_mb=512, iterations=100):
    """Measure simultaneous H2D + D2H bandwidth"""
    size = size_mb * 1024 * 1024 // 4
    
    # Create two streams for overlap
    stream_h2d = torch.cuda.Stream()
    stream_d2h = torch.cuda.Stream()
    
    cpu_tensor_src = torch.randn(size, device='cpu', pin_memory=True)
    cpu_tensor_dst = torch.empty(size, device='cpu', pin_memory=True)
    gpu_tensor_a = torch.empty(size, device='cuda')
    gpu_tensor_b = torch.randn(size, device='cuda')
    
    # Warmup
    warmup_iters = 3 if BENCHMARK_QUICK else 10
    for _ in range(warmup_iters):
        with torch.cuda.stream(stream_h2d):
            gpu_tensor_a.copy_(cpu_tensor_src, non_blocking=True)
        with torch.cuda.stream(stream_d2h):
            cpu_tensor_dst.copy_(gpu_tensor_b, non_blocking=True)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(iterations):
        with torch.cuda.stream(stream_h2d):
            gpu_tensor_a.copy_(cpu_tensor_src, non_blocking=True)
        with torch.cuda.stream(stream_d2h):
            cpu_tensor_dst.copy_(gpu_tensor_b, non_blocking=True)
    torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - start_time
    
    # Total payload: one H2D transfer plus one D2H transfer per iteration.
    bandwidth_gbs = _decimal_gb_per_second(
        size_mb=size_mb,
        iterations=iterations,
        elapsed_seconds=elapsed_sec,
        directions=2,
    )
    
    return bandwidth_gbs, (elapsed_sec * 1000.0) / iterations


def measure_zero_copy_read_bandwidth(size_mb=1024, iterations=50):
    """
    Measure a GPU kernel that reads pinned host memory in place (zero-copy).

    Each iteration negates the pinned host buffer into a device buffer. The
    kernel's loads read host memory directly and no DMA copy runs. On GB10
    both buffers live in the shared LPDDR5X, whose 273 GB/s peak covers the
    reads and the writes together. On other GPUs every load crosses the host
    link: PCIe 5.0 x16 at 64 GB/s per direction on B200 and B300, or NVLink-C2C
    at 450 GB/s per direction on GB200 and GB300.
    """
    size = size_mb * 1024 * 1024 // 4

    cpu_tensor = torch.randn(size, device='cpu', pin_memory=True)
    host_view = pinned_host_cuda_view(cpu_tensor, torch.device("cuda", torch.cuda.current_device()))
    gpu_output = torch.empty(size, device=host_view.device)

    def zero_copy_kernel():
        torch.neg(host_view, out=gpu_output)

    # Warmup
    warmup_iters = 3 if BENCHMARK_QUICK else 10
    for _ in range(warmup_iters):
        zero_copy_kernel()
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        zero_copy_kernel()
    end.record()

    end.synchronize()
    elapsed_ms = start.elapsed_time(end)

    if not torch.equal(gpu_output.cpu(), cpu_tensor.neg()):
        raise RuntimeError(
            "zero-copy read kernel output does not match the negated pinned host input"
        )

    bandwidth_gbs = _decimal_gb_per_second(
        size_mb=size_mb,
        iterations=iterations,
        elapsed_seconds=elapsed_ms / 1000.0,
    )
    return bandwidth_gbs, elapsed_ms / iterations


def theoretical_comparison_lines(
    arch_type: str,
    *,
    h2d_gbps: float,
    d2h_gbps: float,
    bidir_gbps: float,
    zero_copy_gbps: float,
) -> list[str]:
    """Return the reference-peak comparison for measured payload rates."""
    gb10_peak = GB10_UNIFIED_MEMORY_PEAK_GBPS
    measured = (
        ("H2D", h2d_gbps),
        ("D2H", d2h_gbps),
        ("Bidirectional", bidir_gbps),
        ("Zero-copy read", zero_copy_gbps),
    )

    if arch_type == "gb10":
        lines = [
            f"Unified LPDDR5X peak:      {gb10_peak:.0f} GB/s, shared by CPU and GPU",
            "Every payload byte is read from and written to that same memory,",
            "so memory traffic is at least twice the payload rate.",
        ]
        for label, payload in measured:
            traffic = 2.0 * payload
            lines.append(
                f"{label + ':':<27}{payload:.2f} GB/s payload, {traffic:.2f} GB/s traffic "
                f"({(traffic / gb10_peak) * 100:.1f}% of peak)"
            )
        lines += [
            "",
            "[OK] GB10 characteristics:",
            "  • CPU and GPU share one coherent LPDDR5X memory, with no separate GPU memory",
            "  • GPU kernels read CPU-written buffers in place, which suits CPU preprocessing pipelines",
            f"  • The {gb10_peak:.0f} GB/s peak is {gb10_peak / PCIE5_X16_PER_DIRECTION_GBPS:.1f}x the "
            f"{PCIE5_X16_PER_DIRECTION_GBPS:.0f} GB/s per-direction rate of PCIe 5.0 x16",
            "  • An explicit copy costs twice its size in memory traffic, so in-place access is cheaper",
        ]
        return lines

    if arch_type in _HOST_LINK_PEAKS:
        link, per_dir = _HOST_LINK_PEAKS[arch_type]
        both = 2.0 * per_dir
        gb10_copy_ceiling = gb10_peak / 2.0
        name = arch_type.upper()
        lines = [
            f"{link + ' spec peak:':<27}{per_dir:.0f} GB/s per direction, "
            f"{both:.0f} GB/s both directions",
        ]
        for label, payload in measured:
            reference = both if label == "Bidirectional" else per_dir
            lines.append(
                f"{label + ':':<27}{payload:.2f} GB/s ({(payload / reference) * 100:.1f}%)"
            )
        if link == "NVLink-C2C":
            lines.append(
                f"NVIDIA quotes this peak for the NVLink-C2C of a {name} superchip, which "
                "joins one Grace CPU to two GPUs, so one GPU may not reach it."
            )
        lines += [
            "",
            f"ℹ️  Note: GB10 vs {name} comparison:",
            f"  • GB10 unified LPDDR5X:  {gb10_peak:.0f} GB/s peak, shared by CPU and GPU "
            f"({gb10_peak / per_dir:.1f}x {link} per direction)",
            f"  • {name + ' ' + link + ':':<23}{per_dir:.0f} GB/s per direction",
            f"  • A GB10 copy reads and writes the same memory, so its payload ceiling is "
            f"{gb10_copy_ceiling:.1f} GB/s ({gb10_copy_ceiling / per_dir:.1f}x {link} per direction)",
        ]
        return lines

    return []


def main():
    print("=" * 80)
    print("NVLink-C2C Bandwidth Benchmark for Grace-Blackwell GB10")
    print("=" * 80)
    
    arch_type, major, minor = detect_architecture()
    
    print(f"\nArchitecture: ", end="")
    if arch_type == "gb10":
        print(f"Grace-Blackwell GB10 (SM {major}.{minor})")
        print("Interconnect: NVLink-C2C between the CPU and GPU dies")
        print(f"Memory: {GB10_UNIFIED_MEMORY_PEAK_GBPS:.0f} GB/s peak LPDDR5X, shared by CPU and GPU")
    elif arch_type in ("gb200", "gb300"):
        print(f"Grace-Blackwell {arch_type.upper()} (SM {major}.{minor})")
        print("Host link: NVLink-C2C to a Grace CPU, as reported by the CUDA driver")
    elif arch_type in ("b200", "b300"):
        print(f"Blackwell {arch_type.upper()} (SM {major}.{minor})")
        print("Host link: PCIe, because this system has no NVLink-C2C host link")
    else:
        print(f"Generic GPU (SM {major}.{minor})")
        print("Host link: unknown, so no reference peak is shown")
    
    if arch_type != "gb10":
        print("\nWARNING: This benchmark targets Grace-Blackwell GB10 unified memory.")
        print("   Results on this GPU measure its host link, not GB10 unified memory.\n")
    
    test_sizes = [256, 512] if BENCHMARK_QUICK else [256, 512, 1024, 2048]
    h2d_iterations = 5 if BENCHMARK_QUICK else 100
    bidir_iterations = 5 if BENCHMARK_QUICK else 100
    zero_copy_iterations = 3 if BENCHMARK_QUICK else 50
    
    print("\n" + "=" * 80)
    print("Test 1: Host-to-Device (H2D) Bandwidth")
    print("=" * 80)
    print(f"{'Size (MB)':<12} {'Bandwidth (GB/s)':<20} {'Latency (ms)':<15}")
    print("-" * 80)
    
    for size_mb in test_sizes:
        bw, latency = measure_h2d_bandwidth(size_mb=size_mb, iterations=h2d_iterations)
        print(f"{size_mb:<12} {bw:<20.2f} {latency:<15.3f}")
    
    print("\n" + "=" * 80)
    print("Test 2: Device-to-Host (D2H) Bandwidth")
    print("=" * 80)
    print(f"{'Size (MB)':<12} {'Bandwidth (GB/s)':<20} {'Latency (ms)':<15}")
    print("-" * 80)
    
    for size_mb in test_sizes:
        bw, latency = measure_d2h_bandwidth(size_mb=size_mb, iterations=h2d_iterations)
        print(f"{size_mb:<12} {bw:<20.2f} {latency:<15.3f}")
    
    print("\n" + "=" * 80)
    print("Test 3: Bidirectional (H2D + D2H simultaneous)")
    print("=" * 80)
    print(f"{'Size (MB)':<12} {'Combined BW (GB/s)':<20} {'Latency (ms)':<15}")
    print("-" * 80)
    
    bidir_sizes = [256, 512] if BENCHMARK_QUICK else [128, 256, 512, 1024]
    for size_mb in bidir_sizes:
        bw, latency = measure_bidirectional_bandwidth(size_mb=size_mb, iterations=bidir_iterations)
        print(f"{size_mb:<12} {bw:<20.2f} {latency:<15.3f}")
    
    # Peak measurement (large transfer)
    peak_iter = 5 if BENCHMARK_QUICK else 50
    peak_size_mb = 512 if BENCHMARK_QUICK else 4096
    print("\n" + "=" * 80)
    print("Peak Bandwidth Measurement (large transfers)")
    print("=" * 80)

    h2d_peak, _ = measure_h2d_bandwidth(size_mb=peak_size_mb, iterations=peak_iter)
    d2h_peak, _ = measure_d2h_bandwidth(size_mb=peak_size_mb, iterations=peak_iter)
    bidir_peak_size = 256 if BENCHMARK_QUICK else 2048
    bidir_peak, _ = measure_bidirectional_bandwidth(size_mb=bidir_peak_size, iterations=peak_iter)

    print(f"H2D Peak:          {h2d_peak:.2f} GB/s ({peak_size_mb} MB copies)")
    print(f"D2H Peak:          {d2h_peak:.2f} GB/s ({peak_size_mb} MB copies)")
    print(f"Bidirectional:     {bidir_peak:.2f} GB/s ({bidir_peak_size} MB each way)")

    print("\n" + "=" * 80)
    print("Test 4: Zero-Copy Reads of Pinned Host Memory")
    print("=" * 80)
    zero_copy_size_mb = 512 if BENCHMARK_QUICK else 1024
    bw_zero_copy, latency_zero_copy = measure_zero_copy_read_bandwidth(
        size_mb=zero_copy_size_mb, iterations=zero_copy_iterations
    )
    print(f"Size (MB):         {zero_copy_size_mb}")
    print(f"Bandwidth:         {bw_zero_copy:.2f} GB/s (host bytes read by the kernel)")
    print(f"Latency:           {latency_zero_copy:.3f} ms")

    # Theoretical comparison
    print("\n" + "=" * 80)
    print("Theoretical Bandwidth Comparison")
    print("=" * 80)

    for line in theoretical_comparison_lines(
        arch_type,
        h2d_gbps=h2d_peak,
        d2h_gbps=d2h_peak,
        bidir_gbps=bidir_peak,
        zero_copy_gbps=bw_zero_copy,
    ):
        print(line)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
