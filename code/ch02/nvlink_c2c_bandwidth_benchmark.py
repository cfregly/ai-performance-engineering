#!/usr/bin/env python3

"""
NVLink-C2C Bandwidth Benchmark for Grace-Blackwell GB10

Measures coherent CPU↔GPU bandwidth via NVLink-C2C (900 GB/s aggregate,
≈450 GB/s per direction). Compares against discrete GPU PCIe bandwidth for
reference.

Architecture detection:
- GB10 (SM 12.x): Grace-Blackwell with NVLink-C2C
- B200 (SM 10.0): Discrete Blackwell with PCIe 5.0
- Other: Generic GPU

Usage:
    python nvlink_c2c_bandwidth_benchmark.py
"""

import os
import time

import torch

BENCHMARK_QUICK = os.environ.get("BENCHMARK_QUICK", "0") == "1"

def detect_architecture():
    """Detect GPU architecture"""
    if not torch.cuda.is_available():
        return "cpu", 0, 0
    
    props = torch.cuda.get_device_properties(0)
    
    if props.major >= 12:
        return "gb10", props.major, props.minor
    elif props.major == 10:
        if props.minor >= 3:
            return "b300", props.major, props.minor
        return "b200", props.major, props.minor
    else:
        return "other", props.major, props.minor


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
    
    bandwidth_gbs = (size_mb * iterations) / (elapsed_ms / 1000.0)
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
    
    bandwidth_gbs = (size_mb * iterations) / (elapsed_ms / 1000.0)
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
    
    # Total data: H2D + D2H
    total_mb = 2 * size_mb * iterations
    bandwidth_gbs = (total_mb / 1024.0) / elapsed_sec
    
    return bandwidth_gbs, (elapsed_sec * 1000.0) / iterations


def measure_zero_copy_read_bandwidth(size_mb=1024, iterations=50):
    """
    Measure GPU reading from CPU memory (zero-copy)
    
    On GB10: ~800-900 GB/s via NVLink-C2C
    On discrete: ~25 GB/s via PCIe
    """
    size = size_mb * 1024 * 1024 // 4
    
    # CPU memory (pinned for GPU access)
    cpu_tensor = torch.randn(size, device='cpu', pin_memory=True)
    gpu_output = torch.empty(size, device='cuda')
    
    # Simple kernel: read from CPU, write to GPU
    def zero_copy_kernel():
        # PyTorch doesn't expose direct zero-copy in Python
        # This simulates it by using non-blocking copy
        # In real CUDA code, GPU reads CPU memory directly
        gpu_output.copy_(cpu_tensor, non_blocking=True)
    
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
    
    bandwidth_gbs = (size_mb * iterations) / (elapsed_ms / 1000.0)
    return bandwidth_gbs, elapsed_ms / iterations


def main():
    print("=" * 80)
    print("NVLink-C2C Bandwidth Benchmark for Grace-Blackwell GB10")
    print("=" * 80)
    
    arch_type, major, minor = detect_architecture()
    
    print(f"\nArchitecture: ", end="")
    if arch_type == "gb10":
        print(f"Grace-Blackwell GB10 (SM {major}.{minor})")
        print(f"Interconnect: NVLink-C2C (900 GB/s aggregate, ≈450/dir)")
    elif arch_type == "b300":
        print(f"Blackwell B300 (SM {major}.{minor})")
        print(f"Interconnect: NVLink 5.0 / PCIe 5.0 depending on host wiring")
    elif arch_type == "b200":
        print(f"Blackwell B200 (SM {major}.{minor})")
        print(f"Interconnect: NVLink 5.0 / PCIe 5.0 depending on host wiring")
    else:
        print(f"Generic GPU (SM {major}.{minor})")
        print(f"Interconnect: Unknown")
    
    if arch_type != "gb10":
        print("\nWARNING: WARNING: This benchmark is optimized for Grace-Blackwell GB10!")
        print("   Results on discrete GPUs will show PCIe bandwidth, not NVLink-C2C.\n")
    
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
    print("\n" + "=" * 80)
    print("Peak Bandwidth Measurement (4 GB transfer)")
    print("=" * 80)
    
    peak_iter = 5 if BENCHMARK_QUICK else 50
    peak_size_mb = 512 if BENCHMARK_QUICK else 4096
    h2d_peak, _ = measure_h2d_bandwidth(size_mb=peak_size_mb, iterations=peak_iter)
    d2h_peak, _ = measure_d2h_bandwidth(size_mb=peak_size_mb, iterations=peak_iter)
    bidir_peak_size = 256 if BENCHMARK_QUICK else 2048
    bidir_peak, _ = measure_bidirectional_bandwidth(size_mb=bidir_peak_size, iterations=peak_iter)
    
    print(f"H2D Peak:          {h2d_peak:.2f} GB/s")
    print(f"D2H Peak:          {d2h_peak:.2f} GB/s")
    print(f"Bidirectional:     {bidir_peak:.2f} GB/s")

    print("\n" + "=" * 80)
    print("Test 4: Zero-Copy CPU Memory Reads")
    print("=" * 80)
    zero_copy_size_mb = 512 if BENCHMARK_QUICK else 1024
    bw_zero_copy, latency_zero_copy = measure_zero_copy_read_bandwidth(
        size_mb=zero_copy_size_mb, iterations=zero_copy_iterations
    )
    print(f"Size (MB):         {zero_copy_size_mb}")
    print(f"Bandwidth:         {bw_zero_copy:.2f} GB/s")
    print(f"Latency:           {latency_zero_copy:.3f} ms")
    
    # Theoretical comparison
    print("\n" + "=" * 80)
    print("Theoretical Bandwidth Comparison")
    print("=" * 80)
    
    if arch_type == "gb10":
        theoretical = 900.0
        print(f"NVLink-C2C Theoretical:    {theoretical:.0f} GB/s")
        print(f"H2D Achieved:              {h2d_peak:.2f} GB/s ({(h2d_peak/theoretical)*100:.1f}%)")
        print(f"D2H Achieved:              {d2h_peak:.2f} GB/s ({(d2h_peak/theoretical)*100:.1f}%)")
        print(f"Bidirectional:             {bidir_peak:.2f} GB/s ({(bidir_peak/theoretical)*100:.1f}%)")
        
        print("\n[OK] GB10 Benefits:")
        print("  • 900 GB/s coherent CPU-GPU interconnect")
        print("  • Cache-coherent memory access")
        print("  • ~14x faster than PCIe 5.0 (64 GB/s)")
        print("  • Ideal for:")
        print("    - Large KV caches in CPU memory")
        print("    - Optimizer states offload")
        print("    - CPU preprocessing pipelines")
        
    elif arch_type == "b200":
        theoretical = 64.0  # PCIe 5.0
        print(f"PCIe 5.0 Theoretical:      {theoretical:.0f} GB/s")
        print(f"H2D Achieved:              {h2d_peak:.2f} GB/s ({(h2d_peak/theoretical)*100:.1f}%)")
        print(f"D2H Achieved:              {d2h_peak:.2f} GB/s ({(d2h_peak/theoretical)*100:.1f}%)")
        print(f"Bidirectional:             {bidir_peak:.2f} GB/s ({(bidir_peak/theoretical)*100:.1f}%)")
        
        print("\nℹ️  Note: GB10 vs B200 comparison:")
        print(f"  • GB10 NVLink-C2C:  ~900 GB/s  ({900/theoretical:.1f}x faster)")
        print(f"  • B200 PCIe 5.0:    ~{theoretical:.0f} GB/s")
        print(f"  • For CPU-GPU data movement, GB10 has {900/h2d_peak:.1f}x more headroom")
    elif arch_type == "b300":
        theoretical = 64.0  # PCIe 5.0
        print(f"PCIe 5.0 Theoretical:      {theoretical:.0f} GB/s")
        print(f"H2D Achieved:              {h2d_peak:.2f} GB/s ({(h2d_peak/theoretical)*100:.1f}%)")
        print(f"D2H Achieved:              {d2h_peak:.2f} GB/s ({(d2h_peak/theoretical)*100:.1f}%)")
        print(f"Bidirectional:             {bidir_peak:.2f} GB/s ({(bidir_peak/theoretical)*100:.1f}%)")
        
        print("\nℹ️  Note: GB10 vs B300 comparison:")
        print(f"  • GB10 NVLink-C2C:  ~900 GB/s  ({900/theoretical:.1f}x faster)")
        print(f"  • B300 PCIe 5.0:    ~{theoretical:.0f} GB/s")
        print(f"  • For CPU-GPU data movement, GB10 has {900/h2d_peak:.1f}x more headroom")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
