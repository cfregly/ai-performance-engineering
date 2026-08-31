#!/usr/bin/env python3

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

"""
FP8 Quantization with torch.compile Integration

Demonstrates native FP8 usage with torch.compile on Blackwell.

Blackwell B200 dense Tensor Core capabilities (per GPU):
- 4,500 TFLOPS FP8
- 2,250 TFLOPS FP16/BF16
- Native FP8 tensor cores
- Hardware accelerated FP8↔FP16 conversion

Architecture support:
- B200: Full FP8 support
- Other Blackwell GPUs: capability-gated native FP8 path; no B200 utilization denominator
- Older: Explicitly unsupported by this native-FP8 benchmark
"""

from typing import Tuple

import torch


# NVIDIA publishes HGX B200's eight-GPU sparse peaks as 72 PFLOPS FP8 and
# 36 PFLOPS FP16/BF16, and states that dense performance is half of sparse.
# Dividing those dense system figures by eight gives the per-GPU ceilings used
# by this dense GEMM benchmark. Source:
# https://www.nvidia.com/en-us/data-center/hgx/
B200_DENSE_FP8_TFLOPS = 4_500.0
B200_DENSE_FP16_TFLOPS = 2_250.0


def _architecture_name(major: int, minor: int) -> str:
    if (major, minor) == (10, 0):
        return "Blackwell B200"
    if (major, minor) == (10, 3):
        return "Blackwell Ultra"
    if major == 12:
        return "Blackwell SM 12.x (not B200)"
    return "Blackwell"


def _dense_tensor_core_peaks(major: int, minor: int) -> Tuple[float, float] | None:
    """Return published dense per-GPU FP8/FP16 peaks when exactly known."""
    if (major, minor) == (10, 0):
        return B200_DENSE_FP8_TFLOPS, B200_DENSE_FP16_TFLOPS
    return None


def detect_fp8_support() -> Tuple[bool, str]:
    """Check if hardware supports FP8"""
    if not torch.cuda.is_available():
        return False, "No CUDA device"
    if not hasattr(torch, "float8_e4m3fn"):
        return False, "PyTorch does not expose torch.float8_e4m3fn"
    if not callable(getattr(torch, "_scaled_mm", None)):
        return False, "PyTorch does not expose torch._scaled_mm"
    
    props = torch.cuda.get_device_properties(0)
    
    # This benchmark intentionally targets the Blackwell-native scaled GEMM path.
    if props.major >= 10:
        arch_name = _architecture_name(props.major, props.minor)
        return True, f"{arch_name} (SM {props.major}.{props.minor})"
    else:
        return False, f"SM {props.major}.{props.minor} (requires SM 10.0+)"


def quantize_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP32/FP16 tensor to FP8
    
    Returns:
        - Native FP8 tensor
        - Scale factor for dequantization
    """
    # Compute scale factor
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("SKIPPED: torch.float8_e4m3fn is required")
    amax = tensor.abs().max().to(torch.float32)
    scale = (amax / 448.0).clamp_min_(torch.finfo(torch.float32).tiny)
    
    # Quantize
    scaled = tensor / scale
    fp8_tensor = scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
    
    return fp8_tensor, scale


def dequantize_fp8(fp8_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 back to FP32/FP16"""
    return fp8_tensor.to(torch.float32) * scale


# Baseline: FP32 matmul
def fp32_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Standard FP32 matrix multiplication"""
    return torch.matmul(x, w)


# FP8 matmul (naive)
def fp8_matmul_naive(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """FP8 matmul with quantization/dequantization"""
    # Quantize inputs
    x_fp8, x_scale = quantize_fp8(x)
    w_fp8, w_scale = quantize_fp8(w)
    
    # Dequantize for computation
    x_dq = dequantize_fp8(x_fp8, x_scale)
    w_dq = dequantize_fp8(w_fp8, w_scale)
    
    # Matmul
    return torch.matmul(x_dq, w_dq)


# FP8 matmul with torch.compile
@torch.compile(mode='max-autotune', fullgraph=True)
def fp8_matmul_compiled(x_fp8: torch.Tensor, w_fp8: torch.Tensor, 
                        x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """
    Compiled native FP8 matmul through PyTorch's scaled GEMM operator.
    """
    return torch._scaled_mm(
        x_fp8,
        w_fp8,
        scale_a=x_scale,
        scale_b=w_scale,
        out_dtype=torch.float16,
        use_fast_accum=False,
    )


# FP16 matmul with compile (for comparison)
@torch.compile(mode='max-autotune', fullgraph=True)
def fp16_matmul_compiled(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compiled FP16 matmul"""
    return torch.matmul(x, w)


def benchmark_matmul(fn, *args, name="", warmup=50, iters=500):
    """Benchmark matrix multiplication"""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    start.record(current_stream)
    for _ in range(iters):
        _ = fn(*args)
    end.record(current_stream)
    end.synchronize()

    return start.elapsed_time(end) / max(iters, 1)


def main() -> int:
    print("=" * 80)
    print("FP8 Quantization with torch.compile Integration")
    print("=" * 80)
    
    has_fp8, arch_info = detect_fp8_support()
    print(f"\nArchitecture: {arch_info}")
    print(f"FP8 Support: {'[OK] YES' if has_fp8 else 'ERROR: NO'}")
    
    if not has_fp8:
        print(f"\nSKIPPED: Native FP8 benchmark unavailable: {arch_info}")
        return 3
    else:
        props = torch.cuda.get_device_properties(0)
        dense_peaks = _dense_tensor_core_peaks(props.major, props.minor)
        print("\n[OK] Blackwell native FP8 path available")
        if dense_peaks is not None:
            peak_fp8_tflops, peak_fp16_tflops = dense_peaks
            print(f"  • B200 dense FP8 peak: {peak_fp8_tflops:,.0f} TFLOPS")
            print(f"  • B200 dense FP16/BF16 peak: {peak_fp16_tflops:,.0f} TFLOPS")
        else:
            peak_fp8_tflops = None
            peak_fp16_tflops = None
            print("  • Published dense peak denominator: unavailable for this device")
        print("  • Theoretical dense FP8:FP16 ratio on B200: 2x\n")
    
    # Test configuration
    # Use large matrices to saturate tensor cores
    M, K, N = 4096, 4096, 4096
    
    print(f"Matrix dimensions: ({M} x {K}) @ ({K} x {N})")
    print(f"Output size: {M} x {N}")
    
    # Calculate FLOPs
    flops = 2 * M * K * N  # MAD operations
    print(f"FLOPs per matmul: {flops / 1e9:.2f} GFLOPS\n")
    
    # Create test tensors
    print("Preparing tensors...")
    x_fp32 = torch.randn(M, K, device='cuda', dtype=torch.float32)
    w_fp32 = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    x_fp16 = x_fp32.to(torch.float16)
    w_fp16 = w_fp32.to(torch.float16)
    
    # Quantize to FP8
    x_fp8, x_scale = quantize_fp8(x_fp16)
    # _scaled_mm consumes B as a column-major (K, N) view. Quantize a
    # contiguous (N, K) backing tensor and transpose it without materializing.
    w_fp8_storage, w_scale = quantize_fp8(w_fp16.transpose(0, 1).contiguous())
    w_fp8 = w_fp8_storage.transpose(0, 1)
    
    print(f"FP32 memory: {(x_fp32.numel() + w_fp32.numel()) * 4 / 1e6:.2f} MB")
    print(f"FP16 memory: {(x_fp16.numel() + w_fp16.numel()) * 2 / 1e6:.2f} MB")
    print(f"FP8 memory:  {(x_fp8.numel() + w_fp8.numel()) * 1 / 1e6:.2f} MB")
    print(f"Memory savings: {((4-1)/4)*100:.0f}% vs FP32\n")
    
    # ========================================================================
    # Benchmark 1: FP32 baseline
    # ========================================================================
    print("=" * 80)
    print("Benchmark 1: FP32 Baseline")
    print("=" * 80)
    
    time_fp32 = benchmark_matmul(fp32_matmul, x_fp32, w_fp32, name="FP32", warmup=10, iters=100)
    tflops_fp32 = (flops / 1e12) / (time_fp32 / 1000.0)
    
    print(f"Time:       {time_fp32:.3f} ms")
    print(f"TFLOPS:     {tflops_fp32:.2f}")
    print(f"Bandwidth:  {((M*K + K*N + M*N) * 4 / 1e9) / (time_fp32 / 1000.0):.2f} GB/s\n")
    
    # ========================================================================
    # Benchmark 2: FP16 compiled
    # ========================================================================
    print("=" * 80)
    print("Benchmark 2: FP16 Compiled (torch.compile)")
    print("=" * 80)
    
    try:
        time_fp16 = benchmark_matmul(fp16_matmul_compiled, x_fp16, w_fp16, 
                                     name="FP16 Compiled", warmup=50, iters=200)
        tflops_fp16 = (flops / 1e12) / (time_fp16 / 1000.0)
        
        print(f"Time:       {time_fp16:.3f} ms")
        print(f"TFLOPS:     {tflops_fp16:.2f}")
        print(f"Speedup:    {time_fp32 / time_fp16:.2f}x vs FP32\n")
    except Exception as e:
        print(f"Failed to compile: {e}\n")
        time_fp16 = None
        tflops_fp16 = None
    
    # ========================================================================
    # Benchmark 3: FP8 naive (no compile)
    # ========================================================================
    print("=" * 80)
    print("Benchmark 3: FP8 Naive (quantize + dequantize)")
    print("=" * 80)
    
    time_fp8_naive = benchmark_matmul(fp8_matmul_naive, x_fp16, w_fp16,
                                      name="FP8 Naive", warmup=10, iters=100)
    
    print(f"Time:       {time_fp8_naive:.3f} ms")
    print(f"Note:       Overhead from quantization/dequantization")
    print(f"Speedup:    {time_fp32 / time_fp8_naive:.2f}x vs FP32\n")
    
    # ========================================================================
    # Benchmark 4: FP8 compiled ⭐ (KEY OPTIMIZATION)
    # ========================================================================
    print("=" * 80)
    print("Benchmark 4: FP8 Compiled ⭐ (torch.compile + FP8 tensors)")
    print("=" * 80)
    
    try:
        time_fp8_compiled = benchmark_matmul(
            fp8_matmul_compiled, x_fp8, w_fp8, x_scale, w_scale,
            name="FP8 Compiled", warmup=100, iters=200
        )
        tflops_fp8 = (flops / 1e12) / (time_fp8_compiled / 1000.0)
        
        print(f"Time:       {time_fp8_compiled:.3f} ms")
        print(f"TFLOPS:     {tflops_fp8:.2f}")
        print(f"Speedup:    {time_fp32 / time_fp8_compiled:.2f}x vs FP32")
        
        if time_fp16:
            print(f"Speedup:    {time_fp16 / time_fp8_compiled:.2f}x vs FP16")
        
        if peak_fp8_tflops is not None:
            print(
                f"HW Util:    {(tflops_fp8 / peak_fp8_tflops) * 100:.1f}% "
                f"of B200 dense FP8 peak ({peak_fp8_tflops:,.0f} TFLOPS)\n"
            )
        else:
            print("HW Util:    unavailable (no verified dense peak for this device)\n")
            
    except Exception as e:
        print(f"FAILED: Native FP8 compiled matmul did not run: {e}\n")
        return 4
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"{'Method':<25} {'Time (ms)':<12} {'TFLOPS':<10} {'Speedup':<10}")
    print("-" * 80)
    print(f"{'FP32 Baseline':<25} {time_fp32:<12.3f} {tflops_fp32:<10.2f} {'1.00x':<10}")
    
    if time_fp16:
        print(f"{'FP16 Compiled':<25} {time_fp16:<12.3f} {tflops_fp16:<10.2f} {f'{time_fp32/time_fp16:.2f}x':<10}")
    
    print(f"{'FP8 Naive':<25} {time_fp8_naive:<12.3f} {'N/A':<10} {f'{time_fp32/time_fp8_naive:.2f}x':<10}")
    
    if time_fp8_compiled:
        print(f"{'FP8 Compiled ⭐':<25} {time_fp8_compiled:<12.3f} {tflops_fp8:<10.2f} {f'{time_fp32/time_fp8_compiled:.2f}x':<10}")
    
    print("\n" + "=" * 80)
    print("KEY LEARNINGS")
    print("=" * 80)
    
    if has_fp8:
        print("[OK] Blackwell FP8 Benefits:")
        if peak_fp8_tflops is not None and peak_fp16_tflops is not None:
            print(
                "  • 2x theoretical dense Tensor Core peak vs FP16/BF16 "
                f"({peak_fp8_tflops:,.0f} vs {peak_fp16_tflops:,.0f} TFLOPS)"
            )
        print("  • 4x memory savings vs FP32")
        print("  • Native hardware support (no emulation overhead)")
        print("  • torch.compile preserves the native scaled-GEMM path")
    
    print("\nRecommended usage patterns:")
    print("  1. Training: FP16 mixed precision (balance speed & accuracy)")
    print("  2. Inference: FP8 for 2x throughput (Blackwell)")
    print("  3. Always use torch.compile for matmul-heavy workloads")
    print("  4. Quantize weights offline, keep activations FP16")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
