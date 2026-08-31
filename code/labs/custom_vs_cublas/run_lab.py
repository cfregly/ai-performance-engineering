#!/usr/bin/env python3
"""
Lab: Matching cuBLAS on Blackwell
=================================

This is a SELF-CONTAINED lab that demonstrates the performance gap between
a custom tcgen05 kernel and NVIDIA's cuBLAS library.

No imports from other chapters - everything needed is in this directory.

Stages:
  - Stage 0: cuBLAS (the target - highly optimized)
  - Stage 1: Naive CUDA with shared memory (no tensor cores)
  - Stage 2: tcgen05 tensor cores (basic CuTE/CUTLASS implementation)

Comparisons describe this run only; backend failures never substitute another kernel.
"""

import argparse
import ctypes
import math
from pathlib import Path

import torch

# Local imports only
_LAB_DIR = Path(__file__).resolve().parent

# Try to load custom naive kernel
_kernels_lib = None
try:
    _kernels_lib = ctypes.CDLL(str(_LAB_DIR / "kernels.so"))
except OSError:
    pass  # Stage 1 reports an unavailable backend; it never substitutes cuBLAS.


def get_device_info():
    """Get GPU device information."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / 1e9,
    }


def benchmark_kernel(fn, *args, warmup=5, iters=20):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    start.record(current_stream)
    for _ in range(iters):
        fn(*args)
    end.record(current_stream)
    end.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise RuntimeError("CUDA event timing must be finite and positive")
    return elapsed_ms


def calculate_tflops(M, N, K, time_ms):
    """Calculate GEMM throughput from a successful, positive measurement."""
    if not math.isfinite(time_ms) or time_ms <= 0:
        raise ValueError("time_ms must be finite and positive")
    return 2 * M * N * K / (time_ms * 1e9)


def gemm_arithmetic_intensity(m, n, k, *, input_bytes=2, output_bytes=2):
    """Ideal GEMM FLOPs/byte: read A/B once and write C once (beta=0).

    This lower bound on traffic omits repeated tile loads, temporaries, and
    allocation/epilogue traffic; it does not establish a measured bottleneck.
    """
    if min(m, n, k, input_bytes, output_bytes) <= 0:
        raise ValueError("dimensions and element sizes must be positive")
    traffic = input_bytes * (m * k + k * n) + output_bytes * m * n
    return 2 * m * n * k / traffic


# =============================================================================
# Stage Implementations
# =============================================================================

def stage0_cublas(A, B_T):
    """Stage 0: torch.matmul CUDA library baseline; library/kernel selection is runtime dependent."""
    return torch.matmul(A, B_T.T)


def stage1_naive_smem(A, B_T):
    """Stage 1: Native scalar CUDA GEMM with shared-memory tiling and FP32 output."""
    if _kernels_lib is None:
        raise RuntimeError("native kernels.so unavailable; build the lab before selecting stage 1")

    if A.dtype != torch.float16 or B_T.dtype != torch.float16 or not A.is_cuda or A.device != B_T.device:
        raise ValueError("stage 1 requires FP16 inputs on the same CUDA device")
    if A.ndim != 2 or B_T.ndim != 2 or A.shape[1] != B_T.shape[1]:
        raise ValueError("stage 1 requires A[M,K] and B_T[N,K]")
    A, B_T = A.contiguous(), B_T.contiguous()
    M, K = A.shape
    N = B_T.shape[0]
    if min(M, N, K) <= 0:
        raise ValueError("matrix dimensions must be positive")
    C = torch.empty(M, N, device=A.device, dtype=torch.float32)

    _kernels_lib.launch_gemm_naive_smem.restype = None
    with torch.cuda.device(A.device):
        _kernels_lib.launch_gemm_naive_smem(
            ctypes.c_void_p(A.data_ptr()),
            ctypes.c_void_p(B_T.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(K),
            ctypes.c_void_p(torch.cuda.current_stream(A.device).cuda_stream)
        )
    return C


def stage2_tcgen05_basic(A, B_T):
    """Stage 2: Single-stage tcgen05 MMA, TMA loads and TMEM accumulators."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05
    return matmul_tcgen05(A, B_T)


def stage3_tcgen05_pipelined(A, B_T):
    """Stage 3: Two shared-memory pipeline buffers with explicit reuse waits."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_pipelined
    return matmul_tcgen05_pipelined(A, B_T)


def stage4_tcgen05_3stage(A, B_T):
    """Stage 4: Three shared-memory pipeline buffers with explicit reuse waits."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_3stage
    return matmul_tcgen05_3stage(A, B_T)


def stage5_tcgen05_swizzled(A, B_T):
    """Stage 5: Three-stage pipeline with swizzled tile scheduling; cache benefit requires measurement."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_swizzled
    return matmul_tcgen05_swizzled(A, B_T)


def stage6_cluster(A, B_T):
    """Stage 6: Two-CTA cluster structure with ordinary TMA loads; this kernel does not multicast."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_cluster
    return matmul_tcgen05_cluster(A, B_T)


def stage7_4stage_deep(A, B_T):
    """Stage 7: Four shared-memory buffers; pipeline depth is a configuration, not a measured optimum."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_warp_spec
    return matmul_tcgen05_warp_spec(A, B_T)


def stage8_no_wait(A, B_T):
    """Stage 8: Overlapped MMA with an empty barrier wait before every shared-memory stage reuse.

    The legacy no_wait name does not permit overwriting data still read by MMA.
    This describes the synchronization pattern only; any speedup requires a
    verified measurement against the selected baseline on the target workload."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_no_wait
    return matmul_tcgen05_no_wait(A, B_T)


def stage9_no_wait_swizzle(A, B_T):
    """Stage 9: Explicitly synchronized stage reuse with swizzled tile scheduling."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_no_wait_swizzle
    return matmul_tcgen05_no_wait_swizzle(A, B_T)


def stage10_warp_parallel(A, B_T):
    """Stage 10: TMA prefetch with explicit stage-reuse synchronization."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_warp_parallel
    return matmul_tcgen05_warp_parallel(A, B_T)


def stage11_cluster(A, B_T):
    """Stage 11: Alias of stage 6; this kernel does not multicast."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_cluster
    return matmul_tcgen05_cluster(A, B_T)


def stage13_dual_cta(A, B_T):
    """Stage 13: 128x128 tiles and smaller per-CTA storage; achieved occupancy requires measurement."""
    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_dual_cta
    return matmul_tcgen05_dual_cta(A, B_T)


def stage12_cutlass(A, B_T):
    """Stage 12: CUTLASS CollectiveBuilder backend; its performance must be measured on this workload."""
    from labs.custom_vs_cublas.cutlass_gemm import cutlass_gemm
    return cutlass_gemm(A, B_T)


# These are alternatives, not a claim that every stage compounds or improves.
STAGES = {
    0: ("cuBLAS/library baseline", stage0_cublas),
    1: ("Scalar shared-memory", stage1_naive_smem),
    2: ("tcgen05 single-stage", stage2_tcgen05_basic),
    3: ("Two-stage pipeline", stage3_tcgen05_pipelined),
    4: ("Three-stage pipeline", stage4_tcgen05_3stage),
    5: ("Swizzled tiles", stage5_tcgen05_swizzled),
    6: ("Cluster ordinary TMA", stage6_cluster),
    7: ("Four-stage pipeline", stage7_4stage_deep),
    8: ("MMA stage-reuse overlap", stage8_no_wait),
    9: ("Overlap + swizzled tiles", stage9_no_wait_swizzle),
    10: ("TMA prefetch overlap", stage10_warp_parallel),
    11: ("Cluster alias of stage 6", stage11_cluster),
    12: ("CUTLASS CollectiveBuilder", stage12_cutlass),
    13: ("Smaller 128x128 CTA tiles", stage13_dual_cta),
}


def reference_result(A, B_T):
    """Separate FP32 GEMM reference with TF32 disabled for these FP16 inputs."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return torch.matmul(A.float(), B_T.T.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def validate_result(result, reference):
    """Compare every element; nonfinite output, wrong shape/device, or mismatch fails."""
    if result.shape != reference.shape or result.device != reference.device:
        raise AssertionError("output shape/device does not match reference")
    if not result.is_floating_point() or not torch.isfinite(result).all().item():
        raise AssertionError("output must be finite and floating point")
    torch.testing.assert_close(result.float(), reference.float(), rtol=1e-3, atol=1e-2,
                               equal_nan=False, check_dtype=False)


def run_stage(stage_num, A, B_T, M, N, K, verbose=True, reference=None):
    """Verify this backend before timing; failed backends receive no throughput."""
    name, fn = STAGES[stage_num]
    try:
        if reference is None:
            reference = reference_result(A, B_T)
        output = fn(A, B_T)
        validate_result(output, reference)
        output_bytes = output.element_size()
        del output
        time_ms = benchmark_kernel(fn, A, B_T)
        # Detect state-dependent corruption after repeated execution as well.
        validate_result(fn(A, B_T), reference)
        tflops = calculate_tflops(M, N, K, time_ms)
        intensity = gemm_arithmetic_intensity(M, N, K, input_bytes=A.element_size(), output_bytes=output_bytes)
        if verbose:
            print(f"  Stage {stage_num}: {name:<27} {time_ms:>8.3f} ms  {tflops:>7.1f} TFLOPS  full-output PASS")
        return {"stage": stage_num, "name": name, "time_ms": time_ms, "tflops": tflops,
                "output_bytes": output_bytes, "ideal_flops_per_byte": intensity, "verified": True}
    except Exception as exc:
        if verbose:
            print(f"  Stage {stage_num}: {name:<27} FAILED/UNAVAILABLE: {exc}")
        return {"stage": stage_num, "name": name, "time_ms": None, "tflops": None,
                "verified": False, "error": str(exc)}


def verify_correctness(A, B_T, verbose=True, stages=None):
    """Return False if any selected backend is unavailable or has incorrect output."""
    reference = reference_result(A, B_T)
    passed = True
    for stage_num in STAGES if stages is None else stages:
        name, fn = STAGES[stage_num]
        try:
            validate_result(fn(A, B_T), reference)
            if verbose:
                print(f"  Stage {stage_num}: {name:<27} full-output PASS")
        except Exception as exc:
            passed = False
            if verbose:
                print(f"  Stage {stage_num}: {name:<27} FAILED/UNAVAILABLE: {exc}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Matching cuBLAS Lab")
    parser.add_argument("--stage", type=int, choices=tuple(STAGES), help="Run specific stage only")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size (default: 4096)")
    parser.add_argument("--verify", action="store_true", help="Verify selected backends and exit without benchmarking")
    parser.add_argument("--no-naive", action="store_true", help="Skip scalar native kernel")
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    if not torch.cuda.is_available():
        print("UNSUPPORTED: CUDA device unavailable; no benchmark ran")
        return 3
    device_info = get_device_info()
    M = ((args.size + 127) // 128) * 128
    N = ((args.size + 255) // 256) * 256
    K = ((args.size + 63) // 64) * 64
    if (M, N, K) != (args.size,) * 3:
        print(f"Adjusted size to M={M}, N={N}, K={K} for tcgen05 alignment")
    print(f"Device: {device_info['name']} (SM {device_info['compute_capability']})")
    print(f"A[{M},{K}] @ B_T[{N},{K}].T -> C[{M},{N}]; FP16 inputs")
    print(f"FLOPs: {2*M*N*K}; ideal minimum-traffic intensity with FP16 C: "
          f"{gemm_arithmetic_intensity(M, N, K):.3f} FLOPs/byte")
    print("Intensity excludes repeated loads and intermediates; no compute/memory bottleneck is inferred.")
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B_T = torch.randn(N, K, device="cuda", dtype=torch.float16)
    stages = [args.stage] if args.stage is not None else list(STAGES)
    stages = [stage for stage in stages if not (args.no_naive and stage == 1)]
    if not stages:
        parser.error("no stages selected")
    if args.verify:
        return 0 if verify_correctness(A, B_T, stages=stages) else 1
    reference = reference_result(A, B_T)
    results = [run_stage(stage, A, B_T, M, N, K, reference=reference) for stage in stages]
    baseline = next((r for r in results if r["stage"] == 0 and r["verified"]), None)
    if baseline:
        print("Measured comparisons for this run only:")
        for result in results:
            if result["stage"] and result["verified"]:
                pct = 100 * result["tflops"] / baseline["tflops"]
                print(f"  Stage {result['stage']}: {pct:.1f}% of this library baseline; "
                      f"C element size {result['output_bytes']} bytes; ideal intensity "
                      f"{result['ideal_flops_per_byte']:.3f} FLOPs/byte")
    print("Backend selection, occupancy, cache behavior and bottleneck attribution require device profiling.")
    return 0 if all(result["verified"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
