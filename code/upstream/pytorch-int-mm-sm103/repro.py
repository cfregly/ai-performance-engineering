# DRAFT — internal review pending
#
# Standalone reproducer: `torch._int_mm` (INT8 GEMM) is an order of
# magnitude SLOWER than a tf32 fp32 matmul of the same shape on
# NVIDIA GB300 (sm_103, Blackwell Ultra): measured 6.4x slower
# (column-major rhs) / 13.4x slower (row-major rhs, 3.65 ms vs 0.273 ms)
# at M=8192, K=N=4096. The profiler shows why: sm_103 dispatch lands on
# Ampere-era CUTLASS 2.x kernels (`cutlass_80_tensorop_i16832gemm_s8_*`,
# and for row-major rhs the `cutlass_80_wmma_..._forwardCompat` WMMA
# kernel), while the fp32-tf32 path gets a modern
# `cutlass3x_sm100_tensorop_*_2sm` kernel and fp16/bf16 get nvjet_sm103.
#
# This script:
#   1. times torch._int_mm at a representative shape against fp32-tf32 / fp16 /
#      bf16 matmuls of the same shape,
#   2. controls for the known row-major-rhs penalty
#      (pytorch/pytorch#165230) by timing BOTH rhs layouts:
#      column-major (cuBLAS-preferred, `w.t()` view of a contiguous
#      (N, K) tensor) and row-major contiguous (K, N),
#   3. captures the dispatched CUDA kernel name for each variant with the
#      torch profiler (the perf-bug evidence: WHICH int8 kernel sm_103
#      dispatch lands on).
#
# Tested environment: NVIDIA GB300 (sm_103), CUDA 13.2,
# torch 2.12.0a0 (NGC 26.05). No external deps.

import torch

M, K, N = 8192, 4096, 4096
WARMUP, ITERS = 10, 50


def cuda_time_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    beg = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    beg.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return beg.elapsed_time(end) / ITERS


def gpu_kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
    names = {}
    for evt in prof.key_averages():
        if (evt.device_type == torch.autograd.DeviceType.CUDA
                and evt.self_device_time_total > 0):
            names[evt.key] = names.get(evt.key, 0.0) + evt.self_device_time_total / 3.0
    return names


def report(label, ms, kerns=None):
    print(f"{label:42s}: {ms * 1000:9.1f} us/call")
    for name in sorted(kerns or {}, key=lambda n: -kerns[n]):
        print(f"    {kerns[name]:9.1f} us  {name}")


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    cap = torch.cuda.get_device_capability()
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}, "
          f"sm_{cap[0]}{cap[1]}")
    print(f"shape: M={M}, K={K}, N={N}\n")

    dev = "cuda"

    # float baselines ---------------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    x32 = torch.randn(M, K, device=dev)
    w32 = torch.randn(K, N, device=dev)
    t_tf32 = cuda_time_ms(lambda: x32 @ w32)
    report("fp32 matmul (tf32 enabled)", t_tf32,
           gpu_kernel_names(lambda: x32 @ w32))

    x16 = x32.half()
    w16 = w32.half()
    t_fp16 = cuda_time_ms(lambda: x16 @ w16)
    report("fp16 matmul", t_fp16, gpu_kernel_names(lambda: x16 @ w16))

    xb = x32.bfloat16()
    wb = w32.bfloat16()
    t_bf16 = cuda_time_ms(lambda: xb @ wb)
    report("bf16 matmul", t_bf16)

    # int8 --------------------------------------------------------------
    x8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    w8_nk = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
    w8_col = w8_nk.t()                  # (K, N) column-major view
    w8_row = w8_col.contiguous()        # (K, N) row-major

    t_int8_col = cuda_time_ms(lambda: torch._int_mm(x8, w8_col))
    report("torch._int_mm, rhs column-major", t_int8_col,
           gpu_kernel_names(lambda: torch._int_mm(x8, w8_col)))

    t_int8_row = cuda_time_ms(lambda: torch._int_mm(x8, w8_row))
    report("torch._int_mm, rhs row-major", t_int8_row,
           gpu_kernel_names(lambda: torch._int_mm(x8, w8_row)))

    # summary -----------------------------------------------------------
    print(f"\n_int_mm (col-major rhs) vs tf32 : {t_int8_col / t_tf32:6.1f}x slower")
    print(f"_int_mm (row-major rhs) vs tf32 : {t_int8_row / t_tf32:6.1f}x slower")
    print(f"_int_mm (col-major rhs) vs fp16 : {t_int8_col / t_fp16:6.1f}x slower")
    print(f"_int_mm (row-major rhs) vs fp16 : {t_int8_row / t_fp16:6.1f}x slower")
    print(f"row-major-rhs layout penalty    : {t_int8_row / t_int8_col:6.2f}x "
          f"(pytorch/pytorch#165230 covers this part)")

    best_int8 = min(t_int8_col, t_int8_row)
    assert best_int8 > 2.0 * t_tf32, (
        f"_int_mm is no longer pathologically slow vs tf32 here "
        f"({best_int8:.3f} vs {t_tf32:.3f} ms) — reproducer stale?")
    print("\nDONE — INT8 _int_mm is far slower than the fp32 tf32 matmul on "
          "this sm_103 device in BOTH rhs layouts.")


if __name__ == "__main__":
    main()
