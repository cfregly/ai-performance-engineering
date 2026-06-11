# DRAFT — internal review pending
#
# Standalone reproducer: `torch._scaled_grouped_mm` refuses NVFP4
# (torch.float4_e2m1fn_x2) inputs, although the SAME device (sm_103,
# Blackwell Ultra) runs NVFP4 block-scaled GEMMs through the non-grouped
# `torch._scaled_mm` (cuBLASLt `nvjet_sm103 ... Avec16UE4M3_Bvec16UE4M3`
# kernels), and CUTLASS ships grouped NVFP4 block-scaled GEMM kernels for
# sm100/sm103.
#
# Consequence for grouped (MoE-expert-style) NVFP4 workloads: the only
# library path is N separate `_scaled_mm` launches. This script:
#   1. proves the per-group NVFP4 `_scaled_mm` works on this device and
#      prints the dispatched kernel name (profiler),
#   2. captures the exact `_scaled_grouped_mm` NVFP4 refusal errors
#      (both with NVFP4-style e4m3 block scales and with fp32 scales),
#   3. times the 30-launch `_scaled_mm` fallback that the refusal forces
#      (the measured cost of the missing feature),
#   4. (child process, because the failure poisons the CUDA context)
#      shows that even the SUPPORTED dtype, FP8 rowwise, dispatches a
#      CUTLASS kernel that traps at runtime on sm_103 with
#      "Arch conditional MMA instruction used without targeting
#      appropriate compute capability" — i.e. `_scaled_grouped_mm`
#      currently has no working path at all on sm_103 in this build.
#
# Shapes are an MoE-expert grouped workload: g=2 groups per call with
# M=(192, 320), N=3072, K=4096, batched over 15 inputs = 30 sub-problems.
#
# Tested environment: NVIDIA GB300 (sm_103), CUDA 13.2,
# torch 2.12.0a0 (NGC 26.05). No external deps.

import subprocess
import sys
import time

import torch

G_MS = (192, 320)       # per-group M (token counts routed to each expert)
N, K = 3072, 4096
N_INPUTS = 15           # batched workload: 15 inputs x 2 groups = 30 GEMMs
SF_VEC = 16             # NVFP4 scale granularity (16 elements per scale)
WARMUP, ITERS = 10, 50


def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(a, b):
    return ceil_div(a, b) * b


def make_nvfp4_operands(m, n, k, device="cuda"):
    """A (m, k/2) fp4x2 row-major; B as (k/2, n) column-major view;
    scales in the cuBLASLt 128x4-blocked float8_e4m3fn layout (flat).

    Scale VALUES are synthetic (all ones): kernel dispatch and timing
    depend on dtype/shape/layout only, not on scale contents.
    """
    a = torch.randint(-128, 128, (m, k // 2), dtype=torch.int8, device=device)
    b = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8, device=device)
    a = a.view(torch.float4_e2m1fn_x2)
    b = b.view(torch.float4_e2m1fn_x2)
    sf_k = ceil_div(k, SF_VEC)
    numel_a = round_up(m, 128) * round_up(sf_k, 4)
    numel_b = round_up(n, 128) * round_up(sf_k, 4)
    scale_a = torch.ones(numel_a, dtype=torch.float8_e4m3fn, device=device)
    scale_b = torch.ones(numel_b, dtype=torch.float8_e4m3fn, device=device)
    return a, b.transpose(0, 1), scale_a, scale_b


def scaled_mm_nvfp4(a, bt, scale_a, scale_b):
    return torch._scaled_mm(a, bt, scale_a, scale_b, bias=None,
                            out_dtype=torch.float16)


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
            names[evt.key] = names.get(evt.key, 0.0) + evt.self_device_time_total
    return names


# Child snippet for step 4: the FP8 rowwise grouped call. Run isolated —
# on sm_103 the dispatched kernel traps and poisons the CUDA context.
FP8_CHILD = r"""
import torch
gm = (192, 320)
n, k = 3072, 4096
sum_m = sum(gm)
offs = torch.tensor([gm[0], sum_m], dtype=torch.int32, device="cuda")
a8 = torch.randn(sum_m, k, device="cuda").to(torch.float8_e4m3fn)
b8 = torch.randn(len(gm), n, k, device="cuda").to(torch.float8_e4m3fn)
sa8 = torch.ones(sum_m, dtype=torch.float32, device="cuda")
sb8 = torch.ones(len(gm), n, dtype=torch.float32, device="cuda")
out = torch._scaled_grouped_mm(a8, b8.transpose(-2, -1), sa8, sb8,
                               offs=offs, out_dtype=torch.bfloat16)
torch.cuda.synchronize()
print("FP8_GROUPED_OK", tuple(out.shape))
"""


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    cap = torch.cuda.get_device_capability()
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}, "
          f"sm_{cap[0]}{cap[1]}")

    # ------------------------------------------------------------------
    # 1. Per-group NVFP4 _scaled_mm WORKS on this device (the hardware and
    #    the library support NVFP4 GEMM; only the grouped entry refuses it).
    # ------------------------------------------------------------------
    groups = [make_nvfp4_operands(m, N, K) for m in G_MS]
    outs = [scaled_mm_nvfp4(*g) for g in groups]
    torch.cuda.synchronize()
    for m, out in zip(G_MS, outs):
        assert out.shape == (m, N) and out.dtype == torch.float16
    print(f"\n[1] per-group NVFP4 _scaled_mm: OK "
          f"(M={G_MS}, N={N}, K={K}, out fp16)")
    kerns = gpu_kernel_names(lambda: scaled_mm_nvfp4(*groups[0]))
    for name in sorted(kerns, key=lambda n: -kerns[n]):
        print(f"    dispatched kernel: {name}")

    # ------------------------------------------------------------------
    # 2. torch._scaled_grouped_mm REFUSES the same operands in NVFP4.
    #    (Validation-time errors: no kernel is launched, context stays
    #    healthy.)
    # ------------------------------------------------------------------
    sum_m = sum(G_MS)
    offs = torch.tensor([G_MS[0], sum_m], dtype=torch.int32, device="cuda")

    a4 = torch.randint(-128, 128, (sum_m, K // 2), dtype=torch.int8,
                       device="cuda").view(torch.float4_e2m1fn_x2)
    b4 = torch.randint(-128, 128, (len(G_MS), N, K // 2), dtype=torch.int8,
                       device="cuda").view(torch.float4_e2m1fn_x2)
    b4t = b4.transpose(-2, -1)
    sf_k = ceil_div(K, SF_VEC)
    sa_block = torch.ones(round_up(sum_m, 128) * round_up(sf_k, 4),
                          dtype=torch.float8_e4m3fn, device="cuda")
    sb_block = torch.ones(len(G_MS), round_up(N, 128) * round_up(sf_k, 4),
                          dtype=torch.float8_e4m3fn, device="cuda")
    sa_f32 = torch.ones(sum_m, dtype=torch.float32, device="cuda")
    sb_f32 = torch.ones(len(G_MS), N, dtype=torch.float32, device="cuda")

    print("\n[2] torch._scaled_grouped_mm with NVFP4 operands:")
    refusals = []
    for label, sa, sb in (
            ("e4m3 block scales (the NVFP4 recipe)", sa_block, sb_block),
            ("fp32 rowwise-style scales", sa_f32, sb_f32)):
        try:
            torch._scaled_grouped_mm(a4, b4t, sa, sb, offs=offs,
                                     out_dtype=torch.bfloat16)
            print(f"    {label}: UNEXPECTED — accepted NVFP4 "
                  f"(feature exists?!)")
        except Exception as exc:  # noqa: BLE001 - we want the exact message
            msg = f"{type(exc).__name__}: {exc}"
            refusals.append(msg)
            print(f"    {label}:\n        REFUSED -> {msg}")

    # ------------------------------------------------------------------
    # 3. Cost of the forced workaround: 30 separate _scaled_mm launches
    #    (15 inputs x 2 groups) — the only library path for grouped NVFP4.
    # ------------------------------------------------------------------
    batch = [make_nvfp4_operands(m, N, K) for _ in range(N_INPUTS)
             for m in G_MS]

    def run_batch():
        for g in batch:
            scaled_mm_nvfp4(*g)

    for _ in range(WARMUP):
        run_batch()
    torch.cuda.synchronize()
    beg = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    beg.record()
    for _ in range(ITERS):
        run_batch()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0 / ITERS
    gpu_ms = beg.elapsed_time(end) / ITERS
    n_gemms = len(batch)
    flops = sum(2 * m * N * K for _ in range(N_INPUTS) for m in G_MS)
    print(f"\n[3] forced fallback, {n_gemms} separate _scaled_mm launches "
          f"per batched call:")
    print(f"    GPU span : {gpu_ms * 1000:8.1f} us/call "
          f"({gpu_ms * 1000 / n_gemms:6.2f} us/GEMM)")
    print(f"    host wall: {wall_ms * 1000:8.1f} us/call")
    print(f"    achieved : {flops / (gpu_ms * 1e-3) / 1e12:8.1f} TFLOP/s "
          f"(launch-bound; each GEMM is a tiny M={G_MS} problem)")

    # ------------------------------------------------------------------
    # 4. Even the SUPPORTED dtype (FP8 rowwise) has no working grouped
    #    path on sm_103: the dispatched CUTLASS kernel traps at runtime.
    #    Run in a child process — the trap poisons the CUDA context.
    # ------------------------------------------------------------------
    print("\n[4] FP8 rowwise _scaled_grouped_mm (child process):")
    res = subprocess.run([sys.executable, "-c", FP8_CHILD],
                         capture_output=True, text=True, timeout=600)
    combined = res.stdout + res.stderr
    if "FP8_GROUPED_OK" in res.stdout:
        print("    OK — FP8 grouped works on this device "
              "(refusal above is NVFP4-specific)")
    else:
        trap = "Arch conditional MMA instruction" in combined
        print(f"    FAILED (rc={res.returncode}, arch-conditional "
              f"trap={'yes' if trap else 'no'})")
        for line in combined.splitlines():
            if ("Arch conditional" in line or "Error" in line
                    or "error" in line):
                print(f"    {line.strip()}")
                break

    assert refusals, (
        "torch._scaled_grouped_mm accepted NVFP4 — this reproducer is "
        "stale, re-check against the feature request")
    print("\nDONE — _scaled_grouped_mm refuses NVFP4; per-group _scaled_mm "
          "works; the workaround costs one launch per (input, group).")


if __name__ == "__main__":
    main()
