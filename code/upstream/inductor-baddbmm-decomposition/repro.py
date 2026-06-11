# DRAFT — internal review pending
#
# Standalone reproducer: torch.compile (Inductor, max-autotune GEMM tuning)
# never considers the `bmm + pointwise bias` decomposition as an autotune
# choice for `baddbmm` with a BROADCAST bias, leaving measurable performance
# on the table at MoE expert-GEMM shapes.
#
# Mechanism (all autotune numbers honest, none of this is a benchmark bug):
#   * For `baddbmm(bias[B,1,N], x[B,M,K], w[B,K,N])` the extern (cuBLAS/ATen)
#     choice is honestly slow: cuBLAS computes C = alpha*A@B + beta*C, so
#     `at::baddbmm_out` must first MATERIALIZE the broadcast bias into the
#     output buffer (a strided elementwise copy over the full [B,M,N] output)
#     and then run a beta=1 GEMM that re-reads it.
#   * The Triton template, which reads the [B,1,N] bias in its epilogue for
#     free, therefore wins the autotune FOR THE OP AS WRITTEN.
#   * But the decomposition `bmm(x, w) + bias` is faster than BOTH: the pure
#     GEMM routes extern to a beta=0 kernel (C never read), and the rank-1
#     bias add is a cheap pointwise kernel that Inductor can fuse into
#     neighboring ops in real graphs (where it becomes effectively free).
#
# This script compiles both forms at a representative shape, both bare and
# with the activation that follows the GEMM in any MLP/MoE block (GELU),
# prints the kernels each compiled graph actually runs, and asserts:
#   1. compiled baddbmm keeps a Triton GEMM template; the decomposed form
#      routes the GEMM to an extern beta=0 kernel (no Triton template),
#   2. with the activation consumer present (the realistic case), the
#      decomposed form is faster end-to-end: the bias add fuses into the
#      activation pointwise for free,
#   3. eager baddbmm with a broadcast bias is much slower than eager bmm
#      (the ATen materialization cost itself).
#
# The bare (no-consumer) pair is also timed and printed: standalone, the
# unfused bias add costs more than the template's free epilogue read, which
# is exactly why this should be an AUTOTUNE-LEVEL DECOMPOSITION CHOICE
# (benchmarked in graph context, where pointwise fusion applies) rather
# than an unconditional decomposition.
#
# Tested environment: NVIDIA GB300 (sm_103), CUDA 13.2, driver 580.159.03,
# torch 2.12.0a0 (NGC 26.05). No repo dependencies.

import os

# Fresh tune every run: keep the experiment hermetic.
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402

B, M, K, N = 32, 640, 1024, 2048  # MoE expert GEMM1 class (bf16)
DTYPE = torch.bfloat16
WARMUP, ITERS = 20, 200


def cuda_time_us(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    beg = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    beg.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return beg.elapsed_time(end) * 1000.0 / ITERS


def gpu_kernels(fn, *args):
    """Names + total us of CUDA kernels per call (5-call profile)."""
    from torch.profiler import ProfilerActivity, profile

    fn(*args)  # ensure compiled
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn(*args)
        torch.cuda.synchronize()
    out = {}
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
            out[evt.key] = out.get(evt.key, 0.0) + evt.self_device_time_total / 5.0
    return out


def op_baddbmm(bias, x, w):
    return torch.baddbmm(bias, x, w)


def op_bmm_bias(bias, x, w):
    return torch.bmm(x, w) + bias


def op_baddbmm_gelu(bias, x, w):
    return torch.nn.functional.gelu(torch.baddbmm(bias, x, w))


def op_bmm_bias_gelu(bias, x, w):
    return torch.nn.functional.gelu(torch.bmm(x, w) + bias)


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    dev = "cuda"
    x = torch.randn(B, M, K, device=dev, dtype=DTYPE)
    w = torch.randn(B, K, N, device=dev, dtype=DTYPE)
    bias = torch.randn(B, 1, N, device=dev, dtype=DTYPE)  # broadcast over M

    # --- eager: the ATen-level cost of the broadcast-bias materialization ---
    t_eager_baddbmm = cuda_time_us(op_baddbmm, bias, x, w)
    t_eager_bmm = cuda_time_us(torch.bmm, x, w)
    print(f"eager baddbmm (broadcast bias [B,1,N]): {t_eager_baddbmm:8.1f} us/call")
    print(f"eager bmm     (no bias)               : {t_eager_bmm:8.1f} us/call")

    # --- compiled: what Inductor picks for each formulation ---
    compiled_baddbmm = torch.compile(
        op_baddbmm, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs"
    )
    compiled_bmm_bias = torch.compile(
        op_bmm_bias, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs"
    )

    t_c_baddbmm = cuda_time_us(compiled_baddbmm, bias, x, w)
    t_c_bmm_bias = cuda_time_us(compiled_bmm_bias, bias, x, w)
    print(f"compiled baddbmm        (bare)        : {t_c_baddbmm:8.1f} us/call")
    print(f"compiled bmm + bias     (bare)        : {t_c_bmm_bias:8.1f} us/call")
    print("  (bare, the unfused bias add costs more than the template's free"
          " epilogue read — the decomposition only pays in graph context)")

    # --- compiled, with the activation consumer (the realistic MLP/MoE case)
    compiled_baddbmm_gelu = torch.compile(
        op_baddbmm_gelu, fullgraph=True, dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    compiled_decomposed_gelu = torch.compile(
        op_bmm_bias_gelu, fullgraph=True, dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    t_c_baddbmm_gelu = cuda_time_us(compiled_baddbmm_gelu, bias, x, w)
    t_c_decomposed_gelu = cuda_time_us(compiled_decomposed_gelu, bias, x, w)
    print(f"compiled gelu(baddbmm)  (as written)  : {t_c_baddbmm_gelu:8.1f} us/call")
    print(f"compiled gelu(bmm+bias) (decomposed)  : {t_c_decomposed_gelu:8.1f} us/call")
    print(f"decomposition speedup with consumer   : "
          f"{t_c_baddbmm_gelu / t_c_decomposed_gelu:8.3f}x")

    # numerics: identical inputs, compare outputs
    ref = op_baddbmm_gelu(bias.float(), x.float(), w.float())
    out_a = compiled_baddbmm_gelu(bias, x, w).float()
    out_b = compiled_decomposed_gelu(bias, x, w).float()
    for name, out in (("gelu(baddbmm)", out_a), ("gelu(bmm+bias)", out_b)):
        rel = (out - ref).abs().max() / ref.abs().max()
        print(f"max rel err vs fp32 ref, {name:15s}: {rel.item():.3e}")
        assert rel < 2e-2, f"{name} numerics off"

    kerns_a = gpu_kernels(compiled_baddbmm_gelu, bias, x, w)
    kerns_b = gpu_kernels(compiled_decomposed_gelu, bias, x, w)
    print("\nkernels, compiled gelu(baddbmm):")
    for k, v in sorted(kerns_a.items(), key=lambda i: -i[1]):
        print(f"  {v:9.1f} us  {k}")
    print("kernels, compiled gelu(bmm + bias):")
    for k, v in sorted(kerns_b.items(), key=lambda i: -i[1]):
        print(f"  {v:9.1f} us  {k}")

    # 1. template choice: baddbmm keeps a Triton template GEMM; the
    #    decomposition routes the GEMM extern (cuBLAS), no template survives.
    a_has_template = any("triton_tem" in k for k in kerns_a)
    b_has_template = any("triton_tem" in k for k in kerns_b)
    assert a_has_template, (
        "expected compiled gelu(baddbmm) to pick a Triton GEMM template "
        f"(got: {sorted(kerns_a)})"
    )
    assert not b_has_template, (
        "expected compiled gelu(bmm+bias) to route the GEMM extern "
        f"(got: {sorted(kerns_b)})"
    )

    # 2. with the consumer present, the decomposition is faster end-to-end.
    assert t_c_baddbmm_gelu / t_c_decomposed_gelu > 1.05, (
        f"decomposition not faster with consumer: "
        f"{t_c_baddbmm_gelu:.1f} vs {t_c_decomposed_gelu:.1f} us"
    )

    # 3. ATen-level: broadcast-bias baddbmm pays a full output-sized
    #    materialization copy before the GEMM.
    assert t_eager_baddbmm / t_eager_bmm > 1.5, (
        f"expected eager broadcast-bias baddbmm >> eager bmm: "
        f"{t_eager_baddbmm:.1f} vs {t_eager_bmm:.1f} us"
    )

    print("\nALL ASSERTIONS PASSED — Inductor lacks the bmm+pointwise-bias "
          "decomposition choice for broadcast-bias baddbmm.")


if __name__ == "__main__":
    main()
