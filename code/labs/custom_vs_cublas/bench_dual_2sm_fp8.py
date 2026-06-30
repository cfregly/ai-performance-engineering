#!/usr/bin/env python3
"""FP8 (e4m3) dual_2sm kernel-level A/B bench vs cuBLASLt FP8 (Front F8).

Usage (on the GB300 pod, GPU 2 only, under flock /tmp/gpu2.lock):
    export CUDA_VISIBLE_DEVICES=2
    cd /work/ai-performance-engineering/code
    python labs/custom_vs_cublas/bench_dual_2sm_fp8.py [--size 8192] \
        [--interleave N] [--fp8 N,S,WS,MB,K,P,RG,TE ...] [--with-fp16]

Arms:
  - cuBLASLt FP8: torch._scaled_mm with unit fp32 scales (e4m3 x e4m3 ->
    fp16), the ceiling reference and A/B opponent.
  - custom FP8 dual_2sm configs (each a separate JIT build).
  - optional --with-fp16: the FP16 champion on its own fp16 copy of the
    same logical data (absolute cross-dtype reference; same FLOP count).

Correctness (tolerances are EXPLICIT):
  - 'exact' dataset (default for verify): values from {0,+-0.5,...,+-2}
    cast to e4m3. Every product is a multiple of 0.25 with |sum| <= 32768,
    so every fp32 partial sum is exact regardless of order -> kernel out
    (fp16) must equal the fp32 upcast-matmul reference rounded to fp16
    BIT-EXACTLY: required rel_err == 0.0.
  - 'randn' dataset: e4m3-quantized 0.25*randn. fp32 accumulation order
    differs between implementations -> gate rel_err < 1e-3 (expected ~1e-6;
    reported, not asserted to 0).
Reference matmul runs with TF32 disabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _LAB_DIR.parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch  # noqa: E402

# Real-SoL ceiling is CALIBRATED from the cuBLASLt FP8 asymptote (B61
# method), not from a marketing number; this constant is only the fallback
# denominator for the %SoL column until calibrate() overwrites it.
FP8_REAL_SOL_TFLOPS = 3750.0

_FP8_EXACT_VALUES_CPU = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
_FP8_EXACT_VALUES_BY_DEVICE: dict[torch.device, torch.Tensor] = {}


def _median_in_place(values: list[float]) -> float:
    values.sort()
    count = len(values)
    mid = count // 2
    if count & 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) * 0.5


def _paired_speedup_stats(
    candidate_samples: list[float],
    baseline_samples: list[float],
    ratio_scratch: list[float],
) -> tuple[float, int]:
    ratio_scratch.clear()
    wins = 0
    for candidate_ms, baseline_ms in zip(candidate_samples, baseline_samples):
        if candidate_ms < baseline_ms:
            wins += 1
        ratio_scratch.append(baseline_ms / candidate_ms)
    return _median_in_place(ratio_scratch), wins


def fp8_exact_values(device: torch.device | str = "cuda") -> torch.Tensor:
    dev = torch.device(device)
    vals = _FP8_EXACT_VALUES_BY_DEVICE.get(dev)
    if vals is None:
        vals = _FP8_EXACT_VALUES_CPU.to(dev)
        _FP8_EXACT_VALUES_BY_DEVICE[dev] = vals
    return vals


def make_fp8_data(size: int, kind: str):
    torch.manual_seed(42)
    if kind == "exact":
        vals = fp8_exact_values("cuda")
        a = vals[torch.randint(0, 9, (size, size), device="cuda")].to(torch.float8_e4m3fn)
        b = vals[torch.randint(0, 9, (size, size), device="cuda")].to(torch.float8_e4m3fn)
    else:
        a = (0.25 * torch.randn(size, size, device="cuda")).to(torch.float8_e4m3fn)
        b = (0.25 * torch.randn(size, size, device="cuda")).to(torch.float8_e4m3fn)
    return a, b


def fp32_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.matmul(a.float(), b.float().T)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved


_ONE = None


def scaled_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """cuBLASLt FP8: C = A @ B^T, e4m3 x e4m3, unit scales, fp16 out."""
    global _ONE
    if _ONE is None:
        _ONE = torch.ones((), device=a.device, dtype=torch.float32)
    return torch._scaled_mm(a, b.t(), scale_a=_ONE, scale_b=_ONE, out_dtype=torch.float16)


def bench(fn, a, b, warmup=10, iters=50):
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    start.record(current_stream)
    for _ in range(iters):
        fn(a, b)
    end.record(current_stream)
    end.synchronize()
    return start.elapsed_time(end) / iters  # ms


def _relative_error(out: torch.Tensor, ref16: torch.Tensor, ref_abs_max=None, error_stats=None) -> float:
    if ref_abs_max is None:
        ref_abs_max = ref16.abs().max()
    if error_stats is None:
        error_stats = torch.empty(2, device=ref16.device, dtype=ref16.dtype)
    error_stats[0].copy_((ref16 - out.float()).abs().max())
    error_stats[1].copy_(ref_abs_max)
    error_stats_host = error_stats.detach().cpu()
    max_diff = float(error_stats_host[0])
    denom = float(error_stats_host[1])
    return max_diff / denom if denom else max_diff


def check(fn, a, b, ref16, ref_abs_max=None, error_stats=None):
    out = fn(a, b).float()
    return _relative_error(out, ref16, ref_abs_max, error_stats)


def report(name, ms, rel, M, N, K, exact_mode):
    tflops = (2 * M * N * K / 1e12) / (ms / 1e3)
    sol = 100.0 * tflops / FP8_REAL_SOL_TFLOPS
    bad = (rel != 0.0) if exact_mode else (rel >= 1e-3)
    flag = "BAD" if bad else "OK "
    print(f"  {name:<34} {ms*1e3:>9.1f} us  {tflops:>8.1f} TFLOPS  {sol:>5.1f}% calSoL  rel_err={rel:.2e} [{flag}]")
    return tflops


def load_fp8(cfg):
    n, s, ws, mb, k, p, rg, te = cfg[:8]
    dh = cfg[8] if len(cfg) > 8 else 0
    eo = cfg[9] if len(cfg) > 9 else 0
    from labs.custom_vs_cublas.tcgen05_loader import _load_tcgen05_dual_2sm_fp8_module
    mod = _load_tcgen05_dual_2sm_fp8_module(n, s, ws, mb, k, 32, p, rg, te, dh, eo)
    return mod.matmul_tcgen05_dual_2sm_fp8


def main():
    global FP8_REAL_SOL_TFLOPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--data", choices=["exact", "randn"], default="exact")
    parser.add_argument("--interleave", type=int, default=0, metavar="N",
                        help="round-robin all arms N times; report per-arm median")
    parser.add_argument("--fp8", action="append", default=None,
                        metavar="N,S,WS,MB,K,P,RG,TE[,DH[,EO]]",
                        help="custom FP8 arm config; repeatable (DH=1: in-kernel fp16 D; "
                             "EO=2: F8c overlap-epilogue warpgroup)")
    parser.add_argument("--with-fp16", action="store_true",
                        help="add the FP16 champion arm (own fp16 data, same FLOPs)")
    parser.add_argument("--sol", type=float, default=0.0,
                        help="override the calibrated SoL denominator (TFLOPS)")
    args = parser.parse_args()

    M = N = K = args.size
    a, b = make_fp8_data(args.size, args.data)
    ref = fp32_ref(a, b)
    ref16_for_check = ref.to(torch.float16).float()
    del ref
    ref_abs_max = ref16_for_check.abs().max()
    error_stats = torch.empty(2, device=ref16_for_check.device, dtype=ref16_for_check.dtype)
    exact_mode = (args.data == "exact")

    dev = torch.cuda.get_device_properties(0)
    print(f"\nDevice: {dev.name} (SM {dev.major}.{dev.minor}, {dev.multi_processor_count} SMs)")
    print(f"GEMM:   {M}x{N}x{K} e4m3 x e4m3 -> fp32 accum -> fp16, {2*M*N*K/1e12:.2f} TFLOP")
    print(f"Data:   {args.data} (exact-mode requires rel_err == 0.0; randn gates < 1e-3)\n")

    if args.fp8:
        cfgs = [tuple(int(x) for x in s.split(",")) for s in args.fp8]
    else:
        cfgs = [(128, 3, 1, 3, 128, 1, 8, 1)]

    arms = []
    rel = check(scaled_mm, a, b, ref16_for_check, ref_abs_max, error_stats)
    arms.append(("cuBLASLt FP8 (scaled_mm)", scaled_mm, rel, (a, b)))

    for cfg in cfgs:
        n, s, ws, mb, k, p, rg, te = cfg[:8]
        dh = cfg[8] if len(cfg) > 8 else 0
        eo = cfg[9] if len(cfg) > 9 else 0
        name = (f"fp8_2sm n{n}s{s}w{ws}mb{mb}k{k}p{p}rg{rg}te{te}"
                + (f"dh{dh}" if dh else "") + (f"eo{eo}" if eo else ""))
        try:
            fn = load_fp8(cfg)
            rel = check(fn, a, b, ref16_for_check, ref_abs_max, error_stats)
            arms.append((name, fn, rel, (a, b)))
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<34} FAILED: {type(e).__name__}: {e}")

    if args.with_fp16:
        a16 = a.float().to(torch.float16)
        b16 = b.float().to(torch.float16)
        from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_dual_cta_2sm
        ref16 = fp32_ref(a16, b16)
        out = matmul_tcgen05_dual_cta_2sm(a16, b16).float()
        ref16_for_check = ref16.to(torch.float16).float()
        ref16_abs_max = ref16.abs().max()
        rel16 = _relative_error(out, ref16_for_check, ref16_abs_max, error_stats)
        arms.append(("fp16_2sm champion (own data)", matmul_tcgen05_dual_cta_2sm, rel16, (a16, b16)))

    if args.sol > 0:
        FP8_REAL_SOL_TFLOPS = args.sol

    if args.interleave > 0:
        print(f"Interleaved A/B: {args.interleave} round-robin reps/arm (median +/- spread)")
        # warmup every arm once
        for _name, fn, _, data in arms:
            bench(fn, *data, warmup=5, iters=10)
        samples = {name: [] for name, *_ in arms}
        for _ in range(args.interleave):
            for name, fn, _, data in arms:
                samples[name].append(bench(fn, *data, warmup=2, iters=10))
        ratio_scratch: list[float] = []
        base = samples[arms[0][0]]
        paired_vs_base = [
            (
                name,
                *_paired_speedup_stats(samples[name], base, ratio_scratch),
                len(base),
            )
            for name, *_ in arms[1:]
        ]
        paired_vs_champ: list[tuple[str, float, int, int]] = []
        if len(arms) > 2:
            champ_name = arms[1][0]
            champ = samples[champ_name]
            paired_vs_champ = [
                (
                    name,
                    *_paired_speedup_stats(samples[name], champ, ratio_scratch),
                    len(champ),
                )
                for name, *_ in arms[2:]
            ]
        print()
        for name, _fn, rel, _data in arms:
            arm_samples = samples[name]
            med = _median_in_place(arm_samples)
            lo, hi = arm_samples[0], arm_samples[-1]
            report(name, med, rel, M, N, K, exact_mode)
            print(f"  {'':<34} spread [{lo*1e3:.1f}, {hi*1e3:.1f}] us over {args.interleave} reps")
        # paired verdict vs the cuBLASLt arm
        for name, ratio, wins, total in paired_vs_base:
            print(f"  paired {name} vs cuBLASLt FP8: median speedup {ratio:.4f}x, wins {wins}/{total}")
        # paired verdict of every later custom arm vs the FIRST custom arm
        # (list the control/champion first) -- the F8b deep-ring readout.
        if len(arms) > 2:
            champ_name = arms[1][0]
            for name, ratio, wins, total in paired_vs_champ:
                print(f"  paired {name} vs {champ_name}: median speedup {ratio:.4f}x, wins {wins}/{total}")
    else:
        print()
        for name, fn, rel, data in arms:
            ms = bench(fn, *data)
            report(name, ms, rel, M, N, K, exact_mode)


if __name__ == "__main__":
    main()
