#!/usr/bin/env python3
"""NVFP4 (block-scaled e2m1) dual_2sm kernel-level A/B bench vs cuBLASLt NVFP4.

Usage (on the GB300 pod, GPU 1 only, under flock /tmp/gpu1.lock):
    export CUDA_VISIBLE_DEVICES=1
    cd /work/ai-performance-engineering/code
    python labs/custom_vs_cublas/bench_dual_2sm_nvfp4.py [--size 8192] \
        [--interleave N] [--nvfp4 N,S,MB,K,RG,TE,DH ...] [--gates]

Arms:
  - cuBLASLt NVFP4: torch._scaled_mm on float4_e2m1fn_x2 inputs with
    128x4-blocked e4m3 scales (the B61 warmed-asymptote ceiling reference).
  - custom NVFP4 dual_2sm configs (each a separate JIT build).

Correctness (tolerances are EXPLICIT):
  - 'exact' dataset (default): e2m1 codes from {0,+-0.5,+-1,+-1.5,+-2} and
    POWER-OF-2 scales from {0.5,1,2}. Every product is a multiple of 1/16
    with |sum| <= 8192*16, so every fp32 partial sum is exact regardless
    of order -> kernel out (fp16) must equal the fp32 dequant-matmul
    reference rounded to fp16 BIT-EXACTLY: required rel_err == 0.0.
    torch._scaled_mm on the same operands must also be exact.
  - 'randn' dataset: per-16-group amax/6 e4m3 scales, nearest-e2m1 codes
    (matching torch._scaled_mm quantization). fp32 accumulation order
    differs -> gate rel_err < 1e-3 vs the dequant reference (TF32 off),
    and report the direct diff vs torch._scaled_mm.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _LAB_DIR.parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch  # noqa: E402

# Calibrated from the warmed cuBLASLt NVFP4 asymptote at run time (B61);
# this constant is only the fallback denominator until calibrate overwrites.
NVFP4_REAL_SOL_TFLOPS = 7500.0

SF_VEC = 16
E2M1_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(x: torch.Tensor) -> torch.Tensor:
    """(rows, sf_k) scale matrix -> flat cuBLASLt 128x4-blocked layout.

    Layout law (== cute SfAtom ((32,4),(16,4)):((16,4),(0,1)) tiled
    Step<_2,_1>): offset(r, kg) within a 128x4 block = (r%32)*16 +
    ((r//32)%4)*4 + kg; k-blocks consecutive, then m-blocks.
    """
    rows, cols = x.shape
    rb, cb = ceil_div(rows, 128), ceil_div(cols, 4)
    padded = x
    if (rb * 128, cb * 4) != (rows, cols):
        padded = torch.zeros(rb * 128, cb * 4, dtype=x.dtype, device=x.device)
        padded[:rows, :cols] = x
    blocks = padded.view(rb, 128, cb, 4).permute(0, 2, 1, 3)  # (rb, cb, 128, 4)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2)
    return rearranged.reshape(-1).contiguous()


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """(rows, k) uint8 e2m1 codes -> (rows, k/2) packed, low nibble first."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def decode_codes(codes: torch.Tensor) -> torch.Tensor:
    vals = E2M1_VALS.to(codes.device)
    mag = vals[(codes & 7).long()]
    sign = torch.where(codes >= 8, -1.0, 1.0).to(mag.dtype)
    return mag * sign


def make_exact_data(size: int):
    """Exact dataset: small e2m1 values + power-of-2 scales -> exact fp32."""
    torch.manual_seed(42)
    dev = "cuda"
    # codes for {0, +-0.5, +-1, +-1.5, +-2} = {0,1,2,3,4, 9,10,11,12}
    pool = torch.tensor([0, 1, 2, 3, 4, 9, 10, 11, 12], dtype=torch.uint8, device=dev)
    a_codes = pool[torch.randint(0, 9, (size, size), device=dev)]
    b_codes = pool[torch.randint(0, 9, (size, size), device=dev)]
    spool = torch.tensor([0.5, 1.0, 2.0], device=dev)
    sa = spool[torch.randint(0, 3, (size, size // SF_VEC), device=dev)]
    sb = spool[torch.randint(0, 3, (size, size // SF_VEC), device=dev)]
    return (pack_codes(a_codes), pack_codes(b_codes),
            decode_codes(a_codes), decode_codes(b_codes), sa, sb)


def quantize_nvfp4(x: torch.Tensor):
    """fp32 (rows,k) -> (packed codes, dequant fp32, scales fp32) with
    per-16-group amax/6 e4m3 scales and nearest-e2m1 codes."""
    rows, k = x.shape
    xg = x.view(rows, k // SF_VEC, SF_VEC)
    amax = xg.abs().amax(dim=-1)
    scale8 = (amax / 6.0).clamp(min=1e-4).to(torch.float8_e4m3fn)
    s = scale8.float()
    q = (xg / s.unsqueeze(-1)).clamp(-6.0, 6.0)
    cand = E2M1_VALS.to(x.device)
    idx = (q.abs().unsqueeze(-1) - cand).abs().argmin(dim=-1).to(torch.uint8)
    neg = (q < 0) & (idx > 0)
    codes = (idx + neg.to(torch.uint8) * 8).view(rows, k)
    deq = decode_codes(codes)
    return pack_codes(codes), deq, s


def make_randn_data(size: int):
    torch.manual_seed(42)
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    a_pack, a_deq, sa = quantize_nvfp4(a)
    b_pack, b_deq, sb = quantize_nvfp4(b)
    return a_pack, b_pack, a_deq, b_deq, sa, sb


def fp32_ref(a_deq, b_deq, sa, sb):
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        af = a_deq * sa.repeat_interleave(SF_VEC, dim=1)
        bf = b_deq * sb.repeat_interleave(SF_VEC, dim=1)
        return torch.matmul(af, bf.T)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved


def scaled_mm_nvfp4(a_pack, b_pack, sfa_blocked, sfb_blocked):
    a4 = a_pack.view(torch.float4_e2m1fn_x2)
    b4 = b_pack.view(torch.float4_e2m1fn_x2)
    return torch._scaled_mm(a4, b4.transpose(0, 1), sfa_blocked, sfb_blocked,
                            bias=None, out_dtype=torch.float16)


def bench(fn, args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def rel_err(out: torch.Tensor, ref_f32: torch.Tensor) -> float:
    ref16 = ref_f32.to(torch.float16).float()
    max_diff = (ref16 - out.float()).abs().max().item()
    denom = ref16.abs().max().item()
    return max_diff / denom if denom else max_diff


def report(name, ms, rel, M, N, K, exact_mode):
    tflops = (2 * M * N * K / 1e12) / (ms / 1e3)
    sol = 100.0 * tflops / NVFP4_REAL_SOL_TFLOPS
    bad = (rel != 0.0) if exact_mode else (rel >= 1e-3)
    flag = "BAD" if bad else "OK "
    print(f"  {name:<38} {ms*1e3:>9.1f} us  {tflops:>8.1f} TFLOPS  {sol:>5.1f}% calSoL  rel_err={rel:.2e} [{flag}]")
    return tflops


def load_nvfp4(cfg):
    n, s, mb, k, rg, te, dh = cfg[:7]
    from labs.custom_vs_cublas.tcgen05_loader import _load_tcgen05_dual_2sm_nvfp4_module
    mod = _load_tcgen05_dual_2sm_nvfp4_module(n, s, mb, k, rg, te, dh)
    return mod.matmul_tcgen05_dual_2sm_nvfp4


def run_gates(size=4096):
    """FP8 + FP16 champion paths must still PASS (shared loader)."""
    print("\n[gates] FP8 + FP16 champion correctness on the shared loader")
    from labs.custom_vs_cublas.tcgen05_loader import (
        matmul_tcgen05_dual_2sm_fp8, matmul_tcgen05_dual_cta_2sm)
    torch.manual_seed(7)
    vals = torch.tensor([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], device="cuda")
    a8 = vals[torch.randint(0, 9, (size, size), device="cuda")].to(torch.float8_e4m3fn)
    b8 = vals[torch.randint(0, 9, (size, size), device="cuda")].to(torch.float8_e4m3fn)
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref8 = torch.matmul(a8.float(), b8.float().T)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
    r8 = rel_err(matmul_tcgen05_dual_2sm_fp8(a8, b8), ref8)
    print(f"  fp8 champion  rel_err={r8:.2e} [{'OK' if r8 == 0.0 else 'BAD'}]")
    a16 = a8.float().to(torch.float16)
    b16 = b8.float().to(torch.float16)
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref16 = torch.matmul(a16.float(), b16.float().T)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
    r16 = rel_err(matmul_tcgen05_dual_cta_2sm(a16, b16), ref16)
    print(f"  fp16 champion rel_err={r16:.2e} [{'OK' if r16 == 0.0 else 'BAD'}]")
    return r8 == 0.0 and r16 == 0.0


def main():
    global NVFP4_REAL_SOL_TFLOPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--data", choices=["exact", "randn"], default="exact")
    parser.add_argument("--interleave", type=int, default=0, metavar="N")
    parser.add_argument("--nvfp4", action="append", default=None,
                        metavar="N,S,MB,K,RG,TE,DH",
                        help="custom NVFP4 arm config; repeatable")
    parser.add_argument("--gates", action="store_true",
                        help="run FP8 + FP16 champion gates and exit")
    parser.add_argument("--sol", type=float, default=0.0)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if args.gates:
        ok = run_gates()
        sys.exit(0 if ok else 1)

    M = N = K = args.size
    if args.data == "exact":
        a_pack, b_pack, a_deq, b_deq, sa, sb = make_exact_data(args.size)
    else:
        a_pack, b_pack, a_deq, b_deq, sa, sb = make_randn_data(args.size)
    sfa_blocked = to_blocked(sa.to(torch.float8_e4m3fn))
    sfb_blocked = to_blocked(sb.to(torch.float8_e4m3fn))
    ref = fp32_ref(a_deq, b_deq, sa, sb)
    del a_deq, b_deq
    exact_mode = (args.data == "exact")

    dev = torch.cuda.get_device_properties(0)
    print(f"\nDevice: {dev.name} (SM {dev.major}.{dev.minor}, {dev.multi_processor_count} SMs)")
    print(f"GEMM:   {M}x{N}x{K} e2m1 x e2m1 (SF_VEC=16 ue4m3) -> fp32 accum -> fp16, "
          f"{2*M*N*K/1e12:.2f} TFLOP")
    print(f"Data:   {args.data} (exact-mode requires rel_err == 0.0; randn gates < 1e-3)\n")

    if args.nvfp4:
        cfgs = [tuple(int(x) for x in s.split(",")) for s in args.nvfp4]
    else:
        cfgs = [(128, 3, 2, 256, 4, 1, 1)]  # the N4b-ratified champion

    data = (a_pack, b_pack, sfa_blocked, sfb_blocked)

    arms = []
    rel = 0.0 if args.skip_verify else rel_err(scaled_mm_nvfp4(*data), ref)
    arms.append(("cuBLASLt NVFP4 (scaled_mm)", scaled_mm_nvfp4, rel))

    for cfg in cfgs:
        n, s, mb, k, rg, te, dh = cfg[:7]
        name = f"nvfp4_2sm n{n}s{s}mb{mb}k{k}rg{rg}te{te}dh{dh}"
        try:
            fn = load_nvfp4(cfg)
            r = 0.0 if args.skip_verify else rel_err(fn(*data), ref)
            arms.append((name, fn, r))
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<38} FAILED: {type(e).__name__}: {e}")

    if args.sol > 0:
        NVFP4_REAL_SOL_TFLOPS = args.sol
    else:
        # B61 ceiling calibration: warmed cuBLASLt NVFP4 asymptote.
        ms = bench(scaled_mm_nvfp4, data, warmup=20, iters=50)
        NVFP4_REAL_SOL_TFLOPS = (2 * M * N * K / 1e12) / (ms / 1e3)
        print(f"Calibrated NVFP4 ceiling (warmed _scaled_mm): {ms*1e3:.1f} us "
              f"= {NVFP4_REAL_SOL_TFLOPS:.0f} TFLOPS\n")

    if args.interleave > 0:
        print(f"Interleaved A/B: {args.interleave} round-robin reps/arm (median +/- spread)")
        for name, fn, _ in arms:
            bench(fn, data, warmup=5, iters=10)
        samples = {name: [] for name, *_ in arms}
        for _ in range(args.interleave):
            for name, fn, _ in arms:
                samples[name].append(bench(fn, data, warmup=2, iters=10))
        print()
        for name, fn, r in arms:
            med = statistics.median(samples[name])
            lo, hi = min(samples[name]), max(samples[name])
            report(name, med, r, M, N, K, exact_mode)
            print(f"  {'':<38} spread [{lo*1e3:.1f}, {hi*1e3:.1f}] us over {args.interleave} reps")
        base = samples[arms[0][0]]
        for name, *_ in arms[1:]:
            wins = sum(1 for x, y in zip(samples[name], base) if x < y)
            ratio = statistics.median([y / x for x, y in zip(samples[name], base)])
            print(f"  paired {name} vs cuBLASLt NVFP4: median speedup {ratio:.4f}x, wins {wins}/{len(base)}")
        if len(arms) > 2:
            champ_name = arms[1][0]
            champ = samples[champ_name]
            for name, *_ in arms[2:]:
                wins = sum(1 for x, y in zip(samples[name], champ) if x < y)
                ratio = statistics.median([y / x for x, y in zip(samples[name], champ)])
                print(f"  paired {name} vs {champ_name}: median speedup {ratio:.4f}x, wins {wins}/{len(champ)}")
    else:
        print()
        for name, fn, r in arms:
            ms = bench(fn, data)
            report(name, ms, r, M, N, K, exact_mode)


if __name__ == "__main__":
    main()
