#!/usr/bin/env python3
"""Kernel-only A/B bench: dual-CTA occupancy variant vs incumbent cluster vs cuBLAS.

Usage (on the GB300 pod, GPU 2 only):
    export CUDA_VISIBLE_DEVICES=2
    cd /work/ai-performance-engineering/code
    python labs/custom_vs_cublas/bench_dual_cta.py [--size 8192] [--sweep] [--interleave N]

Reports CUDA-event kernel time, TFLOPS, and % of GB300 FP16 dense SoL
(3.75 PFLOPS), plus max relative error vs torch.matmul.

--sweep additionally tries (tile_n, stages, cluster_m) configs
(each config is a separate JIT build; first run compiles).

--interleave N round-robins every arm N times (one bench rep each) and
reports per-arm median +/- spread, so the node's 5-9% thermal drift hits all
arms equally. Use this mode for any perf claim.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _LAB_DIR.parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch  # noqa: E402

GB300_FP16_PEAK_TFLOPS = 3750.0


def bench(fn, a, b, warmup=10, iters=50):
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def check(fn, a, b, ref):
    out = fn(a, b).float()
    max_diff = (ref - out).abs().max().item()
    rel = max_diff / ref.abs().max().item()
    return rel


def report(name, ms, rel, M, N, K, suffix=""):
    tflops = (2 * M * N * K / 1e12) / (ms / 1e3)
    sol = 100.0 * tflops / GB300_FP16_PEAK_TFLOPS
    flag = "OK " if rel < 0.01 else "BAD"
    print(f"  {name:<28} {ms*1e3:>9.1f} us  {tflops:>8.1f} TFLOPS  {sol:>5.1f}% SoL  rel_err={rel:.2e} [{flag}]{suffix}")


def load_dual(tile_n: int, stages: int, cluster_m: int = 1):
    from labs.custom_vs_cublas.tcgen05_loader import _load_tcgen05_dual_cta_module
    mod = _load_tcgen05_dual_cta_module(tile_n, stages, cluster_m)
    return mod.matmul_tcgen05_dual_cta


def load_dual_2sm(tile_n: int, stages: int, warp_split: int = 0, amcast: int = 0):
    from labs.custom_vs_cublas.tcgen05_loader import _load_tcgen05_dual_cta_2sm_module
    mod = _load_tcgen05_dual_cta_2sm_module(tile_n, stages, warp_split, amcast)
    return mod.matmul_tcgen05_dual_cta_2sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--sweep", action="store_true", help="sweep (tile_n, stages, cluster_m) configs")
    parser.add_argument("--interleave", type=int, default=0, metavar="N",
                        help="round-robin all arms N times; report per-arm median (thermal-drift-fair)")
    parser.add_argument("--twosm", action="append", default=None, metavar="N,S[,WS[,AMC]]",
                        help="explicit dual_2sm arm tile_n,stages[,warp_split[,amcast]]; repeatable, "
                             "overrides the default dual_2sm arm list")
    args = parser.parse_args()

    M = N = K = args.size
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref = torch.matmul(a, b.T).float()

    dev = torch.cuda.get_device_properties(0)
    print(f"\nDevice: {dev.name} (SM {dev.major}.{dev.minor}, {dev.multi_processor_count} SMs)")
    print(f"GEMM:   {M}x{N}x{K} FP16, {2*M*N*K/1e12:.2f} TFLOP\n")

    cublas = lambda x, y: torch.matmul(x, y.T)  # noqa: E731

    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_cluster

    # Default = measured-best GB300 config (2026-06-10 sweep) plain + its
    # 2x1-cluster B-multicast variant (E3); --sweep tries all.
    if args.sweep:
        configs = [(128, 3, 1), (128, 2, 1), (256, 2, 1), (128, 3, 2), (256, 2, 2), (256, 2, 4)]
    else:
        configs = [(256, 2, 1), (256, 2, 2)]

    arms = [("cuBLAS (target)", cublas, 0.0)]
    rel = check(matmul_tcgen05_cluster, a, b, ref)
    arms.append(("cluster (incumbent)", matmul_tcgen05_cluster, rel))
    for tile_n, stages, cluster_m in configs:
        name = f"dual_cta n={tile_n} s={stages} cm={cluster_m}"
        try:
            fn = load_dual(tile_n, stages, cluster_m)
            rel = check(fn, a, b, ref)
            arms.append((name, fn, rel))
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<28} FAILED: {e}")

    # 2-SM UMMA pair (cta_group::2) arms: the U-front structural lever.
    # (128,3) is the measured winner (3 CTAs/SM; see loader docstring).
    # 4-tuples are (tile_n, stages, warp_split, amcast); V-front levers.
    if args.twosm:
        twosm_configs = []
        for spec in args.twosm:
            parts = [int(x) for x in spec.split(",")]
            parts += [0] * (4 - len(parts))
            twosm_configs.append(tuple(parts[:4]))
    elif args.sweep:
        twosm_configs = [(256, 2, 0, 0), (128, 3, 0, 0), (128, 2, 0, 0),
                         (128, 3, 1, 0), (128, 3, 1, 1), (128, 3, 0, 1)]
    else:
        twosm_configs = [(128, 3, 0, 0), (256, 2, 0, 0)]
    for tile_n, stages, ws, amc in twosm_configs:
        name = f"dual_2sm n={tile_n} s={stages}"
        if ws or amc:
            name += f" ws={ws} amc={amc}"
        try:
            fn = load_dual_2sm(tile_n, stages, ws, amc)
            rel = check(fn, a, b, ref)
            arms.append((name, fn, rel))
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<28} FAILED: {e}")

    if args.interleave > 0:
        times: dict[str, list[float]] = {name: [] for name, _, _ in arms}
        for rep in range(args.interleave):
            for name, fn, _ in arms:
                times[name].append(bench(fn, a, b, warmup=5, iters=20))
        print(f"Interleaved A/B: {args.interleave} reps x (warmup=5, iters=20) per arm, round-robin\n")
        for name, _, rel in arms:
            ts = times[name]
            med = statistics.median(ts)
            suffix = f"  reps[min..max]=[{min(ts)*1e3:.1f}..{max(ts)*1e3:.1f}]us"
            report(name, med, rel, M, N, K, suffix=suffix)
    else:
        for name, fn, rel in arms:
            report(name, bench(fn, a, b), rel, M, N, K)
    print()


if __name__ == "__main__":
    main()
