#!/usr/bin/env python3
"""Kernel-only A/B bench: dual-CTA occupancy variant vs incumbent cluster vs cuBLAS.

Usage (on the GB300 pod, GPU 2 only):
    export CUDA_VISIBLE_DEVICES=2
    cd /work/ai-performance-engineering/code
    python labs/custom_vs_cublas/bench_dual_cta.py [--size 8192] [--sweep]

Reports CUDA-event kernel time, TFLOPS, and % of GB300 FP16 dense SoL
(3.75 PFLOPS), plus max relative error vs torch.matmul.

--sweep additionally tries (tile_n, stages) in {(128,3), (128,2), (256,2)}
(each config is a separate JIT build; first run compiles).
"""

from __future__ import annotations

import argparse
import os
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


def report(name, ms, rel, M, N, K):
    tflops = (2 * M * N * K / 1e12) / (ms / 1e3)
    sol = 100.0 * tflops / GB300_FP16_PEAK_TFLOPS
    flag = "OK " if rel < 0.01 else "BAD"
    print(f"  {name:<28} {ms*1e3:>9.1f} us  {tflops:>8.1f} TFLOPS  {sol:>5.1f}% SoL  rel_err={rel:.2e} [{flag}]")


def load_dual(tile_n: int, stages: int):
    from labs.custom_vs_cublas.tcgen05_loader import _load_tcgen05_dual_cta_module
    mod = _load_tcgen05_dual_cta_module(tile_n, stages)
    return mod.matmul_tcgen05_dual_cta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--sweep", action="store_true", help="sweep (tile_n, stages) configs")
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
    report("cuBLAS (target)", bench(cublas, a, b), 0.0, M, N, K)

    from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_cluster
    rel = check(matmul_tcgen05_cluster, a, b, ref)
    report("cluster (incumbent)", bench(matmul_tcgen05_cluster, a, b), rel, M, N, K)

    # Default = measured-best GB300 config (2026-06-10 sweep); --sweep tries all.
    configs = [(256, 2)] if not args.sweep else [(128, 3), (128, 2), (256, 2)]
    for tile_n, stages in configs:
        try:
            fn = load_dual(tile_n, stages)
            rel = check(fn, a, b, ref)
            report(f"dual_cta n={tile_n} s={stages}", bench(fn, a, b), rel, M, N, K)
        except Exception as e:  # noqa: BLE001
            print(f"  dual_cta n={tile_n} s={stages:<14} FAILED: {e}")
    print()


if __name__ == "__main__":
    main()
