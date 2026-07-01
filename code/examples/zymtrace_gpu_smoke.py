#!/usr/bin/env python3
"""Long-enough CUDA workload for validating Zymtrace GPU capture.

Run through ``code/core/scripts/profiling/profile.sh --tool zymtrace`` so the
CUDA injection path is set consistently with the rest of the profiling harness.
"""

from __future__ import annotations

import argparse
import os
import time

import torch


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a timed CUDA GEMM loop for Zymtrace smoke tests.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Minimum measured runtime.")
    parser.add_argument("--size", type=int, default=4096, help="Square matrix dimension.")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16", help="Input dtype.")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup GEMMs before timing.")
    parser.add_argument("--device", default="cuda", help="Torch device to run on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Zymtrace GPU smoke workload")
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    if args.size <= 0:
        raise ValueError("--size must be positive")

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)

    a = torch.randn((args.size, args.size), device=device, dtype=dtype)
    b = torch.randn((args.size, args.size), device=device, dtype=dtype)
    out = torch.empty((args.size, args.size), device=device, dtype=dtype)

    for _ in range(args.warmup):
        torch.mm(a, b, out=out)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    deadline = start + args.seconds
    iterations = 0
    while time.perf_counter() < deadline:
        torch.mm(a, b, out=out)
        iterations += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    checksum = float(out[0, 0].float().item())
    injection_path = os.environ.get("CUDA_INJECTION64_PATH", "")
    print(
        "zymtrace_gpu_smoke "
        f"device={torch.cuda.get_device_name(device) if device.type == 'cuda' else device} "
        f"dtype={args.dtype} size={args.size} iterations={iterations} "
        f"elapsed_s={elapsed:.3f} checksum={checksum:.6f} "
        f"cuda_injection={'set' if injection_path else 'unset'}"
    )


if __name__ == "__main__":
    main()
