"""Run the fused NVFP4 add/RMSNorm path with its exact verifier."""

from __future__ import annotations

import argparse

import torch

from labs.nvfp4_quantization.benchmarks import WORKLOADS, NVFP4Benchmark

WORKLOAD_NAME = "rmsnorm_2048"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fused NVFP4 add/RMSNorm correctness demo."
    )
    parser.add_argument("--workload", choices=[WORKLOAD_NAME], default=WORKLOAD_NAME)
    return parser


def build_benchmark(*, workload_name: str) -> NVFP4Benchmark:
    """Build the optimized lab path for the bounded RMSNorm workload."""
    return NVFP4Benchmark(
        "add_rmsnorm",
        optimized=True,
        workload=WORKLOADS[workload_name],
    )


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for labs NVFP4 add/RMSNorm demo")


def main() -> int:
    args = build_parser().parse_args()
    _require_cuda()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    benchmark = build_benchmark(workload_name=args.workload)
    try:
        benchmark.setup()
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        if benchmark.output is None:
            raise RuntimeError("NVFP4 add/RMSNorm did not produce an output")
        finite_scales = bool(
            torch.isfinite(benchmark.output.scales.view(torch.float8_e4m3fn).float()).all().item()
        )
        if not finite_scales:
            raise AssertionError("NVFP4 add/RMSNorm produced nonfinite scales")
        print(f"nvfp4_exact_verifier=pass workload={args.workload} finite_scales=true")
    finally:
        benchmark.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
