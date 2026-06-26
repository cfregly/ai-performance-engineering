"""Quick CLI to benchmark the Blackwell matmul variants."""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from labs.blackwell_matmul import is_cluster_launch_supported

_VARIANT_MODULES = {
    "baseline": "labs.blackwell_matmul.baseline_blackwell_matmul",
    "tma": "labs.blackwell_matmul.optimized_blackwell_matmul_tma",
    "pipeline": "labs.blackwell_matmul.optimized_blackwell_matmul_pipeline",
    "cluster": "labs.blackwell_matmul.optimized_blackwell_matmul_cluster",
    "tcgen05": "labs.blackwell_matmul.optimized_blackwell_matmul_tcgen05",
    "tcgen05_cta2": "labs.blackwell_matmul.optimized_blackwell_matmul_tcgen05",
}

_VARIANT_KERNEL_HINTS = {
    "baseline": "baseline_kernel",
    "tma": "tma_prefetch_kernel",
    "pipeline": "pipeline_prefetch_kernel",
    "cluster": "cluster_kernel",
    "tcgen05": "optimized_matmul_tcgen05",
    "tcgen05_cta2": "optimized_matmul_tcgen05_cta2",
}


def _ensure_cluster_available(requested_variant: str) -> None:
    if requested_variant not in {"cluster", "tcgen05_cta2"}:
        return
    if is_cluster_launch_supported():
        return
    raise SystemExit(
        "[labs/blackwell_matmul] Cluster launch unsupported on this GPU."
    )


def _append_roofline_row(
    csv_path: Path,
    kernel_hint: str,
    arithmetic_intensity: float,
    achieved_tflops: float,
    variant: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists()
    fieldnames = ["kernel", "intensity", "achieved_tflops", "label"]
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "kernel": kernel_hint,
                "intensity": f"{arithmetic_intensity:.6f}",
                "achieved_tflops": f"{achieved_tflops:.4f}",
                "label": variant,
            }
        )
    print(f"[roofline] Appended {variant} metadata to {csv_path}")


def _emit_dual_roofline_meta(
    output_path: Optional[Path],
    benchmark: BaseBenchmark,
    mean_ms: float,
    variant: str,
) -> None:
    if output_path is None or mean_ms <= 0.0:
        return
    if not hasattr(benchmark, "get_problem_shape"):
        print("[roofline] Benchmark lacks get_problem_shape(); skipping metadata export.")
        return
    m, n, k = benchmark.get_problem_shape()  # type: ignore[attr-defined]
    dtype = getattr(benchmark, "tensor_dtype", torch.float16)
    dtype_bytes = torch.finfo(dtype).bits // 8
    flops = float(2 * m * n * k)
    bytes_moved = float((m * k) + (k * n) + (m * n)) * dtype_bytes
    if bytes_moved <= 0:
        print("[roofline] Unable to compute bytes moved; skipping metadata export.")
        return
    seconds = mean_ms / 1000.0
    if seconds <= 0:
        print("[roofline] Invalid runtime; skipping metadata export.")
        return
    arithmetic_intensity = flops / bytes_moved
    achieved_tflops = (flops / 1e12) / seconds
    kernel_hint = _VARIANT_KERNEL_HINTS.get(variant, variant)
    _append_roofline_row(output_path, kernel_hint, arithmetic_intensity, achieved_tflops, variant)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(_VARIANT_MODULES.keys()),
        default="baseline",
        help="Which Modular.org part to replay",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help="Square dimension for the GEMM (must be divisible by 64)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override BenchmarkConfig.iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override BenchmarkConfig.warmup",
    )
    parser.add_argument(
        "--roofline-meta",
        type=Path,
        default=None,
        help="Optional CSV path for kernel,intensity,achieved_tflops rows (dual roofline helper).",
    )
    args = parser.parse_args(argv)

    module_name = _VARIANT_MODULES[args.variant]
    module = importlib.import_module(module_name)
    if args.variant.startswith("tcgen05"):
        benchmark = module.get_benchmark(
            size=args.size, cta2=(args.variant == "tcgen05_cta2")
        )
    else:
        benchmark = module.get_benchmark(size=args.size)
    config = benchmark.get_config()
    if args.iterations is not None and config is not None:
        config.iterations = args.iterations
    if args.warmup is not None and config is not None:
        config.warmup = args.warmup
    if config is not None:
        config.setup_timeout_seconds = max(180, config.setup_timeout_seconds or 0)
        config.measurement_timeout_seconds = max(180, config.measurement_timeout_seconds)
        config.profiling_timeout_seconds = max(300, config.profiling_timeout_seconds or 0)
        config.ncu_timeout_seconds = max(300, config.ncu_timeout_seconds)
        config.nsys_timeout_seconds = max(300, config.nsys_timeout_seconds)

    _ensure_cluster_available(args.variant)

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(
        f"\nBlackwell matmul ({args.variant}, size={args.size}) : {mean_ms:.3f} ms"
    )
    _emit_dual_roofline_meta(args.roofline_meta, benchmark, mean_ms, args.variant)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
