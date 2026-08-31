#!/usr/bin/env python3
"""Build and run all Ozaki scheme variants in one place."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.ozaki_scheme.accuracy_policy import configured_accuracy
from labs.ozaki_scheme.lab_utils import (
    build_lab,
    common_args_from_values,
    detect_binary_suffix,
    format_result_row,
    run_binary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="Skip `make all` before running")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input-scale", type=float, default=0.001)
    parser.add_argument("--dynamic-max-bits", type=int, default=16)
    parser.add_argument("--dynamic-offset", type=int, default=-56)
    parser.add_argument("--fixed-bits", type=int, default=12)
    parser.add_argument(
        "--emulation-strategy",
        choices=("default", "performant", "eager"),
        default="eager",
        help="Control the cuBLAS emulation-strategy pinning for the optimized variants",
    )
    args = parser.parse_args()

    if not args.skip_build:
        build_lab()

    suffix = detect_binary_suffix()

    common_args = common_args_from_values(
        m=args.m,
        n=args.n,
        k=args.k,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        input_scale=args.input_scale,
        emulation_strategy=args.emulation_strategy,
    )
    baseline = run_binary(f"baseline_ozaki_scheme{suffix}", common_args)
    dynamic = run_binary(
        f"optimized_ozaki_scheme_dynamic{suffix}",
        common_args + configured_accuracy("dynamic")[0]
        + ["--dynamic-max-bits", str(args.dynamic_max_bits), "--dynamic-offset", str(args.dynamic_offset)],
    )
    fixed = run_binary(
        f"optimized_ozaki_scheme_fixed{suffix}",
        common_args + configured_accuracy("fixed")[0] + ["--fixed-bits", str(args.fixed_bits)],
    )

    baseline_ms = float(baseline["time_ms"])
    print("| Variant | Time (ms) | TFLOPS | Speedup vs native | Retained bits | Emulation used | Max abs error | Mean abs error |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(format_result_row("Native FP64", baseline, baseline_ms))
    print(format_result_row("Ozaki dynamic", dynamic, baseline_ms))
    print(format_result_row("Ozaki fixed", fixed, baseline_ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
