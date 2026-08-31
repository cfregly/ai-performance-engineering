#!/usr/bin/env python3
"""Run the Ozaki lab's narrative checks from the README."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.ozaki_scheme.accuracy_policy import configured_accuracy
from labs.ozaki_scheme.lab_utils import (
    build_lab,
    common_args_from_values,
    detect_binary_suffix,
    parse_float_csv,
    parse_int_csv,
    run_binary,
    speedup_vs_baseline,
    summarize_reproducibility,
)


def _print_accuracy_table(
    suffix: str,
    common_args: list[str],
    fixed_bits_list: Sequence[int],
) -> None:
    baseline = run_binary(f"baseline_ozaki_scheme{suffix}", common_args)
    baseline_ms = float(baseline["time_ms"])
    print("## Controllable Accuracy")
    print("| Fixed bits | Time (ms) | Speedup vs native | Max abs error | Mean abs error | Emulation used |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for fixed_bits in fixed_bits_list:
        metrics = run_binary(
            f"optimized_ozaki_scheme_fixed{suffix}",
            common_args + configured_accuracy("fixed")[0] + ["--fixed-bits", str(fixed_bits)],
        )
        time_ms = float(metrics["time_ms"])
        print(
            f"| {fixed_bits} | {time_ms:.3f} | {speedup_vs_baseline(baseline_ms, time_ms):.2f}x | "
            f"{float(metrics['max_abs_error']):.3e} | {float(metrics['mean_abs_error']):.3e} | "
            f"{int(metrics['emulation_used'])} |"
        )
    print()


def _print_adaptive_table(
    suffix: str,
    args: argparse.Namespace,
    input_scales: Sequence[float],
) -> None:
    print("## Adaptive Behavior")
    print("| Input scale | Dynamic retained bits | Time (ms) | Speedup vs native | Max abs error |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for input_scale in input_scales:
        common_args = common_args_from_values(
            m=args.m,
            n=args.n,
            k=args.k,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            input_scale=input_scale,
            emulation_strategy=args.emulation_strategy,
        )
        baseline = run_binary(f"baseline_ozaki_scheme{suffix}", common_args)
        dynamic = run_binary(
            f"optimized_ozaki_scheme_dynamic{suffix}",
            common_args + configured_accuracy("dynamic")[0]
            + [
                "--dynamic-max-bits",
                str(args.dynamic_max_bits),
                "--dynamic-offset",
                str(args.dynamic_offset),
            ],
        )
        baseline_ms = float(baseline["time_ms"])
        dynamic_ms = float(dynamic["time_ms"])
        print(
            f"| {input_scale:.1e} | {int(dynamic['retained_bits'])} | {dynamic_ms:.3f} | "
            f"{speedup_vs_baseline(baseline_ms, dynamic_ms):.2f}x | {float(dynamic['max_abs_error']):.3e} |"
        )
    print()


def _binary_name_for_variant(variant: str, suffix: str) -> str:
    if variant == "baseline":
        return f"baseline_ozaki_scheme{suffix}"
    if variant == "dynamic":
        return f"optimized_ozaki_scheme_dynamic{suffix}"
    return f"optimized_ozaki_scheme_fixed{suffix}"


def _variant_args(args: argparse.Namespace, variant: str) -> list[str]:
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
    if variant == "dynamic":
        return common_args + configured_accuracy("dynamic")[0] + [
            "--dynamic-max-bits",
            str(args.dynamic_max_bits),
            "--dynamic-offset",
            str(args.dynamic_offset),
        ]
    if variant == "fixed":
        return common_args + configured_accuracy("fixed")[0] + ["--fixed-bits", str(args.fixed_bits)]
    return common_args


def _print_repro_table(suffix: str, args: argparse.Namespace) -> None:
    binary = _binary_name_for_variant(args.variant, suffix)
    variant_args = _variant_args(args, args.variant)
    records = [run_binary(binary, variant_args) for _ in range(args.repeats)]
    summary = summarize_reproducibility(records)

    print("## Reproducibility")
    print("| Run | Checksum | Retained bits | Emulation used | Time (ms) |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for index, record in enumerate(records, start=1):
        checksum = float(record.get("checksum", 0.0))
        retained_bits = record.get("retained_bits", "-")
        emulation_used = record.get("emulation_used", 0)
        time_ms = float(record["time_ms"])
        print(f"| {index} | {checksum:.10e} | {retained_bits} | {emulation_used} | {time_ms:.3f} |")
    print()
    print(
        "Stable checksums: {checksum_stable} | Stable retained bits: {retained_bits_stable} | "
        "Stable emulation-used flag: {emulation_used_stable}".format(**summary)
    )
    print()

    if not (
        bool(summary["checksum_stable"])
        and bool(summary["retained_bits_stable"])
        and bool(summary["emulation_used_stable"])
    ):
        raise SystemExit("Reproducibility check failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=("all", "accuracy", "adaptive", "reproducibility"),
        default="all",
        help="Which narrative section to run",
    )
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
        "--fixed-bits-list",
        default="6,8,10,12,14",
        help="Comma-separated fixed-bit sweep for the controllable-accuracy section",
    )
    parser.add_argument(
        "--input-scales",
        default="1e-1,1e-2,1e-3,1e-4",
        help="Comma-separated input scales for the adaptive-behavior section",
    )
    parser.add_argument(
        "--variant",
        choices=("baseline", "dynamic", "fixed"),
        default="dynamic",
        help="Variant used by the reproducibility section",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--emulation-strategy",
        choices=("default", "performant", "eager"),
        default="eager",
        help="Control the cuBLAS emulation-strategy pinning for optimized runs",
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

    if args.section in {"all", "accuracy"}:
        _print_accuracy_table(suffix, common_args, parse_int_csv(args.fixed_bits_list))
    if args.section in {"all", "adaptive"}:
        _print_adaptive_table(suffix, args, parse_float_csv(args.input_scales))
    if args.section in {"all", "reproducibility"}:
        _print_repro_table(suffix, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
