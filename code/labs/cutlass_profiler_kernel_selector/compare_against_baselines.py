"""Compare cutlass_profiler baselines against Triton/DeepEP/custom kernels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .shapes import transformer_gemm_shapes


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "cutlass_profiler"
DEFAULT_BASELINE = ARTIFACT_DIR / "cutlass_profiler_results.json"
DEFAULT_TRITON = ARTIFACT_DIR / "triton_matmul_results.json"


def load_results(path: Path, fallback_provider: str | None = None) -> Tuple[str, Dict[str, dict]]:
    with path.open("r") as f:
        payload = json.load(f)
    provider = payload.get("provider", fallback_provider or path.stem)
    results = {item["name"]: item for item in payload.get("results", [])}
    return provider, results


def _compare_loaded(
    baseline_provider: str,
    baseline: Dict[str, dict],
    providers: List[Tuple[str, Dict[str, dict]]],
) -> Dict[str, dict]:
    shapes_of_interest = {s.name for s in transformer_gemm_shapes()}
    comparison: Dict[str, dict] = {}

    for name, base_row in baseline.items():
        if shapes_of_interest and name not in shapes_of_interest:
            continue
        base_tflops = base_row.get("tflops")
        base_runtime = base_row.get("runtime_ms")
        entry = {
            "baseline_provider": baseline_provider,
            "baseline_tflops": base_tflops,
            "baseline_runtime_ms": base_runtime,
            "baseline_kernel": base_row.get("kernel", ""),
            "competitors": [],
        }
        for provider, res_map in providers:
            comp = res_map.get(name)
            if not comp:
                continue
            speedup = None
            if comp.get("tflops") and base_tflops:
                speedup = comp["tflops"] / base_tflops if base_tflops else None
            entry["competitors"].append(
                {
                    "provider": provider,
                    "tflops": comp.get("tflops"),
                    "runtime_ms": comp.get("runtime_ms"),
                    "kernel": comp.get("kernel", ""),
                    "speedup_vs_cutlass": speedup,
                }
            )
        comparison[name] = entry

    return comparison


def compare(
    baseline_path: Path,
    competitor_paths: List[Path],
) -> Dict[str, dict]:
    baseline_provider, baseline = load_results(baseline_path, "cutlass_profiler")
    providers = [load_results(path) for path in competitor_paths]
    return _compare_loaded(baseline_provider, baseline, providers)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare cutlass_profiler baselines with other kernels.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to cutlass_profiler_results.json.",
    )
    parser.add_argument(
        "--providers",
        type=Path,
        nargs="*",
        default=None,
        help="JSON result files to compare (e.g., triton_matmul_results.json, deepep_results.json).",
    )
    parser.add_argument(
        "--include-default-triton",
        action="store_true",
        help="Automatically include artifacts/cutlass_profiler/triton_matmul_results.json if present.",
    )
    args = parser.parse_args(argv)

    provider_paths: List[Path] = []
    if args.providers:
        provider_paths.extend(args.providers)
    if args.include_default_triton and DEFAULT_TRITON.is_file():
        provider_paths.append(DEFAULT_TRITON)

    if not args.baseline.is_file():
        print(f"Baseline not found: {args.baseline}. Run run_cutlass_profiler_sweep.py first.", file=sys.stderr)
        return 1
    if not provider_paths:
        print("No competitor result files provided. Use --providers or --include-default-triton.", file=sys.stderr)
        return 1

    resolved_paths = []
    for path in provider_paths:
        if not path.is_file():
            print(f"Provider results missing: {path}", file=sys.stderr)
            continue
        resolved_paths.append(path)

    if not resolved_paths:
        print("No valid competitor results to compare.", file=sys.stderr)
        return 1

    baseline_provider, baseline = load_results(args.baseline, "cutlass_profiler")
    loaded_providers = [load_results(path) for path in resolved_paths]

    comparison = _compare_loaded(baseline_provider, baseline, loaded_providers)
    output_path = ARTIFACT_DIR / "comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(comparison, f, indent=2)

    print("\nSpeedup vs CUTLASS baseline (TFLOP/s higher is better):")
    for shape, row in comparison.items():
        base = row["baseline_tflops"]
        base_val = base if base is not None else float("nan")
        base_kernel = row["baseline_kernel"]
        line = f"{shape:32s} | base={base_val:.2f} TF/s ({base_kernel})"
        for provider, res_map in loaded_providers:
            comp = res_map.get(shape)
            if comp:
                speedup = comp["tflops"] / base if base else float("nan")
                line += f" | {provider}: {comp['tflops']:.2f} TF/s ({speedup:.2f}x)"
            else:
                line += f" | {provider}: n/a"
        print(line)

    print(f"\nComparison written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
