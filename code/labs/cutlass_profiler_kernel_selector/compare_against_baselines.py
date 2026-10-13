"""Compare cutlass_profiler baselines against Triton/DeepEP/custom kernels.

The baseline and every provider file use the same JSON schema::

    {
      "provider": "triton",
      "results": [
        {"name": "kv_proj_m4096_n4096_k4096", "tflops": 612.0, "runtime_ms": 0.22, "kernel": "..."}
      ],
      "failed": ["attention_qk_m8192_n4096_k4096"]
    }

"provider" defaults to the file name stem and "failed" is optional. Records may
carry extra metadata keys. Every compared shape needs exactly one "results"
record with a positive finite "tflops" value in every file, and must not be
listed in "failed". The compared shapes are all shapes in shapes.py, or the
``--shapes`` subset. Any gap makes the comparison fail.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.cutlass_profiler_kernel_selector.shapes import (  # noqa: E402
    transformer_gemm_shapes,
)


ARTIFACT_DIR = REPO_ROOT / "artifacts" / "cutlass_profiler"
DEFAULT_BASELINE = ARTIFACT_DIR / "cutlass_profiler_results.json"
DEFAULT_TRITON = ARTIFACT_DIR / "triton_matmul_results.json"

# (provider name, records keyed by shape name, shape names the producer reported as failed)
LoadedResults = Tuple[str, Dict[str, dict], List[str]]


def load_results(path: Path, fallback_provider: str | None = None) -> LoadedResults:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'results' list")
    provider = payload.get("provider", fallback_provider or path.stem)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: 'results' must be a list of shape records")
    results: Dict[str, dict] = {}
    for item in rows:
        name = item["name"]
        if name in results:
            raise ValueError(f"{path}: more than one record for shape {name!r}")
        results[name] = item
    failed = payload.get("failed", [])
    if not isinstance(failed, list) or not all(isinstance(name, str) for name in failed):
        raise ValueError(f"{path}: 'failed' must be a list of shape names")
    return provider, results, failed


def expected_shape_names(requested: Sequence[str] | None = None) -> List[str]:
    """Return the shapes every result file must cover, in shapes.py order."""

    known = [shape.name for shape in transformer_gemm_shapes()]
    if requested is None:
        return known
    if not requested:
        raise ValueError("Select at least one shape to compare.")
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"Unknown shape names {unknown}. Known shapes: {known}")
    wanted = set(requested)
    return [name for name in known if name in wanted]


def _missing_record_errors(loaded: LoadedResults, shape_names: Sequence[str]) -> List[str]:
    provider, results, failed = loaded
    errors: List[str] = []
    failed_names = [name for name in shape_names if name in failed]
    if failed_names:
        errors.append(f"{provider} lists these shapes as failed: {', '.join(failed_names)}")
    absent = [name for name in shape_names if name not in results and name not in failed]
    if absent:
        errors.append(f"{provider} has no record for these shapes: {', '.join(absent)}")
    for name in shape_names:
        row = results.get(name)
        if row is None:
            continue
        tflops = row.get("tflops")
        if (
            isinstance(tflops, bool)
            or not isinstance(tflops, (int, float))
            or not math.isfinite(tflops)
            or tflops <= 0
        ):
            errors.append(
                f"{provider} record for {name} has tflops={tflops!r}, not a positive finite number"
            )
    return errors


def record_errors(
    baseline: LoadedResults,
    providers: Sequence[LoadedResults],
    shape_names: Sequence[str],
) -> List[str]:
    """List every missing, failed or unmeasured record across the baseline and providers."""

    errors: List[str] = []
    for loaded in (baseline, *providers):
        errors.extend(_missing_record_errors(loaded, shape_names))
    return errors


def _compare_loaded(
    baseline: LoadedResults,
    providers: Sequence[LoadedResults],
    shape_names: Sequence[str],
) -> Dict[str, dict]:
    baseline_provider, baseline_rows, _ = baseline
    comparison: Dict[str, dict] = {}

    for name in shape_names:
        base_row = baseline_rows[name]
        base_tflops = base_row["tflops"]
        entry = {
            "baseline_provider": baseline_provider,
            "baseline_tflops": base_tflops,
            "baseline_runtime_ms": base_row.get("runtime_ms"),
            "baseline_kernel": base_row.get("kernel", ""),
            "competitors": [],
        }
        for provider, res_map, _ in providers:
            comp = res_map[name]
            entry["competitors"].append(
                {
                    "provider": provider,
                    "tflops": comp["tflops"],
                    "runtime_ms": comp.get("runtime_ms"),
                    "kernel": comp.get("kernel", ""),
                    "speedup_vs_cutlass": comp["tflops"] / base_tflops,
                }
            )
        comparison[name] = entry

    return comparison


def compare(
    baseline_path: Path,
    competitor_paths: List[Path],
    shape_names: Sequence[str] | None = None,
) -> Dict[str, dict]:
    """Compare competitors with the baseline. Raises ValueError if any file is missing a record."""

    if not competitor_paths:
        raise ValueError("No competitor result files provided.")
    expected = expected_shape_names(shape_names)
    baseline = load_results(baseline_path, "cutlass_profiler")
    providers = [load_results(path) for path in competitor_paths]
    errors = record_errors(baseline, providers, expected)
    if errors:
        raise ValueError("Result files are missing records:\n" + "\n".join(errors))
    return _compare_loaded(baseline, providers, expected)


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
        help="Also compare artifacts/cutlass_profiler/triton_matmul_results.json. Fails if that file is missing.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        nargs="+",
        default=None,
        help="Subset of shape names to compare (defaults to every shape in shapes.py).",
    )
    args = parser.parse_args(argv)

    provider_paths: List[Path] = list(args.providers or [])
    if args.include_default_triton:
        provider_paths.append(DEFAULT_TRITON)

    try:
        shape_names = expected_shape_names(args.shapes)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not args.baseline.is_file():
        print(f"Baseline not found: {args.baseline}. Run run_cutlass_profiler_sweep.py first.", file=sys.stderr)
        return 1
    if not provider_paths:
        print("No competitor result files provided. Use --providers or --include-default-triton.", file=sys.stderr)
        return 1

    missing_paths = [path for path in provider_paths if not path.is_file()]
    for path in missing_paths:
        print(f"Provider results missing: {path}", file=sys.stderr)
    if missing_paths:
        return 1

    baseline = load_results(args.baseline, "cutlass_profiler")
    loaded_providers = [load_results(path) for path in provider_paths]

    errors = record_errors(baseline, loaded_providers, shape_names)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        if args.shapes is None:
            print("Pass --shapes to compare only the shapes that every file covers.", file=sys.stderr)
        return 1

    comparison = _compare_loaded(baseline, loaded_providers, shape_names)
    output_path = ARTIFACT_DIR / "comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(comparison, f, indent=2)

    print("\nSpeedup vs CUTLASS baseline (TFLOP/s higher is better):")
    for shape, row in comparison.items():
        line = f"{shape:32s} | base={row['baseline_tflops']:.2f} TF/s ({row['baseline_kernel']})"
        for comp in row["competitors"]:
            line += f" | {comp['provider']}: {comp['tflops']:.2f} TF/s ({comp['speedup_vs_cutlass']:.2f}x)"
        print(line)

    print(f"\nComparison written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
