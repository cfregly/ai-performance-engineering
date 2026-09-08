"""Collect full-cache error metrics on CUDA; does not accept a run or report speedup."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import traceback
from pathlib import Path

import torch

from labs.kv_cache_compression.accuracy import REFERENCE_ID, WORKLOAD
from labs.kv_cache_compression.baseline_kv_cache import BaselineKVCacheBenchmark
from labs.kv_cache_compression.optimized_kv_cache_nvfp4 import OptimizedKVCacheNVFP4Benchmark


COHORTS = ("nominal", "holdout", "alternating", "sparse_outlier")


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _apply_input_cohort(benchmark, cohort: str) -> None:
    """Apply a deterministic distribution change without changing workload shape."""
    if cohort in ("nominal", "holdout"):
        return
    tensors = benchmark.prefill_inputs + benchmark.decode_inputs
    with torch.no_grad():
        if cohort == "alternating":
            feature = torch.ones(benchmark.hidden_dim, dtype=benchmark.tensor_dtype, device=benchmark.device)
            feature[1::2] = -1
            for tensor in tensors:
                tensor.copy_(feature)
                tensor[:, 1::2].neg_()
        elif cohort == "sparse_outlier":
            for tensor in tensors:
                tensor.zero_()
                tensor[..., 0] = 1
                tensor[..., 1] = -1
        else:  # pragma: no cover - argparse owns the public choices.
            raise ValueError(f"Unknown KV accuracy cohort: {cohort}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("fp8", "nvfp4"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cohort", choices=COHORTS, default="nominal")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = {
        "schema_version": 2,
        "status": "failure_not_accepted",
        "variant": args.variant,
        "cohort": args.cohort,
        "seed": args.seed,
        "reference_id": REFERENCE_ID,
        "git_commit": _source_commit(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "workload": {key: value for key, value in WORKLOAD.items() if key != "storage_dtype"},
        "thresholds": None,
    }
    benchmark = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Accuracy calibration requires actual CUDA/Transformer Engine hardware")
        torch.manual_seed(args.seed)
        benchmark = (BaselineKVCacheBenchmark() if args.variant == "fp8"
                     else OptimizedKVCacheNVFP4Benchmark())
        recipe = benchmark.fp8_recipe if args.variant == "fp8" else benchmark.nvfp4_recipe
        benchmark._setup_with_recipe(recipe, require_accuracy_policy=False)
        _apply_input_cohort(benchmark, args.cohort)
        if args.cohort not in ("nominal", "holdout"):
            if recipe.delayed():
                benchmark._calibrate_fp8(recipe)
            benchmark._warmup_runtime(recipe)
        benchmark.benchmark_fn()
        metrics = benchmark.measure_accuracy()
        receipt.update({
            "status": "measurement_only_not_accepted",
            "transformer_engine": importlib.metadata.version("transformer_engine"),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "metrics": metrics,
        })
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    except Exception as exc:
        receipt.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    finally:
        if benchmark is not None:
            benchmark.teardown()


if __name__ == "__main__":
    main()
