"""Collect full-cache error metrics on CUDA; does not accept a run or report speedup."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from labs.kv_cache_compression.baseline_kv_cache import BaselineKVCacheBenchmark
from labs.kv_cache_compression.optimized_kv_cache_nvfp4 import OptimizedKVCacheNVFP4Benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("fp8", "nvfp4"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Accuracy calibration requires actual CUDA/Transformer Engine hardware")
    torch.manual_seed(args.seed)
    benchmark = (BaselineKVCacheBenchmark() if args.variant == "fp8"
                 else OptimizedKVCacheNVFP4Benchmark())
    recipe = benchmark.fp8_recipe if args.variant == "fp8" else benchmark.nvfp4_recipe
    try:
        benchmark._setup_with_recipe(recipe, require_accuracy_policy=False)
        benchmark.benchmark_fn()
        metrics = benchmark.measure_accuracy()
        args.output.write_text(json.dumps({
            "schema_version": 1, "status": "measurement_only_not_accepted",
            "variant": args.variant, "seed": args.seed, "torch": torch.__version__,
            "transformer_engine": importlib.metadata.version("transformer_engine"),
            "gpu": torch.cuda.get_device_name(), "metrics": metrics,
            "workload": {key: getattr(benchmark, key) for key in (
                "batch_size", "hidden_dim", "num_heads", "prefill_seq", "decode_seq", "decode_steps")},
            "thresholds": None,
        }, indent=2) + "\n")
    finally:
        benchmark.teardown()


if __name__ == "__main__":
    main()
