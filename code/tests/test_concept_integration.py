"""Exercise concept ownership, configured targets and shared existing helpers."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.discovery import discover_benchmarks
from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
from labs.nvfp4_quantization.benchmarks import NVFP4Benchmark

CODE = Path(__file__).resolve().parents[1]


def test_concept_targets_discover_in_their_owners():
    expected = {
        "ch09": {"online_softmax"},
        "ch16": {"fair_scheduler"},
        "labs/decode_optimization": {"grammar_mask"},
        "labs/prefix_scan": {"prefix_scan", "prefix_scan_lookback"},
        "labs/nvfp4_quantization": {"add_rmsnorm", "quantize", "silu_mul"},
        "labs/kv_cache_compression": {"kivi"},
    }
    for owner, targets in expected.items():
        pairs = discover_benchmarks(CODE / owner, warn_missing=False)
        found = {name: opts for _, opts, name in pairs}
        assert targets <= found.keys()
        assert all(found[name] for name in targets)
        if owner in {"labs/prefix_scan", "labs/nvfp4_quantization"}:
            assert targets == found.keys()


def test_real_harness_workload_overrides_and_invalid_family_fail_fast():
    target = "labs/nvfp4_quantization:add_rmsnorm"
    config = BenchmarkConfig(
        target_label=target, target_extra_args={target: ["--workload", "rmsnorm_8192"]}
    )
    harness = BenchmarkHarness(config=config)
    for optimized in (False, True):
        benchmark = NVFP4Benchmark("add_rmsnorm", optimized)
        harness._apply_target_overrides(benchmark, config)
        assert benchmark.workload.cols == 8192
        invalid = BenchmarkConfig(
            target_label=target, target_extra_args={target: ["--workload", "silu_mul_7168"]}
        )
        with pytest.raises(SystemExit):
            harness._apply_target_overrides(benchmark, invalid)
        assert benchmark.workload.cols == 8192


def test_existing_gemm_reference_uses_shared_blocking_with_identical_addresses():
    # Import the real existing submission with its own task/utils modules in an
    # isolated interpreter, preserving the parent test runner's module namespace.
    script = """
import torch
import reference_submission
from core.utils.nvfp4_layout import to_blocked
assert reference_submission.to_blocked is to_blocked
for rows, cols in [(128, 4), (256, 12)]:
    source = torch.arange(rows * cols).reshape(rows, cols)
    blocked = reference_submission.to_blocked(source)
    for row in range(rows):
        for col in range(cols):
            address = (row // 128) * cols * 128 + (col // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
            assert blocked[address] == source[row, col]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=CODE / "labs/nvfp4_gemm",
        env={**os.environ, "PYTHONPATH": str(CODE)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
