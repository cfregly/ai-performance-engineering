"""Seed-ownership regressions for informational benchmark pairs."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from ch13.baseline_kv_cache_naive import BaselineKVCacheNaiveBenchmark
from ch13.kv_cache_workload import KVCacheWorkload
from ch13.optimized_kv_cache_naive_flash_blockwise import (
    OptimizedKVCacheNaiveFlashBlockwiseBenchmark,
)
from ch17.baseline_inference_full import BaselineInferenceFullBenchmark
from ch17.optimized_inference_full import OptimizedInferenceFullBenchmark
from core.benchmark.verify_runner import VerifyConfig
from tests.protection_test_utils import make_runner

CODE_ROOT = Path(__file__).resolve().parents[1]
SEED_OWNED_PAIRS = (
    "ch13/baseline_kv_cache_naive.py",
    "ch13/optimized_kv_cache_naive_flash_blockwise.py",
    "ch16/baseline_piece_graphs.py",
    "ch16/optimized_piece_graphs.py",
    "ch17/baseline_inference_full.py",
    "ch17/optimized_inference_full.py",
)


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


@pytest.mark.parametrize("relative_path", SEED_OWNED_PAIRS)
def test_setup_does_not_replace_the_harness_seed(relative_path: str) -> None:
    source = (CODE_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    setup_calls = {
        _call_name(call.func)
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
        and function.name == "setup"
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
    }

    assert "torch.manual_seed" not in setup_calls
    assert "torch.cuda.manual_seed_all" not in setup_calls
    assert "random.seed" not in setup_calls


def _tiny_kv_snapshot(benchmark_type: type, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    benchmark = benchmark_type()
    benchmark.device = torch.device("cpu")
    workload = KVCacheWorkload(
        batch_size=1,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        sequence_lengths=(2, 4),
        dtype=torch.float32,
        page_size=2,
        block_size=2,
    )
    benchmark.workload = workload
    benchmark.num_layers = workload.num_layers
    benchmark.num_heads = workload.num_heads
    benchmark.head_dim = workload.head_dim
    benchmark.hidden_dim = workload.hidden_dim
    benchmark.batch_size = workload.batch_size
    benchmark.sequence_lengths = list(workload.lengths())
    benchmark.page_size = workload.page_size
    benchmark.block_size = workload.block_size
    if hasattr(benchmark, "max_seq_len"):
        benchmark.max_seq_len = workload.max_seq_len

    torch.manual_seed(seed)
    try:
        benchmark.setup()
        assert torch.initial_seed() == seed
        assert benchmark.inputs is not None
        inputs = torch.cat([value.detach().flatten().clone() for value in benchmark.inputs])
        modules = benchmark.model if hasattr(benchmark, "model") else benchmark.layers
        assert modules is not None
        parameter = next(modules.parameters()).detach().flatten().clone()
        return inputs, parameter
    finally:
        benchmark.teardown()


@pytest.mark.parametrize(
    "benchmark_type",
    (BaselineKVCacheNaiveBenchmark, OptimizedKVCacheNaiveFlashBlockwiseBenchmark),
)
def test_kv_setup_uses_the_active_torch_seed(benchmark_type: type) -> None:
    seed_state = torch.random.get_rng_state()
    try:
        inputs_42, parameter_42 = _tiny_kv_snapshot(benchmark_type, 42)
        inputs_1042, parameter_1042 = _tiny_kv_snapshot(benchmark_type, 1042)
    finally:
        torch.random.set_rng_state(seed_state)

    assert not torch.equal(inputs_42, inputs_1042)
    assert not torch.equal(parameter_42, parameter_1042)


def _tiny_inference_snapshot(
    benchmark_type: type,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    benchmark = benchmark_type()
    benchmark.device = torch.device("cpu")
    benchmark.batch_size = 2
    benchmark.hidden_dim = 16
    benchmark.num_layers = 4
    benchmark.identity_start_layer = 2
    if isinstance(benchmark, OptimizedInferenceFullBenchmark):
        benchmark.exit_layer = 2

    torch.manual_seed(seed)
    try:
        benchmark.setup()
        assert torch.initial_seed() == seed
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        assert benchmark.inputs is not None
        return benchmark.inputs.detach().clone(), benchmark.get_verify_output()
    finally:
        benchmark.teardown()


def _tiny_inference_benchmark(benchmark_type: type):
    benchmark = benchmark_type()
    benchmark.device = torch.device("cpu")
    benchmark.batch_size = 2
    benchmark.hidden_dim = 16
    benchmark.num_layers = 4
    benchmark.identity_start_layer = 2
    if isinstance(benchmark, OptimizedInferenceFullBenchmark):
        benchmark.exit_layer = 2
    return benchmark


def test_inference_full_pair_preserves_active_seed_and_real_outputs() -> None:
    seed_state = torch.random.get_rng_state()
    try:
        baseline_42 = _tiny_inference_snapshot(BaselineInferenceFullBenchmark, 42)
        optimized_42 = _tiny_inference_snapshot(OptimizedInferenceFullBenchmark, 42)
        baseline_1042 = _tiny_inference_snapshot(BaselineInferenceFullBenchmark, 1042)
        optimized_1042 = _tiny_inference_snapshot(OptimizedInferenceFullBenchmark, 1042)
    finally:
        torch.random.set_rng_state(seed_state)

    torch.testing.assert_close(baseline_42[0], optimized_42[0], rtol=0, atol=0)
    torch.testing.assert_close(baseline_42[1], optimized_42[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(baseline_1042[0], optimized_1042[0], rtol=0, atol=0)
    torch.testing.assert_close(baseline_1042[1], optimized_1042[1], rtol=1e-6, atol=1e-6)
    assert not torch.equal(baseline_42[0], baseline_1042[0])
    assert not torch.equal(baseline_42[1], baseline_1042[1])


def test_inference_full_pair_passes_real_fresh_and_jitter_checks(tmp_path: Path) -> None:
    baseline = _tiny_inference_benchmark(BaselineInferenceFullBenchmark)
    optimized = _tiny_inference_benchmark(OptimizedInferenceFullBenchmark)

    result = make_runner(tmp_path).verify_pair(baseline, optimized, VerifyConfig())

    assert result.passed
    assert result.reason is None
    assert not (result.details or {}).get("warnings")
