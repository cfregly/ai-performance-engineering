"""Seed transport controls for the cache-aware multi-GPU worker."""

from __future__ import annotations

import ast
import importlib
import shutil
from pathlib import Path

import torch

from core.harness.benchmark_harness import BenchmarkConfig
from core.utils.worker_seed import apply_worker_seed
from labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_common import (
    CacheAwareDisaggMultiGPUConfig,
    _build_reference_state,
    _parse_args,
)
from tests.protection_test_utils import preserve_rng_state

CODE_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = (
    CODE_ROOT
    / "labs"
    / "cache_aware_disagg_inference"
    / "cache_aware_disagg_multigpu_common.py"
)


def _call_name(call: ast.Call) -> str:
    return ast.unparse(call.func)


def test_cache_aware_cli_preserves_default_and_accepts_fresh_seed() -> None:
    default_args = _parse_args(["--variant", "baseline"], require_variant=True)
    fresh_args = _parse_args(
        ["--variant", "optimized", "--seed", "1042"],
        require_variant=True,
    )

    assert default_args.seed == 42
    assert fresh_args.seed == 1042


def test_cache_aware_cpu_reference_state_uses_transported_seed() -> None:
    cfg = CacheAwareDisaggMultiGPUConfig(hidden_size=4, num_layers=1)
    with preserve_rng_state():
        apply_worker_seed(42)
        legacy_state = _build_reference_state(cfg)
        apply_worker_seed(42)
        repeated_state = _build_reference_state(cfg)
        apply_worker_seed(1042)
        fresh_state = _build_reference_state(cfg)

    assert legacy_state.keys() == repeated_state.keys() == fresh_state.keys()
    assert all(
        torch.equal(legacy_state[name], repeated_state[name])
        for name in legacy_state
    )
    assert any(
        not torch.equal(legacy_state[name], fresh_state[name])
        for name in legacy_state
    )


def test_cache_aware_launch_spec_maps_harness_seed_to_worker_cli() -> None:
    cfg = CacheAwareDisaggMultiGPUConfig(
        hidden_size=2,
        num_layers=1,
        batch_size=1,
        requests_per_rank=2,
        context_window=4,
        chunk_size=2,
        decode_tokens=2,
        warm_request_ratio=0.5,
        warm_prefix_ratio=0.5,
        prefill_ranks=1,
    )
    wrappers = (
        ("labs.cache_aware_disagg_inference.baseline_cache_aware_disagg_multigpu", "baseline"),
        ("labs.cache_aware_disagg_inference.optimized_cache_aware_disagg_multigpu", "optimized"),
    )
    for module_name, expected_variant in wrappers:
        benchmark = importlib.import_module(module_name).get_benchmark()
        benchmark.cfg = cfg
        try:
            spec = benchmark.get_torchrun_spec(
                BenchmarkConfig(
                    nproc_per_node=2,
                    iterations=1,
                    warmup=5,
                    seed=1042,
                    multi_gpu_required=True,
                )
            )

            assert spec.config_arg_map["seed"] == "--seed"
            assert spec.script_args[5:7] == ["--variant", expected_variant]
        finally:
            context = benchmark._cache_aware_result_context
            if context is not None:
                shutil.rmtree(context["result_dir"], ignore_errors=True)


def test_cache_aware_worker_and_thin_entrypoint_bind_seed_end_to_end() -> None:
    tree = ast.parse(COMMON_PATH.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_torchrun_worker"
    )
    worker_args = {argument.arg for argument in worker.args.args + worker.args.kwonlyargs}
    assert "seed" in worker_args
    helper_calls = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call) and _call_name(node) == "apply_worker_seed"
    ]
    assert len(helper_calls) == 1
    assert ast.unparse(helper_calls[0].args[0]) == "seed"

    benchmark_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CacheAwareDisaggMultiGPUBenchmark"
    )
    setup = next(
        node
        for node in benchmark_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup"
    )
    literal_seed_calls = {
        "torch.manual_seed",
        "torch.cuda.manual_seed",
        "torch.cuda.manual_seed_all",
    }
    assert not any(
        isinstance(node, ast.Call)
        and _call_name(node) in literal_seed_calls
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 42
        for node in ast.walk(setup)
    )

    run_cli = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_cli"
    )
    worker_call = next(
        node
        for node in ast.walk(run_cli)
        if isinstance(node, ast.Call) and _call_name(node) == "_run_torchrun_worker"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in worker_call.keywords}
    assert keywords["seed"] == "args.seed"

    worker_entrypoint = (
        CODE_ROOT
        / "labs"
        / "cache_aware_disagg_inference"
        / "cache_aware_disagg_multigpu_worker.py"
    )
    entry_tree = ast.parse(worker_entrypoint.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.Call) and _call_name(node) == "run_cli"
        for node in ast.walk(entry_tree)
    )
