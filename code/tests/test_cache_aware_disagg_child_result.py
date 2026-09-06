"""Controls for the cache-aware torchrun entrypoint and child-result contract."""

from __future__ import annotations

import ast
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch

import labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_common as cache_common
from core.harness.benchmark_harness import BenchmarkConfig
from labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_common import (
    CACHE_AWARE_BASELINE_NVTX_RANGE,
    CACHE_AWARE_OPTIMIZED_NVTX_RANGE,
    CacheAwareDisaggMultiGPUBenchmark,
    CacheAwareDisaggMultiGPUConfig,
)
from labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_result import (
    CACHE_AWARE_LAUNCH_WALL_NS_ENV,
    CACHE_AWARE_RESULT_CALLBACK,
    CACHE_AWARE_RESULT_DIR_ENV,
    write_cache_aware_child_result,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
WORKER_MODULE = (
    "labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_worker"
)


def _benchmark(*, optimized: bool = False) -> CacheAwareDisaggMultiGPUBenchmark:
    return CacheAwareDisaggMultiGPUBenchmark(
        optimized=optimized,
        label=(
            "optimized_cache_aware_disagg_multigpu"
            if optimized
            else "baseline_cache_aware_disagg_multigpu"
        ),
        cfg=CacheAwareDisaggMultiGPUConfig(
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
        ),
    )


def _prepare_spec(
    benchmark: CacheAwareDisaggMultiGPUBenchmark,
    *,
    iterations: int = 3,
):
    return benchmark.get_torchrun_spec(
        BenchmarkConfig(
            nproc_per_node=2,
            iterations=iterations,
            warmup=1,
            multi_gpu_required=True,
        )
    )


def _write_valid_receipts(
    benchmark: CacheAwareDisaggMultiGPUBenchmark,
    spec: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    corrupt_output: bool = False,
    omit_request: bool = False,
    token_override: str | None = None,
) -> tuple[int, torch.Tensor]:
    context = benchmark._cache_aware_result_context
    assert context is not None
    launch_wall_ns = time.time_ns() - 1_000_000
    for key, value in spec.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(CACHE_AWARE_LAUNCH_WALL_NS_ENV, str(launch_wall_ns))
    if token_override is not None:
        monkeypatch.setenv("AISP_CACHE_AWARE_DISAGG_RESULT_TOKEN", token_override)

    reference0 = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    reference1 = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    actual0 = reference0.clone()
    if corrupt_output:
        actual0.add_(2.0)
    actual_outputs = {0: actual0}
    if not omit_request:
        actual_outputs[1] = reference1.clone()
    prompt = torch.arange(8, dtype=torch.bfloat16).view(1, 4, 2)
    config = dict(context["config"])
    write_cache_aware_child_result(
        variant="baseline",
        label="baseline_cache_aware_disagg_multigpu",
        rank=0,
        world_size=2,
        iterations_completed=int(context["iterations"]),
        config=config,
        actual_outputs={},
        reference_outputs={0: reference0, 1: reference1},
        verification_prompt=prompt,
        custom_metrics={
            "cache_aware.time_per_iter_ms": 1.25,
            "cache_aware.cache_hit_rate": 0.5,
        },
    )
    write_cache_aware_child_result(
        variant="baseline",
        label="baseline_cache_aware_disagg_multigpu",
        rank=1,
        world_size=2,
        iterations_completed=int(context["iterations"]),
        config=config,
        actual_outputs=actual_outputs,
        reference_outputs={},
        verification_prompt=None,
        custom_metrics=None,
    )
    return launch_wall_ns, prompt


@pytest.mark.parametrize(
    ("optimized", "expected_variant", "expected_range"),
    (
        (False, "baseline", CACHE_AWARE_BASELINE_NVTX_RANGE),
        (True, "optimized", CACHE_AWARE_OPTIMIZED_NVTX_RANGE),
    ),
)
def test_cache_aware_spec_uses_explicit_worker_fresh_results_and_iteration_timing(
    optimized: bool,
    expected_variant: str,
    expected_range: str,
) -> None:
    benchmark = _benchmark(optimized=optimized)
    spec = _prepare_spec(benchmark, iterations=3)
    try:
        assert spec.script_path is None
        assert spec.module_name == "core.harness.benchmark_worker"
        assert spec.script_args[:5] == [
            "--module",
            WORKER_MODULE,
            "--callable",
            "main",
            "--",
        ]
        assert spec.script_args[5:7] == ["--variant", expected_variant]
        assert spec.result_callback == CACHE_AWARE_RESULT_CALLBACK
        assert spec.timing_source == "rank0_time_per_iter_ms"
        assert spec.timing_iterations_per_sample == 3
        assert CACHE_AWARE_RESULT_DIR_ENV in spec.env
        assert not hasattr(benchmark, "_subprocess_verify_output")
        config = benchmark.get_config()
        assert config.profiling.nsys_nvtx_include == (expected_range,)
        assert config.profiling.ncu_replay_mode == "app-range"
    finally:
        context = benchmark._cache_aware_result_context
        if context is not None:
            shutil.rmtree(context["result_dir"], ignore_errors=True)


@pytest.mark.parametrize(
    ("wrapper_module", "expected_variant"),
    (
        (
            "labs.cache_aware_disagg_inference.baseline_cache_aware_disagg_multigpu",
            "baseline",
        ),
        (
            "labs.cache_aware_disagg_inference.optimized_cache_aware_disagg_multigpu",
            "optimized",
        ),
    ),
)
def test_registered_cache_aware_wrappers_route_to_the_explicit_worker(
    wrapper_module: str,
    expected_variant: str,
) -> None:
    module = importlib.import_module(wrapper_module)
    benchmark = module.get_benchmark()
    benchmark.cfg = _benchmark().cfg
    spec = _prepare_spec(benchmark)
    try:
        assert spec.module_name == "core.harness.benchmark_worker"
        assert spec.script_args[5:7] == ["--variant", expected_variant]
        assert spec.result_callback == CACHE_AWARE_RESULT_CALLBACK
    finally:
        context = benchmark._cache_aware_result_context
        if context is not None:
            shutil.rmtree(context["result_dir"], ignore_errors=True)


def test_cache_aware_callback_assembles_and_validates_full_child_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _benchmark()
    spec = _prepare_spec(benchmark)
    launch_wall_ns, prompt = _write_valid_receipts(benchmark, spec, monkeypatch)
    finish_wall_ns = time.time_ns() + 1_000_000

    benchmark.consume_cache_aware_child_results(
        launch_wall_ns=launch_wall_ns,
        finish_wall_ns=finish_wall_ns,
        returncode=0,
    )

    expected = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(benchmark.get_verify_output(), expected)
    torch.testing.assert_close(benchmark.get_verify_inputs()["prompt"], prompt)
    assert benchmark.get_input_signature().shapes["output"] == (2, 1, 2)
    assert benchmark.get_custom_metrics() == {
        "cache_aware.time_per_iter_ms": 1.25,
        "cache_aware.cache_hit_rate": 0.5,
    }
    assert benchmark.validate_result() is None
    context = benchmark._cache_aware_result_context
    assert context is not None
    assert context["retention"] == "cleaned-after-success"
    assert not Path(context["result_dir"]).exists()


def test_cache_aware_full_reference_matches_chunked_decode_on_cpu() -> None:
    torch.manual_seed(42)
    model = cache_common.TinyPrefillDecode(
        hidden_size=4,
        num_layers=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).eval()
    prompt = torch.randn((1, 6, 4), dtype=torch.float32)

    chunk_caches = []
    chunk_seed = None
    for chunk in prompt.split(2, dim=1):
        chunk_cache, chunk_seed = model.prefill(chunk)
        chunk_caches.append(chunk_cache)
    assert chunk_seed is not None
    chunked_output = model.decode(
        chunk_seed,
        torch.cat(chunk_caches, dim=1),
        decode_tokens=3,
    )
    full_cache, full_seed = model.prefill(prompt)
    reference_output = model.decode(full_seed, full_cache, decode_tokens=3)

    torch.testing.assert_close(chunked_output, reference_output, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("receipt_change", "match"),
    (
        ({"corrupt_output": True}, "full timed output differs"),
        ({"omit_request": True}, "ownership is incomplete or incorrect"),
        ({"token_override": "stale-token"}, "token mismatch"),
    ),
)
def test_cache_aware_callback_rejects_invalid_child_results(
    monkeypatch: pytest.MonkeyPatch,
    receipt_change: dict[str, Any],
    match: str,
) -> None:
    benchmark = _benchmark()
    spec = _prepare_spec(benchmark)
    launch_wall_ns, _ = _write_valid_receipts(
        benchmark,
        spec,
        monkeypatch,
        **receipt_change,
    )
    try:
        with pytest.raises(RuntimeError, match=match):
            benchmark.consume_cache_aware_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=time.time_ns() + 1_000_000,
                returncode=0,
            )
        assert not hasattr(benchmark, "_subprocess_verify_output")
    finally:
        context = benchmark._cache_aware_result_context
        assert context is not None
        shutil.rmtree(context["result_dir"], ignore_errors=True)


def test_cache_aware_callback_requires_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _benchmark()
    spec = _prepare_spec(benchmark)
    context = benchmark._cache_aware_result_context
    assert context is not None
    launch_wall_ns = time.time_ns() - 1_000_000
    for key, value in spec.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(CACHE_AWARE_LAUNCH_WALL_NS_ENV, str(launch_wall_ns))
    write_cache_aware_child_result(
        variant="baseline",
        label="baseline_cache_aware_disagg_multigpu",
        rank=0,
        world_size=2,
        iterations_completed=int(context["iterations"]),
        config=dict(context["config"]),
        actual_outputs={},
        reference_outputs={
            0: torch.ones((1, 2), dtype=torch.bfloat16),
            1: torch.ones((1, 2), dtype=torch.bfloat16),
        },
        verification_prompt=torch.ones((1, 4, 2), dtype=torch.bfloat16),
        custom_metrics={"cache_aware.time_per_iter_ms": 1.0},
    )
    try:
        with pytest.raises(RuntimeError, match="rank quorum is incomplete"):
            benchmark.consume_cache_aware_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=time.time_ns() + 1_000_000,
                returncode=0,
            )
    finally:
        shutil.rmtree(context["result_dir"], ignore_errors=True)


def test_cache_aware_worker_entrypoint_executes_and_fails_without_cuda() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.benchmark_worker",
            "--module",
            WORKER_MODULE,
            "--callable",
            "main",
            "--",
            "--variant",
            "baseline",
            "--iters",
            "1",
            "--warmup",
            "0",
        ],
        cwd=CODE_ROOT,
        env={**os.environ, "PYTHONPATH": str(CODE_ROOT), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "CUDA required for cache-aware disaggregated inference" in (
        result.stdout + result.stderr
    )


def test_cache_aware_worker_has_one_measured_range_with_final_rank_drain() -> None:
    source = Path(cache_common.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_torchrun_worker"
    )
    measured_ranges = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "nvtx_range"
            for item in node.items
        )
    ]

    assert len(measured_ranges) == 1
    measured_source = ast.unparse(measured_ranges[0])
    assert "run_iteration(" in measured_source
    assert "_sync_and_barrier(device)" in measured_source
    assert source.count("rank0 time_per_iter_ms:") == 1
