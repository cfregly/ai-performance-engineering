from __future__ import annotations

import ast
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ch04.nvshmem_profile_ranges import (
    COLLECTIVE_BASELINE_NVTX_RANGE,
    COLLECTIVE_OPTIMIZED_NVTX_RANGE,
    PIPELINE_BASELINE_NVTX_RANGE,
    PIPELINE_OPTIMIZED_NVTX_RANGE,
    PROFILE_NVTX_RANGE_ENV,
    TRAINING_EXAMPLE_BASELINE_NVTX_RANGE,
    TRAINING_EXAMPLE_OPTIMIZED_NVTX_RANGE,
    TRAINING_PATTERNS_BASELINE_NVTX_RANGE,
    TRAINING_PATTERNS_OPTIMIZED_NVTX_RANGE,
)
from ch04.symmetric_memory_example import (
    SYMMETRIC_RING_NVTX_RANGE,
    TRADITIONAL_RING_NVTX_RANGE,
)
from ch04.symmetric_memory_perf_common import (
    SYMMETRIC_MEMORY_PERF_BASELINE_NVTX_RANGE,
    SYMMETRIC_MEMORY_PERF_OPTIMIZED_NVTX_RANGE,
    SYMMETRIC_MEMORY_PERF_RESULT_CALLBACK,
    SYMMETRIC_MEMORY_PERF_RESULT_DIR_ENV,
    SYMMETRIC_MEMORY_PERF_RESULT_SCHEMA,
)
from core.benchmark.verification import InputSignature, PrecisionFlags
from core.harness.benchmark_harness import BenchmarkConfig
from core.profiling.nvtx_helper import canonicalize_nvtx_name
from core.profiling.profiler_config import build_profiler_config_from_benchmark

NVSHMEM_CASES = (
    (
        "ch04.baseline_nvshmem_pipeline_parallel_multigpu",
        PIPELINE_BASELINE_NVTX_RANGE,
    ),
    (
        "ch04.optimized_nvshmem_pipeline_parallel_multigpu",
        PIPELINE_OPTIMIZED_NVTX_RANGE,
    ),
    (
        "ch04.baseline_nvshmem_training_example_multigpu",
        TRAINING_EXAMPLE_BASELINE_NVTX_RANGE,
    ),
    (
        "ch04.optimized_nvshmem_training_example_multigpu",
        TRAINING_EXAMPLE_OPTIMIZED_NVTX_RANGE,
    ),
    (
        "ch04.baseline_nvshmem_training_patterns_multigpu",
        TRAINING_PATTERNS_BASELINE_NVTX_RANGE,
    ),
    (
        "ch04.optimized_nvshmem_training_patterns_multigpu",
        TRAINING_PATTERNS_OPTIMIZED_NVTX_RANGE,
    ),
    (
        "ch04.baseline_nvshmem_vs_nccl_benchmark_multigpu",
        COLLECTIVE_BASELINE_NVTX_RANGE,
    ),
    (
        "ch04.optimized_nvshmem_vs_nccl_benchmark_multigpu",
        COLLECTIVE_OPTIMIZED_NVTX_RANGE,
    ),
)

SYMMETRIC_MEMORY_CASES = (
    ("ch04.baseline_symmetric_memory_multigpu", TRADITIONAL_RING_NVTX_RANGE),
    ("ch04.optimized_symmetric_memory_multigpu", SYMMETRIC_RING_NVTX_RANGE),
    (
        "ch04.baseline_symmetric_memory_perf_multigpu",
        SYMMETRIC_MEMORY_PERF_BASELINE_NVTX_RANGE,
    ),
    (
        "ch04.optimized_symmetric_memory_perf_multigpu",
        SYMMETRIC_MEMORY_PERF_OPTIMIZED_NVTX_RANGE,
    ),
)


@pytest.mark.parametrize(
    ("module_name", "expected_range"),
    (*NVSHMEM_CASES, *SYMMETRIC_MEMORY_CASES),
)
def test_distributed_profile_config_uses_one_real_app_range(
    module_name: str,
    expected_range: str,
) -> None:
    benchmark = importlib.import_module(module_name).get_benchmark()
    config = benchmark.get_config()

    assert benchmark.preferred_ncu_replay_mode == "app-range"
    assert config.profiling.nsys_nvtx_include == (expected_range,)
    assert config.profiling.ncu_replay_mode == "app-range"
    assert config.profiling.ncu_replay_mode_override is True
    assert canonicalize_nvtx_name(expected_range) == expected_range

    profiler = build_profiler_config_from_benchmark(config)
    command = profiler.get_ncu_command_for_target(
        "/tmp/ch04-distributed-app-range",
        ["python", f"{module_name}.py"],
    )
    assert command.count("--nvtx-include") == 1
    assert command[command.index("--nvtx-include") + 1] == expected_range
    assert command[command.index("--replay-mode") + 1] == "app-range"
    assert "--launch-count" not in command


@pytest.mark.parametrize(("module_name", "expected_range"), NVSHMEM_CASES)
def test_nvshmem_wrapper_passes_its_declared_range_to_the_worker(
    module_name: str,
    expected_range: str,
) -> None:
    benchmark = importlib.import_module(module_name).get_benchmark()
    config = benchmark.get_config()
    config.nproc_per_node = 2
    config.nnodes = 1
    spec = benchmark.get_torchrun_spec(config)
    try:
        assert spec.script_path is None
        assert spec.module_name == "core.harness.benchmark_worker"
        assert spec.script_args[:5] == [
            "--module",
            "ch04.nvshmem_worker",
            "--callable",
            "main",
            "--",
        ]
        assert spec.env[PROFILE_NVTX_RANGE_ENV] == expected_range
    finally:
        for name, value in spec.env.items():
            if name.endswith("_RESULT_DIR"):
                shutil.rmtree(value, ignore_errors=True)


@pytest.mark.parametrize(
    ("module_name", "function_name", "timed_expression", "warmup_expression"),
    (
        (
            "ch04.nvshmem_pipeline_parallel_multigpu",
            "demo_1f1b_pipeline",
            "engine.run_1f1b_schedule(input_batches)",
            None,
        ),
        (
            "ch04.nvshmem_training_example",
            "demo_pipeline_parallel",
            "range(steps)",
            None,
        ),
        (
            "ch04.nvshmem_training_patterns",
            "demo_gradient_sync",
            "range(num_steps)",
            None,
        ),
        (
            "ch04.nvshmem_vs_nccl_benchmark",
            "_measure_nccl_broadcast",
            "range(iterations)",
            "range(5)",
        ),
        (
            "ch04.nvshmem_vs_nccl_benchmark",
            "_measure_symmetric_broadcast",
            "range(iterations)",
            "range(5)",
        ),
    ),
)
def test_nvshmem_worker_range_encloses_real_timed_work(
    module_name: str,
    function_name: str,
    timed_expression: str,
    warmup_expression: str | None,
) -> None:
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == function_name
    )
    ranges = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and ast.unparse(node.items[0].context_expr.func) == "selected_nvtx_range"
    ]
    assert len(ranges) == 1
    profile_range = ranges[0]
    assert any(
        (
            isinstance(node, ast.For)
            and ast.unparse(node.iter) == timed_expression
        )
        or (isinstance(node, ast.Call) and ast.unparse(node) == timed_expression)
        for node in ast.walk(profile_range)
    )
    if warmup_expression is not None:
        warmup = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and ast.unparse(node.iter) == warmup_expression
        )
        assert warmup.end_lineno < profile_range.lineno


@pytest.mark.parametrize(
    ("function_name", "constant_name", "warmup_iterator", "timed_iterator"),
    (
        (
            "benchmark_traditional_ring",
            "TRADITIONAL_RING_NVTX_RANGE",
            "range(5)",
            "range(iterations)",
        ),
        (
            "benchmark_symmetric_ring",
            "SYMMETRIC_RING_NVTX_RANGE",
            "range(5)",
            "range(iterations)",
        ),
    ),
)
def test_symmetric_memory_range_excludes_warmup_and_encloses_timed_loop(
    function_name: str,
    constant_name: str,
    warmup_iterator: str,
    timed_iterator: str,
) -> None:
    module = importlib.import_module("ch04.symmetric_memory_example")
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    ranges = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and ast.unparse(node.items[0].context_expr.func) == "nvtx_range"
    ]
    assert len(ranges) == 1
    profile_range = ranges[0]
    range_call = profile_range.items[0].context_expr
    assert isinstance(range_call, ast.Call)
    assert ast.unparse(range_call.args[0]) == constant_name
    assert any(
        keyword.arg == "enable"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in range_call.keywords
    )

    warmup = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == warmup_iterator
    )
    timed = next(
        node
        for node in ast.walk(profile_range)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == timed_iterator
    )
    assert warmup.end_lineno < profile_range.lineno < timed.lineno


@pytest.mark.parametrize(
    ("variant", "expected_range"),
    (
        ("baseline", SYMMETRIC_MEMORY_PERF_BASELINE_NVTX_RANGE),
        ("optimized", SYMMETRIC_MEMORY_PERF_OPTIMIZED_NVTX_RANGE),
    ),
)
def test_symmetric_memory_perf_spec_uses_real_worker_and_fresh_result_callback(
    variant: str,
    expected_range: str,
) -> None:
    module = importlib.import_module(
        f"ch04.{variant}_symmetric_memory_perf_multigpu"
    )
    benchmark = module.get_benchmark()
    config = BenchmarkConfig(
        nproc_per_node=2,
        warmup=6,
        iterations=4,
        nsys_nvtx_include=[expected_range],
        ncu_replay_mode="app-range",
        ncu_replay_mode_override=True,
    )
    spec = benchmark.get_torchrun_spec(config)
    result_dir = Path(spec.env[SYMMETRIC_MEMORY_PERF_RESULT_DIR_ENV])
    try:
        assert spec.script_path is not None
        assert spec.script_path.name == "symmetric_memory_perf_worker.py"
        assert spec.script_args == [
            "--variant",
            variant,
            "--warmup",
            "6",
            "--iterations",
            "4",
        ]
        assert spec.result_callback == SYMMETRIC_MEMORY_PERF_RESULT_CALLBACK
        assert result_dir.is_dir()
        assert not any(result_dir.iterdir())
        with pytest.raises(RuntimeError, match="rank quorum is incomplete"):
            benchmark.consume_symmetric_memory_perf_child_results(
                launch_wall_ns=1,
                finish_wall_ns=2,
                returncode=0,
            )
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


def test_symmetric_memory_perf_worker_profiles_only_measured_iterations() -> None:
    module = importlib.import_module("ch04.symmetric_memory_perf_worker")
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    profile_range = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and ast.unparse(node.items[0].context_expr.func) == "nvtx_range"
    )
    warmup = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(args.warmup)"
    )
    measured = next(
        node
        for node in ast.walk(profile_range)
        if isinstance(node, ast.For)
        and ast.unparse(node.iter) == "range(args.iterations)"
    )
    assert warmup.end_lineno < profile_range.lineno < measured.lineno
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "torch.cuda.synchronize"
        for node in ast.walk(profile_range)
    )
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "benchmark.benchmark_fn"
        for node in ast.walk(profile_range)
    )
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "measured_times_ms.append"
        for node in ast.walk(profile_range)
    )


def test_symmetric_memory_perf_timing_aggregates_real_measured_samples() -> None:
    from ch04.symmetric_memory_perf_worker import _mean_measured_iteration_ms

    assert _mean_measured_iteration_ms([2.0, 4.0, 9.0]) == pytest.approx(5.0)


@pytest.mark.parametrize("samples", ([], [0.0], [-1.0], [float("nan")]))
def test_symmetric_memory_perf_timing_rejects_invalid_samples(
    samples: list[float],
) -> None:
    from ch04.symmetric_memory_perf_worker import _mean_measured_iteration_ms

    with pytest.raises(RuntimeError, match="timing samples|No measured"):
        _mean_measured_iteration_ms(samples)


@pytest.mark.parametrize("variant", ("baseline", "optimized"))
def test_symmetric_memory_perf_capture_rejects_missing_timed_output(
    variant: str,
) -> None:
    module = importlib.import_module(
        f"ch04.{variant}_symmetric_memory_perf_multigpu"
    )
    benchmark = module.get_benchmark()
    benchmark._verify_input = torch.ones((2, 2), dtype=torch.float32)
    benchmark._verify_output_buffer = torch.empty((2, 2), dtype=torch.float32)
    benchmark._verify_output = None

    with pytest.raises(RuntimeError, match="Timed receive output was not produced"):
        benchmark.capture_verification_payload()


@pytest.mark.parametrize("variant", ("baseline", "optimized"))
def test_symmetric_memory_perf_worker_genuinely_skips_without_cuda(
    variant: str,
) -> None:
    code_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(code_root)
    completed = subprocess.run(
        [
            sys.executable,
            str(code_root / "ch04/symmetric_memory_perf_worker.py"),
            "--variant",
            variant,
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 3
    assert "SKIPPED:" in completed.stdout


@pytest.mark.parametrize("dtype_name", ("torch.float32", "float32", "float16"))
@pytest.mark.parametrize("invalid_rank", (None, 0, 1))
def test_symmetric_memory_perf_callback_checks_actual_output_and_canonical_dtype(
    tmp_path: Path,
    invalid_rank: int | None,
    dtype_name: str,
) -> None:
    module = importlib.import_module("ch04.baseline_symmetric_memory_perf_multigpu")
    benchmark = module.get_benchmark()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    token = "unit-test-token"
    launch_wall_ns = 10
    finish_wall_ns = 30
    benchmark._symmetric_memory_perf_result_context = {
        "result_dir": result_dir,
        "token": token,
        "variant": "baseline",
        "world_size": 2,
        "retention": "pending-child-result",
    }
    input_tensors = [
        torch.arange(4, dtype=torch.float32).view(2, 2) + rank * 10
        for rank in range(2)
    ]
    signature = InputSignature(
        shapes={"tensor": (2, 2), "output": (2, 2)},
        dtypes={"tensor": dtype_name, "output": dtype_name},
        batch_size=2,
        parameter_count=0,
        precision_flags=PrecisionFlags(),
        world_size=2,
    )
    for rank in range(2):
        output = input_tensors[(rank - 1) % 2].clone()
        if rank == invalid_rank:
            output[0, 0] += 1
        torch.save(
            {
                "schema": SYMMETRIC_MEMORY_PERF_RESULT_SCHEMA,
                "token": token,
                "variant": "baseline",
                "rank": rank,
                "world_size": 2,
                "launch_wall_ns": launch_wall_ns,
                "created_wall_ns": 20,
                "verify_inputs": {"tensor": input_tensors[rank]},
                "verify_output": output,
                "input_signature": signature.to_dict(),
                "output_tolerance": [1e-5, 1e-5],
            },
            result_dir / f"rank-{rank}.pt",
        )

    if dtype_name != "float16" and invalid_rank is None:
        benchmark.consume_symmetric_memory_perf_child_results(
            launch_wall_ns=launch_wall_ns,
            finish_wall_ns=finish_wall_ns,
            returncode=0,
        )
        torch.testing.assert_close(benchmark._subprocess_verify_output, input_tensors[1])
        assert benchmark._subprocess_input_signature.dtypes == {
            "tensor": "float32", "output": "float32"
        }
        return
    expected_error = (
        "signature dtype mismatch"
        if dtype_name == "float16"
        else rf"does not match its measured sender input at rank {invalid_rank}"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        benchmark.consume_symmetric_memory_perf_child_results(
            launch_wall_ns=launch_wall_ns,
            finish_wall_ns=finish_wall_ns,
            returncode=0,
        )
