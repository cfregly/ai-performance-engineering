from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.harness import benchmark_worker
from core.profiling.nvtx_helper import canonicalize_nvtx_name
from core.profiling.profiler_config import build_profiler_config_from_benchmark

PROFILE_CASES = (
    (
        "ch13.baseline_context_parallel_multigpu",
        "BaselineContextParallelMultigpuBenchmark",
        "PROFILE_NVTX_RANGE",
        "compute_kernel:context_parallel_allgather_attention",
        "_world_size",
    ),
    (
        "ch13.optimized_context_parallel_multigpu",
        "OptimizedContextParallelMultigpuBenchmark",
        "PROFILE_NVTX_RANGE",
        "compute_kernel:context_parallel_ring_attention",
        "_world_size",
    ),
    (
        "ch13.baseline_expert_parallel_multigpu",
        "BaselineExpertParallelMultigpuBenchmark",
        "PROFILE_NVTX_RANGE",
        "compute_kernel:expert_parallel_list_all_to_all",
        "_world_size",
    ),
    (
        "ch13.optimized_expert_parallel_multigpu",
        "OptimizedExpertParallelMultigpuBenchmark",
        "PROFILE_NVTX_RANGE",
        "compute_kernel:expert_parallel_single_all_to_all",
        "_world_size",
    ),
    (
        "ch13.baseline_sequence_parallel_multigpu",
        "BaselineSequenceParallelMultigpuBenchmark",
        "PROFILE_NVTX_RANGE",
        "compute_kernel:sequence_parallel_full_gather",
        "_world_size",
    ),
    (
        "ch13.optimized_sequence_parallel_multigpu",
        "OptimizedSequenceParallelMultigpuBenchmark",
        "PROFILE_NVTX_RANGE",
        "compute_kernel:sequence_parallel_sharded",
        "_world_size",
    ),
    (
        "ch15.baseline_disaggregated_inference_multigpu",
        "BaselineDisaggregatedInferenceMultiGPUBenchmark",
        "BASELINE_PROFILE_NVTX_RANGE",
        "compute_kernel:disaggregated_inference_serialized_pipeline",
        "world_size",
    ),
    (
        "ch15.optimized_disaggregated_inference_multigpu",
        "OptimizedDisaggregatedInferenceMultiGPUBenchmark",
        "OPTIMIZED_PROFILE_NVTX_RANGE",
        "compute_kernel:disaggregated_inference_overlapped_pipeline",
        "world_size",
    ),
)


@pytest.mark.parametrize(
    ("module_name", "benchmark_class", "constant_name", "expected_range", "world_size_attr"),
    PROFILE_CASES,
)
def test_multigpu_profile_config_selects_the_variant_owned_range(
    module_name: str,
    benchmark_class: str,
    constant_name: str,
    expected_range: str,
    world_size_attr: str,
) -> None:
    module = importlib.import_module(module_name)
    benchmark_type = getattr(module, benchmark_class)
    benchmark = benchmark_type.__new__(benchmark_type)
    setattr(benchmark, world_size_attr, 2)

    config = benchmark.get_config()
    assert getattr(module, constant_name) == expected_range
    assert canonicalize_nvtx_name(expected_range) == expected_range
    assert config.profiling.nsys_nvtx_include == (expected_range,)
    assert benchmark.preferred_ncu_replay_mode == "app-range"
    assert config.profiling.ncu_replay_mode == "app-range"
    assert config.profiling.ncu_replay_mode_override is True

    profiler = build_profiler_config_from_benchmark(config)
    command = profiler.get_ncu_command_for_target(
        "/tmp/multigpu-direct-profile",
        ["python", str(Path(module.__file__).resolve())],
    )
    assert command.count("--nvtx-include") == 1
    include_index = command.index("--nvtx-include")
    assert command[include_index + 1] == expected_range
    assert command[command.index("--replay-mode") + 1] == "app-range"


@pytest.mark.parametrize(
    ("module_name", "benchmark_class", "_constant_name", "_expected_range", "world_size_attr"),
    PROFILE_CASES,
)
def test_torchrun_spec_calls_the_owned_main_through_the_explicit_adapter(
    module_name: str,
    benchmark_class: str,
    _constant_name: str,
    _expected_range: str,
    world_size_attr: str,
) -> None:
    module = importlib.import_module(module_name)
    benchmark_type = getattr(module, benchmark_class)
    benchmark = benchmark_type.__new__(benchmark_type)
    setattr(benchmark, world_size_attr, 2)
    benchmark._prepare_verification_payload = lambda: None
    if module_name.startswith("ch15."):
        benchmark.label = module_name.rsplit(".", maxsplit=1)[-1]
        benchmark.worker_module = module_name

    spec = benchmark.get_torchrun_spec()
    assert spec.script_path == Path(benchmark_worker.__file__).resolve()
    assert spec.module_name is None
    assert spec.script_args == ["--module", module_name, "--callable", "main", "--"]
    assert spec.config_arg_map == {"iterations": "--iters", "warmup": "--warmup"}


def test_adapter_reaches_real_ch13_main_and_preserves_its_capability_failure() -> None:
    code_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.benchmark_worker",
            "--module",
            "ch13.baseline_context_parallel_multigpu",
            "--callable",
            "main",
            "--",
            "--iters",
            "1",
            "--warmup",
            "0",
        ],
        cwd=code_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "SKIPPED: Requires >= 2 GPUs (found 0 GPU)" in completed.stderr


def test_adapter_reaches_real_ch15_argument_parser() -> None:
    code_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.benchmark_worker",
            "--module",
            "ch15.optimized_disaggregated_inference_multigpu",
            "--callable",
            "main",
            "--",
            "--iters",
            "not-an-int",
        ],
        cwd=code_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "argument --iters: invalid int value: 'not-an-int'" in completed.stderr


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    return ast.unparse(node.func)


WORKER_CASES = (
    ("ch13/context_parallel_benchmark_common.py", "run_context_parallel", False),
    ("ch13/expert_parallel_common.py", "run_expert_parallel", False),
    ("ch13/sequence_parallel_benchmark_common.py", "run_sequence_parallel", False),
    ("ch15/baseline_disaggregated_inference_multigpu.py", "_run_torchrun_worker", True),
)


@pytest.mark.parametrize(("relative_path", "function_name", "includes_barrier"), WORKER_CASES)
def test_real_worker_range_encloses_only_the_post_warmup_timed_loop(
    relative_path: str,
    function_name: str,
    includes_barrier: bool,
) -> None:
    source_path = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function_name
    )
    profile_ranges = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.With)
        and any(_call_name(item.context_expr) == "nvtx_range" for item in node.items)
    ]
    assert len(profile_ranges) == 1
    profile_range = profile_ranges[0]
    range_call = profile_range.items[0].context_expr
    assert isinstance(range_call, ast.Call)
    assert isinstance(range_call.args[0], ast.Name)
    assert range_call.args[0].id == "profile_nvtx_range"
    assert any(
        keyword.arg == "enable"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in range_call.keywords
    )

    warmup_loops = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(max(warmup, 0))"
    ]
    timed_loops = [
        node
        for node in ast.walk(profile_range)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(max(iters, 1))"
    ]
    assert len(warmup_loops) == 1
    assert len(timed_loops) == 1
    assert warmup_loops[0].end_lineno < profile_range.lineno < timed_loops[0].lineno

    range_calls = {
        _call_name(node)
        for node in ast.walk(profile_range)
        if isinstance(node, ast.Call)
    }
    assert "time.perf_counter" in range_calls
    assert "torch.cuda.synchronize" in range_calls
    assert ("_barrier" in range_calls) is includes_barrier
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "elapsed" for target in node.targets)
        for node in profile_range.body
    )


ENTRYPOINT_CASES = (
    ("ch13.baseline_context_parallel_multigpu", "run_context_parallel", "PROFILE_NVTX_RANGE"),
    ("ch13.optimized_context_parallel_multigpu", "run_context_parallel", "PROFILE_NVTX_RANGE"),
    ("ch13.baseline_expert_parallel_multigpu", "run_expert_parallel", "PROFILE_NVTX_RANGE"),
    ("ch13.optimized_expert_parallel_multigpu", "run_expert_parallel", "PROFILE_NVTX_RANGE"),
    ("ch13.baseline_sequence_parallel_multigpu", "run_sequence_parallel", "PROFILE_NVTX_RANGE"),
    ("ch13.optimized_sequence_parallel_multigpu", "run_sequence_parallel", "PROFILE_NVTX_RANGE"),
    (
        "ch15.baseline_disaggregated_inference_multigpu",
        "_run_torchrun_worker",
        "BASELINE_PROFILE_NVTX_RANGE",
    ),
    (
        "ch15.optimized_disaggregated_inference_multigpu",
        "_run_torchrun_worker",
        "OPTIMIZED_PROFILE_NVTX_RANGE",
    ),
)


@pytest.mark.parametrize(("module_name", "worker_name", "constant_name"), ENTRYPOINT_CASES)
def test_worker_entrypoint_forwards_the_configured_range(
    module_name: str,
    worker_name: str,
    constant_name: str,
) -> None:
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    worker_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and _call_name(node) == worker_name
    ]
    assert len(worker_calls) == 1
    range_keywords = [
        keyword for keyword in worker_calls[0].keywords if keyword.arg == "profile_nvtx_range"
    ]
    assert len(range_keywords) == 1
    assert isinstance(range_keywords[0].value, ast.Name)
    assert range_keywords[0].value.id == constant_name


def test_profile_range_labels_are_distinct_and_canonical() -> None:
    labels = [case[3] for case in PROFILE_CASES]
    assert len(labels) == len(set(labels))
    assert [canonicalize_nvtx_name(label) for label in labels] == labels
