from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from ch04.nvshmem_child_result import (
    NVSHMEM_CHILD_RESULT_CALLBACK,
    NVSHMEM_CHILD_RESULT_DIR_ENV,
    NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV,
    NVSHMEMChildResultMixin,
    NVSHMEMWorkloadResult,
    write_nvshmem_child_result,
)
from core.harness.benchmark_harness import BenchmarkConfig


class _ResultConsumer(NVSHMEMChildResultMixin):
    pass


def _cpu_result(rank: int, *, iterations: int = 7) -> NVSHMEMWorkloadResult:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).add_(rank)
    reference = source.square()
    return NVSHMEMWorkloadResult(
        workload="collective",
        rank=rank,
        world_size=2,
        iterations=iterations,
        time_per_iter_ms=1.25 + rank,
        configuration={"mode": "nccl", "iterations": iterations},
        verify_inputs={"source": source},
        verify_output=reference.clone(),
        reference_output=reference,
        batch_size=3,
        parameter_count=0,
        collective_type="broadcast",
        output_tolerance=(1e-5, 1e-6),
    )


def _write_cpu_quorum(
    monkeypatch: pytest.MonkeyPatch,
    consumer: _ResultConsumer,
) -> tuple[Path, int, int]:
    env = consumer.prepare_nvshmem_child_result(
        variant="baseline",
        workload="collective",
        world_size=2,
        iterations=7,
        configuration={"mode": "nccl", "iterations": 7},
    )
    launch_wall_ns = time.time_ns()
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(
        NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV,
        str(launch_wall_ns),
    )
    assert write_nvshmem_child_result(_cpu_result(0), variant="baseline")
    assert write_nvshmem_child_result(_cpu_result(1), variant="baseline")
    finish_wall_ns = time.time_ns()
    return Path(env[NVSHMEM_CHILD_RESULT_DIR_ENV]), launch_wall_ns, finish_wall_ns


def test_cpu_child_result_requires_and_exposes_full_rank_quorum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = _ResultConsumer()
    result_dir, launch_wall_ns, finish_wall_ns = _write_cpu_quorum(
        monkeypatch,
        consumer,
    )

    consumer.consume_nvshmem_child_results(
        launch_wall_ns=launch_wall_ns,
        finish_wall_ns=finish_wall_ns,
        returncode=0,
    )

    assert not result_dir.exists()
    assert set(consumer._subprocess_verify_inputs) == {
        "rank_0_source",
        "rank_1_source",
    }
    assert consumer._subprocess_verify_output.shape == (24,)
    assert consumer._subprocess_input_signature.dtypes == {
        "rank_0_source": "float32",
        "rank_1_source": "float32",
        "output": "float32",
    }
    assert consumer._subprocess_input_signature.world_size == 2
    assert consumer.validate_result() is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing-rank", "rank quorum is incomplete"),
        ("stale", "Stale NVSHMEM child-result"),
        ("wrong-output", "differs from its oracle"),
        ("wrong-configuration", "configuration mismatch"),
    ),
)
def test_cpu_child_result_rejects_incomplete_or_mutated_receipts(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    consumer = _ResultConsumer()
    result_dir, launch_wall_ns, finish_wall_ns = _write_cpu_quorum(
        monkeypatch,
        consumer,
    )
    try:
        rank1_path = result_dir / "rank-1.pt"
        if mutation == "missing-rank":
            rank1_path.unlink()
        else:
            payload = torch.load(rank1_path, map_location="cpu", weights_only=True)
            if mutation == "stale":
                payload["created_wall_ns"] = launch_wall_ns - 1
            elif mutation == "wrong-output":
                payload["verify_output"][0, 0] += 1
            else:
                payload["configuration"]["iterations"] = 8
            torch.save(payload, rank1_path)

        with pytest.raises(RuntimeError, match=message):
            consumer.consume_nvshmem_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=finish_wall_ns,
                returncode=0,
            )
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


_WRAPPER_CASES = (
    ("baseline_nvshmem_training_example_multigpu", "training-example", 240),
    ("optimized_nvshmem_training_example_multigpu", "training-example", 240),
    ("baseline_nvshmem_training_patterns_multigpu", "training-patterns", 100),
    ("optimized_nvshmem_training_patterns_multigpu", "training-patterns", 100),
    ("baseline_nvshmem_vs_nccl_benchmark_multigpu", "collective", 500),
    ("optimized_nvshmem_vs_nccl_benchmark_multigpu", "collective", 500),
    ("baseline_symmetric_memory_multigpu", "symmetric-ring", 400),
    ("optimized_symmetric_memory_multigpu", "symmetric-ring", 400),
)


@pytest.mark.parametrize(("module_name", "workload", "iterations"), _WRAPPER_CASES)
def test_wrappers_use_explicit_worker_full_result_and_iteration_timing(
    module_name: str,
    workload: str,
    iterations: int,
) -> None:
    benchmark = importlib.import_module(f"ch04.{module_name}").get_benchmark()
    config = BenchmarkConfig(
        nproc_per_node=2,
        nnodes=1,
        iterations=1,
        warmup=0,
        measurement_timeout_seconds=300,
    )
    spec = benchmark.get_torchrun_spec(config)
    result_dir = Path(spec.env[NVSHMEM_CHILD_RESULT_DIR_ENV])
    try:
        assert spec.script_path is None
        assert spec.module_name == "core.harness.benchmark_worker"
        assert spec.script_args[:7] == [
            "--module",
            "ch04.nvshmem_worker",
            "--callable",
            "main",
            "--",
            "--workload",
            workload,
        ]
        assert spec.result_callback == NVSHMEM_CHILD_RESULT_CALLBACK
        assert spec.timing_source == "rank0_time_per_iter_ms"
        assert spec.timing_iterations_per_sample == iterations
        assert result_dir.is_dir() and not any(result_dir.iterdir())
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


def test_explicit_worker_genuinely_skips_without_cuda() -> None:
    code_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(code_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.benchmark_worker",
            "--module",
            "ch04.nvshmem_worker",
            "--callable",
            "main",
            "--",
            "--workload",
            "training-example",
            "--variant",
            "baseline",
            "--demo",
            "pipeline",
            "--batch-size",
            "2",
            "--seq-len",
            "4",
            "--dim",
            "8",
            "--steps",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 3
    assert "SKIPPED:" in completed.stdout + completed.stderr
