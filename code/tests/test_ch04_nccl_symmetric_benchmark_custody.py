"""Harness-level custody checks for the NCCL symmetric-memory pair."""

from __future__ import annotations

from array import array
from pathlib import Path
from types import SimpleNamespace

from ch04 import nccl_symmetric_memory_benchmark as benchmark_module


def test_harness_prepares_one_cohort_and_reuses_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark = benchmark_module.NcclSymmetricMemoryBenchmark(arm="cuda_malloc")
    benchmark.elements = 4
    benchmark.native_iterations = 1
    benchmark.native_warmup = 0
    benchmark.requested_ranks = 2
    benchmark._config = SimpleNamespace(
        execution_timeout_seconds=0,
        enable_profiling=False,
        profile_mode="none",
        profile_type="none",
        profiling_output_dir=str(tmp_path / "artifacts"),
    )
    nccl_home = tmp_path / "nccl"
    nccl_home.mkdir()
    benchmark.nccl_home = nccl_home

    prepared_cohort = object()
    preparation_calls: list[tuple[Path, Path]] = []
    invocation_cohorts: list[object] = []

    monkeypatch.setattr(benchmark_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(benchmark_module, "detect_supported_arch", lambda: "sm_100")
    monkeypatch.setattr(benchmark_module.shutil, "which", lambda _name: "/usr/bin/fixture")
    monkeypatch.setattr(
        benchmark_module,
        "validate_build_receipt",
        lambda _binary: {"architecture": "sm_100", "nccl_home": str(nccl_home)},
    )

    def prepare(binary: Path, directory: Path):
        preparation_calls.append((binary, directory))
        return prepared_cohort

    def invoke(_workload, **kwargs):
        invocation_cohorts.append(kwargs["cohort"])
        kwargs["dump_output"].write_bytes(array("f", [1.0, 2.0, 3.0, 4.0]).tobytes())
        return {"time_per_iteration_ms": 1.25}

    monkeypatch.setattr(benchmark_module, "prepare_run_cohort", prepare)
    monkeypatch.setattr(benchmark_module, "invoke_native", invoke)

    benchmark.setup()
    benchmark._run_once()
    benchmark._run_once()

    assert len(preparation_calls) == 1
    assert preparation_calls[0][1].name == "run_cohort"
    assert invocation_cohorts == [prepared_cohort, prepared_cohort]
