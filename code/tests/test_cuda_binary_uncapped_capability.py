"""Reject unsupported binary execution before build or launch work on a real CPU path."""

from pathlib import Path
import sys

import pytest
import torch

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.benchmark_harness import BenchmarkConfig, ExecutionMode, ReadOnlyBenchmarkConfigView


class CpuBinaryCapabilityFixture(CudaBinaryBenchmark):
    """Use a CPU executable only to observe rejection and command dispatch."""

    allow_cpu = True

    def __init__(self, directory: Path) -> None:
        super().__init__(
            chapter_dir=directory, binary_name="cpu_capability_fixture",
            friendly_name="CPU capability rejection fixture", workload_params={"N": 1},
        )
        self.device = torch.device("cpu")
        self.marker = directory / "command-ran.txt"
        self.exec_path = Path(sys.executable)
        self._verify_exec_path = self.exec_path
        self.run_args = [
            "-c",
            "from pathlib import Path\n"
            f"Path({str(self.marker)!r}).write_text('CPU command ran')\n"
            "print('CPU capability fixture, no performance result')\n",
        ]


def _assign_harness_config(benchmark: CudaBinaryBenchmark, timeout: int | None) -> None:
    config = BenchmarkConfig(
        device=torch.device("cpu"), execution_timeout_seconds=timeout,
        execution_mode=ExecutionMode.SUBPROCESS, lock_gpu_clocks=False,
    )
    # This is the same read-only configuration handoff the harness uses.
    benchmark._config = ReadOnlyBenchmarkConfigView.from_config(config)


@pytest.mark.parametrize("entrypoint", [
    "setup", "_build_binary", "benchmark_fn", "_run_once", "run_verify", "_run_verify",
])
def test_stock_binary_rejects_uncapped_before_build_or_command(tmp_path: Path, entrypoint: str) -> None:
    benchmark = CpuBinaryCapabilityFixture(tmp_path)
    assert benchmark.device.type == "cpu"
    _assign_harness_config(benchmark, 0)
    with pytest.raises(RuntimeError, match="does not support execution_timeout_seconds=0"):
        getattr(benchmark, entrypoint)()
    assert not benchmark.marker.exists()
    assert benchmark.last_time_ms is None


@pytest.mark.parametrize("timeout", [None, 7])
def test_default_and_finite_binary_paths_still_launch_cpu_command(
    tmp_path: Path, timeout: int | None,
) -> None:
    benchmark = CpuBinaryCapabilityFixture(tmp_path)
    _assign_harness_config(benchmark, timeout)
    # This fixture emits no timing. Parsing must fail after the real command runs.
    with pytest.raises(RuntimeError, match="Could not parse execution time"):
        benchmark.benchmark_fn()
    assert benchmark.marker.read_text() == "CPU command ran"
    assert benchmark.last_time_ms is None


def test_explicit_cooperative_class_capability_allows_config_validation(tmp_path: Path) -> None:
    class CooperativeCapabilityDeclaration(CpuBinaryCapabilityFixture):
        supports_uncapped_execution = True

    benchmark = CooperativeCapabilityDeclaration(tmp_path)
    _assign_harness_config(benchmark, 0)
    benchmark._require_supported_execution_timeout()
    assert not benchmark.marker.exists()
    assert benchmark.last_time_ms is None
