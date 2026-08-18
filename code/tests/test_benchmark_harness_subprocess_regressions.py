from __future__ import annotations

import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    LaunchVia,
    TorchrunLaunchSpec,
    _benchmark_child_preexec,
)


def _raise_timeout(_signum, _frame):
    raise TimeoutError("parent alarm fired")


def test_benchmark_child_preexec_clears_inherited_alarm() -> None:
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM unavailable on this platform")

    previous_handler = signal.getsignal(signal.SIGALRM)
    process: subprocess.Popen[str] | None = None
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(1)
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(1.5)"],
            preexec_fn=_benchmark_child_preexec,
        )
        # Cancel the parent timer immediately; the child should already have
        # cleared its inherited alarm in the pre-exec hook.
        signal.alarm(0)
        process.wait(timeout=5)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process is not None
    assert process.returncode == 0


class _TorchrunSkipBenchmark(BaseBenchmark):
    allow_cpu = True

    def __init__(self, script_path: Path) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self._script_path = script_path
        self.input_tensor = torch.zeros(1, device=self.device)

    def setup(self) -> None:  # pragma: no cover - not executed in torchrun mode
        pass

    def benchmark_fn(self) -> None:  # pragma: no cover - not executed in torchrun mode
        pass

    def validate_result(self) -> None:
        return None

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        return {"input": self.input_tensor}

    def get_verify_output(self) -> torch.Tensor:
        return self.input_tensor

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_input_signature(self) -> dict[str, object]:
        return {"shape": tuple(self.input_tensor.shape), "dtype": str(self.input_tensor.dtype)}

    def get_torchrun_spec(self, config: BenchmarkConfig) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=self._script_path,
            script_args=[],
            env={},
            parse_rank0_only=True,
            multi_gpu_required=False,
            name="torchrun_skip_reason",
            config_arg_map={},
        )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="requires Linux torchrun rendezvous",
)
def test_torchrun_skip_reason_propagates_without_generic_exit_error(tmp_path: Path) -> None:
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available in PATH")

    script = tmp_path / "skip_script.py"
    script.write_text(
        "import sys\n"
        "print('SKIPPED: unit-test torchrun skip path', flush=True)\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )

    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=1,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=1,
        multi_gpu_required=False,
        enforce_environment_validation=False,
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(_TorchrunSkipBenchmark(script))

    joined = "\n".join(result.errors)
    assert "SKIPPED: unit-test torchrun skip path" in joined
    assert "torchrun exited with code" not in joined
