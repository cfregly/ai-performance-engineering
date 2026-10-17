"""Regression tests for audit work unit r12.

W1-128 residual: the ch04 torchrun_multigpu and nvshmem pair tests in
``test_validate_benchmark_pairs_tools`` kept a ``result.valid`` branch that
standalone validation can never reach for torchrun wrappers. On multi-GPU
hosts, the torchrun_multigpu test accepted any SKIPPED text. The nvshmem test
accepted either of two specific skip reasons regardless of whether symmetric
memory was available. Each host state now has exactly one accepted skip
reason, and the baseline and optimized sides must each report it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.harness.benchmark_harness import LaunchVia
from core.scripts.validate_benchmark_pairs import load_benchmark_class, validate_pair
from tests import test_validate_benchmark_pairs_tools as pair_tests

_CH04_DIR = Path(__file__).resolve().parents[1] / "ch04"

_CH04_TORCHRUN_PAIR_WRAPPERS = (
    "baseline_pipeline_parallel_multigpu.py",
    "optimized_pipeline_parallel_multigpu_1f1b.py",
    "baseline_tensor_parallel_allgather_multigpu.py",
    "optimized_tensor_parallel_allgather_multigpu.py",
    "baseline_tensor_parallel_multigpu.py",
    "optimized_tensor_parallel_multigpu.py",
    "baseline_torchcomms_multigpu.py",
    "optimized_torchcomms_multigpu.py",
    "baseline_nvshmem_training_example.py",
    "optimized_nvshmem_training_example.py",
    "baseline_nvshmem_training_patterns.py",
    "optimized_nvshmem_training_patterns.py",
)

_CPU_PAYLOAD_PAIR_SOURCE = '''
from __future__ import annotations

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, LaunchVia


class CpuPayloadBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.x = None
        self.y = None

    def setup(self) -> None:
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, device=self.device)
        self.y = None

    def benchmark_fn(self) -> None:
        if self.x is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self.y = self.x * 2

    def capture_verification_payload(self) -> None:
        if self.x is None or self.y is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"x": self.x},
            output=self.y,
            batch_size=int(self.x.shape[0]),
            parameter_count=0,
            output_tolerance=(0.0, 0.0),
        )


class TorchrunCpuPayloadBenchmark(CpuPayloadBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(launch_via=LaunchVia.TORCHRUN, nproc_per_node=2)

    def benchmark_fn(self) -> None:
        raise AssertionError("standalone validation must not run a torchrun parent proxy")


def get_benchmark() -> BaseBenchmark:
    return {factory}()
'''


_SETUP_SKIP_SOURCE = '''
from __future__ import annotations

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark


class SetupSkipBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def setup(self) -> None:
        raise RuntimeError({message})

    def benchmark_fn(self) -> None:
        raise AssertionError("setup() must skip before benchmark_fn()")


def get_benchmark() -> BaseBenchmark:
    return SetupSkipBenchmark()
'''


def _write_cpu_payload_benchmark(path: Path, *, torchrun: bool) -> Path:
    factory = "TorchrunCpuPayloadBenchmark" if torchrun else "CpuPayloadBenchmark"
    path.write_text(_CPU_PAYLOAD_PAIR_SOURCE.replace("{factory}", factory))
    return path


def _write_setup_skip_benchmark(path: Path, message: str) -> Path:
    path.write_text(_SETUP_SKIP_SOURCE.replace("{message}", repr(message)))
    return path


@pytest.mark.parametrize("wrapper_name", _CH04_TORCHRUN_PAIR_WRAPPERS)
def test_ch04_pair_wrappers_launch_via_torchrun_outside_distributed_env(
    wrapper_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in pair_tests._DISTRIBUTED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    benchmark, load_error = load_benchmark_class(_CH04_DIR / wrapper_name)

    assert load_error is None, load_error
    assert benchmark is not None
    assert benchmark.get_config().launch_via == LaunchVia.TORCHRUN


def test_torchrun_pair_skip_must_carry_the_expected_reason(tmp_path: Path) -> None:
    result = validate_pair(
        "ch01",
        "torchrun_cpu_payload",
        _write_cpu_payload_benchmark(tmp_path / "baseline_torchrun_cpu_payload.py", torchrun=True),
        _write_cpu_payload_benchmark(tmp_path / "optimized_torchrun_cpu_payload.py", torchrun=True),
    )

    pair_tests._assert_both_sides_skipped_for(result, pair_tests._TORCHRUN_WORKER_REFUSAL)
    for wrong_reason in (
        "SKIPPED: torchrun_cpu_payload requires NVSHMEM or SymmetricMemory support",
        "SKIPPED: Requires >= 2 GPUs",
    ):
        with pytest.raises(AssertionError):
            pair_tests._assert_both_sides_skipped_for(result, wrong_reason)


def test_one_sided_torchrun_skip_does_not_pass_as_a_pair_skip(tmp_path: Path) -> None:
    result = validate_pair(
        "ch01",
        "mixed_cpu_payload",
        _write_cpu_payload_benchmark(tmp_path / "baseline_mixed_cpu_payload.py", torchrun=True),
        _write_cpu_payload_benchmark(tmp_path / "optimized_mixed_cpu_payload.py", torchrun=False),
    )

    # The pre-fix multi-GPU branch accepted this result because it is skipped
    # and its error contains SKIPPED.
    assert result.skipped is True
    assert result.error is not None and "SKIPPED" in result.error
    assert result.optimized_has_signature is True
    with pytest.raises(AssertionError):
        pair_tests._assert_both_sides_skipped_for(result, pair_tests._TORCHRUN_WORKER_REFUSAL)


def test_reason_repeated_on_one_side_does_not_cover_the_other(tmp_path: Path) -> None:
    reason = pair_tests._TORCHRUN_WORKER_REFUSAL
    result = validate_pair(
        "ch01",
        "repeated_reason",
        _write_setup_skip_benchmark(
            tmp_path / "baseline_repeated_reason.py", f"{reason}. {reason}"
        ),
        _write_setup_skip_benchmark(
            tmp_path / "optimized_repeated_reason.py", "SKIPPED: Requires >= 2 GPUs (found 0 GPU)"
        ),
    )

    # A whole-message count of the reason accepts this result. Only the
    # baseline side reports the reason.
    assert result.skipped is True
    assert result.error is not None and result.error.count(reason) == 2
    with pytest.raises(AssertionError, match="optimized_error="):
        pair_tests._assert_both_sides_skipped_for(result, reason)
