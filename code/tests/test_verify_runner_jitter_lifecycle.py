"""Regression coverage for live jitter benchmark lifecycles."""

from __future__ import annotations

from pathlib import Path

import torch

from ch05.baseline_ai import BaselineAIBenchmark
from ch05.optimized_ai import OptimizedAIBenchmark
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.verify_runner import VerifyConfig
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata
from tests.protection_test_utils import make_runner


class _TrackedOptimizedAI(OptimizedAIBenchmark):
    """Run the real storage/model path while recording lifecycle boundaries."""

    def __init__(self) -> None:
        super().__init__()
        self.setup_calls = 0
        self.benchmark_calls = 0
        self.teardown_calls = 0
        self.created_input_path: Path | None = None

    def setup(self) -> None:
        self.setup_calls += 1
        super().setup()
        if self.inputs_path is None:
            raise RuntimeError("AI setup did not create its storage input")
        self.created_input_path = Path(self.inputs_path)

    def benchmark_fn(self) -> None:
        self.benchmark_calls += 1
        super().benchmark_fn()

    def teardown(self) -> None:
        self.teardown_calls += 1
        super().teardown()


class _FailingJitterBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Small real tensor workload that fails only on the perturbed rerun."""

    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.original_input: torch.Tensor | None = None
        self.setup_calls = 0
        self.benchmark_calls = 0
        self.teardown_calls = 0
        self.input_restored_at_teardown = False
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=32.0,
        )

    def setup(self) -> None:
        self.setup_calls += 1
        torch.manual_seed(42)
        self.input = torch.linspace(-1.0, 1.0, 32, device=self.device).reshape(4, 8)
        self.original_input = self.input.clone()

    def benchmark_fn(self) -> None:
        if self.input is None:
            raise RuntimeError("setup() must run first")
        self.benchmark_calls += 1
        self.output = torch.sin(self.input).mul(2.0)
        if self.benchmark_calls == 2:
            raise RuntimeError("injected perturbed-rerun failure")

    def capture_verification_payload(self) -> None:
        if self.input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run first")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output,
            batch_size=self.input.shape[0],
            output_tolerance=(1e-5, 1e-5),
        )

    def get_workload_metadata(self) -> WorkloadMetadata:
        return self._workload

    def teardown(self) -> None:
        self.teardown_calls += 1
        if self.input is not None and self.original_input is not None:
            self.input_restored_at_teardown = torch.equal(self.input, self.original_input)
        self.input = None
        self.output = None
        self.original_input = None
        super().teardown()


class _PayloadClearingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Real CPU workload whose tolerance exists only before teardown."""

    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.setup_calls = 0
        self.benchmark_calls = 0
        self.teardown_calls = 0
        self.live_tolerance_reads = 0
        self.payload_present_at_teardown: list[bool] = []
        self.register_workload_metadata(
            requests_per_iteration=4.0,
            samples_per_iteration=4.0,
        )

    def setup(self) -> None:
        self.setup_calls += 1
        self.input = torch.randn(4, 8, device=self.device)
        self.output = None

    def benchmark_fn(self) -> None:
        if self.input is None:
            raise RuntimeError("setup() must run first")
        self.benchmark_calls += 1
        self.output = torch.sin(self.input).mul(2.0)

    def capture_verification_payload(self) -> None:
        if self.input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run first")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output,
            batch_size=self.input.shape[0],
            output_tolerance=(1e-5, 1e-5),
        )

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._verification_payload is not None:
            self.live_tolerance_reads += 1
        return VerificationPayloadMixin.get_output_tolerance(self)

    def teardown(self) -> None:
        self.teardown_calls += 1
        self.payload_present_at_teardown.append(self._verification_payload is not None)
        self.input = None
        self.output = None
        self._verification_payload = None
        super().teardown()


class _MissingLiveToleranceBenchmark(_PayloadClearingBenchmark):
    def get_output_tolerance(self) -> tuple[float, float]:
        if self._verification_payload is not None:
            self.live_tolerance_reads += 1
        raise RuntimeError("declared live tolerance unavailable")


def test_verify_pair_runs_real_ch05_jitter_in_a_live_lifecycle(tmp_path: Path) -> None:
    baseline = BaselineAIBenchmark()
    optimized = _TrackedOptimizedAI()
    baseline.device = torch.device("cpu")
    optimized.device = torch.device("cpu")

    result = make_runner(tmp_path).verify_pair(baseline, optimized, VerifyConfig())

    # Main verification, fresh-input verification, then an independently owned
    # live jitter lifecycle containing its initial and perturbed executions.
    assert optimized.setup_calls == 3
    assert optimized.benchmark_calls == 4
    assert optimized.teardown_calls == 3
    assert optimized.created_input_path is not None
    assert not optimized.created_input_path.exists()
    assert result.passed
    assert result.reason is None
    assert not (result.details or {}).get("warnings")


def test_live_jitter_cleans_up_once_after_perturbed_rerun_failure(tmp_path: Path) -> None:
    benchmark = _FailingJitterBenchmark()

    passed, reason = make_runner(tmp_path)._run_live_jitter_check(
        benchmark,
        VerifyConfig(),
    )

    assert not passed
    assert reason is not None
    assert "injected perturbed-rerun failure" in reason
    assert benchmark.setup_calls == 1
    assert benchmark.benchmark_calls == 2
    assert benchmark.teardown_calls == 1
    assert benchmark.input_restored_at_teardown
    assert benchmark.input is None
    assert benchmark.output is None
    assert benchmark.original_input is None


def test_verify_pair_snapshots_payload_tolerance_before_cleanup(tmp_path: Path) -> None:
    baseline = _PayloadClearingBenchmark()
    optimized = _PayloadClearingBenchmark()

    result = make_runner(tmp_path).verify_pair(baseline, optimized, VerifyConfig())

    assert result.passed
    assert result.reason is None
    assert baseline.live_tolerance_reads == 1
    assert optimized.live_tolerance_reads == 1
    assert baseline.payload_present_at_teardown == [True]
    assert optimized.payload_present_at_teardown == [True, True, True]
    assert baseline._verification_payload is None
    assert optimized._verification_payload is None
    assert baseline.input is None
    assert optimized.input is None


def test_missing_live_tolerance_keeps_semantic_failure_after_cleanup(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    missing_baseline = _MissingLiveToleranceBenchmark()

    baseline_result = runner.verify_baseline(missing_baseline, VerifyConfig())

    assert not baseline_result.passed
    assert baseline_result.reason == "missing_output_tolerance"
    assert baseline_result.details == {"error": "declared live tolerance unavailable"}
    assert missing_baseline.teardown_calls == 1
    assert missing_baseline._verification_payload is None

    good_baseline = _PayloadClearingBenchmark()
    assert runner.verify_baseline(good_baseline, VerifyConfig()).passed
    missing_optimized = _MissingLiveToleranceBenchmark()

    optimized_result = runner.verify_optimized(missing_optimized, VerifyConfig())

    assert not optimized_result.passed
    assert optimized_result.reason == "missing_output_tolerance"
    assert optimized_result.details == {"error": "declared live tolerance unavailable"}
    assert missing_optimized.teardown_calls == 1
    assert missing_optimized._verification_payload is None
