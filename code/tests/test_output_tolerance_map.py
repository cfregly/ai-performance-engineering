"""Real CPU coverage for exact-keyed output tolerance policies."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from core.benchmark.verification import (
    get_output_tolerances,
    normalize_output_tolerances,
    output_tolerances_from_dict,
    output_tolerances_to_dict,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.verified_comparison import VerifiedComparisonError
from core.benchmark.verify_runner import VerifyConfig, VerifyRunner
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    ExecutionMode,
    _parse_subprocess_output_tolerances,
    compare_benchmarks,
)

_OUTPUT_TOLERANCES = {
    "prediction": (0.0, 1e-6),
    "parameter.weight": (0.0, 1e-6),
}


class _TrainingOutputBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(
        self,
        *,
        parameter_step: float = 0.0,
        prediction_offset: float = 0.0,
        output_tolerances: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.parameter_step = parameter_step
        self.prediction_offset = prediction_offset
        self.declared_output_tolerances = output_tolerances or dict(_OUTPUT_TOLERANCES)
        self.input_tensor: torch.Tensor | None = None
        self.prediction: torch.Tensor | None = None
        self.weight: torch.Tensor | None = None
        self.full_output: dict[str, torch.Tensor] | None = None
        self.benchmark_calls = 0
        self.register_workload_metadata(samples_per_iteration=8)

    def setup(self) -> None:
        self.input_tensor = torch.arange(8, dtype=torch.float32)
        self.weight = torch.tensor([0.02], dtype=torch.float32)

    def benchmark_fn(self) -> torch.Tensor:
        if self.input_tensor is None or self.weight is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self.benchmark_calls += 1
        self.prediction = self.input_tensor * 2.0 + self.prediction_offset
        self.weight.add_(self.parameter_step)
        return self.prediction

    def capture_verification_payload(self) -> None:
        if self.input_tensor is None or self.prediction is None or self.weight is None:
            raise RuntimeError("timed execution did not produce training outputs")
        self.full_output = {
            "prediction": self.prediction.detach().clone(),
            "parameter.weight": self.weight.detach().clone(),
        }
        self._set_verification_payload(
            inputs={"input": self.input_tensor},
            output=self.prediction,
            batch_size=self.input_tensor.shape[0],
            parameter_count=self.weight.numel(),
            output_tolerance=(0.5, 5.0),
            output_tolerances=self.declared_output_tolerances,
        )

    def get_verify_output(self) -> dict[str, torch.Tensor]:
        if self.full_output is None:
            raise RuntimeError("capture_verification_payload() must run first")
        return {name: tensor.detach().clone() for name, tensor in self.full_output.items()}

    def validate_result(self) -> None:
        return None


def _cpu_harness() -> BenchmarkHarness:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=2,
        warmup=5,
        use_subprocess=False,
        execution_mode=ExecutionMode.THREAD,
        enable_profiling=False,
        enable_memory_tracking=False,
        enable_cleanup=False,
        lock_gpu_clocks=False,
        enforce_environment_validation=False,
        detect_setup_precomputation=False,
        adaptive_iterations=False,
        clear_compile_cache=False,
        measurement_timeout_seconds=15,
    )
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)


def _verify_config() -> VerifyConfig:
    return VerifyConfig(
        skip_jitter_check=True,
        skip_fresh_input_check=True,
        skip_workload_check=True,
    )


def test_real_cpu_comparison_records_exact_keyed_tolerances() -> None:
    baseline = _TrainingOutputBenchmark()
    optimized = _TrainingOutputBenchmark()

    result = compare_benchmarks(baseline, optimized, harness=_cpu_harness())

    assert baseline.benchmark_calls > 5
    assert optimized.benchmark_calls > 5
    assert result["verification"]["passed"] is True
    assert result["verification"]["output_tolerances"] == {
        "parameter.weight": {"rtol": 0.0, "atol": 1e-6},
        "prediction": {"rtol": 0.0, "atol": 1e-6},
    }


@pytest.mark.parametrize(
    "optimized",
    [
        pytest.param(_TrainingOutputBenchmark(parameter_step=0.01), id="parameter.weight"),
        pytest.param(_TrainingOutputBenchmark(prediction_offset=0.01), id="prediction"),
    ],
)
def test_real_cpu_comparison_rejects_each_corrupted_output_with_loose_global_tolerance(
    optimized: _TrainingOutputBenchmark,
) -> None:
    baseline = _TrainingOutputBenchmark()

    with pytest.raises(VerifiedComparisonError, match="output_mismatch") as exc_info:
        compare_benchmarks(baseline, optimized, harness=_cpu_harness())

    assert exc_info.value.code == "output_mismatch"
    assert exc_info.value.evidence["passed"] is False
    assert exc_info.value.evidence["output_tolerances"]["parameter.weight"] == {
        "rtol": 0.0,
        "atol": 1e-6,
    }


def test_real_cpu_comparison_rejects_different_maps() -> None:
    baseline = _TrainingOutputBenchmark()
    optimized_tolerances = dict(_OUTPUT_TOLERANCES)
    optimized_tolerances["parameter.weight"] = (0.0, 2e-6)
    optimized = _TrainingOutputBenchmark(output_tolerances=optimized_tolerances)

    with pytest.raises(VerifiedComparisonError, match="output_tolerances_mismatch") as exc_info:
        compare_benchmarks(baseline, optimized, harness=_cpu_harness())

    assert exc_info.value.code == "output_tolerances_mismatch"


def test_real_cpu_comparison_requires_exact_output_key_coverage() -> None:
    incomplete = {"prediction": (0.0, 1e-6)}

    with pytest.raises(VerifiedComparisonError, match="must exactly cover") as exc_info:
        compare_benchmarks(
            _TrainingOutputBenchmark(output_tolerances=incomplete),
            _TrainingOutputBenchmark(output_tolerances=incomplete),
            harness=_cpu_harness(),
        )

    assert exc_info.value.code == "output_tolerances_mismatch"


def test_verify_runner_roundtrips_map_and_rejects_corrupted_parameter(tmp_path: Path) -> None:
    runner = VerifyRunner(cache_dir=tmp_path / "cache")
    baseline = _TrainingOutputBenchmark()

    baseline_result = runner.verify_baseline(baseline, config=_verify_config())
    assert baseline_result.passed is True
    assert baseline_result.signature_hash is not None

    cached = runner.cache.get(baseline_result.signature_hash)
    assert cached is not None
    assert cached.output_tolerances == _OUTPUT_TOLERANCES

    optimized_result = runner.verify_optimized(
        _TrainingOutputBenchmark(parameter_step=0.01),
        config=_verify_config(),
    )
    assert optimized_result.passed is False
    assert optimized_result.reason == "output_mismatch"
    assert optimized_result.comparison_details is not None
    assert optimized_result.comparison_details.output_tolerances == _OUTPUT_TOLERANCES


def test_verify_runner_rejects_non_tensor_entries_in_actual_output_dict(tmp_path: Path) -> None:
    class MixedOutputBenchmark(_TrainingOutputBenchmark):
        def get_verify_output(self) -> dict[str, object]:
            outputs: dict[str, object] = super().get_verify_output()
            outputs["parameter.bias"] = None
            return outputs

    result = VerifyRunner(cache_dir=tmp_path / "cache").verify_baseline(
        MixedOutputBenchmark(),
        config=_verify_config(),
    )

    assert result.passed is False
    assert "get_verify_output()['parameter.bias'] must be a torch.Tensor" in str(result.reason)


def test_real_subprocess_transports_map_through_json(tmp_path: Path) -> None:
    module_path = tmp_path / "subprocess_output_tolerances.py"
    module_path.write_text(
        """
import torch
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark

class ChildBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.input_tensor = None
        self.output = None
        self.full_output = None
        self.output_value = 1.0
        self.policy_atol = 1e-6
        self.use_map = True
        self.should_fail = False
        self.register_workload_metadata(samples_per_iteration=4)
    def setup(self):
        self.input_tensor = torch.arange(4, dtype=torch.float32)
    def benchmark_fn(self):
        if self.should_fail:
            raise RuntimeError('intentional second-run failure')
        self.output = self.input_tensor + self.output_value
        return self.output
    def capture_verification_payload(self):
        self.full_output = {
            'prediction': self.output.detach().clone(),
            'parameter.weight': torch.tensor([0.02]),
        }
        self._set_verification_payload(
            inputs={'input': self.input_tensor}, output=self.output,
            batch_size=4, parameter_count=1, output_tolerance=(0.5, 5.0),
            output_tolerances=(
                {
                    'prediction': (0.0, self.policy_atol),
                    'parameter.weight': (0.0, 1e-6),
                }
                if self.use_map else None
            ),
        )
    def get_verify_output(self):
        if self.full_output is None:
            raise RuntimeError('capture_verification_payload() must run first')
        return {name: tensor.detach().clone() for name, tensor in self.full_output.items()}
    def validate_result(self):
        return None

class DeclaredChildBenchmark(ChildBenchmark):
    def get_output_tolerances(self):
        if not self.use_map:
            return None
        return {
            'prediction': (0.0, self.policy_atol),
            'parameter.weight': (0.0, 1e-6),
        }
""",
        encoding="utf-8",
    )
    module_name = "_test_subprocess_output_tolerances"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    config = _cpu_harness().config
    config.use_subprocess = True
    config.execution_mode = ExecutionMode.SUBPROCESS
    config.subprocess_stderr_dir = str(tmp_path / "logs")
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    # A first-run payload-only mixin has no policy until its post-measurement
    # capture hook. It remains a valid source of a strict child receipt.
    payload_only_benchmark = module.ChildBenchmark()
    payload_only_result = harness.benchmark(payload_only_benchmark)
    assert not payload_only_result.errors
    assert get_output_tolerances(payload_only_benchmark) == _OUTPUT_TOLERANCES

    # Torchrun coordinators receive actual child output and the required global
    # tolerance but do not execute capture_verification_payload() locally. An
    # absent optional map receipt must therefore remain a legitimate None.
    transported_parent = module.ChildBenchmark()
    transported_parent.use_map = False
    transported_parent_result = harness.benchmark(transported_parent)
    assert not transported_parent_result.errors
    assert transported_parent._verification_payload is None
    assert isinstance(transported_parent._subprocess_verify_output, dict)
    assert transported_parent._subprocess_output_tolerance == (0.5, 5.0)
    assert transported_parent._subprocess_output_tolerances is None
    vars(transported_parent).pop("_subprocess_output_tolerances")
    assert get_output_tolerances(transported_parent) is None

    benchmark = module.DeclaredChildBenchmark()
    result = harness.benchmark(benchmark)

    assert not result.errors
    assert result.execution_process_ids[0] != __import__("os").getpid()
    transported = get_output_tolerances(benchmark)
    assert transported == _OUTPUT_TOLERANCES
    serialized = json.loads(json.dumps(output_tolerances_to_dict(transported)))
    assert output_tolerances_from_dict(serialized) == _OUTPUT_TOLERANCES

    actual_receipt = {"output_tolerances": serialized}
    has_policy, parsed = _parse_subprocess_output_tolerances(
        copy.deepcopy(actual_receipt),
        declared_output_tolerances=_OUTPUT_TOLERANCES,
    )
    assert has_policy is True
    assert parsed == _OUTPUT_TOLERANCES

    missing_receipt = copy.deepcopy(actual_receipt)
    missing_receipt.pop("output_tolerances")
    none_receipt = copy.deepcopy(actual_receipt)
    none_receipt["output_tolerances"] = None
    different_receipt = copy.deepcopy(actual_receipt)
    different_receipt["output_tolerances"]["parameter.weight"]["atol"] = 2e-6
    for corrupted_receipt in (missing_receipt, none_receipt, different_receipt):
        with pytest.raises(ValueError, match="pre-dispatch"):
            _parse_subprocess_output_tolerances(
                corrupted_receipt,
                declared_output_tolerances=_OUTPUT_TOLERANCES,
            )

    runner_benchmark = module.DeclaredChildBenchmark()
    subprocess_result = harness.benchmark(runner_benchmark)
    assert not subprocess_result.errors
    assert "_subprocess_output_tolerances" in vars(runner_benchmark)
    runner_benchmark.output_value = 7.0
    runner_benchmark.policy_atol = 3e-6
    runner = VerifyRunner(cache_dir=tmp_path / "reused-instance-cache")
    runner_result = runner.verify_baseline(runner_benchmark, config=_verify_config())
    assert runner_result.passed is True
    assert "_subprocess_verify_output" not in vars(runner_benchmark)
    assert "_subprocess_output_tolerances" not in vars(runner_benchmark)
    assert runner_result.signature_hash is not None
    reused_golden = runner.cache.get(runner_result.signature_hash)
    assert reused_golden is not None
    assert reused_golden.output_tolerances == {
        "prediction": (0.0, 3e-6),
        "parameter.weight": (0.0, 1e-6),
    }

    benchmark.use_map = False
    no_map_result = harness.benchmark(benchmark)
    assert not no_map_result.errors
    assert "_subprocess_output_tolerances" in vars(benchmark)
    assert benchmark._subprocess_output_tolerances is None
    assert get_output_tolerances(benchmark) is None

    benchmark.output_value = 4.0
    benchmark.policy_atol = 2e-6
    benchmark.use_map = True
    thread_result = _cpu_harness().benchmark(benchmark)
    assert not thread_result.errors
    assert "_subprocess_verify_output" not in vars(benchmark)
    assert "_subprocess_output_tolerances" not in vars(benchmark)
    assert torch.equal(
        benchmark.get_verify_output()["prediction"],
        torch.arange(4, dtype=torch.float32) + 4.0,
    )
    assert get_output_tolerances(benchmark) == {
        "prediction": (0.0, 2e-6),
        "parameter.weight": (0.0, 1e-6),
    }

    failed_benchmark = module.ChildBenchmark()
    first_result = harness.benchmark(failed_benchmark)
    assert not first_result.errors
    failed_benchmark.should_fail = True
    failed_result = _cpu_harness().benchmark(failed_benchmark)
    assert any("intentional second-run failure" in error for error in failed_result.errors)
    assert "_subprocess_verify_output" not in vars(failed_benchmark)
    assert "_subprocess_output_tolerance" not in vars(failed_benchmark)
    assert "_subprocess_output_tolerances" not in vars(failed_benchmark)
    assert "_subprocess_input_signature" not in vars(failed_benchmark)
    with pytest.raises(RuntimeError, match="capture_verification_payload"):
        failed_benchmark.get_verify_output()


@pytest.mark.parametrize(
    "value, error",
    [
        ({"output": (float("nan"), 0.0)}, "finite and nonnegative"),
        ({"output": (-1.0, 0.0)}, "finite and nonnegative"),
        ({"output": [0.0, 0.0]}, r"must be an \(rtol, atol\) tuple"),
        ({}, "must not return an empty dictionary"),
    ],
)
def test_tolerance_map_rejects_invalid_numeric_policy(value: object, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        normalize_output_tolerances(value)
