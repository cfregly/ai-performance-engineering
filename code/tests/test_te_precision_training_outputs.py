from __future__ import annotations

import os
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
import torch.nn as nn

from ch13.baseline_precisionfp8_te import BaselineTEFP8Benchmark
from ch13.optimized_precisionfp8_te import OptimizedTEFP8Benchmark
from core.benchmark.verification import get_output_tolerances
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.verify_runner import VerifyRunner

EXPECTED_OUTPUT_TOLERANCES = {
    "prediction": (0.4, 1.0),
    "parameter.fc1.weight": (0.001, 0.00075),
    "parameter.fc1.bias": (0.001, 0.00005),
    "parameter.fc2.weight": (0.001, 0.00075),
    "parameter.fc2.bias": (0.001, 0.00005),
}


def _assert_calibrated_output_policy(benchmark, output) -> None:
    tolerances = get_output_tolerances(benchmark)
    assert benchmark.get_output_tolerance() == (0.4, 1.0)
    assert tolerances == EXPECTED_OUTPUT_TOLERANCES
    assert VerificationPayloadMixin.get_output_tolerances(benchmark) == tolerances
    assert set(tolerances) == set(output)
    runner = VerifyRunner()
    assert runner.compare_perf_outputs(output, output, tolerances).passed

    for name, expected in output.items():
        tolerance = {name: tolerances[name]}
        zeroed = expected.clone()
        zeroed.zero_()
        assert not runner.compare_perf_outputs(
            {name: expected}, {name: zeroed}, tolerance
        ).passed

        perturbed = expected.clone()
        index = int(torch.argmax(expected.reshape(-1).abs()))
        reference = float(expected.reshape(-1)[index])
        rtol, atol = tolerances[name]
        perturbation = 2.0 * (atol + rtol * abs(reference))
        perturbed.reshape(-1)[index].add_(perturbation)
        assert not runner.compare_perf_outputs(
            {name: expected}, {name: perturbed}, tolerance
        ).passed


@pytest.mark.parametrize(
    "benchmark_type",
    [BaselineTEFP8Benchmark, OptimizedTEFP8Benchmark],
)
def test_training_payload_contains_target_prediction_and_every_named_parameter(
    benchmark_type,
) -> None:
    benchmark = benchmark_type()
    benchmark.model = nn.Sequential(
        OrderedDict(
            (
                ("fc1", nn.Linear(3, 4)),
                ("activation", nn.GELU()),
                ("fc2", nn.Linear(4, 2)),
            )
        )
    )
    verify_input = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    verify_target = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 100
    with torch.no_grad():
        for parameter in benchmark.model.parameters():
            parameter.fill_(0.25)
    prediction = benchmark.model(verify_input).detach()
    benchmark._verify_input = verify_input
    benchmark._verify_target = verify_target
    benchmark.output = prediction
    benchmark._verify_output_buffer = torch.empty_like(prediction)
    benchmark.parameter_count = sum(parameter.numel() for parameter in benchmark.model.parameters())

    parameter_snapshots = {
        name: parameter.detach().clone()
        for name, parameter in benchmark.model.named_parameters()
    }
    benchmark.capture_verification_payload()

    inputs = benchmark.get_verify_inputs()
    assert set(inputs) == {"input", "target"}
    assert inputs["input"].data_ptr() == verify_input.data_ptr()
    assert inputs["target"].data_ptr() == verify_target.data_ptr()
    torch.testing.assert_close(inputs["input"], verify_input)
    torch.testing.assert_close(inputs["target"], verify_target)
    signature = benchmark.get_input_signature()
    assert signature.shapes["target"] == tuple(verify_target.shape)
    assert signature.dtypes["target"] == "float32"

    output = benchmark.get_verify_output()
    assert set(output) == {
        "prediction",
        *(f"parameter.{name}" for name in parameter_snapshots),
    }
    torch.testing.assert_close(output["prediction"], prediction)
    assert all(tensor.device.type == "cpu" for tensor in output.values())
    for name, expected in parameter_snapshots.items():
        torch.testing.assert_close(output[f"parameter.{name}"], expected)
    _assert_calibrated_output_policy(benchmark, output)
    corrupted = {name: tensor.clone() for name, tensor in output.items()}
    corrupted[next(name for name in corrupted if name.startswith("parameter."))].add_(1)
    assert not VerifyRunner().compare_perf_outputs(output, corrupted, (0.0, 0.0)).passed

    with torch.no_grad():
        for parameter in benchmark.model.parameters():
            parameter.add_(10)
    retained = benchmark.get_verify_output()
    for name, expected in parameter_snapshots.items():
        torch.testing.assert_close(retained[f"parameter.{name}"], expected)

    benchmark.teardown()
    retained_inputs = benchmark.get_verify_inputs()
    assert all(tensor.device.type == "cpu" for tensor in retained_inputs.values())
    torch.testing.assert_close(retained_inputs["target"], verify_target)
    assert all(tensor.device.type == "cpu" for tensor in benchmark.get_verify_output().values())


def test_fp16_baseline_metrics_do_not_label_the_run_as_fp32() -> None:
    benchmark = BaselineTEFP8Benchmark()
    benchmark._last_elapsed_ms = 1.25

    metrics = benchmark.get_custom_metrics()

    assert metrics["precision.reduced_ms"] == 1.25
    assert metrics["precision.theoretical_storage_reduction_factor"] == 2.0
    assert "precision.fp32_ms" not in metrics


def _require_real_te218_cuda() -> None:
    if os.environ.get("AISP_RUN_TE218_CUDA_TESTS") != "1":
        pytest.skip("set AISP_RUN_TE218_CUDA_TESTS=1 for the real TE2.18 CUDA harness test")
    if not torch.cuda.is_available():
        pytest.skip("real Transformer Engine test requires CUDA")
    try:
        te_version = version("transformer_engine")
    except PackageNotFoundError:
        pytest.skip("real Transformer Engine test requires transformer_engine")
    if not te_version.startswith("2.18."):
        pytest.skip(f"real Transformer Engine test requires 2.18.x, found {te_version}")


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "benchmark_type",
    [BaselineTEFP8Benchmark, OptimizedTEFP8Benchmark],
)
def test_real_te218_cuda_harness_retains_full_training_output(benchmark_type) -> None:
    _require_real_te218_cuda()
    from core.harness.benchmark_harness import (
        BenchmarkHarness,
        BenchmarkMode,
        ExecutionMode,
    )

    benchmark = benchmark_type()
    config = benchmark.get_config()
    config.device = torch.device("cuda")
    config.use_subprocess = False
    config.execution_mode = ExecutionMode.THREAD
    config.enable_profiling = False
    config.enable_memory_tracking = False
    config.enforce_environment_validation = False
    config.allow_virtualization = True
    config.clear_l2_cache = False
    config.monitor_gpu_state = False
    config.track_memory_allocations = False
    config.single_gpu = True
    assert config.backend_policy == "fp32_strict"
    assert config.iterations == 50
    assert config.warmup == 10
    result = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config).benchmark(benchmark)

    assert not result.errors, result.errors
    assert result.timing.iterations == 50
    assert result.timing.warmup_iterations == 10
    output = benchmark.get_verify_output()
    assert "prediction" in output
    parameter_outputs = {
        name.removeprefix("parameter."): tensor
        for name, tensor in output.items()
        if name.startswith("parameter.")
    }
    assert parameter_outputs
    assert sum(tensor.numel() for tensor in parameter_outputs.values()) == benchmark.parameter_count
    assert all(torch.isfinite(tensor).all() for tensor in output.values())
    _assert_calibrated_output_policy(benchmark, output)
    retained_inputs = benchmark.get_verify_inputs()
    assert set(retained_inputs) == {"input", "target"}
    assert all(tensor.device.type == "cpu" for tensor in retained_inputs.values())
    assert all(tensor.device.type == "cpu" for tensor in output.values())


def _run_real_te218_step_with_target_delta(benchmark_type, target_delta: float):
    torch.manual_seed(7391)
    torch.cuda.manual_seed_all(7391)
    benchmark = benchmark_type()
    benchmark.batch_size = 16
    benchmark.hidden_dim = 128
    benchmark.setup()
    try:
        if isinstance(benchmark, BaselineTEFP8Benchmark):
            training_target = benchmark.targets
        else:
            training_target = benchmark.target_pool[0]
        assert training_target is not None
        live_target = benchmark.get_verify_inputs()["target"]
        assert live_target.data_ptr() == training_target.data_ptr()
        initial_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in benchmark.model.named_parameters()
        }
        with torch.no_grad():
            live_target.add_(target_delta)
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        final_parameters = {
            name.removeprefix("parameter."): tensor
            for name, tensor in benchmark.get_verify_output().items()
            if name.startswith("parameter.")
        }
        return initial_parameters, final_parameters
    finally:
        benchmark.teardown()


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "benchmark_type",
    [BaselineTEFP8Benchmark, OptimizedTEFP8Benchmark],
)
def test_real_te218_live_target_changes_post_sgd_parameters(benchmark_type) -> None:
    _require_real_te218_cuda()
    control_initial, control_final = _run_real_te218_step_with_target_delta(
        benchmark_type, 0.0
    )
    changed_initial, changed_final = _run_real_te218_step_with_target_delta(
        benchmark_type, 8.0
    )

    assert control_initial.keys() == changed_initial.keys()
    assert control_final.keys() == changed_final.keys() == control_initial.keys()
    for name in control_initial:
        torch.testing.assert_close(control_initial[name], changed_initial[name], rtol=0, atol=0)
    assert any(
        not torch.equal(control_final[name], changed_final[name])
        for name in control_final
    )
