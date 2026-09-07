from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from ch13 import optimized_precisionfp8 as tensorwise_module
from ch13 import optimized_precisionfp8_rowwise as rowwise_module
from ch13 import optimized_precisionfp8_rowwise_gw_hp as gw_hp_module
from ch13.baseline_precisionfp8 import BaselinePrecisionFP8Benchmark
from ch13.optimized_precisionfp8 import (
    OptimizedFP8Benchmark,
    _capture_fp8_verification_output,
)
from ch13.optimized_precisionfp8_rowwise import OptimizedFP8RowwiseBenchmark
from ch13.optimized_precisionfp8_rowwise_gw_hp import (
    OptimizedFP8RowwiseGWHpBenchmark,
)

BENCHMARK_TYPES = (
    BaselinePrecisionFP8Benchmark,
    OptimizedFP8Benchmark,
    OptimizedFP8RowwiseBenchmark,
    OptimizedFP8RowwiseGWHpBenchmark,
)
OPTIMIZED_TYPES = (
    OptimizedFP8Benchmark,
    OptimizedFP8RowwiseBenchmark,
    OptimizedFP8RowwiseGWHpBenchmark,
)


class _Float8LikeOutput:
    def __init__(self, high_precision: torch.Tensor) -> None:
        self.high_precision = high_precision
        self.dequantized = False

    def to_original_precision(self) -> torch.Tensor:
        self.dequantized = True
        return self.high_precision


class _CaptureModel(nn.Module):
    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output
        self.seen_input: torch.Tensor | None = None
        self.grad_enabled: bool | None = None
        self.inference_mode_enabled: bool | None = None

    def forward(self, value: torch.Tensor) -> object:
        self.seen_input = value.detach().clone()
        self.grad_enabled = torch.is_grad_enabled()
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        return self.output


class _InputSensitiveModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 1.5


@pytest.mark.parametrize("benchmark_type", BENCHMARK_TYPES)
def test_declared_verification_input_drives_each_fp8_capture(benchmark_type) -> None:
    benchmark = benchmark_type()
    benchmark.model = _InputSensitiveModel()
    benchmark._verify_input = torch.arange(63, dtype=torch.float32).reshape(9, 7)
    benchmark._verify_output_buffer = torch.empty(4, 5, dtype=torch.float32)
    benchmark.parameter_count = 0

    benchmark.capture_verification_payload()
    first_output = benchmark.get_verify_output()

    declared_input = benchmark.get_verify_inputs()["input"]
    with torch.no_grad():
        declared_input.add_(0.5)
    benchmark.capture_verification_payload()
    second_output = benchmark.get_verify_output()

    assert not hasattr(benchmark, "_verify_input_fp16")
    assert not torch.equal(first_output, second_output)
    torch.testing.assert_close(
        second_output,
        (declared_input.half() * 1.5)[:4, :5].float(),
        rtol=0,
        atol=0,
    )
    assert benchmark.get_output_tolerance() == (0.25, 2.0)


@pytest.mark.parametrize("benchmark_type", OPTIMIZED_TYPES)
def test_optimized_payload_converts_full_declared_input_and_dequantizes(
    benchmark_type,
) -> None:
    full_output = torch.arange(63, dtype=torch.float16).reshape(9, 7)
    subclass_output = _Float8LikeOutput(full_output)
    model = _CaptureModel(subclass_output)
    benchmark = benchmark_type()
    benchmark.model = model
    benchmark._verify_input = torch.randn(9, 7, dtype=torch.float32)
    benchmark._verify_output_buffer = torch.empty(4, 5, dtype=torch.float32)
    benchmark.parameter_count = 0

    with torch.inference_mode():
        benchmark.capture_verification_payload()

    assert model.seen_input is not None
    assert model.seen_input.shape == (9, 7)
    assert model.seen_input.dtype is torch.float16
    torch.testing.assert_close(
        model.seen_input,
        benchmark._verify_input.half(),
        rtol=0,
        atol=0,
    )
    assert model.grad_enabled is False
    assert model.inference_mode_enabled is False
    assert subclass_output.dequantized is True
    torch.testing.assert_close(
        benchmark.output,
        full_output[:4, :5].float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "recipe_name",
    ["TENSORWISE", "ROWWISE", "ROWWISE_WITH_GW_HP"],
)
def test_pinned_torchao_recipe_capture_is_sensitive_to_declared_input(
    recipe_name: str,
) -> None:
    torchao_config = pytest.importorskip("torchao.float8.config")
    torchao_utils = pytest.importorskip("torchao.float8.float8_linear_utils")

    config = torchao_config.Float8LinearConfig.from_recipe_name(
        getattr(torchao_config.Float8LinearRecipeName, recipe_name)
    )
    config = replace(config, emulate=True)
    linear = nn.Linear(16, 16, bias=False).half()
    with torch.no_grad():
        linear.weight.copy_(torch.eye(16, dtype=torch.float16))
    model = torchao_utils.convert_to_float8_training(
        nn.Sequential(linear),
        config=config,
    )
    verify_input = torch.linspace(0.125, 2.0, 8 * 16).reshape(8, 16)
    output_buffer = torch.empty(8, 16, dtype=torch.float32)

    with torch.inference_mode():
        first_output = _capture_fp8_verification_output(
            model,
            verify_input,
            output_buffer,
        ).clone()
        verify_input.add_(0.5)
        second_output = _capture_fp8_verification_output(
            model,
            verify_input,
            output_buffer,
        ).clone()

    assert type(first_output) is torch.Tensor
    assert first_output.shape == (8, 16)
    assert torch.isfinite(first_output).all()
    assert torch.isfinite(second_output).all()
    assert not torch.equal(first_output, second_output)


def _small_seeded_run(benchmark_type, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    benchmark = benchmark_type()
    benchmark._device = torch.device("cpu")
    benchmark.batch_size = 4
    benchmark.hidden_dim = 16
    torch.manual_seed(seed)
    try:
        benchmark.setup()
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        return (
            benchmark.get_verify_inputs()["input"].clone(),
            benchmark.get_verify_output(),
        )
    finally:
        benchmark.teardown()


def test_baseline_setup_respects_external_verification_seed() -> None:
    first_input, first_output = _small_seeded_run(
        BaselinePrecisionFP8Benchmark,
        42,
    )
    second_input, second_output = _small_seeded_run(
        BaselinePrecisionFP8Benchmark,
        1042,
    )

    assert not torch.equal(first_input, second_input)
    assert not torch.equal(first_output, second_output)


@pytest.mark.parametrize(
    ("module", "benchmark_type"),
    [
        (tensorwise_module, OptimizedFP8Benchmark),
        (rowwise_module, OptimizedFP8RowwiseBenchmark),
        (gw_hp_module, OptimizedFP8RowwiseGWHpBenchmark),
    ],
)
def test_pinned_torchao_setup_respects_external_verification_seed(
    monkeypatch,
    module,
    benchmark_type,
) -> None:
    torchao_config = pytest.importorskip("torchao.float8.config")
    pytest.importorskip("torchao.float8.float8_linear_utils")

    class _EmulatedConfigFactory:
        @staticmethod
        def from_recipe_name(recipe):
            config = torchao_config.Float8LinearConfig.from_recipe_name(recipe)
            return replace(config, emulate=True)

    monkeypatch.setattr(module, "Float8LinearConfig", _EmulatedConfigFactory)

    baseline_input, baseline_output = _small_seeded_run(
        BaselinePrecisionFP8Benchmark,
        42,
    )
    first_input, first_output = _small_seeded_run(benchmark_type, 42)
    second_input, second_output = _small_seeded_run(benchmark_type, 1042)

    torch.testing.assert_close(first_input, baseline_input, rtol=0, atol=0)
    torch.testing.assert_close(first_output, baseline_output, rtol=0.25, atol=2.0)
    assert not torch.equal(first_input, second_input)
    assert not torch.equal(first_output, second_output)
