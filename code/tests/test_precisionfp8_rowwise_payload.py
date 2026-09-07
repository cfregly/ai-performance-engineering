from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from ch13.optimized_precisionfp8_rowwise import (
    OptimizedFP8RowwiseBenchmark,
    _capture_rowwise_verification_output,
)
from ch13.optimized_precisionfp8_rowwise_gw_hp import (
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
        self.seen_shape: tuple[int, ...] | None = None
        self.grad_enabled: bool | None = None
        self.inference_mode_enabled: bool | None = None

    def forward(self, value: torch.Tensor) -> object:
        self.seen_shape = tuple(value.shape)
        self.grad_enabled = torch.is_grad_enabled()
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        return self.output


@pytest.mark.parametrize(
    "benchmark_type",
    [OptimizedFP8RowwiseBenchmark, OptimizedFP8RowwiseGWHpBenchmark],
)
def test_rowwise_payload_runs_full_input_outside_inference_mode_and_dequantizes(
    benchmark_type: type[OptimizedFP8RowwiseBenchmark] | type[OptimizedFP8RowwiseGWHpBenchmark],
) -> None:
    full_output = torch.arange(63, dtype=torch.float16).reshape(9, 7)
    subclass_output = _Float8LikeOutput(full_output)
    model = _CaptureModel(subclass_output)
    benchmark = benchmark_type()
    benchmark.model = model
    benchmark._verify_input = torch.randn(9, 7)
    benchmark._verify_input_fp16 = benchmark._verify_input.half()
    benchmark._verify_output_buffer = torch.empty(4, 5, dtype=torch.float32)
    benchmark.parameter_count = 0

    with torch.inference_mode():
        benchmark.capture_verification_payload()

    assert model.seen_shape == (9, 7)
    assert model.grad_enabled is False
    assert model.inference_mode_enabled is False
    assert subclass_output.dequantized is True
    torch.testing.assert_close(
        benchmark.output,
        full_output[:4, :5].float(),
        rtol=0,
        atol=0,
    )
    assert benchmark.get_output_tolerance() == (0.25, 2.0)


@pytest.mark.parametrize("recipe_name", ["ROWWISE", "ROWWISE_WITH_GW_HP"])
def test_pinned_torchao_rowwise_recipe_capture_avoids_axiswise_inference_path(
    recipe_name: str,
) -> None:
    torchao_config = pytest.importorskip("torchao.float8.config")
    torchao_utils = pytest.importorskip("torchao.float8.float8_linear_utils")

    config = torchao_config.Float8LinearConfig.from_recipe_name(
        getattr(torchao_config.Float8LinearRecipeName, recipe_name)
    )
    config = replace(config, emulate=True)
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 16),
    ).half()
    model = torchao_utils.convert_to_float8_training(model, config=config)
    verify_input = torch.randn(8, 16, dtype=torch.float16)
    output_buffer = torch.empty(8, 16, dtype=torch.float32)

    with torch.inference_mode():
        output = _capture_rowwise_verification_output(model, verify_input, output_buffer)

    assert type(output) is torch.Tensor
    assert output.shape == (8, 16)
    assert torch.isfinite(output).all()
