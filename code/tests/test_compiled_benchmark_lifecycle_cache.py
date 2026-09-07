"""Real CPU regressions for target-owned Dynamo cache lifecycles."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch._dynamo.utils import counters

from ch13.optimized_torchao_quantization_compiled import (
    OptimizedTorchAOQuantizationCompiledBenchmark,
    _compile_torchao_module,
)
from ch17.optimized_pipeline_parallelism import (
    OptimizedPipelineParallelismBenchmark,
    _compile_pipeline_module,
)
from core.harness.benchmark_harness import BaseBenchmark


@pytest.mark.parametrize(
    ("benchmark_factory", "compile_module", "compiled_attribute"),
    [
        (
            OptimizedTorchAOQuantizationCompiledBenchmark,
            _compile_torchao_module,
            "compiled_model",
        ),
        (
            OptimizedPipelineParallelismBenchmark,
            _compile_pipeline_module,
            "_compiled_model",
        ),
    ],
)
def test_compiled_benchmark_teardown_retraces_each_fresh_model(
    benchmark_factory: Callable[[], BaseBenchmark],
    compile_module: Callable[[nn.Module], nn.Module],
    compiled_attribute: str,
) -> None:
    """A fresh model must not reuse a graph bound to the previous parameters."""
    torch.compiler.reset()
    counters.clear()
    outputs: list[torch.Tensor] = []

    try:
        for seed in (42, 1042, 42):
            torch.manual_seed(seed)
            inner = nn.Sequential(
                nn.Linear(8, 16),
                nn.GELU(),
                nn.Linear(16, 8),
            ).eval()
            value = torch.randn(2, 8)
            benchmark = benchmark_factory()
            benchmark.device = torch.device("cpu")
            compiled = compile_module(inner)
            setattr(benchmark, compiled_attribute, compiled)

            try:
                with torch.inference_mode():
                    benchmark.output = compiled(value).clone()
                benchmark.parameter_count = sum(parameter.numel() for parameter in inner.parameters())
                if isinstance(benchmark, OptimizedTorchAOQuantizationCompiledBenchmark):
                    benchmark.model = inner
                    benchmark.data = value
                    benchmark._verify_input = value
                    benchmark._verify_output_buffer = torch.empty_like(benchmark.output)
                else:
                    benchmark._input_data = value
                benchmark.capture_verification_payload()
                outputs.append(benchmark.get_verify_output())
            finally:
                benchmark.teardown()

            assert getattr(benchmark, compiled_attribute) is None
            assert benchmark.output is None
            assert benchmark._verification_payload is None

        assert counters["stats"]["unique_graphs"] == 3
        torch.testing.assert_close(outputs[0], outputs[2], rtol=0.0, atol=0.0)
        assert not torch.equal(outputs[0], outputs[1])
    finally:
        torch.compiler.reset()
