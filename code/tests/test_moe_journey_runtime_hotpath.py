from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from labs.moe_optimization_journey import get_config
from labs.moe_optimization_journey.level4_triton import Level4Triton
from labs.moe_optimization_journey.level6_full_stack import Level6FullStack

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _FakeForward:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.calls = 0

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        assert input_ids.is_cuda
        return self.logits


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("benchmark_factory", "model_attr", "input_attr"),
    [
        (Level4Triton, "model", "input_ids"),
        (Level6FullStack, "compiled_model", "static_input"),
    ],
)
def test_moe_journey_timing_events_and_verification_clone_stay_out_of_hot_path(
    benchmark_factory: Callable,
    model_attr: str,
    input_attr: str,
) -> None:
    config = get_config(
        "tiny",
        batch_size=1,
        seq_len=2,
        vocab_size=16,
        warmup_iterations=5,
        benchmark_iterations=1,
    )
    bench = benchmark_factory(config)
    input_ids = torch.zeros((config.batch_size, config.seq_len), device="cuda", dtype=torch.long)
    logits = torch.arange(
        config.batch_size * config.seq_len * config.vocab_size,
        device="cuda",
        dtype=torch.float32,
    ).reshape(config.batch_size, config.seq_len, config.vocab_size).to(torch.bfloat16)
    fake_forward = _FakeForward(logits)

    setattr(bench, model_attr, fake_forward)
    setattr(bench, input_attr, input_ids)
    if isinstance(bench, Level6FullStack):
        bench.model = fake_forward
        bench.input_ids = input_ids
        bench.graph = None
        bench.graph_output = None
    bench.parameter_count = 123

    bench.benchmark_fn()
    torch.cuda.synchronize()
    first_events = bench._timing_events
    assert first_events is not None
    assert bench._pending_events is first_events
    assert fake_forward.calls == 1
    assert bench.output is not None
    assert bench.output.dtype == torch.bfloat16
    assert bench.output.data_ptr() == logits.data_ptr()

    metrics = bench.finalize_iteration_metrics()
    assert metrics is not None
    assert metrics["latency_ms"] >= 0.0

    bench.capture_verification_payload()
    payload = bench._verification_payload
    assert payload.output.dtype == torch.float32
    assert payload.output.data_ptr() != bench.output.data_ptr()

    bench.benchmark_fn()
    torch.cuda.synchronize()
    assert bench._timing_events is first_events
    assert fake_forward.calls == 2
