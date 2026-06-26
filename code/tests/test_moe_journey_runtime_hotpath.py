from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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


def test_level4_grouped_moe_batches_expert_count_metadata_reads() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "moe_optimization_journey"
        / "level4_triton.py"
    ).read_text(encoding="utf-8")
    grouped_section = source.split("class GroupedMoEExperts", maxsplit=1)[1].split(
        "class TritonMoELayer",
        maxsplit=1,
    )[0]

    assert "expert_counts_cpu = [int(count) for count in expert_counts.detach().cpu().tolist()]" in grouped_section
    assert "expert_offsets_cpu = [int(offset) for offset in expert_offsets.detach().cpu().tolist()]" in grouped_section
    assert "for expert_id, (start, count) in enumerate(zip(expert_offsets_cpu, expert_counts_cpu))" in grouped_section
    assert "expert_offsets[expert_id].item()" not in grouped_section
    assert "expert_counts[expert_id].item()" not in grouped_section


def test_level4_grouped_moe_overwrites_sorted_expert_output() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "moe_optimization_journey"
        / "level4_triton.py"
    ).read_text(encoding="utf-8")
    grouped_section = source.split("class GroupedMoEExperts", maxsplit=1)[1].split(
        "class TritonMoELayer",
        maxsplit=1,
    )[0]
    expert_loop_section = grouped_section.split("# Process each expert's tokens", maxsplit=1)[1].split(
        "# Apply weights",
        maxsplit=1,
    )[0]

    assert "output = torch.empty_like(sorted_x)" in expert_loop_section
    assert "torch.zeros_like(sorted_x)" not in expert_loop_section


def test_triton_fused_moe_benchmark_reuses_precomputed_max_tokens() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "moe_optimization_journey"
        / "triton_fused_moe.py"
    ).read_text(encoding="utf-8")
    function_section = source.split("def triton_fused_moe", maxsplit=1)[1].split(
        "def benchmark_triton_moe",
        maxsplit=1,
    )[0]
    benchmark_section = source.split("def benchmark_triton_moe", maxsplit=1)[1]

    assert "max_tokens: int | None = None" in function_section
    assert "if max_tokens is None:" in function_section
    assert "max_tokens = int(counts.max().item())" in benchmark_section
    assert benchmark_section.count("max_tokens=max_tokens") == 3


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
