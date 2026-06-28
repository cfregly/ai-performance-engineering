from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from labs.moe_optimization_journey import get_config
from labs.moe_optimization_journey.level4_triton import GroupedMoEExperts, Level4Triton
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

    assert "self._expert_metadata_workspace: Optional[torch.Tensor] = None" in grouped_section
    assert "self._expert_metadata_host: Optional[torch.Tensor] = None" in grouped_section
    assert "self._sorted_output_buffer: Optional[torch.Tensor] = None" in grouped_section
    assert "self._unsorted_output_buffer: Optional[torch.Tensor] = None" in grouped_section
    assert "def _expert_metadata_buffers(self, device: torch.device)" in grouped_section
    assert "def _sorted_output_like(self, sorted_x: torch.Tensor) -> torch.Tensor" in grouped_section
    assert "def _unsorted_output_like(self, output: torch.Tensor) -> torch.Tensor" in grouped_section
    assert "if torch.is_grad_enabled() and sorted_x.requires_grad:" in grouped_section
    assert "if torch.is_grad_enabled() and output.requires_grad:" in grouped_section
    assert "torch.cumsum(expert_counts, dim=0, out=expert_offsets)" in grouped_section
    assert "expert_offsets.sub_(expert_counts)" in grouped_section
    assert "expert_metadata[1].copy_(expert_counts)" in grouped_section
    assert "expert_metadata_host.copy_(expert_metadata)" in grouped_section
    assert "expert_offsets_cpu, expert_counts_cpu = expert_metadata_host.tolist()" in grouped_section
    assert "for expert_id, (start, count) in enumerate(zip(expert_offsets_cpu, expert_counts_cpu))" in grouped_section
    assert "expert_counts.detach().cpu().tolist()" not in grouped_section
    assert "expert_offsets.detach().cpu().tolist()" not in grouped_section
    assert "expert_offsets = torch.cumsum(expert_counts, dim=0) - expert_counts" not in grouped_section
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
    apply_weights_section = grouped_section.split("# Apply weights", maxsplit=1)[1].split(
        "# Unsort back to original order",
        maxsplit=1,
    )[0]
    unsort_section = grouped_section.split("# Unsort back to original order", maxsplit=1)[1].split(
        "# Sum over top-k experts",
        maxsplit=1,
    )[0]

    assert "output = self._sorted_output_like(sorted_x)" in expert_loop_section
    assert "output = torch.empty_like(sorted_x)" not in expert_loop_section
    assert "torch.zeros_like(sorted_x)" not in expert_loop_section
    assert "F.silu(gate, inplace=True)" in expert_loop_section
    assert "gate.mul_(up)" in expert_loop_section
    assert "hidden = gate * up" not in expert_loop_section
    assert "output.mul_(sorted_weights.unsqueeze(-1))" in apply_weights_section
    assert "output = output * sorted_weights.unsqueeze(-1)" not in grouped_section
    assert "unsorted_output = self._unsorted_output_like(output)" in unsort_section
    assert "unsorted_output.index_copy_(0, sorted_indices, output)" in unsort_section
    assert "torch.argsort(sorted_indices)" not in grouped_section


def test_level4_grouped_moe_unsort_scatter_matches_reference() -> None:
    torch.manual_seed(123)
    layer = GroupedMoEExperts(num_experts=3, hidden_size=4, intermediate_size=8).eval()
    x = torch.randn(5, 4)
    expert_indices = torch.tensor(
        [
            [2, 0],
            [1, 2],
            [0, 1],
            [2, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )
    expert_weights = torch.rand(5, 2)
    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

    with torch.inference_mode():
        output = layer(x, expert_indices, expert_weights)
        first_unsorted_ptr = layer._unsorted_output_buffer.data_ptr()
        output_again = layer(x, expert_indices, expert_weights)

        reference = torch.zeros_like(x)
        for token_idx in range(x.shape[0]):
            token = x[token_idx : token_idx + 1]
            for route_idx in range(expert_indices.shape[1]):
                expert_id = int(expert_indices[token_idx, route_idx])
                gate = token @ layer.w1[expert_id]
                torch.nn.functional.silu(gate, inplace=True)
                gate.mul_(token @ layer.w3[expert_id])
                expert_out = gate @ layer.w2[expert_id]
                reference[token_idx].add_(expert_out.squeeze(0) * expert_weights[token_idx, route_idx])

    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(output_again, reference)
    assert layer._unsorted_output_buffer.data_ptr() == first_unsorted_ptr


def test_moe_route_weight_normalization_uses_inplace_inference_guard() -> None:
    targets = (
        (
            "moe_model.py",
            "class MoELayer",
            "class MoEBlock",
        ),
        (
            "level4_triton.py",
            "class TritonMoELayer",
            "class TritonMoEBlock",
        ),
        (
            "level6_full_stack.py",
            "class CUDAGraphMoELayer",
            "class CUDAGraphMoEBlock",
        ),
    )

    for filename, start_marker, end_marker in targets:
        source = (
            Path(__file__).resolve().parents[1]
            / "labs"
            / "moe_optimization_journey"
            / filename
        ).read_text(encoding="utf-8")
        layer_section = source.split(start_marker, maxsplit=1)[1].split(
            end_marker,
            maxsplit=1,
        )[0]

        assert "expert_weight_sums = expert_weights.sum(dim=-1, keepdim=True)" in layer_section
        assert "if torch.is_grad_enabled() and expert_weights.requires_grad:" in layer_section
        assert "expert_weights = expert_weights / expert_weight_sums" in layer_section
        assert "expert_weights.div_(expert_weight_sums)" in layer_section
        assert "expert_weights / expert_weights.sum" not in layer_section
        assert "(expert_weights / expert_weights.sum" not in layer_section


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
    assert "sorted_ids" not in function_section
    assert "Sorted_ids_ptr" not in source
    assert "def _flat_topk_token_ids" in source
    assert 'token_ids.div_(top_k, rounding_mode="floor")' in source
    assert "x.repeat_interleave(K" not in benchmark_section
    assert "sorted_token_ids = flat_token_ids.index_select(0, sorted_order)" in benchmark_section
    assert "sorted_tokens = x.index_select(0, sorted_token_ids)" in benchmark_section
    assert "max_tokens = int(counts.max().item())" in benchmark_section
    assert benchmark_section.count("max_tokens=max_tokens") == 3
    assert benchmark_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start.elapsed_time(end) / 10" in benchmark_section
    assert "time.perf_counter()" not in benchmark_section


def test_moe_journey_run_level_uses_reused_cuda_events() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "moe_optimization_journey"
        / "moe_benchmark.py"
    ).read_text(encoding="utf-8")
    run_level_section = source.split("def run_level", maxsplit=1)[1].split(
        'if __name__ == "__main__":',
        maxsplit=1,
    )[0]
    timing_loop = run_level_section.split("for i in range(5):", maxsplit=1)[1].split(
        "avg = sum(times) / len(times)",
        maxsplit=1,
    )[0]

    assert run_level_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start_event.record()" in timing_loop
    assert "end_event.record()" in timing_loop
    assert "end_event.synchronize()" in timing_loop
    assert "elapsed_ms = start_event.elapsed_time(end_event)" in timing_loop
    assert "time.perf_counter()" not in run_level_section


def test_moe_bmm_fusion_reuses_offset_buffer_without_cat() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "moe_optimization_journey"
        / "moe_model.py"
    ).read_text(encoding="utf-8")
    bmm_fusion_section = source.split("def forward_bmm_fused", maxsplit=1)[1].split(
        "def _forward_bmm_fused_graphable",
        maxsplit=1,
    )[0]

    assert '"_bmm_expert_starts",' in bmm_fusion_section
    assert "starts[0] = 0" in bmm_fusion_section
    assert "starts[1:].copy_(cumsum[:-1])" in bmm_fusion_section
    assert "torch.cumsum(counts, dim=0, out=cumsum)" in bmm_fusion_section
    assert "starts = torch.cat(" not in bmm_fusion_section
    assert "x.repeat_interleave(top_k" not in bmm_fusion_section
    assert "flat_token_ids = self._flat_topk_token_ids_for(batch_seq, top_k, device)" in bmm_fusion_section
    assert "torch.index_select(flat_token_ids, 0, sorted_order, out=sorted_token_ids)" in bmm_fusion_section
    assert "torch.index_select(x, 0, sorted_token_ids, out=sorted_tokens)" in bmm_fusion_section
    assert "torch.index_select(starts, 0, sorted_expert_ids, out=expert_offsets)" in bmm_fusion_section
    assert "position_ids = self._position_ids_for(sorted_expert_ids.numel(), device)" in bmm_fusion_section
    assert "torch.sub(position_ids, expert_offsets, out=positions)" in bmm_fusion_section
    assert "torch.mul(sorted_expert_ids, max_count, out=padded_indices)" in bmm_fusion_section
    assert "out.mul_(padded_weights)" in bmm_fusion_section
    assert "if torch.is_grad_enabled() and flat_out.requires_grad:" in bmm_fusion_section
    assert "valid_out = flat_out.index_select(0, padded_indices)" in bmm_fusion_section
    assert "torch.index_select(flat_out, 0, padded_indices, out=valid_out)" in bmm_fusion_section
    assert "unsort[sorted_order] = position_ids" in bmm_fusion_section
    assert "if torch.is_grad_enabled() and valid_out.requires_grad:" in bmm_fusion_section
    assert "restored = valid_out.index_select(0, unsort)" in bmm_fusion_section
    assert "torch.index_select(valid_out, 0, unsort, out=restored)" in bmm_fusion_section
    assert "torch.arange(len(sorted_expert_ids)" not in bmm_fusion_section
    assert "torch.argsort(sorted_order)" not in bmm_fusion_section


def test_triton_fused_moe_uses_overwritten_output_and_inplace_offsets() -> None:
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

    assert "output = torch.empty_like(x)" in function_section
    assert "torch.zeros_like(x)" not in function_section
    assert "expert_offsets = torch.empty(E + 1, device=device, dtype=torch.long)" in benchmark_section
    assert "expert_offsets[0] = 0" in benchmark_section
    assert "expert_offsets[1:].copy_(counts.cumsum(0))" in benchmark_section
    assert "expert_offsets = torch.cat(" not in benchmark_section
    assert "x.repeat_interleave(K" not in benchmark_section


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
