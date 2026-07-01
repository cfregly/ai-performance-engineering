from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

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
    assert "self._sorted_token_ids_buffer: Optional[torch.Tensor] = None" in grouped_section
    assert "self._sorted_expert_ids_buffer: Optional[torch.Tensor] = None" in grouped_section
    assert "self._sorted_x_buffer: Optional[torch.Tensor] = None" in grouped_section
    assert "self._sorted_weight_buffer: Optional[torch.Tensor] = None" in grouped_section
    assert "self._route_token_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}" in grouped_section
    assert "self._sorted_weight_column_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}" in grouped_section
    assert "def _expert_metadata_buffers(self, device: torch.device)" in grouped_section
    assert "def _workspace(" in grouped_section
    assert "def _sorted_output_like(self, sorted_x: torch.Tensor) -> torch.Tensor" in grouped_section
    assert "def _unsorted_output_like(self, output: torch.Tensor) -> torch.Tensor" in grouped_section
    assert "or cached.numel() < numel" in grouped_section
    assert "return cached[:numel].view(shape)" in grouped_section
    assert "or self._sorted_output_buffer.numel() < numel" in grouped_section
    assert "return self._sorted_output_buffer[:numel].view(shape)" in grouped_section
    assert "or self._unsorted_output_buffer.numel() < numel" in grouped_section
    assert "return self._unsorted_output_buffer[:numel].view(shape)" in grouped_section
    assert "def _sorted_weight_column(self, sorted_weights: torch.Tensor)" in grouped_section
    assert "def _route_token_ids(self, batch_seq: int, top_k: int, device: torch.device)" in grouped_section
    assert "route_token_ids = self._route_token_ids(batch_seq, top_k, x.device)" in grouped_section
    assert 'sorted_token_ids = self._workspace(\n                "_sorted_token_ids_buffer",' in grouped_section
    assert "torch.index_select(route_token_ids, 0, sorted_indices, out=sorted_token_ids)" in grouped_section
    assert 'sorted_x = self._workspace(\n                "_sorted_x_buffer",' in grouped_section
    assert "torch.index_select(x, 0, sorted_token_ids, out=sorted_x)" in grouped_section
    assert 'sorted_weights = self._workspace(\n                "_sorted_weight_buffer",' in grouped_section
    assert "torch.index_select(flat_weights, 0, sorted_indices, out=sorted_weights)" in grouped_section
    assert "use_workspace = not (" in grouped_section
    assert "sorted_token_ids = route_token_ids.index_select(0, sorted_indices)" in grouped_section
    assert "sorted_x = x.index_select(0, sorted_token_ids)" in grouped_section
    assert "if torch.is_grad_enabled() and sorted_x.requires_grad:" in grouped_section
    assert "if torch.is_grad_enabled() and output.requires_grad:" in grouped_section
    assert "torch.cumsum(expert_counts, dim=0, out=expert_offsets)" in grouped_section
    assert "expert_offsets.sub_(expert_counts)" in grouped_section
    assert "expert_metadata[1].copy_(expert_counts)" in grouped_section
    assert 'expert_metadata_host.copy_(expert_metadata, non_blocking=expert_counts.device.type == "cuda")' in grouped_section
    assert "expert_offsets_cpu = expert_metadata_host[0]" in grouped_section
    assert "expert_counts_cpu = expert_metadata_host[1]" in grouped_section
    assert "for expert_id in range(self.num_experts):" in grouped_section
    assert "count = int(expert_counts_cpu[expert_id])" in grouped_section
    assert "start = int(expert_offsets_cpu[expert_id])" in grouped_section
    assert grouped_section.index("count = int(expert_counts_cpu[expert_id])") < grouped_section.index(
        "start = int(expert_offsets_cpu[expert_id])"
    )
    assert "expert_metadata_host.tolist()" not in grouped_section
    assert "zip(expert_offsets_cpu, expert_counts_cpu)" not in grouped_section
    assert "expert_counts.detach().cpu().tolist()" not in grouped_section
    assert "expert_offsets.detach().cpu().tolist()" not in grouped_section
    assert "expert_offsets = torch.cumsum(expert_counts, dim=0) - expert_counts" not in grouped_section
    assert "expert_offsets[expert_id].item()" not in grouped_section
    assert "expert_counts[expert_id].item()" not in grouped_section
    assert "x_repeated =" not in grouped_section
    assert "x.unsqueeze(1).expand" not in grouped_section


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
    reduce_section = grouped_section.split("# Sum over top-k experts", maxsplit=1)[1].split(
        "return reduced",
        maxsplit=1,
    )[0]

    assert "output = self._sorted_output_like(sorted_x)" in expert_loop_section
    assert "output = torch.empty_like(sorted_x)" not in expert_loop_section
    assert "torch.zeros_like(sorted_x)" not in expert_loop_section
    assert "F.silu(gate, inplace=True)" in expert_loop_section
    assert "gate.mul_(up)" in expert_loop_section
    assert "hidden = gate * up" not in expert_loop_section
    assert "if use_workspace:" in apply_weights_section
    assert "sorted_weight_column = self._sorted_weight_column(sorted_weights)" in apply_weights_section
    assert "sorted_weight_column = sorted_weights.unsqueeze(-1)" in apply_weights_section
    assert "output.mul_(sorted_weight_column)" in apply_weights_section
    assert "output.mul_(sorted_weights.unsqueeze(-1))" not in apply_weights_section
    assert "output = output * sorted_weights.unsqueeze(-1)" not in grouped_section
    assert "unsorted_output = self._unsorted_output_like(output)" in unsort_section
    assert "unsorted_output.index_copy_(0, sorted_indices, output)" in unsort_section
    assert "torch.argsort(sorted_indices)" not in grouped_section
    assert "output = output.view(batch_seq, top_k, -1)" in reduce_section
    assert "if torch.is_grad_enabled() and output.requires_grad:" in reduce_section
    assert "return output.sum(dim=1)" in reduce_section
    assert "reduced = output[:, 0, :]" in reduce_section
    assert "for route_idx in range(1, top_k):" in reduce_section
    assert "reduced.add_(output[:, route_idx, :])" in reduce_section
    assert "output.view(batch_seq, top_k, -1).sum(dim=1)" not in grouped_section


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
        route_token_cache = next(iter(layer._route_token_cache.values()))
        sorted_token_ids_ptr = layer._sorted_token_ids_buffer.data_ptr()
        sorted_expert_ids_ptr = layer._sorted_expert_ids_buffer.data_ptr()
        sorted_x_ptr = layer._sorted_x_buffer.data_ptr()
        sorted_weight_ptr = layer._sorted_weight_buffer.data_ptr()
        sorted_output_ptr = layer._sorted_output_buffer.data_ptr()
        sorted_weight_column = next(iter(layer._sorted_weight_column_cache.values()))
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
    assert next(iter(layer._route_token_cache.values())).data_ptr() == route_token_cache.data_ptr()
    assert layer._sorted_token_ids_buffer.data_ptr() == sorted_token_ids_ptr
    assert layer._sorted_expert_ids_buffer.data_ptr() == sorted_expert_ids_ptr
    assert layer._sorted_x_buffer.data_ptr() == sorted_x_ptr
    assert layer._sorted_weight_buffer.data_ptr() == sorted_weight_ptr
    assert layer._sorted_output_buffer.data_ptr() == sorted_output_ptr
    assert next(iter(layer._sorted_weight_column_cache.values())) is sorted_weight_column
    torch.testing.assert_close(sorted_weight_column[:, 0], layer._sorted_weight_buffer)

    with torch.inference_mode():
        smaller_output = layer(x[:3], expert_indices[:3], expert_weights[:3])

    assert smaller_output.shape == (3, 4)
    assert layer._sorted_token_ids_buffer.data_ptr() == sorted_token_ids_ptr
    assert layer._sorted_expert_ids_buffer.data_ptr() == sorted_expert_ids_ptr
    assert layer._sorted_x_buffer.data_ptr() == sorted_x_ptr
    assert layer._sorted_weight_buffer.data_ptr() == sorted_weight_ptr
    assert layer._sorted_output_buffer.data_ptr() == sorted_output_ptr
    assert layer._unsorted_output_buffer.data_ptr() == first_unsorted_ptr
    assert layer._sorted_x_buffer.numel() >= x.numel() * expert_indices.shape[1]


def test_moe_route_weight_normalization_uses_selected_logit_softmax() -> None:
    targets = (
        (
            "moe_model.py",
            "class MoELayer",
            "class MoEBlock",
            "self.num_experts_per_tok",
        ),
        (
            "level4_triton.py",
            "class TritonMoELayer",
            "class TritonMoEBlock",
            "self.top_k",
        ),
        (
            "level6_full_stack.py",
            "class CUDAGraphMoELayer",
            "class CUDAGraphMoEBlock",
            "self.top_k",
        ),
    )

    for filename, start_marker, end_marker, top_k_expr in targets:
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

        assert f"top_logits, expert_indices = torch.topk(router_logits.float(), {top_k_expr}, dim=-1)" in layer_section
        assert "expert_weights = F.softmax(top_logits, dim=-1)" in layer_section
        assert "routing_weights = F.softmax(" not in layer_section
        assert "torch.topk(routing_weights" not in layer_section
        assert "expert_weight_sums =" not in layer_section
        assert "expert_weights / expert_weights.sum" not in layer_section
        assert "(expert_weights / expert_weights.sum" not in layer_section

    logits = torch.randn(6, 8)
    full_probs = torch.softmax(logits.float(), dim=-1)
    old_weights, old_indices = torch.topk(full_probs, 2, dim=-1)
    old_weights = old_weights / old_weights.sum(dim=-1, keepdim=True)
    top_logits, new_indices = torch.topk(logits.float(), 2, dim=-1)
    new_weights = F.softmax(top_logits, dim=-1)

    torch.testing.assert_close(new_indices, old_indices)
    torch.testing.assert_close(new_weights, old_weights)


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
    assert "max_tokens = total_tokens" in function_section
    assert ".max().item()" not in function_section
    assert "sorted_ids" not in function_section
    assert "Sorted_ids_ptr" not in source
    assert "def _flat_topk_token_ids" in source
    assert 'token_ids.div_(top_k, rounding_mode="floor")' in source
    assert "expert_indices_cpu = torch.randint(0, E, (batch_seq, K), dtype=torch.int64)" in benchmark_section
    assert "counts_cpu = torch.bincount(expert_indices_cpu.reshape(-1), minlength=E)" in benchmark_section
    assert "max_tokens = int(counts_cpu.max())" in benchmark_section
    assert "expert_indices = expert_indices_cpu.to(device=device, non_blocking=True)" in benchmark_section
    assert "x.repeat_interleave(K" not in benchmark_section
    assert "sorted_token_ids = flat_token_ids.index_select(0, sorted_order)" in benchmark_section
    assert "sorted_tokens = x.index_select(0, sorted_token_ids)" in benchmark_section
    assert "max_tokens = int(counts.max().item())" not in benchmark_section
    assert ".max().item()" not in benchmark_section
    assert benchmark_section.count("max_tokens=max_tokens") == 3
    assert benchmark_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "current_stream = torch.cuda.current_stream()" in benchmark_section
    assert "start.record(current_stream)" in benchmark_section
    assert "end.record(current_stream)" in benchmark_section
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
        "avg = total_ms / 5",
        maxsplit=1,
    )[0]

    assert run_level_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "current_stream = torch.cuda.current_stream()" in run_level_section
    assert "start_event.record(current_stream)" in timing_loop
    assert "end_event.record(current_stream)" in timing_loop
    assert "start_event.record()" not in timing_loop
    assert "end_event.record()" not in timing_loop
    assert "end_event.synchronize()" in timing_loop
    assert "elapsed_ms = start_event.elapsed_time(end_event)" in timing_loop
    assert "total_ms += elapsed_ms" in timing_loop
    assert "avg = total_ms / 5" in run_level_section
    assert "times = []" not in run_level_section
    assert "times.append(" not in run_level_section
    assert "sum(times)" not in run_level_section
    assert "time.perf_counter()" not in run_level_section


def test_moe_journey_level_benchmarks_record_events_on_captured_stream() -> None:
    targets = ("level4_triton.py", "level6_full_stack.py")

    for filename in targets:
        source = (
            Path(__file__).resolve().parents[1]
            / "labs"
            / "moe_optimization_journey"
            / filename
        ).read_text(encoding="utf-8")
        benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
            "def finalize_iteration_metrics",
            maxsplit=1,
        )[0]

        assert "current_stream = torch.cuda.current_stream(self.device)" in benchmark_section
        assert "start_event.record(current_stream)" in benchmark_section
        assert "end_event.record(current_stream)" in benchmark_section
        assert "start_event.record()" not in benchmark_section
        assert "end_event.record()" not in benchmark_section


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
    assert "padded_token_index = self._padded_token_index_view(padded_indices, self.hidden_size)" in bmm_fusion_section
    assert "padded_tokens.scatter_(0, padded_token_index, sorted_tokens)" in bmm_fusion_section
    assert "padded_weight_index = self._padded_column_index(padded_indices)" in bmm_fusion_section
    assert "sorted_weight_column = self._sorted_weight_column(sorted_weights)" in bmm_fusion_section
    assert "padded_weights.scatter_(0, padded_weight_index, sorted_weight_column)" in bmm_fusion_section
    assert "padded_indices.unsqueeze(1).expand(-1, self.hidden_size)" not in bmm_fusion_section
    assert "padded_indices.unsqueeze(1), sorted_weights.unsqueeze(1)" not in bmm_fusion_section
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
    assert metrics is bench._iteration_metrics
    assert metrics["latency_ms"] >= 0.0

    bench.capture_verification_payload()
    payload = bench._verification_payload
    assert payload.output.dtype == torch.float32
    assert payload.output.data_ptr() != bench.output.data_ptr()

    bench.benchmark_fn()
    torch.cuda.synchronize()
    assert bench._timing_events is first_events
    assert fake_forward.calls == 2
    next_metrics = bench.finalize_iteration_metrics()
    assert next_metrics is metrics
