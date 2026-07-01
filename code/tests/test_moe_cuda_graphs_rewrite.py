from __future__ import annotations

from pathlib import Path
from types import MethodType

import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark
from labs.moe_optimization_journey.level7_compiled import Level7Compiled
from labs.moe_optimization_journey.moe_model import ConfigurableMoEModel, MoEExperts, MoEOptimizations
from labs.moe_optimization_journey.optimized_moe import get_benchmark as get_main_optimized_benchmark


def _make_experts(*, use_cuda_graphs: bool) -> MoEExperts:
    opts = MoEOptimizations(use_bmm_fused=True, use_cuda_graphs=use_cuda_graphs)
    return MoEExperts(num_experts=2, hidden_size=4, intermediate_size=8, opts=opts)


def test_moe_expert_workspaces_reuse_larger_capacity_views() -> None:
    experts = _make_experts(use_cuda_graphs=False)
    device = torch.device("cpu")

    large = experts._bmm_workspace(
        "_capacity_test_workspace",
        (5, 4),
        device=device,
        dtype=torch.float32,
    )
    workspace_ptr = experts._capacity_test_workspace.data_ptr()
    small = experts._bmm_workspace(
        "_capacity_test_workspace",
        (2, 4),
        device=device,
        dtype=torch.float32,
    )

    assert large.shape == (5, 4)
    assert small.shape == (2, 4)
    assert small.data_ptr() == workspace_ptr
    assert experts._capacity_test_workspace.data_ptr() == workspace_ptr
    assert experts._capacity_test_workspace.numel() >= large.numel()

    large_output = experts._naive_output_like(torch.empty(5, 4))
    output_ptr = experts._naive_output.data_ptr()
    small_output = experts._naive_output_like(torch.empty(2, 4))

    assert large_output.shape == (5, 4)
    assert small_output.shape == (2, 4)
    assert small_output.data_ptr() == output_ptr
    assert experts._naive_output.numel() >= large_output.numel()


def test_moe_forward_prefers_cuda_graph_path_when_enabled() -> None:
    experts = _make_experts(use_cuda_graphs=True)
    x = torch.randn(3, 4)
    expert_indices = torch.zeros(3, 1, dtype=torch.long)
    expert_weights = torch.ones(3, 1)

    experts.forward_cuda_graphs = MethodType(lambda self, *_args: "graph", experts)
    experts.forward_bmm_fused = MethodType(lambda self, *_args: "bmm", experts)

    assert experts.forward(x, expert_indices, expert_weights, num_experts_per_tok=1) == "graph"


def test_moe_forward_prefers_graphable_bmm_path_while_torch_compile_is_active() -> None:
    experts = _make_experts(use_cuda_graphs=True)
    experts.opts.use_compile = True
    x = torch.randn(3, 4)
    expert_indices = torch.zeros(3, 1, dtype=torch.long)
    expert_weights = torch.ones(3, 1)

    experts._is_torch_compiling = MethodType(lambda self: True, experts)
    experts._forward_bmm_fused_graphable = MethodType(lambda self, *_args: "graphable", experts)
    experts.forward_bmm_fused = MethodType(lambda self, *_args: "bmm", experts)

    assert experts.forward(x, expert_indices, expert_weights, num_experts_per_tok=1) == "graphable"


def test_moe_cuda_graphs_fallback_is_visible_on_cpu() -> None:
    experts = _make_experts(use_cuda_graphs=True)
    x = torch.randn(3, 4)
    expert_indices = torch.tensor([[0], [1], [0]], dtype=torch.long)
    expert_weights = torch.ones(3, 1)

    output = experts.forward_cuda_graphs(x, expert_indices, expert_weights)
    metrics = experts.get_cuda_graph_metrics()

    assert output.shape == (3, 4)
    assert metrics["cuda_graph_attempted"] == 0.0
    assert metrics["cuda_graph_captured"] == 0.0
    assert metrics["cuda_graph_fallback"] == 1.0


def test_graphable_bmm_fused_path_matches_dynamic_bmm_path() -> None:
    experts = _make_experts(use_cuda_graphs=True)
    x = torch.randn(4, 4)
    expert_indices = torch.tensor([[0], [1], [0], [1]], dtype=torch.long)
    expert_weights = torch.ones(4, 1)

    with torch.inference_mode():
        dynamic = experts.forward_bmm_fused(x, expert_indices, expert_weights)
        token_workspace_ptr = experts._bmm_padded_tokens.data_ptr()
        weight_workspace_ptr = experts._bmm_padded_weights.data_ptr()
        index_workspace_ptr = experts._bmm_padded_indices.data_ptr()
        unsort_workspace_ptr = experts._bmm_unsort.data_ptr()
        restored_workspace_ptr = experts._bmm_restored.data_ptr()
        padded_token_index_view = next(iter(experts._bmm_padded_token_index_view_cache.values()))
        padded_weight_index_view = next(iter(experts._bmm_padded_column_index_cache.values()))
        sorted_weight_column_view = next(iter(experts._bmm_sorted_weight_column_cache.values()))
        flat_token_ids_ptr = experts._bmm_flat_token_ids_cache[(4, 1, torch.device("cpu"))].data_ptr()
        position_ids_ptr = experts._bmm_position_ids_cache[(4, torch.device("cpu"))].data_ptr()
        dynamic_again = experts.forward_bmm_fused(x, expert_indices, expert_weights)
        graphable = experts._forward_bmm_fused_graphable(x, expert_indices, expert_weights)
        graph_workspace_ptr = experts._graph_padded_tokens.data_ptr()
        graphable_again = experts._forward_bmm_fused_graphable(x, expert_indices, expert_weights)

    assert experts._bmm_padded_tokens.data_ptr() == token_workspace_ptr
    assert experts._bmm_padded_weights.data_ptr() == weight_workspace_ptr
    assert experts._bmm_padded_indices.data_ptr() == index_workspace_ptr
    assert experts._bmm_unsort.data_ptr() == unsort_workspace_ptr
    assert experts._bmm_restored.data_ptr() == restored_workspace_ptr
    assert dynamic_again.data_ptr() == restored_workspace_ptr
    assert next(iter(experts._bmm_padded_token_index_view_cache.values())) is padded_token_index_view
    assert next(iter(experts._bmm_padded_column_index_cache.values())) is padded_weight_index_view
    assert next(iter(experts._bmm_sorted_weight_column_cache.values())) is sorted_weight_column_view
    torch.testing.assert_close(padded_token_index_view[:, 0], experts._bmm_padded_indices)
    torch.testing.assert_close(padded_weight_index_view[:, 0], experts._bmm_padded_indices)
    torch.testing.assert_close(sorted_weight_column_view[:, 0], experts._bmm_sorted_weights)
    assert experts._bmm_flat_token_ids_cache[(4, 1, torch.device("cpu"))].data_ptr() == flat_token_ids_ptr
    assert experts._bmm_position_ids_cache[(4, torch.device("cpu"))].data_ptr() == position_ids_ptr
    assert experts._graph_padded_tokens.data_ptr() == graph_workspace_ptr
    torch.testing.assert_close(dynamic_again, dynamic)
    torch.testing.assert_close(graphable, dynamic)
    torch.testing.assert_close(graphable_again, dynamic)


def test_moe_benchmark_metrics_surface_model_cuda_graph_state() -> None:
    bench = MoEJourneyBenchmark()
    bench.opts = MoEOptimizations(use_bmm_fused=True, use_cuda_graphs=True)

    model = ConfigurableMoEModel(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        opts=bench.opts,
    )
    experts = model.blocks[0].moe.experts
    x = torch.randn(2, 8)
    expert_indices = torch.tensor([[0], [1]], dtype=torch.long)
    expert_weights = torch.ones(2, 1)
    experts.forward_cuda_graphs(x, expert_indices, expert_weights)

    bench.model = model
    metrics = bench.get_custom_metrics()

    assert metrics["use_cuda_graphs"] == 1.0
    assert metrics["cuda_graph_fallback"] == 1.0
    assert metrics["cuda_graph_captured"] == 0.0


def test_main_optimized_moe_entrypoint_now_targets_level7_compile_stage() -> None:
    benchmark = get_main_optimized_benchmark()
    assert isinstance(benchmark, Level7Compiled)


def test_graphable_moe_path_uses_fixed_capacity_dense_dispatch() -> None:
    source = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey" / "moe_model.py"
    text = source.read_text(encoding="utf-8")

    graphable_section = text.split("def _forward_bmm_fused_graphable", maxsplit=1)[1].split(
        "def forward_cuda_graphs", maxsplit=1
    )[0]
    implementation = graphable_section.split('"""', maxsplit=2)[-1]

    assert "F.one_hot(" in graphable_section
    assert "counts.max().item()" not in implementation
    assert "torch.argsort" not in implementation
    assert "repeat_interleave" not in implementation
    assert "x[:, None, :].expand(batch_seq, top_k, self.hidden_size).reshape(" in implementation
    assert "self._graph_padded_tokens: Optional[torch.Tensor] = None" in text
    assert 'padded_tokens = self._bmm_workspace(\n                "_graph_padded_tokens",' in implementation
    assert "torch.mul(expert_mask_column, expanded_x_broadcast, out=padded_tokens)" in implementation
    assert "padded_tokens = expert_mask.unsqueeze(-1) * expanded_x.unsqueeze(0)" not in implementation


def test_level5_bmm_path_reuses_padding_workspaces() -> None:
    source = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey" / "moe_model.py"
    text = source.read_text(encoding="utf-8")

    bmm_section = text.split("def forward_bmm_fused", maxsplit=1)[1].split(
        "def _forward_bmm_fused_graphable", maxsplit=1
    )[0]

    assert "def _bmm_workspace" in text
    assert "cached = getattr(self, name, None)" in text
    assert "or cached.numel() < numel" in text
    assert "return cached[:numel].view(shape)" in text
    assert "or self._naive_output.numel() < numel" in text
    assert "return self._naive_output[:numel].view(shape)" in text
    assert "self._bmm_padded_tokens: Optional[torch.Tensor] = None" in text
    assert "self._bmm_padded_weights: Optional[torch.Tensor] = None" in text
    assert "self._bmm_valid_out: Optional[torch.Tensor] = None" in text
    assert "self._bmm_restored: Optional[torch.Tensor] = None" in text
    assert "self._bmm_reduced: Optional[torch.Tensor] = None" not in text
    assert "self._bmm_flat_token_ids_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}" in text
    assert "self._bmm_position_ids_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}" in text
    assert "self._bmm_padded_token_index_view_cache: Dict[Tuple[int, int, int, torch.device], torch.Tensor] = {}" in text
    assert "self._bmm_padded_column_index_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}" in text
    assert "self._bmm_sorted_weight_column_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}" in text
    assert "def _flat_topk_token_ids_for(self, num_tokens: int, top_k: int, device: torch.device)" in text
    assert "def _position_ids_for(self, length: int, device: torch.device)" in text
    assert "def _padded_token_index_view(self, padded_indices: torch.Tensor, width: int)" in text
    assert "def _padded_column_index(self, padded_indices: torch.Tensor)" in text
    assert "def _sorted_weight_column(self, sorted_weights: torch.Tensor)" in text
    assert "flat_token_ids = self._flat_topk_token_ids_for(batch_seq, top_k, device)" in bmm_section
    assert "torch.index_select(flat_token_ids, 0, sorted_order, out=sorted_token_ids)" in bmm_section
    assert "torch.index_select(x, 0, sorted_token_ids, out=sorted_tokens)" in bmm_section
    assert "torch.index_select(expert_weights.view(-1), 0, sorted_order, out=sorted_weights)" in bmm_section
    assert "torch.index_select(flat_idx, 0, sorted_order, out=sorted_expert_ids)" in bmm_section
    assert "torch.cumsum(counts, dim=0, out=cumsum)" in bmm_section
    assert "torch.index_select(starts, 0, sorted_expert_ids, out=expert_offsets)" in bmm_section
    assert "position_ids = self._position_ids_for(sorted_expert_ids.numel(), device)" in bmm_section
    assert "torch.sub(position_ids, expert_offsets, out=positions)" in bmm_section
    assert "torch.mul(sorted_expert_ids, max_count, out=padded_indices)" in bmm_section
    assert "out.mul_(padded_weights)" in bmm_section
    assert "if torch.is_grad_enabled() and flat_out.requires_grad:" in bmm_section
    assert "valid_out = flat_out.index_select(0, padded_indices)" in bmm_section
    assert '"_bmm_valid_out",' in bmm_section
    assert "torch.index_select(flat_out, 0, padded_indices, out=valid_out)" in bmm_section
    assert "unsort[sorted_order] = position_ids" in bmm_section
    assert "if torch.is_grad_enabled() and valid_out.requires_grad:" in bmm_section
    assert "restored = valid_out.index_select(0, unsort)" in bmm_section
    assert '"_bmm_restored",' in bmm_section
    assert "torch.index_select(valid_out, 0, unsort, out=restored)" in bmm_section
    assert '"_bmm_reduced",' not in bmm_section
    assert "return _sum_routes_in_place_if_safe(restored)" in bmm_section
    assert "torch.sum(restored, dim=1, out=reduced)" not in bmm_section
    assert "return restored.sum(dim=1)" not in bmm_section
    assert 'padded_tokens = self._bmm_workspace(' in bmm_section
    assert '"_bmm_padded_tokens",' in bmm_section
    assert 'padded_weights = self._bmm_workspace(' in bmm_section
    assert '"_bmm_padded_weights",' in bmm_section
    assert "padded_token_index = self._padded_token_index_view(padded_indices, self.hidden_size)" in bmm_section
    assert "padded_tokens.scatter_(0, padded_token_index, sorted_tokens)" in bmm_section
    assert "padded_weight_index = self._padded_column_index(padded_indices)" in bmm_section
    assert "sorted_weight_column = self._sorted_weight_column(sorted_weights)" in bmm_section
    assert "padded_weights.scatter_(0, padded_weight_index, sorted_weight_column)" in bmm_section
    assert "padded_indices.unsqueeze(1).expand(-1, self.hidden_size)" not in bmm_section
    assert "padded_indices.unsqueeze(1), sorted_weights.unsqueeze(1)" not in bmm_section
    assert "padded_tokens.zero_()" not in bmm_section
    assert "padded_weights.zero_()" not in bmm_section
    assert "padded_tokens.scatter_(" in bmm_section
    assert "padded_weights.scatter_(" in bmm_section
    assert "torch.zeros(self.num_experts * max_count" not in bmm_section
    assert "torch.arange(len(sorted_expert_ids)" not in bmm_section
    assert "torch.argsort(sorted_order)" not in bmm_section


def test_naive_moe_path_seeds_output_from_first_route() -> None:
    source = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey" / "moe_model.py"
    text = source.read_text(encoding="utf-8")

    naive_section = text.split("def forward_naive", maxsplit=1)[1].split(
        "def forward_batched",
        maxsplit=1,
    )[0]

    assert "self._naive_output: Optional[torch.Tensor] = None" in text
    assert "def _naive_output_like(self, x: torch.Tensor) -> torch.Tensor:" in text
    assert "if torch.is_grad_enabled() and x.requires_grad:" in text
    assert "output = self._naive_output_like(x)" in naive_section
    assert "output = torch.empty_like(x)" not in naive_section
    assert "torch.zeros_like(x)" not in naive_section
    assert "for k in range(num_experts_per_tok):" in naive_section
    assert "token_ids = (expert_indices[:, k] == expert_idx).nonzero(as_tuple=True)[0]" in naive_section
    assert "if token_ids.numel() == 0:" in naive_section
    assert "if k == 0:" in naive_section
    assert "output[token_ids] = weighted_output" in naive_section
    assert "output[token_ids] += weighted_output" in naive_section
    assert "mask.any()" not in naive_section

    torch.manual_seed(0)
    opts = MoEOptimizations()
    experts = MoEExperts(num_experts=3, hidden_size=4, intermediate_size=8, opts=opts)
    x = torch.randn(5, 4)
    expert_indices = torch.tensor(
        [[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]],
        dtype=torch.long,
    )
    expert_weights = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.8, 0.2], [0.25, 0.75]],
        dtype=torch.float32,
    )

    first = experts.forward_naive(x, expert_indices, expert_weights, num_experts_per_tok=2)
    first_ptr = first.data_ptr()
    second = experts.forward_naive(x, expert_indices, expert_weights, num_experts_per_tok=2)

    assert second.data_ptr() == first_ptr
    torch.testing.assert_close(second, first)

    x_requires_grad = x.detach().requires_grad_(True)
    grad_output = experts.forward_naive(
        x_requires_grad,
        expert_indices,
        expert_weights,
        num_experts_per_tok=2,
    )
    assert grad_output.data_ptr() != first_ptr


def test_moe_expert_paths_weight_outputs_in_place_when_grad_disabled() -> None:
    source = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey" / "moe_model.py"
    text = source.read_text(encoding="utf-8")

    assert "def _weight_routes_in_place_if_safe" in text
    assert "if torch.is_grad_enabled() and out.requires_grad:" in text
    assert "return out * weights" in text
    assert "out.mul_(weights)" in text
    assert "def _sum_routes_in_place_if_safe" in text
    assert "reduced = out[:, 0, :]" in text
    assert "for route_idx in range(1, out.shape[1]):" in text
    assert "reduced.add_(out[:, route_idx, :])" in text

    naive_section = text.split("def forward_naive", maxsplit=1)[1].split(
        "def forward_batched",
        maxsplit=1,
    )[0]
    batched_section = text.split("def forward_batched", maxsplit=1)[1].split(
        "def forward_fused",
        maxsplit=1,
    )[0]
    fused_section = text.split("def forward_fused", maxsplit=1)[1].split(
        "def forward_mem_efficient",
        maxsplit=1,
    )[0]
    mem_section = text.split("def forward_mem_efficient", maxsplit=1)[1].split(
        "def forward_grouped",
        maxsplit=1,
    )[0]
    grouped_section = text.split("def forward_grouped", maxsplit=1)[1].split(
        "def forward_bmm_fused",
        maxsplit=1,
    )[0]
    graphable_section = text.split("def _forward_bmm_fused_graphable", maxsplit=1)[1].split(
        "def forward_cuda_graphs",
        maxsplit=1,
    )[0]

    assert "weighted_output = _weight_routes_in_place_if_safe(expert_output, weights)" in naive_section
    for section in (batched_section, fused_section, mem_section):
        assert "out = _weight_routes_in_place_if_safe(out, expert_weights.unsqueeze(-1))" in section
        assert "(out * expert_weights.unsqueeze(-1)).sum(dim=1)" not in section
    assert "weighted_out = _weight_routes_in_place_if_safe(expert_out, weights_e)" in grouped_section
    assert "expert_out * weights_e" not in grouped_section
    assert "expert_mask_column = expert_mask.unsqueeze(-1)" in graphable_section
    assert "out = _weight_routes_in_place_if_safe(out, expert_mask_column)" in graphable_section
    assert "out = _weight_routes_in_place_if_safe(out, flat_weights)" in graphable_section
    assert "out = out * expert_mask.unsqueeze(-1) * flat_weights" not in graphable_section
    assert "self._mem_out_buffer: Optional[torch.Tensor] = None" in text
    assert "self._mem_reduced_buffer: Optional[torch.Tensor] = None" not in text
    assert 'out_flat = self._bmm_workspace(\n            "_mem_out_buffer",' in mem_section
    assert "torch.bmm(hidden.unsqueeze(1), w2_sel, out=out_flat.unsqueeze(1))" in mem_section
    assert 'reduced = self._bmm_workspace(\n            "_mem_reduced_buffer",' not in mem_section
    assert "return _sum_routes_in_place_if_safe(out)" in mem_section
    assert "torch.sum(out, dim=1, out=reduced)" not in mem_section
    assert "out = torch.bmm(hidden.unsqueeze(1), w2_sel).squeeze(1)" not in mem_section
    assert "return out.sum(dim=1)" not in mem_section


def test_moe_expert_paths_fuse_swiglu_in_place_when_grad_disabled() -> None:
    source = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey" / "moe_model.py"
    text = source.read_text(encoding="utf-8")

    assert "def _silu_mul_in_place_if_safe" in text
    assert "if torch.is_grad_enabled() and gate.requires_grad:" in text
    assert "return F.silu(gate) * up" in text
    assert "F.silu(gate, inplace=True)" in text
    assert "gate.mul_(up)" in text

    naive_section = text.split("def forward_naive", maxsplit=1)[1].split(
        "def forward_batched",
        maxsplit=1,
    )[0]
    batched_section = text.split("def forward_batched", maxsplit=1)[1].split(
        "def forward_fused",
        maxsplit=1,
    )[0]
    grouped_section = text.split("def forward_grouped", maxsplit=1)[1].split(
        "def forward_bmm_fused",
        maxsplit=1,
    )[0]
    mem_section = text.split("def forward_mem_efficient", maxsplit=1)[1].split(
        "def forward_grouped",
        maxsplit=1,
    )[0]
    bmm_section = text.split("def forward_bmm_fused", maxsplit=1)[1].split(
        "def _forward_bmm_fused_graphable",
        maxsplit=1,
    )[0]
    graphable_section = text.split("def _forward_bmm_fused_graphable", maxsplit=1)[1].split(
        "def forward_cuda_graphs",
        maxsplit=1,
    )[0]

    assert "hidden = _silu_mul_in_place_if_safe(gate, up)" in naive_section
    assert "expert_output = expert['w2'](hidden)" in naive_section
    assert "expert['w2'](gate * up)" not in naive_section
    for section in (batched_section, bmm_section, graphable_section):
        assert "hidden = _silu_mul_in_place_if_safe(gate, up)" in section
        assert "hidden = gate * up" not in section
    assert "hidden = _silu_mul_in_place_if_safe(gate_buffer, up_buffer)" in mem_section
    assert "hidden = fused_silu_mul(self._gate_buffer, self._up_buffer)" not in mem_section
    assert "hidden = _silu_mul_in_place_if_safe(gate, up)" in grouped_section
    assert "expert_out = hidden @ self.w2_stacked[expert_id]" in grouped_section
    assert "expert_out = (gate * up) @ self.w2_stacked[expert_id]" not in grouped_section


def test_mem_efficient_moe_path_reuses_workspaces() -> None:
    torch.manual_seed(0)
    opts = MoEOptimizations(use_mem_efficient=True)
    experts = MoEExperts(num_experts=3, hidden_size=4, intermediate_size=8, opts=opts)
    x = torch.randn(5, 4)
    expert_indices = torch.tensor(
        [[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]],
        dtype=torch.long,
    )
    expert_weights = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.8, 0.2], [0.25, 0.75]],
        dtype=torch.float32,
    )

    with torch.inference_mode():
        expected = experts.forward_naive(x, expert_indices, expert_weights, num_experts_per_tok=2)
        actual = experts.forward_mem_efficient(x, expert_indices, expert_weights)
        gate_ptr = experts._gate_buffer.data_ptr()
        up_ptr = experts._up_buffer.data_ptr()
        out_ptr = experts._mem_out_buffer.data_ptr()
        actual_reused = experts.forward_mem_efficient(x, expert_indices, expert_weights)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_reused, expected, atol=1e-5, rtol=1e-5)
    assert experts._gate_buffer.data_ptr() == gate_ptr
    assert experts._up_buffer.data_ptr() == up_ptr
    assert experts._mem_out_buffer.data_ptr() == out_ptr
    assert actual_reused.data_ptr() == out_ptr


def test_moe_inplace_weighted_paths_match_naive_reference() -> None:
    torch.manual_seed(0)
    opts = MoEOptimizations(
        use_batched=True,
        use_fused=True,
        use_mem_efficient=True,
        use_grouped=True,
        use_bmm_fused=True,
        use_cuda_graphs=True,
    )
    experts = MoEExperts(num_experts=3, hidden_size=4, intermediate_size=8, opts=opts)
    x = torch.randn(5, 4)
    expert_indices = torch.tensor(
        [[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]],
        dtype=torch.long,
    )
    expert_weights = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.8, 0.2], [0.25, 0.75]],
        dtype=torch.float32,
    )

    with torch.inference_mode():
        expected = experts.forward_naive(x, expert_indices, expert_weights, num_experts_per_tok=2)
        actuals = (
            experts.forward_batched(x, expert_indices, expert_weights),
            experts.forward_grouped(x, expert_indices, expert_weights),
            experts._forward_bmm_fused_graphable(x, expert_indices, expert_weights),
        )

    for actual in actuals:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_graphable_moe_path_matches_level5_bmm_fused_outputs() -> None:
    experts = _make_experts(use_cuda_graphs=True)
    x = torch.randn(3, 4)
    expert_indices = torch.tensor([[0], [1], [0]], dtype=torch.long)
    expert_weights = torch.ones(3, 1)

    expected = experts.forward_bmm_fused(x, expert_indices, expert_weights)
    actual = experts._forward_bmm_fused_graphable(x, expert_indices, expert_weights)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_grouped_moe_path_matches_naive_reference() -> None:
    torch.manual_seed(0)
    opts = MoEOptimizations(
        use_batched=True,
        use_fused=True,
        use_mem_efficient=True,
        use_grouped=True,
    )
    experts = MoEExperts(num_experts=4, hidden_size=4, intermediate_size=8, opts=opts)
    x = torch.randn(3, 4)
    expert_indices = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.long)
    expert_weights = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]], dtype=torch.float32)

    with torch.inference_mode():
        expected = experts.forward_naive(x, expert_indices, expert_weights, num_experts_per_tok=2)
        actual = experts.forward_grouped(x, expert_indices, expert_weights)
        first_output_ptr = experts._grouped_output.data_ptr()
        first_restored_ptr = experts._grouped_restored.data_ptr()
        actual_reused = experts.forward_grouped(x, expert_indices, expert_weights)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_reused, expected, atol=1e-5, rtol=1e-5)
    assert experts._grouped_output.data_ptr() == first_output_ptr
    assert experts._grouped_restored.data_ptr() == first_restored_ptr
    assert actual_reused.data_ptr() == first_restored_ptr


def test_grouped_moe_path_uses_shared_bucket_helpers() -> None:
    source = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey" / "moe_model.py"
    text = source.read_text(encoding="utf-8")

    grouped_section = text.split("def forward_grouped", maxsplit=1)[1].split(
        "def forward_bmm_fused", maxsplit=1
    )[0]

    assert "bucket_grouped_tokens(" in grouped_section
    assert "def _flat_topk_token_ids" in text
    assert "flat_token_ids = self._flat_topk_token_ids_for(batch_seq, top_k, x.device)" in grouped_section
    assert "token_ids=flat_token_ids" in grouped_section
    assert "return_expert_order_list=True" in grouped_section
    assert "self._grouped_output: Optional[torch.Tensor] = None" in text
    assert "self._grouped_restored: Optional[torch.Tensor] = None" in text
    assert "self._grouped_reduced: Optional[torch.Tensor] = None" not in text
    assert 'output = self._bmm_workspace(\n            "_grouped_output",' in grouped_section
    assert 'restored = self._bmm_workspace(\n            "_grouped_restored",' in grouped_section
    assert 'reduced = self._bmm_workspace(\n            "_grouped_reduced",' not in grouped_section
    assert "output = torch.empty(" not in grouped_section
    assert "restored = torch.empty(" not in grouped_section
    assert "return restored.view(batch_seq, top_k, -1).sum(dim=1)" not in grouped_section
    assert "return _sum_routes_in_place_if_safe(restored)" in grouped_section
    assert "torch.sum(restored, dim=1, out=reduced)" not in grouped_section
    assert "torch.zeros(sorted_tokens.shape[0]" not in grouped_section
    assert "repeat_interleave" not in grouped_section
    assert "for expert_id, count in zip(expert_order_host, counts)" in grouped_section
    assert "expert_order.tolist()" not in grouped_section
    assert "restore_grouped_tokens(" in grouped_section
    assert "torch.argsort" not in grouped_section
