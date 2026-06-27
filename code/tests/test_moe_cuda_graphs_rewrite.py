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

    dynamic = experts.forward_bmm_fused(x, expert_indices, expert_weights)
    token_workspace_ptr = experts._bmm_padded_tokens.data_ptr()
    weight_workspace_ptr = experts._bmm_padded_weights.data_ptr()
    dynamic_again = experts.forward_bmm_fused(x, expert_indices, expert_weights)
    graphable = experts._forward_bmm_fused_graphable(x, expert_indices, expert_weights)

    assert experts._bmm_padded_tokens.data_ptr() == token_workspace_ptr
    assert experts._bmm_padded_weights.data_ptr() == weight_workspace_ptr
    torch.testing.assert_close(dynamic_again, dynamic)
    torch.testing.assert_close(graphable, dynamic)


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


def test_level5_bmm_path_reuses_padding_workspaces() -> None:
    source = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey" / "moe_model.py"
    text = source.read_text(encoding="utf-8")

    bmm_section = text.split("def forward_bmm_fused", maxsplit=1)[1].split(
        "def _forward_bmm_fused_graphable", maxsplit=1
    )[0]

    assert "def _bmm_workspace" in text
    assert "self._bmm_padded_tokens: Optional[torch.Tensor] = None" in text
    assert "self._bmm_padded_weights: Optional[torch.Tensor] = None" in text
    assert 'padded_tokens = self._bmm_workspace(' in bmm_section
    assert '"_bmm_padded_tokens",' in bmm_section
    assert 'padded_weights = self._bmm_workspace(' in bmm_section
    assert '"_bmm_padded_weights",' in bmm_section
    assert "padded_tokens.zero_()" not in bmm_section
    assert "padded_weights.zero_()" not in bmm_section
    assert "padded_tokens.scatter_(" in bmm_section
    assert "padded_weights.scatter_(" in bmm_section
    assert "torch.zeros(self.num_experts * max_count" not in bmm_section


def test_naive_moe_path_seeds_output_from_first_route() -> None:
    source = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey" / "moe_model.py"
    text = source.read_text(encoding="utf-8")

    naive_section = text.split("def forward_naive", maxsplit=1)[1].split(
        "def forward_batched",
        maxsplit=1,
    )[0]

    assert "output = torch.empty_like(x)" in naive_section
    assert "torch.zeros_like(x)" not in naive_section
    assert "for k in range(num_experts_per_tok):" in naive_section
    assert "token_ids = (expert_indices[:, k] == expert_idx).nonzero(as_tuple=True)[0]" in naive_section
    assert "if token_ids.numel() == 0:" in naive_section
    assert "if k == 0:" in naive_section
    assert "output[token_ids] = weighted_output" in naive_section
    assert "output[token_ids] += weighted_output" in naive_section
    assert "mask.any()" not in naive_section


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

    expected = experts.forward_naive(x, expert_indices, expert_weights, num_experts_per_tok=2)
    actual = experts.forward_grouped(x, expert_indices, expert_weights)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_grouped_moe_path_uses_shared_bucket_helpers() -> None:
    source = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey" / "moe_model.py"
    text = source.read_text(encoding="utf-8")

    grouped_section = text.split("def forward_grouped", maxsplit=1)[1].split(
        "def forward_bmm_fused", maxsplit=1
    )[0]

    assert "bucket_grouped_tokens(" in grouped_section
    assert "def _flat_topk_token_ids" in text
    assert "flat_token_ids = _flat_topk_token_ids(batch_seq, top_k, x.device)" in grouped_section
    assert "token_ids=flat_token_ids" in grouped_section
    assert "return_expert_order_list=True" in grouped_section
    assert "output = torch.empty(" in grouped_section
    assert "torch.zeros(sorted_tokens.shape[0]" not in grouped_section
    assert "repeat_interleave" not in grouped_section
    assert "for expert_id, count in zip(expert_order_host, counts)" in grouped_section
    assert "expert_order.tolist()" not in grouped_section
    assert "restore_grouped_tokens(" in grouped_section
    assert "torch.argsort" not in grouped_section
