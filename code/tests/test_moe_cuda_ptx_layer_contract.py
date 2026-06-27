from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

import labs.moe_cuda_ptx.moe_cuda_ptx_common as moe_common


def test_mxfp8_pow2_scales_masks_zero_blocks_in_place() -> None:
    source = inspect.getsource(moe_common._pow2_scales)

    assert "torch.where(" not in source
    assert "torch.full_like(scales" not in source
    assert "scales.masked_fill_(amax <= 0, _MXFP8_E8M0_MIN)" in source

    blocks = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -2.0, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    scales = moe_common._pow2_scales(blocks)

    assert scales[0].item() == moe_common._MXFP8_E8M0_MIN
    assert scales[1].item() >= moe_common._MXFP8_E8M0_MIN


def test_run_layer_cuda_forward_skips_standalone_quantize_roundtrip(monkeypatch) -> None:
    workload = moe_common.MoECudaPtxWorkload(mode="forward")
    sentinel_packed = SimpleNamespace(packed_tokens="packed_tokens")
    calls = {"pack": 0, "grouped": 0, "combine": 0}

    def _pack_topk_routes(*args, **kwargs):
        calls["pack"] += 1
        return sentinel_packed

    def _grouped_ffn_cuda(
        packed_tokens,
        packed,
        gate_proj,
        up_proj,
        down_proj,
        *,
        padded_tokens_buffer=None,
    ):
        calls["grouped"] += 1
        assert packed_tokens == "packed_tokens"
        assert packed is sentinel_packed
        return "sorted_outputs"

    def _combine_weighted_outputs(
        sorted_outputs,
        packed,
        num_tokens,
        *,
        output_buffer=None,
        consume_sorted_outputs=False,
    ):
        calls["combine"] += 1
        assert sorted_outputs == "sorted_outputs"
        assert packed is sentinel_packed
        assert num_tokens == workload.num_tokens
        assert consume_sorted_outputs is True
        return "combined_outputs"

    monkeypatch.setattr(moe_common, "pack_topk_routes", _pack_topk_routes)
    monkeypatch.setattr(moe_common, "grouped_ffn_cuda", _grouped_ffn_cuda)
    monkeypatch.setattr(moe_common, "combine_weighted_outputs", _combine_weighted_outputs)
    monkeypatch.setattr(
        moe_common,
        "quantize_mxfp8_optimized",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected quantize path")),
    )
    monkeypatch.setattr(
        moe_common,
        "dequantize_mxfp8",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected dequantize path")),
    )

    state = SimpleNamespace(
        x="x",
        expert_indices="expert_indices",
        expert_weights="expert_weights",
        gate_proj="gate_proj",
        up_proj="up_proj",
        down_proj="down_proj",
    )

    result = moe_common.run_layer_cuda(state, workload)

    assert result == "combined_outputs"
    assert calls == {"pack": 1, "grouped": 1, "combine": 1}
    source = inspect.getsource(moe_common.run_layer_cuda)
    assert "consume_sorted_outputs=True" in source


def test_combine_weighted_outputs_can_consume_sorted_outputs() -> None:
    source = inspect.getsource(moe_common.combine_weighted_outputs)
    assert "consume_sorted_outputs: bool = False" in source
    assert "weighted_outputs.mul_(weights)" in source
    assert "combined = torch.empty(" in source
    assert "combined.zero_()" not in source
    assert "combined = torch.zeros(" not in source
    assert "combined.index_add_(" not in source
    assert 'combined.scatter_reduce_(0, combine_index, weighted_outputs, reduce="sum", include_self=False)' in source
    assert "sorted_outputs * packed.packed_weights.unsqueeze(-1)" not in source

    sorted_outputs = torch.tensor([[2.0, 4.0], [3.0, 6.0], [5.0, 10.0]])
    original = sorted_outputs.clone()
    packed = SimpleNamespace(
        token_indices=torch.tensor([0, 0, 1], dtype=torch.long),
        packed_weights=torch.tensor([0.25, 0.75, 0.5]),
    )

    combined = moe_common.combine_weighted_outputs(
        sorted_outputs,
        packed,
        num_tokens=2,
        consume_sorted_outputs=True,
    )

    torch.testing.assert_close(sorted_outputs, original * packed.packed_weights.unsqueeze(-1))
    torch.testing.assert_close(combined, torch.tensor([[2.75, 5.5], [2.5, 5.0]]))

    output_buffer = torch.full((2, 2), float("nan"))
    combined_reused = moe_common.combine_weighted_outputs(
        original,
        packed,
        num_tokens=2,
        output_buffer=output_buffer,
    )
    assert combined_reused.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(combined_reused, torch.tensor([[2.75, 5.5], [2.5, 5.0]]))


def test_pack_topk_routes_reuses_start_offsets_without_cat() -> None:
    source = inspect.getsource(moe_common.pack_topk_routes)
    module_source = inspect.getsource(moe_common)
    assert "starts = torch.cat(" not in source
    assert "starts = torch.empty_like(counts)" in source
    assert "counts_cpu: Optional[Sequence[int]] = None" in source
    assert "counts = torch.tensor(counts_cpu, device=x.device, dtype=torch.long)" in source
    assert "repeat_interleave(top_k)" not in source
    assert "def _flat_topk_token_ids" in module_source
    assert 'token_ids.div_(top_k, rounding_mode="floor")' in module_source

    torch.testing.assert_close(
        moe_common._flat_topk_token_ids(3, 1, torch.device("cpu")),
        torch.tensor([0, 1, 2], dtype=torch.long),
    )
    torch.testing.assert_close(
        moe_common._flat_topk_token_ids(3, 2, torch.device("cpu")),
        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
    )

    x = torch.arange(20, dtype=torch.float32).view(5, 4)
    expert_indices = torch.tensor(
        [
            [1, 0],
            [2, 1],
            [0, 2],
            [1, 2],
            [0, 1],
        ],
        dtype=torch.long,
    )
    expert_weights = torch.ones_like(expert_indices, dtype=torch.float32)

    packed = moe_common.pack_topk_routes(
        x,
        expert_indices,
        expert_weights,
        num_experts=3,
    )

    expected_counts = torch.bincount(expert_indices.reshape(-1), minlength=3)
    expected_starts = torch.tensor(
        [0, int(expected_counts[0]), int(expected_counts[0] + expected_counts[1])],
        dtype=torch.long,
    )
    torch.testing.assert_close(packed.starts.cpu(), expected_starts)
    torch.testing.assert_close(
        packed.token_indices.cpu(),
        torch.tensor([0, 2, 4, 0, 1, 3, 4, 1, 2, 3], dtype=torch.long),
    )


def test_grouped_ffn_cuda_does_not_clear_discarded_padding_rows() -> None:
    source = inspect.getsource(moe_common.grouped_ffn_cuda)
    assert "padded_tokens = torch.empty(" in source
    assert "padded_tokens.zero_()" not in source
    assert "torch.zeros(flat_slots" not in source

    torch.manual_seed(0)
    x = torch.randn(4, 3)
    expert_indices = torch.tensor([[0], [0], [0], [1]], dtype=torch.long)
    expert_weights = torch.ones(4, 1)
    packed = moe_common.pack_topk_routes(
        x,
        expert_indices,
        expert_weights,
        num_experts=3,
        counts_cpu=(3, 1, 0),
    )
    gate_proj = torch.randn(3, 3, 5)
    up_proj = torch.randn(3, 3, 5)
    down_proj = torch.randn(3, 5, 3)
    padded = torch.empty(3 * packed.max_count, 3).fill_(float("nan"))

    expected = moe_common.grouped_ffn_reference(
        packed.packed_tokens,
        packed.counts_cpu,
        gate_proj,
        up_proj,
        down_proj,
    )
    actual = moe_common.grouped_ffn_cuda(
        packed.packed_tokens,
        packed,
        gate_proj,
        up_proj,
        down_proj,
        padded_tokens_buffer=padded,
    )

    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected)


def test_moe_cuda_ptx_skewed_routes_use_cpu_counts_without_fragment_cat() -> None:
    source = inspect.getsource(moe_common._build_primary_routes)
    count_source = "\n".join(
        (
            inspect.getsource(moe_common._counts_from_weights),
            inspect.getsource(moe_common._primary_route_counts_cpu),
            inspect.getsource(moe_common._route_counts_cpu),
        )
    )
    assert "int(count.item())" not in source
    assert "torch.cat(routes" not in source
    assert "torch.repeat_interleave(" in source
    assert "torch.linspace(" not in count_source
    assert ".sum().item()" not in count_source
    assert ".tolist()" not in count_source
    assert "_primary_route_ids_cpu" not in count_source

    balanced = moe_common.MoECudaPtxWorkload(
        num_tokens=19,
        num_experts=4,
        hidden_dim=32,
        expert_ffn_dim=64,
        capacity_factor=1.5,
        histogram="balanced",
    )
    balanced_indices, _ = moe_common.build_routes(balanced, torch.device("cpu"))
    balanced_counts_cpu = moe_common._route_counts_cpu(balanced)

    torch.testing.assert_close(
        torch.bincount(balanced_indices.reshape(-1), minlength=balanced.num_experts),
        torch.tensor(balanced_counts_cpu, dtype=torch.long),
    )

    workload = moe_common.MoECudaPtxWorkload(
        num_tokens=19,
        num_experts=4,
        hidden_dim=32,
        expert_ffn_dim=64,
        capacity_factor=1.5,
        histogram="skewed",
    )
    expert_indices, expert_weights = moe_common.build_routes(workload, torch.device("cpu"))
    counts_cpu = moe_common._route_counts_cpu(workload)

    torch.testing.assert_close(
        torch.bincount(expert_indices.reshape(-1), minlength=workload.num_experts),
        torch.tensor(counts_cpu, dtype=torch.long),
    )

    x = torch.arange(workload.num_tokens * 4, dtype=torch.float32).view(
        workload.num_tokens,
        4,
    )
    packed = moe_common.pack_topk_routes(
        x,
        expert_indices,
        expert_weights,
        num_experts=workload.num_experts,
        counts_cpu=counts_cpu,
    )

    assert packed.counts_cpu == counts_cpu
    torch.testing.assert_close(
        packed.counts.cpu(),
        torch.tensor(counts_cpu, dtype=torch.long),
    )


def test_moe_cuda_ptx_layer_forward_reuses_prepacked_routes() -> None:
    setup_source = inspect.getsource(moe_common.MoECudaPtxBenchmark.setup)
    forward_source = inspect.getsource(moe_common.MoECudaPtxBenchmark._benchmark_layer_forward)

    assert 'self.target != "moe_layer" or (' in setup_source
    assert 'self.backend == "cuda" and self.workload.mode == "forward"' in setup_source
    assert "counts_cpu=route_counts_cpu" in setup_source
    assert "packed=self.packed" in forward_source
    assert "padded_tokens_buffer=self._padded_tokens_buffer" in forward_source


def test_moe_cuda_ptx_backward_reuses_prepared_autograd_leaves() -> None:
    setup_source = inspect.getsource(moe_common.MoECudaPtxBenchmark.setup)
    prepare_source = inspect.getsource(moe_common.MoECudaPtxBenchmark._prepare_backward_tensors)
    grouped_bwd_source = inspect.getsource(moe_common.MoECudaPtxBenchmark._benchmark_grouped_gemm_bwd)
    layer_bwd_source = inspect.getsource(moe_common.MoECudaPtxBenchmark._benchmark_layer_fwd_bwd)
    grouped_run_source = inspect.getsource(moe_common.MoECudaPtxBenchmark._run_grouped_gemm_backward)

    assert "self._prepare_backward_tensors()" in setup_source
    assert "self._bwd_tokens = self.packed.packed_tokens.detach().clone().requires_grad_(True)" in prepare_source
    assert "self._bwd_x = self.state.x.detach().clone().requires_grad_(True)" in prepare_source
    assert "self._packed_loss_grad = self.state.loss_grad.index_select" in prepare_source
    assert ".detach().clone().requires_grad_(True)" not in grouped_bwd_source
    assert ".detach().clone().requires_grad_(True)" not in layer_bwd_source
    assert "self._packed_loss_grad" in grouped_run_source
    assert "self.state.loss_grad.index_select" not in grouped_run_source


def test_moe_cuda_ptx_build_state_reuses_route_count_tuple() -> None:
    source = inspect.getsource(moe_common.build_state)

    assert "_build_routes_with_counts(workload, device)" in source
    assert "route_counts_cpu=route_counts_cpu" in source
    assert "route_counts_cpu=_route_counts_cpu(workload)" not in source


def test_moe_cuda_ptx_baseline_output_buffer_avoids_double_zero_fill() -> None:
    source = inspect.getsource(moe_common.run_layer_baseline)

    assert "torch.empty_like(state.x)" in source
    assert "torch.zeros_like(state.x)" not in source
    assert "output.zero_()" in source


def test_moe_cuda_ptx_baseline_skips_redundant_mask_any_sync() -> None:
    source = inspect.getsource(moe_common.run_layer_baseline)

    assert "token_ids = (state.expert_indices[:, slot_idx] == expert_idx).nonzero(as_tuple=True)[0]" in source
    assert "if token_ids.numel() == 0:" in source
    assert "state.x[token_ids]" in source
    assert "state.expert_weights[token_ids, slot_idx]" in source
    assert "torch.any(mask)" not in source
    assert "mask]" not in source
