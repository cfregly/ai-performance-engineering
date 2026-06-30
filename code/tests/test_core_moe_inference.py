from __future__ import annotations

import inspect

import torch

from core.optimization import moe_inference
from core.optimization.moe_inference import (
    MoEFeedForward,
    MoEFeedForwardNoHostSync,
    MoEFeedForwardSortedDispatch,
    SimpleMoEBlock,
    allocate_kv_cache,
    dtype_bytes,
)


def test_allocate_kv_cache_avoids_zero_fill() -> None:
    source = inspect.getsource(allocate_kv_cache)
    assert "torch.empty(" in source
    assert "torch.zeros(" not in source

    cache = allocate_kv_cache(2, 3, 4, torch.float32, torch.device("cpu"))
    assert cache.shape == (2, 3, 4)
    assert cache.dtype == torch.float32


def test_dtype_bytes_uses_dtype_metadata_without_tensor_materialization() -> None:
    source = inspect.getsource(moe_inference.dtype_bytes)

    assert "torch.finfo(dt).bits // 8" in source
    assert "torch.iinfo(dt).bits // 8" in source
    assert "torch.tensor([], dtype=dt)" not in source
    assert dtype_bytes("float32") == 4
    assert dtype_bytes(torch.bfloat16) == 2
    assert dtype_bytes(torch.int64) == 8
    assert dtype_bytes(torch.bool) == 1


def test_simple_moe_block_reuses_attention_norm_once() -> None:
    source = inspect.getsource(SimpleMoEBlock.forward)
    assert "attn_input = self.ln_attn(hidden)" in source
    assert "self.attn(attn_input, attn_input, attn_input" in source
    assert source.count("self.ln_attn(hidden)") == 1


def test_sorted_dispatch_reuses_flat_token_id_cache_on_cpu() -> None:
    source = inspect.getsource(MoEFeedForwardSortedDispatch)
    forward_source = inspect.getsource(MoEFeedForwardSortedDispatch.forward)
    base_source = inspect.getsource(MoEFeedForward)

    assert "def _scaled_expert_output(" in base_source
    assert "expert_out.mul_(weights)" in base_source
    assert "return expert_out * weights" in base_source
    assert "def _expert_metadata_lists(" in source
    assert "def _route_workspaces(" in source
    assert "metadata_slice[0].copy_(unique_experts)" in source
    assert "metadata_slice[1].copy_(counts)" in source
    assert "host_slice.copy_(metadata_slice)" in source
    assert "expert_list, count_list = self._expert_metadata_lists(unique_experts, counts)" in forward_source
    assert "unique_experts.tolist()" not in forward_source
    assert "counts.tolist()" not in forward_source

    torch.manual_seed(123)
    layer = MoEFeedForwardSortedDispatch(
        hidden=8,
        ffn=16,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    x = torch.randn(2, 3, 8)

    out1 = layer(x)
    cache1 = layer._token_ids_cache
    metadata_ptr = layer._expert_metadata_buffer.data_ptr()
    assert cache1.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    out2 = layer(x)
    cache2 = layer._token_ids_cache

    assert cache2.data_ptr() == cache1.data_ptr()
    assert layer._expert_metadata_buffer.data_ptr() == metadata_ptr
    torch.testing.assert_close(out2, out1)

    _ = layer(torch.randn(1, 2, 8))

    assert layer._token_ids_cache.numel() == 4
    assert layer._token_ids_cache.data_ptr() != cache1.data_ptr()


def test_sorted_dispatch_inference_reuses_route_workspaces() -> None:
    forward_source = inspect.getsource(MoEFeedForwardSortedDispatch.forward)

    assert "if torch.is_grad_enabled():" in forward_source
    assert "self._route_workspaces(" in forward_source
    assert "torch.index_select(token_ids, 0, perm, out=sorted_token_ids)" in forward_source
    assert "torch.index_select(weights, 0, perm, out=sorted_weights)" in forward_source
    assert "torch.index_select(flat, 0, sorted_token_ids, out=sorted_flat)" in forward_source
    assert "expert_input = sorted_flat.narrow(0, segment_start, count)" in forward_source
    assert "weighted_out = self._scaled_expert_output(expert_out, segment_weights)" in forward_source
    assert "weighted_out = expert_out * segment_weights" not in forward_source
    source = inspect.getsource(MoEFeedForward)
    assert "def _topk_route_scores(" in source
    assert "self._topk_scores: Optional[torch.Tensor] = None" in source
    assert "self._topk_indices: Optional[torch.Tensor] = None" in source
    assert "torch.topk(logits, k=self.top_k, dim=-1, out=(self._topk_scores, self._topk_indices))" in source
    assert "self._topk_scores.sub_(torch.logsumexp(logits, dim=-1, keepdim=True)).exp_()" in source
    assert "top_scores = torch.exp(top_logits - torch.logsumexp(logits, dim=-1, keepdim=True))" in source
    assert "self._topk_route_scores(logits, reusable=not collect_router_stats)" in forward_source
    assert "probs = torch.softmax(logits, dim=-1)" not in source

    torch.manual_seed(123)
    baseline = MoEFeedForward(
        hidden=8,
        ffn=16,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    sorted_dispatch = MoEFeedForwardSortedDispatch(
        hidden=8,
        ffn=16,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    sorted_dispatch.load_state_dict(baseline.state_dict(), strict=True)

    logits = torch.randn(5, 4)
    with torch.inference_mode():
        top_scores, top_indices = sorted_dispatch._topk_route_scores(logits, reusable=True)
    expected_scores, expected_indices = torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1)
    torch.testing.assert_close(top_scores, expected_scores)
    torch.testing.assert_close(top_indices, expected_indices)

    x = torch.randn(2, 3, 8)

    with torch.inference_mode():
        expected = baseline(x)
        actual1 = sorted_dispatch(x)
        topk_score_ptr = sorted_dispatch._topk_scores.data_ptr()
        topk_index_ptr = sorted_dispatch._topk_indices.data_ptr()
        token_ptr = sorted_dispatch._sorted_token_ids_workspace.data_ptr()
        weight_ptr = sorted_dispatch._sorted_weights_workspace.data_ptr()
        flat_ptr = sorted_dispatch._sorted_flat_workspace.data_ptr()
        actual2 = sorted_dispatch(x)

    torch.testing.assert_close(actual1, expected)
    torch.testing.assert_close(actual2, expected)
    assert sorted_dispatch._topk_scores.data_ptr() == topk_score_ptr
    assert sorted_dispatch._topk_indices.data_ptr() == topk_index_ptr
    assert sorted_dispatch._sorted_token_ids_workspace.data_ptr() == token_ptr
    assert sorted_dispatch._sorted_weights_workspace.data_ptr() == weight_ptr
    assert sorted_dispatch._sorted_flat_workspace.data_ptr() == flat_ptr


def test_sorted_dispatch_top1_uses_write_once_output_path() -> None:
    forward_source = inspect.getsource(MoEFeedForwardSortedDispatch.forward)

    assert "single_route = self.top_k == 1" in forward_source
    assert "torch.empty_like(flat) if single_route else torch.zeros_like(flat)" in forward_source
    assert "combined.index_copy_(0, segment_tokens, weighted_out)" in forward_source
    assert "combined.index_add_(0, segment_tokens, weighted_out)" in forward_source

    torch.manual_seed(123)
    baseline = MoEFeedForward(
        hidden=8,
        ffn=16,
        num_experts=4,
        top_k=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    sorted_dispatch = MoEFeedForwardSortedDispatch(
        hidden=8,
        ffn=16,
        num_experts=4,
        top_k=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    sorted_dispatch.load_state_dict(baseline.state_dict(), strict=True)
    x = torch.randn(2, 3, 8)

    torch.testing.assert_close(sorted_dispatch(x), baseline(x))


def test_no_host_sync_top1_uses_write_once_output_path() -> None:
    forward_source = inspect.getsource(MoEFeedForwardNoHostSync.forward)

    assert "single_route = self.top_k == 1" in forward_source
    assert "torch.empty_like(flat) if single_route else torch.zeros_like(flat)" in forward_source
    assert "weighted_out = self._scaled_expert_output(expert_out, selected_weights)" in forward_source
    assert "combined.index_copy_(0, indices, weighted_out)" in forward_source
    assert "combined.index_add_(0, indices, weighted_out)" in forward_source

    torch.manual_seed(123)
    baseline = MoEFeedForward(
        hidden=8,
        ffn=16,
        num_experts=4,
        top_k=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    no_host_sync = MoEFeedForwardNoHostSync(
        hidden=8,
        ffn=16,
        num_experts=4,
        top_k=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    no_host_sync.load_state_dict(baseline.state_dict(), strict=True)
    x = torch.randn(2, 3, 8)

    torch.testing.assert_close(no_host_sync(x), baseline(x))


def test_moe_capacity_mask_avoids_float_mask_materialization() -> None:
    source = inspect.getsource(moe_inference)
    assert "(~drop_mask).float()" not in source
    assert source.count("top_scores.masked_fill_(drop_mask, 0.0)") == 3

    x = torch.randn(2, 3, 8)
    for layer_cls in (MoEFeedForward, MoEFeedForwardNoHostSync, MoEFeedForwardSortedDispatch):
        torch.manual_seed(123)
        layer = layer_cls(
            hidden=8,
            ffn=16,
            num_experts=2,
            top_k=2,
            capacity_factor=0.25,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        out = layer(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()
