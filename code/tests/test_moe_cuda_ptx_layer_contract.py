from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

import labs.moe_cuda_ptx.moe_cuda_ptx_common as moe_common


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

    def _combine_weighted_outputs(sorted_outputs, packed, num_tokens, *, output_buffer=None):
        calls["combine"] += 1
        assert sorted_outputs == "sorted_outputs"
        assert packed is sentinel_packed
        assert num_tokens == workload.num_tokens
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


def test_pack_topk_routes_reuses_start_offsets_without_cat() -> None:
    source = inspect.getsource(moe_common.pack_topk_routes)
    module_source = inspect.getsource(moe_common)
    assert "starts = torch.cat(" not in source
    assert "starts = torch.empty_like(counts)" in source
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
