from __future__ import annotations

from typing import List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from ch16.inference_serving_multigpu import DemoCausalLM, TensorParallelAttention


def _reference_attention(
    module: TensorParallelAttention,
    x: torch.Tensor,
    kv_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    input_lengths: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a pure scaled-dot product attention reference for comparison."""
    batch_size, seq_len, _ = x.shape

    qkv = module.qkv_proj(x)
    qkv = qkv.reshape(batch_size, seq_len, 3, module.heads_per_gpu, module.head_dim)
    q, k, v = qkv.unbind(2)

    q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
    key_local = k.transpose(1, 2).contiguous()
    value_local = v.transpose(1, 2).contiguous()
    token_lengths = [seq_len] * batch_size if input_lengths is None else input_lengths

    if kv_cache is None and all(length == seq_len for length in token_lengths):
        attn_k = key_local
        attn_v = value_local
        attn_bias = None
        out = F.scaled_dot_product_attention(
            q,
            attn_k,
            attn_v,
            dropout_p=0.0,
            attn_mask=attn_bias,
            is_causal=True,
        )
    else:
        sample_outputs: List[torch.Tensor] = []
        for batch_idx in range(batch_size):
            delta_len = token_lengths[batch_idx]
            if delta_len == 0:
                sample_outputs.append(q[batch_idx : batch_idx + 1].mul(0.0))
                continue
            cache_entry = (
                kv_cache[batch_idx]
                if kv_cache is not None and batch_idx < len(kv_cache)
                else None
            )
            if cache_entry is None:
                cache_len = 0
                sample_k = key_local[batch_idx : batch_idx + 1, :, :delta_len, :]
                sample_v = value_local[batch_idx : batch_idx + 1, :, :delta_len, :]
            else:
                cache_k, cache_v = cache_entry
                cache_len = cache_k.shape[1]
                sample_k = torch.cat(
                    (
                        cache_k.unsqueeze(0),
                        key_local[batch_idx : batch_idx + 1, :, :delta_len, :],
                    ),
                    dim=2,
                )
                sample_v = torch.cat(
                    (
                        cache_v.unsqueeze(0),
                        value_local[batch_idx : batch_idx + 1, :, :delta_len, :],
                    ),
                    dim=2,
                )

            query_positions = torch.arange(delta_len, device=x.device).view(delta_len, 1)
            key_positions = torch.arange(sample_k.shape[2], device=x.device).view(1, -1)
            sample_mask = key_positions <= cache_len + query_positions
            sample_out = F.scaled_dot_product_attention(
                q[batch_idx : batch_idx + 1, :, :delta_len, :],
                sample_k,
                sample_v,
                dropout_p=0.0,
                attn_mask=sample_mask.view(1, 1, delta_len, -1),
                is_causal=False,
            )
            if delta_len < seq_len:
                sample_out = F.pad(sample_out, (0, 0, 0, seq_len - delta_len))
            sample_outputs.append(sample_out)
        out = torch.cat(sample_outputs, dim=0)
    out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
    out = module.out_proj(out)
    return out, key_local, value_local


def _clone_kv_cache(
    kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    cloned: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    for entry in kv_cache:
        if entry is None:
            cloned.append(None)
        else:
            cache_k, cache_v = entry
            cloned.append((cache_k.clone(), cache_v.clone()))
    return cloned


def test_tensor_parallel_attention_rejects_small_head_dim():
    with pytest.raises(ValueError, match="head_dim"):
        TensorParallelAttention(
            d_model=32,
            num_heads=4,
            num_gpus=1,
            max_batch_size=4,
            max_seq_len=16,
        )


def test_tensor_parallel_attention_requires_even_head_sharding():
    with pytest.raises(ValueError, match="divisible"):
        TensorParallelAttention(
            d_model=64,
            num_heads=6,
            num_gpus=4,
            max_batch_size=4,
            max_seq_len=16,
        )


def test_tensor_parallel_attention_forward_matches_reference():
    torch.manual_seed(0)
    attn = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=8,
        max_seq_len=32,
    )
    x = torch.randn(2, 5, 64)

    module_out, key_local, value_local = attn(x)
    ref_out, ref_key, ref_value = _reference_attention(attn, x)

    torch.testing.assert_close(module_out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(key_local, ref_key)
    torch.testing.assert_close(value_local, ref_value)
    assert key_local.shape == (2, attn.heads_per_gpu, 5, attn.head_dim)
    assert value_local.shape == (2, attn.heads_per_gpu, 5, attn.head_dim)


def test_tensor_parallel_attention_reuses_layout_projection_buffers():
    torch.manual_seed(3)
    attn = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=8,
        max_seq_len=32,
    )
    x = torch.randn(2, 5, 64)

    with torch.inference_mode():
        first_out, first_key, first_value = attn(x)
        key_ptr = attn._local_key_workspace.data_ptr()
        value_ptr = attn._local_value_workspace.data_ptr()
        merge_ptr = attn._attn_merge_buffer.data_ptr()
        output_ptr = attn._attn_output_buffer.data_ptr()
        weight_view_ptr = attn._out_proj_weight_t.data_ptr()

        second_out, second_key, second_value = attn(x)
    ref_out, ref_key, ref_value = _reference_attention(attn, x)

    torch.testing.assert_close(second_out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(second_key, ref_key)
    torch.testing.assert_close(second_value, ref_value)
    assert first_out.data_ptr() == output_ptr
    assert second_out.data_ptr() == output_ptr
    assert first_key.data_ptr() == key_ptr
    assert first_value.data_ptr() == value_ptr
    assert second_key.data_ptr() == key_ptr
    assert second_value.data_ptr() == value_ptr
    assert attn._local_key_workspace.data_ptr() == key_ptr
    assert attn._local_value_workspace.data_ptr() == value_ptr
    assert attn._attn_merge_buffer.data_ptr() == merge_ptr
    assert attn._attn_output_buffer.data_ptr() == output_ptr
    assert attn._out_proj_weight_t.data_ptr() == weight_view_ptr


def test_tensor_parallel_attention_with_kv_cache_matches_reference():
    torch.manual_seed(1)
    attn = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=8,
        max_seq_len=32,
    )
    x = torch.randn(2, 4, 64)

    kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
        (
            torch.randn(attn.heads_per_gpu, 3, attn.head_dim),
            torch.randn(attn.heads_per_gpu, 3, attn.head_dim),
        ),
        None,
    ]

    # Reused padding may contain arbitrary payloads from an earlier allocation.
    # NaNs make any accidental read deterministic instead of allocator-order dependent.
    with torch.inference_mode():
        attn(x, kv_cache=kv_cache)
        assert attn._attn_k_workspace is not None
        assert attn._attn_v_workspace is not None
        attn._attn_k_workspace.fill_(float("nan"))
        attn._attn_v_workspace.fill_(float("nan"))
        module_out, key_local, value_local = attn(x, kv_cache=kv_cache)
    ref_out, ref_key, ref_value = _reference_attention(attn, x, kv_cache=_clone_kv_cache(kv_cache))

    assert torch.isfinite(module_out).all()
    torch.testing.assert_close(module_out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(key_local, ref_key)
    torch.testing.assert_close(value_local, ref_value)


def test_tensor_parallel_attention_cached_queries_include_full_cache_prefix():
    attn = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=2,
        max_seq_len=8,
    )
    with torch.no_grad():
        attn.qkv_proj.weight.zero_()
        attn.out_proj.weight.copy_(torch.eye(64))

    x = torch.zeros(2, 2, 64)
    cache_k = torch.zeros(attn.heads_per_gpu, 3, attn.head_dim)
    cache_v = torch.stack(
        (
            torch.ones(attn.heads_per_gpu, attn.head_dim),
            torch.full((attn.heads_per_gpu, attn.head_dim), 3.0),
            torch.full((attn.heads_per_gpu, attn.head_dim), 5.0),
        ),
        dim=1,
    )

    out, _, _ = attn(x, kv_cache=[(cache_k, cache_v), None])

    torch.testing.assert_close(out[0, 0], torch.full((64,), 2.25))
    torch.testing.assert_close(out[1], torch.zeros_like(out[1]))


@pytest.mark.parametrize("invalid_length", [True, -1, 5, 1.0])
def test_tensor_parallel_attention_rejects_invalid_input_lengths(invalid_length):
    attn = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=2,
        max_seq_len=8,
    )

    with pytest.raises(ValueError, match=r"input_lengths\[0\]"):
        attn(torch.randn(2, 4, 64), input_lengths=[invalid_length, 4])


def test_tensor_parallel_attention_honors_lengths_without_kv_cache():
    torch.manual_seed(4)
    attn = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=3,
        max_seq_len=8,
    )
    x = torch.randn(3, 4, 64)
    input_lengths = [4, 2, 0]

    out, key_local, value_local = attn(x, input_lengths=input_lengths)
    ref_out, ref_key, ref_value = _reference_attention(
        attn,
        x,
        input_lengths=input_lengths,
    )

    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(key_local, ref_key)
    torch.testing.assert_close(value_local, ref_value)
    torch.testing.assert_close(out[1, 2:], torch.zeros_like(out[1, 2:]))
    torch.testing.assert_close(out[2], torch.zeros_like(out[2]))

    fast_path = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=3,
        max_seq_len=8,
    ).eval()
    with torch.inference_mode():
        fast_path(x, input_lengths=[4, 4, 4])
    assert fast_path._attn_k_workspace is None
    assert fast_path._attn_v_workspace is None
    assert fast_path._causal_mask_workspace is None


def test_tensor_parallel_attention_cached_workspaces_grow_monotonically():
    torch.manual_seed(5)
    attn = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=2,
        max_seq_len=64,
    ).eval()
    long_cache = [
        (
            torch.randn(attn.heads_per_gpu, 40, attn.head_dim),
            torch.randn(attn.heads_per_gpu, 40, attn.head_dim),
        )
        for _ in range(2)
    ]

    with torch.inference_mode():
        long_input = torch.randn(2, 1, 64)
        long_out, _, _ = attn(long_input, kv_cache=long_cache)
        ref_long_out, _, _ = _reference_attention(
            attn,
            long_input,
            kv_cache=long_cache,
        )
        torch.testing.assert_close(long_out, ref_long_out, rtol=1e-3, atol=1e-3)
        assert attn._attn_k_workspace is not None
        assert attn._attn_v_workspace is not None
        assert attn._causal_mask_workspace is None
        key_ptr = attn._attn_k_workspace.data_ptr()
        value_ptr = attn._attn_v_workspace.data_ptr()
        key_capacity = attn._attn_k_workspace.shape
        value_capacity = attn._attn_v_workspace.shape

        attn(
            torch.randn(2, 12, 64),
            kv_cache=[None, None],
            input_lengths=[12, 11],
        )
        assert attn._causal_mask_workspace is not None
        mask_ptr = attn._causal_mask_workspace.data_ptr()
        mask_capacity = attn._causal_mask_workspace.numel()
        assert attn._attn_k_workspace.data_ptr() == key_ptr
        assert attn._attn_v_workspace.data_ptr() == value_ptr
        assert attn._attn_k_workspace.shape == key_capacity
        assert attn._attn_v_workspace.shape == value_capacity

        attn(torch.randn(2, 1, 64), kv_cache=long_cache)

    assert attn._attn_k_workspace.data_ptr() == key_ptr
    assert attn._attn_v_workspace.data_ptr() == value_ptr
    assert attn._causal_mask_workspace.data_ptr() == mask_ptr
    assert attn._attn_k_workspace.shape == key_capacity
    assert attn._attn_v_workspace.shape == value_capacity
    assert attn._causal_mask_workspace.numel() == mask_capacity


def test_tensor_parallel_attention_large_heterogeneous_mask_uses_bounded_fallback():
    torch.manual_seed(6)
    attn = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=2,
        max_seq_len=16,
        max_explicit_mask_elements=1,
    )
    x = torch.randn(2, 4, 64, requires_grad=True)
    cache_k = torch.randn(attn.heads_per_gpu, 3, attn.head_dim, requires_grad=True)
    cache_v = torch.randn(attn.heads_per_gpu, 3, attn.head_dim, requires_grad=True)
    kv_cache = [(cache_k, cache_v), None]
    input_lengths = [4, 3]

    out, key_local, value_local = attn(
        x,
        kv_cache=kv_cache,
        input_lengths=input_lengths,
    )
    ref_out, ref_key, ref_value = _reference_attention(
        attn,
        x,
        kv_cache=kv_cache,
        input_lengths=input_lengths,
    )

    assert attn._causal_mask_workspace is None
    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(key_local, ref_key)
    torch.testing.assert_close(value_local, ref_value)
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert cache_k.grad is not None and torch.isfinite(cache_k.grad).all()
    assert cache_v.grad is not None and torch.isfinite(cache_v.grad).all()


@pytest.mark.parametrize("uniform_cache", [False, True])
def test_tensor_parallel_attention_cached_autograd_supports_repeated_backward(
    uniform_cache,
):
    torch.manual_seed(7)
    attn = TensorParallelAttention(
        d_model=64,
        num_heads=4,
        num_gpus=1,
        max_batch_size=2,
        max_seq_len=8,
    )

    for _ in range(2):
        attn.zero_grad(set_to_none=True)
        x = torch.randn(2, 3, 64, requires_grad=True)
        cache_k = torch.randn(attn.heads_per_gpu, 2, attn.head_dim, requires_grad=True)
        cache_v = torch.randn(attn.heads_per_gpu, 2, attn.head_dim, requires_grad=True)
        second_cache_k = torch.randn(
            attn.heads_per_gpu,
            2,
            attn.head_dim,
            requires_grad=True,
        )
        second_cache_v = torch.randn(
            attn.heads_per_gpu,
            2,
            attn.head_dim,
            requires_grad=True,
        )
        second_cache = (
            (second_cache_k, second_cache_v) if uniform_cache else None
        )
        out, _, _ = attn(
            x,
            kv_cache=[(cache_k, cache_v), second_cache],
            input_lengths=[3, 3] if uniform_cache else [3, 2],
        )
        out.sum().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert cache_k.grad is not None and torch.isfinite(cache_k.grad).all()
        assert cache_v.grad is not None and torch.isfinite(cache_v.grad).all()
        if uniform_cache:
            assert second_cache_k.grad is not None and torch.isfinite(second_cache_k.grad).all()
            assert second_cache_v.grad is not None and torch.isfinite(second_cache_v.grad).all()


def test_demo_causal_lm_forward_shape():
    torch.manual_seed(2)
    model = DemoCausalLM(
        vocab_size=128,
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_gpus=1,
    )
    input_ids = torch.randint(0, 128, (2, 6))

    logits, keys, values = model(input_ids)

    assert logits.shape == (2, 128)
    heads_per_gpu = model.num_heads // model.num_gpus
    expected_shape = (model.num_layers, 2, heads_per_gpu, input_ids.size(1), model.head_dim)
    assert keys.shape == expected_shape
    assert values.shape == expected_shape
    assert model._key_stack_buffer is None
    assert model._value_stack_buffer is None


def test_demo_causal_lm_reuses_kv_stack_buffers_in_inference():
    torch.manual_seed(3)
    model = DemoCausalLM(
        vocab_size=128,
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_gpus=1,
    ).eval()
    input_ids = torch.randint(0, 128, (2, 6))

    with torch.inference_mode():
        _logits, keys, values = model(input_ids)
        key_ptr = keys.data_ptr()
        value_ptr = values.data_ptr()
        keys_snapshot = keys.clone()
        values_snapshot = values.clone()
        _logits_again, keys_again, values_again = model(input_ids)

    assert keys_again.data_ptr() == key_ptr
    assert values_again.data_ptr() == value_ptr
    torch.testing.assert_close(keys_again, keys_snapshot)
    torch.testing.assert_close(values_again, values_snapshot)


def test_demo_causal_lm_invalid_head_dim_propagates():
    with pytest.raises(ValueError, match="head_dim"):
        DemoCausalLM(
            vocab_size=128,
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_gpus=1,
        )
