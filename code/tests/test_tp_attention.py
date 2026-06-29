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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a pure scaled-dot product attention reference for comparison."""
    batch_size, seq_len, _ = x.shape

    qkv = module.qkv_proj(x)
    qkv = qkv.reshape(batch_size, seq_len, 3, module.heads_per_gpu, module.head_dim)
    q, k, v = qkv.unbind(2)

    q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
    key_local = k.transpose(1, 2).contiguous()
    value_local = v.transpose(1, 2).contiguous()

    if kv_cache is None:
        attn_k = key_local
        attn_v = value_local
        attn_bias = None
    else:
        cache_shapes: List[Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]] = []
        max_cache_len = 0
        for cache_entry in kv_cache:
            if cache_entry is None:
                cache_shapes.append((0, None, None))
                continue
            cache_k, cache_v = cache_entry
            cache_len = cache_k.shape[1]
            cache_shapes.append((cache_len, cache_k, cache_v))
            max_cache_len = max(max_cache_len, cache_len)

        current_len = key_local.shape[2]
        total_k_len = max_cache_len + current_len
        attn_k = key_local.new_zeros(
            (batch_size, module.heads_per_gpu, total_k_len, module.head_dim)
        )
        attn_v = torch.zeros_like(attn_k)
        valid_mask = torch.zeros(
            (batch_size, total_k_len), dtype=torch.bool, device=attn_k.device
        )

        for batch_idx, (cache_len, cache_k, cache_v) in enumerate(cache_shapes):
            write_pos = 0
            if cache_len > 0:
                attn_k[batch_idx, :, :cache_len, :].copy_(cache_k)
                attn_v[batch_idx, :, :cache_len, :].copy_(cache_v)
                write_pos = cache_len

            attn_k[batch_idx, :, write_pos : write_pos + current_len, :].copy_(
                key_local[batch_idx]
            )
            attn_v[batch_idx, :, write_pos : write_pos + current_len, :].copy_(
                value_local[batch_idx]
            )
            valid_mask[batch_idx, : write_pos + current_len] = True

        attn_bias = None
        if not valid_mask.all():
            attn_bias = valid_mask.view(batch_size, 1, 1, total_k_len)
    out = F.scaled_dot_product_attention(
        q,
        attn_k,
        attn_v,
        dropout_p=0.0,
        attn_mask=attn_bias,
        is_causal=True,
    )
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

    module_out, key_local, value_local = attn(x, kv_cache=kv_cache)
    ref_out, ref_key, ref_value = _reference_attention(attn, x, kv_cache=_clone_kv_cache(kv_cache))

    torch.testing.assert_close(module_out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(key_local, ref_key)
    torch.testing.assert_close(value_local, ref_value)


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
