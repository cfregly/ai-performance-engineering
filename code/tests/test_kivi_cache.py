"""Asymmetric quantization axes, actual storage ownership, and decode parity."""

import pytest
import torch

from labs.kv_cache_compression.kivi_cache import encode_cache


@pytest.mark.parametrize("tokens", [1, 32, 33, 97, 512])
def test_packing_preserves_every_quantized_value_and_tail(tokens):
    generator = torch.Generator().manual_seed(602)
    keys = torch.randn(1, 2, tokens, 64, generator=generator).bfloat16()
    values = torch.randn(keys.shape, generator=generator).bfloat16()
    unpacked = encode_cache(keys, values, packed=False)
    packed = encode_cache(keys, values, packed=True)
    actual_k, actual_v = packed.decode()
    expected_k, expected_v = unpacked.decode()
    assert torch.equal(actual_k, expected_k)
    assert torch.equal(actual_v, expected_v)
    assert torch.equal(actual_k[:, :, packed.prefix_tokens :], keys[:, :, packed.prefix_tokens :])
    assert torch.equal(actual_v[:, :, packed.prefix_tokens :], values[:, :, packed.prefix_tokens :])
    assert unpacked.nbytes - packed.nbytes == 2 * 1 * 2 * packed.prefix_tokens * 64 * 3 // 4
    assert packed.key_tail.untyped_storage().data_ptr() != keys.untyped_storage().data_ptr()
    assert (
        packed.key_tail.untyped_storage().nbytes()
        == packed.key_tail.numel() * packed.key_tail.element_size()
    )


def test_key_channel_and_value_token_grouping_are_distinct_and_lossless_for_constants():
    keys = torch.arange(32, dtype=torch.bfloat16)[None, None, None, :].expand(1, 1, 96, 32).clone()
    values = (
        torch.arange(96, dtype=torch.bfloat16)[None, None, :, None].expand(1, 1, 96, 32).clone()
    )
    cache = encode_cache(keys, values)
    recovered_k, recovered_v = cache.decode()
    assert torch.equal(keys, recovered_k)
    assert torch.equal(values, recovered_v)
    assert cache.keys.minimum.shape == (1, 1, 2, 32, 1)
    assert cache.values.minimum.shape == (1, 1, 64, 1, 1)


def test_decoded_cache_runs_real_attention_with_finite_outputs():
    generator = torch.Generator().manual_seed(701)
    keys = torch.randn(1, 2, 97, 32, generator=generator).bfloat16()
    values = torch.randn(keys.shape, generator=generator).bfloat16()
    query = torch.randn(1, 2, 1, 32, generator=generator)
    cache = encode_cache(keys, values)
    k, v = cache.decode()
    output = torch.nn.functional.scaled_dot_product_attention(query, k.float(), v.float())
    scalar_scores = (query @ k.float().transpose(-1, -2)) / (32**0.5)
    reference = scalar_scores.softmax(-1) @ v.float()
    torch.testing.assert_close(output, reference)
    assert torch.isfinite(output).all()
    assert cache.nbytes < keys.numel() * keys.element_size() * 2


def test_inference_cache_does_not_retain_input_autograd_graph():
    keys = torch.randn(1, 1, 96, 32, dtype=torch.bfloat16, requires_grad=True)
    values = torch.randn_like(keys, requires_grad=True)
    cache = encode_cache(keys, values)
    for tensor in [
        cache.key_tail,
        cache.value_tail,
        cache.keys.minimum,
        cache.keys.scale,
        cache.values.minimum,
        cache.values.scale,
    ]:
        assert tensor.grad_fn is None and not tensor.requires_grad


def test_bad_kv_contracts_fail_and_goal_is_memory():
    from labs.kv_cache_compression.kivi_benchmark import KiviBenchmark

    x = torch.zeros(1, 1, 64, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        encode_cache(x, x, group_size=3)
    with pytest.raises(ValueError):
        encode_cache(x, x, residual_tokens=-1)
    with pytest.raises(ValueError):
        encode_cache(x.float(), x.float())
    for packed in [False, True]:
        bench = KiviBenchmark(packed)
        assert bench.get_optimization_goal() == "memory"
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="SKIPPED:.*CUDA"):
                bench.setup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real NVIDIA CUDA GPU required")
def test_actual_cuda_pair_verifies_complete_decoded_cache():
    from labs.kv_cache_compression.kivi_benchmark import KiviBenchmark

    for packed in [False, True]:
        bench = KiviBenchmark(packed, shape=(1, 2, 97, 32))
        bench.setup()
        bench.benchmark_fn()
        bench.capture_verification_payload()
        assert bench.get_verify_output().shape == (2, 1, 2, 97, 32)
        bench.teardown()
