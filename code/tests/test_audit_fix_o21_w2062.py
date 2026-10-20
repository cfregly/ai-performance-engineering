"""Regression tests for W2-062: radix-tree KV cache appends.

Shared prefixes must stay intact when branches extend them, capacity must
grow only when an allocation is full, and a token loop must not copy the
whole prefix on every append.
"""

from __future__ import annotations

import re

import torch

from ch16.radix_attention_example import (
    KVCache,
    ModelState,
    RadixTree,
    SimpleTransformerModel,
    benchmark_prefix_caching,
    generate_with_radix,
)

BLOCK_TOKENS = 4


def _small_model() -> SimpleTransformerModel:
    torch.manual_seed(0)
    model = SimpleTransformerModel(
        vocab_size=64,
        hidden_dim=16,
        num_heads=4,
        device=torch.device("cpu"),
    )
    model._cache_block_tokens = BLOCK_TOKENS
    return model


@torch.inference_mode()
def _expected_kv(model: SimpleTransformerModel, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token K and V rows computed directly from the model's own projections."""
    x = model.ln(model.embedding(torch.tensor(tokens, dtype=torch.long, device=model.device)))
    shape = (len(tokens), model.num_heads, model.head_dim)
    return model.k_proj(x).view(shape), model.v_proj(x).view(shape)


def _assert_cache_holds(model: SimpleTransformerModel, cache: KVCache, tokens: list[int]) -> None:
    expected_k, expected_v = _expected_kv(model, tokens)
    assert cache.seq_len == len(tokens)
    torch.testing.assert_close(cache.key_view, expected_k)
    torch.testing.assert_close(cache.value_view, expected_v)


def _empty_state(model: SimpleTransformerModel) -> ModelState:
    return ModelState(hidden_dim=model.hidden_dim, num_heads=model.num_heads, device=model.device)


def _prefill_into_tree(
    model: SimpleTransformerModel,
    radix: RadixTree,
    tokens: list[int],
) -> ModelState:
    """Prefill the uncached suffix of ``tokens`` and cache every prefix, as generate_with_radix does."""
    node, prefix_len = radix.longest_prefix(tokens)
    state = (
        ModelState.from_cache(node.cache, model.hidden_dim, model.num_heads)
        if node.cache
        else _empty_state(model)
    )
    for i, token in enumerate(tokens[prefix_len:]):
        state = model.forward(token, state)
        radix.insert(tokens[: prefix_len + i + 1], cache=state.kv_cache)
    return state


def _cached(radix: RadixTree, tokens: list[int]) -> KVCache:
    node, matched = radix.longest_prefix(tokens)
    assert matched == len(tokens)
    assert node.cache is not None
    return node.cache


def test_token_loop_grows_capacity_only_when_full() -> None:
    model = _small_model()
    tokens = list(range(1, 41))
    state = _empty_state(model)
    states = []
    for token in tokens:
        state = model.forward(token, state)
        states.append(state)
        cache = state.kv_cache
        # Doubling only when full keeps capacity within 2x of the live length.
        assert cache.capacity <= max(BLOCK_TOKENS, 2 * cache.seq_len), (
            f"capacity {cache.capacity} for {cache.seq_len} tokens"
        )

    # 40 tokens from a 4-row block need allocations of 4, 8, 16, 32 and 64 rows.
    # Copying on every append would make one allocation per token.
    allocations = {s.kv_cache.keys.data_ptr() for s in states}
    assert len(allocations) == 5
    assert states[-1].kv_cache.capacity == 64

    for length, earlier in enumerate(states, start=1):
        _assert_cache_holds(model, earlier.kv_cache, tokens[:length])


def test_sibling_branches_keep_their_own_rows() -> None:
    model = _small_model()
    radix = RadixTree()
    prefix = [1, 2, 3]
    branch_a = prefix + [10, 11]
    branch_b = prefix + [20, 21]

    _prefill_into_tree(model, radix, prefix)
    prefix_cache = _cached(radix, prefix)
    _prefill_into_tree(model, radix, branch_a)
    _prefill_into_tree(model, radix, branch_b)

    for branch in (branch_a, branch_b):
        for length in range(1, len(branch) + 1):
            _assert_cache_holds(model, _cached(radix, branch[:length]), branch[:length])

    # The first branch owns the unused row after the prefix and writes it in place.
    # The second branch finds that row taken and copies the prefix instead.
    assert _cached(radix, prefix + [10]).keys.data_ptr() == prefix_cache.keys.data_ptr()
    assert _cached(radix, prefix + [20]).keys.data_ptr() != prefix_cache.keys.data_ptr()


def test_branch_from_middle_of_cached_chain_does_not_overwrite_chain() -> None:
    model = _small_model()
    model._cache_block_tokens = 8
    radix = RadixTree()
    chain = [1, 2, 3, 4, 5, 6, 7]
    fork = [1, 2, 3, 30, 31]

    _prefill_into_tree(model, radix, chain)
    chain_allocation = _cached(radix, chain).keys.data_ptr()
    _prefill_into_tree(model, radix, fork)

    for length in range(1, len(chain) + 1):
        _assert_cache_holds(model, _cached(radix, chain[:length]), chain[:length])
    _assert_cache_holds(model, _cached(radix, fork), fork)
    assert _cached(radix, chain).keys.data_ptr() == chain_allocation
    assert _cached(radix, fork).keys.data_ptr() != chain_allocation


def test_direct_kvcache_siblings_and_capacity() -> None:
    keys = torch.zeros(2, 1, 1)
    values = torch.zeros_like(keys)
    keys[:, 0, 0] = torch.tensor([1.0, 2.0])
    values[:, 0, 0] = torch.tensor([10.0, 20.0])
    prefix = KVCache(keys=keys, values=values, seq_len=2, capacity=2)

    left = prefix.clone_ref().append(torch.tensor([[[3.0]]]), torch.tensor([[[30.0]]]))
    right = prefix.clone_ref().append(torch.tensor([[[4.0]]]), torch.tensor([[[40.0]]]))
    left_more = left.append(torch.tensor([[[5.0]]]), torch.tensor([[[50.0]]]))

    torch.testing.assert_close(prefix.key_view[:, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(left.key_view[:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(left_more.value_view[:, 0, 0], torch.tensor([10.0, 20.0, 30.0, 50.0]))
    torch.testing.assert_close(right.key_view[:, 0, 0], torch.tensor([1.0, 2.0, 4.0]))
    torch.testing.assert_close(right.value_view[:, 0, 0], torch.tensor([10.0, 20.0, 40.0]))
    assert left.capacity == 4
    assert right.capacity == 4
    assert left_more.keys.data_ptr() == left.keys.data_ptr()
    assert right.keys.data_ptr() != left.keys.data_ptr()


def test_generate_with_radix_keeps_shared_prefix_branches_intact() -> None:
    model = _small_model()
    radix = RadixTree()
    prompt_a = [1, 2, 3, 4, 5, 6, 7]
    prompt_b = [1, 2, 3, 4, 5, 8, 9]

    generate_with_radix(prompt_a, model, radix, max_generated_tokens=2)
    generate_with_radix(prompt_b, model, radix, max_generated_tokens=2)

    _assert_cache_holds(model, _cached(radix, prompt_a), prompt_a)
    _assert_cache_holds(model, _cached(radix, prompt_b), prompt_b)


def test_benchmark_prefix_caching_completes_with_demo_sized_prompts(capsys) -> None:
    torch.manual_seed(0)
    benchmark_prefix_caching()
    out = capsys.readouterr().out

    reused = [(int(a), int(b)) for a, b in re.findall(r"Reused (\d+)/(\d+) tokens from cache", out)]
    assert reused == [(64, 80), (64, 88), (64, 72), (64, 96), (80, 80), (88, 88), (72, 72), (96, 96)]
    assert "RadixAttention approach took:" in out
