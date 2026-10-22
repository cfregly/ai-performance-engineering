"""Regressions for W2-061 and the W2-021 residual in ch16/inference_optimizations_blackwell.py.

W2-061: every layer of a multi-layer model shared one cache length per row, so
layer 1 wrote after layer 0's tokens and read slots it never wrote. These tests
drive the real DynamicQuantizedKVCache, OptimizedDecoderLayer and
BlackwellInferencePipeline on CPU and compare cached decode with a full
recompute. They also check that the pipeline rejects models that skip the
cache or reuse one layer slot, and that the in-file demos run.

CPU PyTorch has no float8 kernel for tensor-index reads ("index_cpu"), so the
CPU tests either disable FP8 storage or use float16 as a stand-in storage dtype
to exercise the per-token scale path.
The CUDA tests at the end cover FlexAttention and real FP8 storage.
"""

from __future__ import annotations

import re
import time

import pytest
import torch
import torch.nn as nn

import ch16.inference_optimizations_blackwell as blackwell
from core.utils.compile_utils import configure_tf32, restore_tf32

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SKIPPED: CUDA GPU required for FlexAttention and FP8 KV storage",
)
# Compiled FlexAttention needs a head dimension of at least 16.
CUDA_D_MODEL = 64


@pytest.fixture
def ieee_fp32_matmul():
    precision = torch.get_float32_matmul_precision()
    previous = configure_tf32(enable_matmul=False, enable_cudnn=False)
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        restore_tf32(previous)
        torch.set_float32_matmul_precision(precision)


@pytest.fixture
def plain_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blackwell, "FP8_AVAILABLE", False)


@pytest.fixture
def scaled_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blackwell, "FP8_AVAILABLE", True)
    monkeypatch.setattr(blackwell, "FP8_E4M3", torch.float16)


def _cache(*, num_layers: int, max_batch_size: int, max_seq_len: int, num_heads: int = 1,
           head_dim: int = 2, device: str = "cpu") -> blackwell.DynamicQuantizedKVCache:
    return blackwell.DynamicQuantizedKVCache(
        num_layers=num_layers,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
        dtype=torch.float32,
    )


def _tokens(*magnitudes: float) -> torch.Tensor:
    """Build [1, 1, len, 2] keys whose per-token scale is the magnitude."""
    rows = [[magnitude, -magnitude / 2] for magnitude in magnitudes]
    return torch.tensor(rows, dtype=torch.float32).view(1, 1, len(magnitudes), 2)


@pytest.mark.parametrize("storage", ["plain_storage", "scaled_storage"])
def test_each_layer_appends_and_reads_only_its_own_history(
    storage: str,
    request: pytest.FixtureRequest,
) -> None:
    request.getfixturevalue(storage)
    # Capacity fits one prefill plus one decode token per layer. With a shared
    # length the second layer would overflow at prefill.
    cache = _cache(num_layers=3, max_batch_size=1, max_seq_len=5)
    prefill = [_tokens(1, 2, 3, 4), _tokens(10, 20, 30, 40), _tokens(5, 6, 7, 8)]
    decode = [_tokens(9), _tokens(50), _tokens(11)]

    for layer_idx, keys in enumerate(prefill):
        cached_key, cached_value = cache.update(layer_idx, keys, keys * 2)
        torch.testing.assert_close(cached_key, keys)
        torch.testing.assert_close(cached_value, keys * 2)

    for layer_idx, keys in enumerate(decode):
        cached_key, _ = cache.update(layer_idx, keys, keys)
        expected = torch.cat([prefill[layer_idx], keys], dim=2)
        torch.testing.assert_close(cached_key, expected)

    assert cache._layer_seq_lens_host == [[5], [5], [5]]
    assert cache._seq_lens_host == [5]
    torch.testing.assert_close(cache.seq_lens, torch.tensor([5]))

    with pytest.raises(ValueError, match="KV cache overflow"):
        cache.update(0, _tokens(1), _tokens(1))


def test_mixed_length_rows_use_per_layer_offsets(plain_storage: None) -> None:
    cache = _cache(num_layers=2, max_batch_size=2, max_seq_len=4)
    row0_layer0 = _tokens(1, 2)
    row0_layer1 = _tokens(10, 20)
    cache.update(0, row0_layer0, row0_layer0, batch_idx=0)
    cache.update(1, row0_layer1, row0_layer1, batch_idx=0)

    step = torch.cat([_tokens(3), _tokens(4)], dim=0)
    rows = torch.tensor([0, 1], dtype=torch.long)
    layer0_key, _ = cache.update(0, step, step, batch_indices=rows)
    layer0_key = layer0_key.clone()
    layer1_key, _ = cache.update(1, step * 10, step * 10, batch_indices=rows)

    torch.testing.assert_close(layer0_key[0], torch.cat([row0_layer0, _tokens(3)], dim=2)[0])
    torch.testing.assert_close(layer0_key[1, :, :1], _tokens(4)[0])
    torch.testing.assert_close(layer1_key[0], torch.cat([row0_layer1, _tokens(30)], dim=2)[0])
    torch.testing.assert_close(layer1_key[1, :, :1], _tokens(40)[0])
    # Row 1 is shorter, so its padded slots are zero.
    assert torch.count_nonzero(layer1_key[1, :, 1:]) == 0
    assert cache._layer_seq_lens_host == [[3, 1], [3, 1]]
    assert cache._seq_lens_host == [3, 1]
    torch.testing.assert_close(cache.seq_lens, torch.tensor([3, 1]))

    cache.clear(batch_idx=0)
    assert cache._layer_seq_lens_host == [[0, 1], [0, 1]]
    assert cache._seq_lens_host == [0, 1]
    cache.clear()
    assert cache._layer_seq_lens_host == [[0, 0], [0, 0]]
    torch.testing.assert_close(cache.seq_lens, torch.tensor([0, 0]))


def test_row_length_reports_the_furthest_layer(plain_storage: None) -> None:
    cache = _cache(num_layers=2, max_batch_size=1, max_seq_len=4)
    cache.update(0, _tokens(1, 2, 3), _tokens(1, 2, 3))
    layer1_key, _ = cache.update(1, _tokens(4), _tokens(4))

    torch.testing.assert_close(layer1_key, _tokens(4))
    assert cache._seq_lens_host == [3]
    torch.testing.assert_close(cache.seq_lens, torch.tensor([3]))
    assert cache._layer_seq_lens_host == [[3], [1]]


@pytest.mark.parametrize("layer_idx", [-1, 2])
def test_update_rejects_layer_outside_cache(plain_storage: None, layer_idx: int) -> None:
    cache = _cache(num_layers=2, max_batch_size=1, max_seq_len=4)
    with pytest.raises(IndexError, match="outside the cache's 2 layers"):
        cache.update(layer_idx, _tokens(1), _tokens(1))
    assert cache._layer_seq_lens_host == [[0], [0]]


def test_update_rejects_tensors_with_autograd_history(plain_storage: None) -> None:
    cache = _cache(num_layers=1, max_batch_size=1, max_seq_len=4)
    key = _tokens(1).requires_grad_()
    with pytest.raises(RuntimeError, match="require grad"):
        cache.update(0, key, key)
    with torch.no_grad():
        cache.update(0, key, key)
    assert not cache.cache.requires_grad

    layer = blackwell.OptimizedDecoderLayer(8, 2, device="cpu", use_flex_attention=False)
    layer_cache = _cache(num_layers=1, max_batch_size=1, max_seq_len=4, num_heads=2, head_dim=4)
    with pytest.raises(RuntimeError, match="require grad"):
        layer(torch.randn(1, 2, 8), kv_cache=layer_cache, layer_idx=0)


class _CacheAwareDecoderStack(nn.Module):
    """Small causal LM built from the real OptimizedDecoderLayer."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        window_size: int,
        device: str = "cpu",
        use_flex_attention: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.embed = nn.Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList(
            blackwell.OptimizedDecoderLayer(
                d_model,
                num_heads,
                window_size=window_size,
                device=device,
                use_flex_attention=use_flex_attention,
            )
            for _ in range(num_layers)
        )
        self.lm_head = nn.Linear(d_model, vocab_size, device=device)

    def forward(self, input_ids: torch.Tensor, kv_cache=None) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden = hidden + layer(hidden, kv_cache=kv_cache, layer_idx=layer_idx)
        return self.lm_head(hidden)


def _decoder_stack(
    window_size: int,
    *,
    vocab_size: int = 13,
    d_model: int = 16,
    device: str = "cpu",
    use_flex_attention: bool = False,
) -> _CacheAwareDecoderStack:
    torch.manual_seed(20261012)
    return _CacheAwareDecoderStack(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        num_layers=3,
        window_size=window_size,
        device=device,
        use_flex_attention=use_flex_attention,
    ).eval()


def _cached_and_recomputed_logits(
    model: nn.Module,
    cache: blackwell.DynamicQuantizedKVCache,
    prompt: torch.Tensor,
    decode_steps: int,
    reference_model: nn.Module | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return (cached, full recompute) logits for the prefill and each decoded position."""
    reference_model = model if reference_model is None else reference_model
    pairs = []
    with torch.inference_mode():
        logits = model(prompt, kv_cache=cache)
        pairs.append((logits.clone(), reference_model(prompt)))
        sequence = prompt
        for _ in range(decode_steps):
            next_token = logits[:, -1:].argmax(dim=-1)
            sequence = torch.cat([sequence, next_token], dim=1)
            logits = model(next_token, kv_cache=cache)
            pairs.append((logits.clone(), reference_model(sequence)[:, -1:]))
    return pairs


def test_compiled_flex_attention_bounds_dynamic_decode_recompilation(
    plain_storage: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the guard sequence that exhausted the shared compile cache on B200."""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    compiled_flex_attention = torch.compile(
        blackwell._flex_attention_wrapper,
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    monkeypatch.setattr(blackwell, "_FLEX_ATTENTION_FN", compiled_flex_attention)

    prompt = torch.tensor([[3, 7, 1, 12, 5]])
    decode_steps = 12
    try:
        for window_size in (3, 64):
            model = _decoder_stack(
                window_size=window_size,
                d_model=CUDA_D_MODEL,
                use_flex_attention=True,
            )
            cache = _cache(
                num_layers=model.num_layers,
                max_batch_size=1,
                max_seq_len=prompt.size(1) + decode_steps,
                num_heads=model.num_heads,
                head_dim=model.d_model // model.num_heads,
            )
            for cached, reference in _cached_and_recomputed_logits(
                model,
                cache,
                prompt,
                decode_steps,
            ):
                torch.testing.assert_close(cached, reference)

        model = _decoder_stack(
            window_size=64,
            d_model=CUDA_D_MODEL,
            use_flex_attention=True,
        ).to(torch.bfloat16)
        pipeline = blackwell.BlackwellInferencePipeline(
            model,
            max_seq_len=prompt.size(1) + decode_steps,
            compile=False,
        )
        generated = pipeline.generate(prompt, max_new_tokens=decode_steps)
        assert generated.shape == (1, prompt.size(1) + decode_steps)

        pipeline.kv_cache.clear()
        for cached, reference in _cached_and_recomputed_logits(
            model,
            pipeline.kv_cache,
            prompt,
            decode_steps,
        ):
            torch.testing.assert_close(cached, reference)
        unique_graphs = counters["stats"]["unique_graphs"]
        assert 1 <= unique_graphs <= 8
    finally:
        torch._dynamo.reset()


def test_dynamic_decode_compile_policy_disables_cuda_graph_capture(
    plain_storage: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Inductor compilation while excluding changing K/V shapes from graph replay."""
    callable_calls: list[tuple[object, dict[str, object]]] = []
    model_calls: list[tuple[nn.Module, dict[str, object]]] = []

    def record_callable(fn, **kwargs):
        callable_calls.append((fn, kwargs))
        return fn

    def record_model(module, **kwargs):
        model_calls.append((module, kwargs))
        return module

    def marker() -> None:
        return None
    monkeypatch.setattr(blackwell, "compile_callable", record_callable)
    assert blackwell._compile_dynamic_flex_attention(marker) is marker
    assert callable_calls == [
        (
            marker,
            {
                "mode": None,
                "fullgraph": True,
                "dynamic": True,
                "backend": "inductor",
                "options": {"triton.cudagraphs": False},
            },
        )
    ]

    monkeypatch.setattr(blackwell, "compile_model", record_model)
    model = _decoder_stack(window_size=3)
    pipeline = blackwell.BlackwellInferencePipeline(model, max_seq_len=17, compile=True)

    assert pipeline.model is model
    assert model_calls == [
        (
            model,
            {
                "mode": "max-autotune-no-cudagraphs",
                "fullgraph": False,
                "dynamic": True,
                "backend": "inductor",
            },
        )
    ]


@pytest.mark.parametrize("window_size", [2, 64])
@pytest.mark.parametrize("headroom", [0, 32])
def test_cached_decode_matches_full_recompute_for_every_layer(
    plain_storage: None,
    window_size: int,
    headroom: int,
) -> None:
    # headroom=0 sizes the cache exactly. headroom=32 leaves room for a shared
    # length to read slots another layer owns instead of overflowing.
    model = _decoder_stack(window_size)
    prompt = torch.tensor([[3, 7, 1, 12, 5]])
    decode_steps = 4
    cache = _cache(
        num_layers=model.num_layers,
        max_batch_size=1,
        max_seq_len=prompt.size(1) + decode_steps + headroom,
        num_heads=model.num_heads,
        head_dim=model.d_model // model.num_heads,
    )

    for cached, reference in _cached_and_recomputed_logits(model, cache, prompt, decode_steps):
        torch.testing.assert_close(cached, reference, rtol=1e-5, atol=1e-5)

    total = prompt.size(1) + decode_steps
    assert cache._layer_seq_lens_host == [[total]] * model.num_layers
    assert cache._seq_lens_host == [total]


def test_pipeline_generate_matches_greedy_recompute(plain_storage: None) -> None:
    model = _decoder_stack(window_size=3)
    prompt = torch.tensor([[2, 9, 4, 4, 11]])
    new_tokens = 4
    pipeline = blackwell.BlackwellInferencePipeline(
        model,
        max_batch_size=1,
        max_seq_len=prompt.size(1) + new_tokens,
        compile=False,
    )

    generated = pipeline.generate(prompt, max_new_tokens=new_tokens).clone()

    expected = prompt
    with torch.inference_mode():
        for _ in range(new_tokens):
            next_token = model(expected)[:, -1].argmax(dim=-1, keepdim=True)
            expected = torch.cat([expected, next_token], dim=1)
    assert torch.equal(generated, expected)
    assert pipeline.kv_cache.cache.dtype == torch.float32
    # The last generated token is never fed back, so each layer holds
    # prompt + new_tokens - 1 positions.
    expected_length = prompt.size(1) + new_tokens - 1
    assert pipeline.kv_cache._layer_seq_lens_host == [[expected_length]] * model.num_layers


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="SKIPPED: CUDA GPU required for CUDA argmax buffer coverage",
            ),
        ),
    ],
)
def test_pipeline_argmax_handles_strided_logits_and_reuses_output_buffers(
    plain_storage: None,
    batch_size: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    model = _decoder_stack(window_size=3, device=device).to(dtype=dtype)
    pipeline = blackwell.BlackwellInferencePipeline(
        model,
        max_batch_size=2,
        max_seq_len=4,
        compile=False,
    )
    vocab_size = 13

    def strided_logits(winners: torch.Tensor) -> torch.Tensor:
        storage = torch.full(
            (batch_size, 1, vocab_size),
            -4,
            dtype=dtype,
            device=device,
        )
        rows = torch.arange(batch_size, device=device)
        storage[rows, 0, winners] = 4
        logits = storage[:, -1, :]
        assert logits.stride() == (vocab_size, 1)
        return logits

    first_expected = torch.arange(batch_size, device=device, dtype=torch.long) + 2
    first_tokens = pipeline._next_token_from_logits(strided_logits(first_expected)).clone()
    values_ptr = pipeline._next_token_values.data_ptr()
    tokens_ptr = pipeline._next_token_buffer.data_ptr()

    torch.testing.assert_close(first_tokens, first_expected[:, None], rtol=0, atol=0)
    assert pipeline._next_token_values.is_contiguous()
    assert pipeline._next_token_values.stride() == pipeline._next_token_buffer.stride()

    second_expected = first_expected + 3
    second_tokens = pipeline._next_token_from_logits(strided_logits(second_expected))

    torch.testing.assert_close(second_tokens, second_expected[:, None], rtol=0, atol=0)
    assert pipeline._next_token_values.data_ptr() == values_ptr
    assert pipeline._next_token_buffer.data_ptr() == tokens_ptr


class _MiswiredDecoderStack(_CacheAwareDecoderStack):
    """Accepts kv_cache through **kwargs but does not give each layer its own slot."""

    def __init__(self, mode: str, *, vocab_size: int = 13) -> None:
        if mode not in ("ignore_cache", "omit_layer_idx", "shared_layer_idx"):
            raise ValueError(f"unknown mode {mode!r}")
        torch.manual_seed(20261012)
        super().__init__(vocab_size=vocab_size, d_model=16, num_heads=2, num_layers=3, window_size=64)
        self.mode = mode
        self.eval()

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        kv_cache = kwargs.get("kv_cache")
        hidden = self.embed(input_ids)
        for layer in self.layers:
            if self.mode == "ignore_cache":
                out = layer(hidden)
            elif self.mode == "omit_layer_idx":
                out = layer(hidden, kv_cache=kv_cache)
            else:
                out = layer(hidden, kv_cache=kv_cache, layer_idx=0)
            hidden = hidden + out
        return self.lm_head(hidden)


_PROMPT = torch.tensor([[2, 9, 4, 4, 11]])


def test_pipeline_generate_rejects_model_that_ignores_the_cache(plain_storage: None) -> None:
    pipeline = blackwell.BlackwellInferencePipeline(
        _MiswiredDecoderStack("ignore_cache"),
        max_seq_len=64,
        compile=False,
    )
    with pytest.raises(RuntimeError, match=re.escape("After prefill") + ".*" + re.escape("{0: [0], 1: [0], 2: [0]}")):
        pipeline.generate(_PROMPT, max_new_tokens=6)


def test_pipeline_generate_rejects_stack_that_omits_layer_idx(plain_storage: None) -> None:
    pipeline = blackwell.BlackwellInferencePipeline(
        _MiswiredDecoderStack("omit_layer_idx"),
        max_seq_len=64,
        compile=False,
    )
    with pytest.raises(ValueError, match="explicit layer_idx"):
        pipeline.generate(_PROMPT, max_new_tokens=6)
    assert pipeline.kv_cache._layer_seq_lens_host == [[0], [0], [0]]


def test_pipeline_generate_rejects_layers_sharing_one_cache_slot(plain_storage: None) -> None:
    pipeline = blackwell.BlackwellInferencePipeline(
        _MiswiredDecoderStack("shared_layer_idx"),
        max_seq_len=64,
        compile=False,
    )
    # Three layers each append the 5 prompt tokens to slot 0.
    with pytest.raises(RuntimeError, match=re.escape("{0: [15], 1: [0], 2: [0]}")):
        pipeline.generate(_PROMPT, max_new_tokens=6)


def test_pipeline_generate_rejects_more_rows_than_the_cache_holds(plain_storage: None) -> None:
    pipeline = blackwell.BlackwellInferencePipeline(
        _MiswiredDecoderStack("ignore_cache"),
        max_batch_size=1,
        max_seq_len=64,
        compile=False,
    )
    with pytest.raises(ValueError, match="got 2 rows.*max_batch_size=1"):
        pipeline.generate(_PROMPT.repeat(2, 1), max_new_tokens=2)


def test_scaled_storage_returns_the_requested_dtype(scaled_storage: None) -> None:
    cache = blackwell.DynamicQuantizedKVCache(
        num_layers=1,
        max_batch_size=2,
        max_seq_len=4,
        num_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.bfloat16,
    )
    prefill = _tokens(1, 2).to(torch.bfloat16)
    key, value = cache.update(0, prefill, prefill, batch_idx=0)
    assert key.dtype == value.dtype == torch.bfloat16
    torch.testing.assert_close(key, prefill)

    # Rows of different lengths take the padded path.
    step = torch.cat([_tokens(3), _tokens(4)], dim=0).to(torch.bfloat16)
    key, value = cache.update(0, step, step, batch_indices=torch.tensor([0, 1]))
    assert key.dtype == value.dtype == torch.bfloat16
    torch.testing.assert_close(key[0], torch.cat([prefill, _tokens(3).to(torch.bfloat16)], dim=2)[0])


def test_bf16_model_reads_the_scaled_cache_in_its_dtype(scaled_storage: None) -> None:
    model = _decoder_stack(window_size=64).to(torch.bfloat16)
    prompt = torch.tensor([[3, 7, 1, 12, 5]])
    decode_steps = 4
    cache = blackwell.DynamicQuantizedKVCache(
        num_layers=model.num_layers,
        max_batch_size=1,
        max_seq_len=prompt.size(1) + decode_steps,
        num_heads=model.num_heads,
        head_dim=model.d_model // model.num_heads,
        device="cpu",
        dtype=torch.bfloat16,
    )

    for cached, reference in _cached_and_recomputed_logits(model, cache, prompt, decode_steps):
        assert cached.dtype == torch.bfloat16
        assert torch.isfinite(cached).all()
        relative_error = float(
            (cached.float() - reference.float()).norm() / reference.float().norm()
        )
        # The CPU path uses float16 as a lossy stand-in for E4M3 storage. Keep
        # the cache's documented sub-0.5% logit-error contract explicit rather
        # than relying on dtype-specific elementwise defaults near zero.
        assert relative_error < 0.005, relative_error


class _NoGeometryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, input_ids: torch.Tensor, kv_cache=None) -> torch.Tensor:
        return self.proj(input_ids)


def test_pipeline_requires_model_geometry(plain_storage: None) -> None:
    with pytest.raises(ValueError, match="Missing: num_layers, d_model, num_heads"):
        blackwell.BlackwellInferencePipeline(_NoGeometryModel(), max_seq_len=4, compile=False)

    model = _NoGeometryModel()
    model.num_layers = 2
    model.d_model = 10
    model.num_heads = 4
    with pytest.raises(ValueError, match="positive multiple of num_heads"):
        blackwell.BlackwellInferencePipeline(model, max_seq_len=4, compile=False)


def test_tensor_parallel_forward_rejects_kv_cache() -> None:
    parallel = object.__new__(blackwell.TensorParallelMultiGPU)
    with pytest.raises(ValueError, match="does not consume a KV cache"):
        parallel.forward(torch.zeros(1, 1, dtype=torch.long), kv_cache=object())


def test_comparison_demo_runs_with_matching_dtype_and_fresh_prefills(
    scaled_storage: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the demo on CPU with its CUDA timing, sync and compile seams replaced.

    FlexAttention's compiled CUDA kernel is unavailable in this CPU orchestration
    test, so the layer uses its SDPA path. The real demo needs a B200 run.
    """
    timed_iterations: list[int] = []

    def cpu_latency_ms(fn, iterations: int) -> float:
        timed_iterations.append(iterations)
        for _ in range(iterations):
            fn()
        return 1.0

    monkeypatch.setattr(blackwell, "_benchmark_cuda_latency_ms", cpu_latency_ms)
    monkeypatch.setattr(blackwell, "_FLEX_ATTENTION_FN", None)
    monkeypatch.setattr(blackwell, "compile_model", lambda module, **_: module)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    blackwell.compare_inference_methods()

    assert timed_iterations == [10, 10, 10]


class _HostClockEvent:
    """Host-clock stand-in for torch.cuda.Event so the timing code can run on CPU."""

    def __init__(self, enable_timing: bool = False) -> None:
        del enable_timing
        self._seconds = 0.0

    def record(self, stream=None) -> None:
        del stream
        self._seconds = time.perf_counter()

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, end: _HostClockEvent) -> float:
        return (end._seconds - self._seconds) * 1000.0


@pytest.fixture
def host_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "Event", _HostClockEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)


def test_pipeline_benchmark_writes_cache_in_inference_mode(
    plain_storage: None,
    host_clock: None,
) -> None:
    """benchmark() drives the real model and cache. Only the CUDA clock seams are replaced."""
    # benchmark() draws prompt ids from [0, 32000).
    model = _decoder_stack(window_size=4, vocab_size=32000)
    pipeline = blackwell.BlackwellInferencePipeline(model, max_seq_len=8, compile=False)

    pipeline.benchmark(seq_len=8, num_iterations=2)

    assert pipeline.kv_cache._layer_seq_lens_host == [[8]] * model.num_layers


def test_pipeline_benchmark_rejects_model_that_ignores_the_cache(
    plain_storage: None,
    host_clock: None,
) -> None:
    model = _MiswiredDecoderStack("ignore_cache", vocab_size=32000)
    pipeline = blackwell.BlackwellInferencePipeline(model, max_seq_len=8, compile=False)

    with pytest.raises(RuntimeError, match="After warmup prefill"):
        pipeline.benchmark(seq_len=8, num_iterations=2)


def test_multigpu_shard_benchmark_runs_fresh_prefills_and_reports_the_slowest_rank(
    scaled_storage: None,
    host_clock: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run the real layer, cache and loop on CPU with the launch, all-reduce and clock seams replaced."""
    caches: list[blackwell.DynamicQuantizedKVCache] = []

    class RecordingCache(blackwell.DynamicQuantizedKVCache):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            caches.append(self)

    def all_reduce_with_slower_peer(tensor: torch.Tensor, op) -> None:
        assert op == torch.distributed.ReduceOp.MAX
        tensor.mul_(2.0)

    destroyed: list[bool] = []
    monkeypatch.setattr(
        blackwell,
        "_tensor_parallel_launch",
        lambda: (0, 2, torch.device("cpu"), True),
    )
    monkeypatch.setattr(blackwell.dist, "all_reduce", all_reduce_with_slower_peer)
    monkeypatch.setattr(blackwell.dist, "destroy_process_group", lambda: destroyed.append(True))
    monkeypatch.setattr(blackwell, "DynamicQuantizedKVCache", RecordingCache)
    # The compiled CUDA kernel is unavailable in this CPU orchestration test.
    monkeypatch.setattr(blackwell, "_FLEX_ATTENTION_FN", None)

    # The cache holds exactly one prefill, so a missing clear() overflows on the second call.
    blackwell.benchmark_multigpu_tensor_parallel(seq_len=32, num_iterations=3, warmup_iterations=2)

    (cache,) = caches
    assert (cache.num_heads, cache.head_dim, cache.max_seq_len) == (8, 128, 32)
    assert cache.compute_dtype == torch.float32
    assert cache._layer_seq_lens_host == [[32]]
    assert destroyed == [True]

    output = capsys.readouterr().out
    rank_ms = float(re.search(r"Rank 0 shard: ([0-9.]+) ms/iteration", output).group(1))
    slowest_ms = float(re.search(r"Slowest rank: ([0-9.]+) ms/iteration", output).group(1))
    group_tokens_per_sec = float(re.search(r"Group throughput: ([0-9]+) tokens/sec", output).group(1))
    assert slowest_ms == pytest.approx(2 * rank_ms, rel=0.02)
    assert group_tokens_per_sec == pytest.approx(32 * 1000 / slowest_ms, rel=0.02)
    assert "Total throughput" not in output


def test_multigpu_benchmark_requires_torchrun_with_two_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(RuntimeError, match="torchrun --nproc_per_node"):
        blackwell.benchmark_multigpu_tensor_parallel()

    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(RuntimeError, match="SKIPPED: .*at least 2 ranks"):
        blackwell.benchmark_multigpu_tensor_parallel()


def test_multigpu_launch_joins_the_torchrun_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(("set_device", device)))
    monkeypatch.setattr(blackwell.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(blackwell.dist, "init_process_group", lambda **kwargs: calls.append(("init", kwargs)))
    monkeypatch.setattr(blackwell.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(blackwell.dist, "get_world_size", lambda: 2)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    device = torch.device("cuda", 1)

    assert blackwell._tensor_parallel_launch() == (1, 2, device, True)
    assert calls == [("set_device", device), ("init", {"backend": "nccl", "device_id": device})]

    monkeypatch.setenv("LOCAL_RANK", "2")
    with pytest.raises(RuntimeError, match="LOCAL_RANK=2 has no GPU"):
        blackwell._tensor_parallel_launch()


@requires_cuda
def test_cuda_flex_attention_cached_decode_matches_full_recompute(
    plain_storage: None,
    ieee_fp32_matmul: None,
) -> None:
    model = _decoder_stack(window_size=3, d_model=CUDA_D_MODEL, device="cuda", use_flex_attention=True)
    assert all(layer.flex_attention_fn is not None for layer in model.layers)
    prompt = torch.tensor([[3, 7, 1, 12, 5]], device="cuda")
    # This exceeds the default eight-entry Dynamo cache and exercises twelve
    # increasing K/V lengths without CUDA Graph Trees replay.
    decode_steps = 12
    cache = _cache(
        num_layers=model.num_layers,
        max_batch_size=1,
        max_seq_len=prompt.size(1) + decode_steps,
        num_heads=model.num_heads,
        head_dim=model.d_model // model.num_heads,
        device="cuda",
    )

    for cached, reference in _cached_and_recomputed_logits(model, cache, prompt, decode_steps):
        torch.testing.assert_close(cached, reference, rtol=1e-4, atol=1e-4)
    total = prompt.size(1) + decode_steps
    assert cache._layer_seq_lens_host == [[total]] * model.num_layers


@requires_cuda
def test_cuda_fp8_cache_serves_every_layer(ieee_fp32_matmul: None) -> None:
    if not blackwell.FP8_AVAILABLE:
        pytest.skip("SKIPPED: this PyTorch build has no float8_e4m3fn dtype")
    model = _decoder_stack(window_size=64, d_model=CUDA_D_MODEL, device="cuda", use_flex_attention=True)
    prompt = torch.tensor([[3, 7, 1, 12, 5]], device="cuda")
    decode_steps = 4
    cache = blackwell.DynamicQuantizedKVCache(
        num_layers=model.num_layers,
        max_batch_size=1,
        max_seq_len=prompt.size(1) + decode_steps,
        num_heads=model.num_heads,
        head_dim=model.d_model // model.num_heads,
        device="cuda",
        dtype=torch.float32,
    )
    assert cache.cache.dtype == torch.float8_e4m3fn

    for cached, reference in _cached_and_recomputed_logits(model, cache, prompt, decode_steps):
        relative_error = float((cached - reference).norm() / reference.norm())
        # Emulating E4M3 K/V rounding on CPU for this geometry moved these
        # logits by at most 1.8 percent over 40 seeded runs. The bound leaves
        # margin for that rounding.
        assert relative_error < 0.05, relative_error
    total = prompt.size(1) + decode_steps
    assert cache._layer_seq_lens_host == [[total]] * model.num_layers


@requires_cuda
@pytest.mark.parametrize("compile_model", [False, True], ids=["eager-model", "compiled-model"])
def test_cuda_fp8_cache_serves_a_bf16_model_through_the_pipeline(compile_model: bool) -> None:
    if not blackwell.FP8_AVAILABLE:
        pytest.skip("SKIPPED: this PyTorch build has no float8_e4m3fn dtype")
    model = _decoder_stack(
        window_size=64,
        d_model=CUDA_D_MODEL,
        device="cuda",
        use_flex_attention=True,
    ).to(torch.bfloat16)
    prompt = torch.tensor([[3, 7, 1, 12, 5]], device="cuda")
    decode_steps = 12
    pipeline = blackwell.BlackwellInferencePipeline(
        model,
        max_seq_len=prompt.size(1) + decode_steps,
        compile=compile_model,
    )
    assert pipeline.kv_cache.cache.dtype == torch.float8_e4m3fn
    assert pipeline.kv_cache.compute_dtype == torch.bfloat16

    generated = pipeline.generate(prompt, max_new_tokens=decode_steps)
    assert generated.shape == (1, prompt.size(1) + decode_steps)

    pipeline.kv_cache.clear()
    for cached, reference in _cached_and_recomputed_logits(
        pipeline.model,
        pipeline.kv_cache,
        prompt,
        decode_steps,
        reference_model=model,
    ):
        assert cached.dtype == torch.bfloat16
        relative_error = float((cached.float() - reference.float()).norm() / reference.float().norm())
        # Emulating E4M3 K/V rounding on CPU for this bf16 model moved these
        # logits by at most 1.9 percent over 40 seeded runs.
        assert relative_error < 0.05, relative_error
