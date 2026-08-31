"""Nanochat cache/control tests; CUDA tests never substitute CPU or fake graphs."""
from __future__ import annotations

import pytest
import torch

import labs.nanochat_fullstack  # register the historical nanochat import name
from nanochat.engine import Engine, KVCache
from nanochat.gpt import GPT, GPTConfig


def make_model(device="cpu", **flags):
    torch.manual_seed(718)
    flags.setdefault("use_flash_sdp", False)  # FP32/GQA oracle uses supported math SDPA
    config = GPTConfig(sequence_len=32, vocab_size=64, n_layer=2, n_head=2,
                       n_kv_head=1, n_embd=16, use_flash3=False, **flags)
    # Constructor weights are nonzero; init_weights zeros the head/projections,
    # which would make a cache parity test vacuous.
    model = GPT(config).to(device).eval()
    assert torch.count_nonzero(model.lm_head.weight) > 0
    return model


def make_cache(model, batch=2, capacity=16, **kwargs):
    cfg = model.config
    return KVCache(batch, cfg.n_kv_head, capacity, cfg.n_embd // cfg.n_head,
                   cfg.n_layer, **kwargs)


@pytest.mark.parametrize("batch,prompt_len", [(1, 1), (2, 3), (3, 5)])
@torch.inference_mode()
def test_device_position_cache_matches_eager_and_full_sequence_each_step(batch, prompt_len):
    model = make_model()
    tokens = torch.randint(0, 64, (batch, prompt_len + 6))
    eager, static = make_cache(model, batch), make_cache(model, batch)
    model(tokens[:, :prompt_len], kv_cache=eager)
    model(tokens[:, :prompt_len], kv_cache=static)
    # A fixed-capacity attention path must not expose uninitialized/NaN tail.
    static.kv_cache[..., prompt_len:, :].fill_(float("nan"))
    static.prepare_graph_decode()
    for position in range(prompt_len, tokens.size(1)):
        step = tokens[:, position:position + 1]
        expected = model(step, kv_cache=eager).clone()
        actual = model(step, kv_cache=static).clone()
        static.advance_graph_position()
        full = model(tokens[:, :position + 1])[:, -1:]
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual, full, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(static.kv_cache[..., :position + 1, :],
                                   eager.kv_cache[..., :position + 1, :])
        assert static.get_pos() == position + 1
        assert static.graph_position.item() == position + 1


@torch.inference_mode()
def test_graph_cache_reset_clears_mode_and_invalidates_capture_generation():
    model = make_model()
    cache = make_cache(model)
    model(torch.tensor([[1, 2], [3, 4]]), kv_cache=cache)
    cache.prepare_graph_decode()
    generation = cache.cache_gen
    cache.reset()
    assert cache.get_pos() == 0
    assert not cache.graph_mode
    assert cache.cache_gen != generation
    model(torch.tensor([[5, 6], [7, 8]]), kv_cache=cache)
    assert cache.get_pos() == 2


@torch.inference_mode()
def test_prefill_copies_valid_prefix_from_rounded_storage():
    model = make_model()
    source = make_cache(model, batch=1, capacity=3, block_size=4)
    model(torch.tensor([[1, 2, 3]]), kv_cache=source)
    target = make_cache(model, batch=2, capacity=12, block_size=4)
    target.prefill(source)
    assert target.get_pos() == 3
    torch.testing.assert_close(target.kv_cache[..., :3, :],
                               source.kv_cache[..., :3, :].expand(-1, -1, 2, -1, -1, -1))


def test_requested_cuda_graph_does_not_silently_run_eager_on_cpu(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    engine = Engine(make_model(use_cuda_graphs=True), None)
    with pytest.raises(RuntimeError, match="CUDA"):
        engine._execute_decode(torch.tensor([[1], [2]]), make_cache(engine.model))


def test_requested_side_stream_does_not_silently_run_eager_on_cpu(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    engine = Engine(make_model(enable_persistent_decode=True), None)
    with pytest.raises(RuntimeError, match="requires a CUDA model"):
        engine._execute_decode(torch.tensor([[1], [2]]), make_cache(engine.model))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA graphs and streams")
@pytest.mark.parametrize("batch,prompt_len", [(1, 1), (2, 3), (3, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_real_cuda_replay_matches_eager_per_step_and_recaptures_new_cache(monkeypatch, batch, prompt_len, dtype):
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    model = make_model("cuda", use_cuda_graphs=True)
    if dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            pytest.skip("device does not support BF16")
        model = model.to(dtype=dtype)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-4
    engine = Engine(model, None)
    old_graph = None
    for sequence in range(2):
        tokens = torch.randint(0, 64, (batch, prompt_len + 6), device="cuda")
        eager, graphed = make_cache(model, batch), make_cache(model, batch)
        model(tokens[:, :prompt_len], kv_cache=eager)
        model(tokens[:, :prompt_len], kv_cache=graphed)
        capture = None
        for position in range(prompt_len, tokens.size(1)):
            step = tokens[:, position:position + 1]
            expected = model(step, kv_cache=eager).clone()
            actual = engine._execute_decode(step, graphed).clone()
            assert engine.decode_execution_mode == "cuda_graph"
            assert engine._decode_graph is not None
            if capture is None:
                capture = engine._decode_graph
                assert capture is not old_graph
            assert engine._decode_graph is capture, "must replay, not recapture at every position"
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(graphed.kv_cache[..., :position + 1, :],
                                       eager.kv_cache[..., :position + 1, :], atol=tolerance, rtol=tolerance)
            assert graphed.get_pos() == position + 1
            assert graphed.graph_position.item() == position + 1
        old_graph = capture


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA stream ordering and D2H")
@torch.inference_mode()
def test_real_side_stream_decode_and_host_sampling_match_eager(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    model = make_model("cuda", enable_persistent_decode=True)
    engine = Engine(model, None)
    assert engine._persistent_stream is not None
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        eager, side = make_cache(model), make_cache(model)
        tokens = torch.randint(0, 64, (2, 10), device="cuda")
        model(tokens[:, :3], kv_cache=eager)
        model(tokens[:, :3], kv_cache=side)
        for position in range(3, 10):
            step = tokens[:, position:position + 1].clone()
            expected = model(step, kv_cache=eager).clone()
            actual = engine._execute_decode(step, side)
            del step  # exercises allocator lifetime across streams
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            assert engine.decode_execution_mode == "side_stream"
            expected_ids = expected[:, -1].argmax(dim=-1).tolist()
            assert engine._token_tensor_to_list(actual[:, -1].argmax(dim=-1)) == expected_ids
            rng = torch.Generator(device="cuda").manual_seed(22)
            sampled = engine._sample_batch_tokens(actual[:, -1], rng, [0.0, 0.0],
                [None, None], torch.ones(2, dtype=torch.bool, device="cuda"), 0,
                active_rows=[0, 1], uniform_sampling=False)
            assert sampled == expected_ids
    torch.cuda.current_stream().wait_stream(producer)


@torch.inference_mode()
def test_graph_rejects_masks_and_capacity_exhaustion_before_capture(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    model = make_model(use_cuda_graphs=True)
    engine = Engine(model, None)
    cache = make_cache(model, capacity=2)
    ids = torch.tensor([[1], [2]])
    model(ids, kv_cache=cache)
    with pytest.raises(ValueError, match="padded/masked"):
        engine._graph_decode(ids, cache, token_mask=torch.ones_like(ids, dtype=torch.bool))
    model(ids, kv_cache=cache)
    with pytest.raises(ValueError, match="capacity exhausted"):
        engine._graph_decode(ids, cache)


def _tiny_benchmark():
    from labs.nanochat_fullstack.benchmark_incremental_optimizations import IncrementalBenchmark
    benchmark = IncrementalBenchmark(device=torch.device("cpu"), warmup=1, iterations=2)
    benchmark.batch_size = 1
    benchmark.prompt_len = 3
    benchmark.decode_len = 4
    benchmark.vocab_size = 64
    flags = dict(n_layer=1, n_head=2, n_kv_head=2, n_embd=16,
                 use_flash3=False, use_flash_sdp=False)
    return benchmark, flags


def test_real_cpu_benchmark_uses_engine_and_reports_execution(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    benchmark, flags = _tiny_benchmark()
    prefill, decode, elapsed = benchmark.benchmark_inference(flags)
    assert prefill > 0 and decode > 0 and elapsed > 0
    assert benchmark.last_decode_execution == "eager"


@pytest.mark.parametrize("flag", ["enable_persistent_decode", "use_cuda_graphs"])
def test_cpu_benchmark_rejects_unavailable_cuda_rows(monkeypatch, flag):
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    benchmark, flags = _tiny_benchmark()
    flags[flag] = True
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        benchmark.benchmark_inference(flags)
    assert benchmark.last_decode_execution == "not_run"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA work on a side stream")
def test_cuda_wall_timer_includes_unjoined_side_stream_work():
    from labs.nanochat_fullstack.scripts.bench_b200_flags import _time_cuda_region_seconds
    side = torch.cuda.Stream()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = torch.randn((2048, 2048), device="cuda")
    output = torch.empty_like(values)
    side.wait_stream(torch.cuda.current_stream())

    def work():
        with torch.cuda.stream(side):
            start.record(side)
            for _ in range(8):
                torch.mm(values, values, out=output)
            end.record(side)
        # Deliberately no caller-stream join: the all-stream timer must wait.

    seconds = _time_cuda_region_seconds(work)
    assert end.query(), "timer returned while the side stream was pending"
    assert seconds >= start.elapsed_time(end) / 1000.0
