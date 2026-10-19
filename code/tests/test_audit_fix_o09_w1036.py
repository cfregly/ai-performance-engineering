"""Regressions for W1-036 and W1-085: NanoChat flag sweeps must hold torch.compile fixed.

Engine compiles by default on Blackwell. The two flag-sweep benchmarks must
keep every row uncompiled, and Engine must not label compiled decode as eager.
"""
from __future__ import annotations

import pytest
import torch

from labs.nanochat_fullstack import benchmark_incremental_optimizations as incremental
from labs.nanochat_fullstack.nanochat.engine import Engine, KVCache
from labs.nanochat_fullstack.nanochat.gpt import GPT, GPTConfig
from labs.nanochat_fullstack.scripts import bench_b200_flags


def _small_model(device="cpu", **flags) -> GPT:
    torch.manual_seed(718)
    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        use_flash_sdp=False,
        use_flash_attn_varlen=False,
        **flags,
    )
    return GPT(config).to(device).eval()


def _cache(engine: Engine, batch_size: int = 1) -> KVCache:
    return KVCache(**engine._kv_cache_params(batch_size=batch_size, seq_len=16))


def _compile_with_eager_backend(engine: Engine) -> None:
    # Real Dynamo capture through the same method Engine.__init__ uses on
    # Blackwell. The eager backend keeps it runnable on any host.
    engine._COMPILE_BACKEND = "eager"
    engine._COMPILE_MODE = "default"
    engine._configure_compiled_model()


@pytest.fixture
def compile_reset():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def test_engine_allow_compile_false_is_explicit_and_keeps_model_eager(monkeypatch) -> None:
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    model = _small_model()
    engine = Engine(model, tokenizer=None, allow_compile=False)

    diagnostics = engine.get_compile_diagnostics()
    assert diagnostics["status"] == "disabled"
    assert diagnostics["reason"] == "Engine(allow_compile=False)"
    assert diagnostics["mode"] is None
    assert engine.model is model

    kv_cache = _cache(engine)
    with torch.inference_mode():
        engine._model_forward(torch.tensor([[1, 2]]), kv_cache=kv_cache)
        logits = engine._execute_decode(torch.tensor([[3]]), kv_cache)
    assert logits.shape == (1, 1, 64)
    assert engine.decode_execution_mode == "eager"


def test_compiled_decode_is_labeled_torch_compile_not_eager(monkeypatch, compile_reset) -> None:
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    eager_model = _small_model()
    compiled_source = _small_model()
    compiled_source.load_state_dict(eager_model.state_dict())
    engine = Engine(compiled_source, tokenizer=None)
    _compile_with_eager_backend(engine)

    eager_cache = _cache(engine)
    compiled_cache = _cache(engine)
    with torch.inference_mode():
        eager_model(torch.tensor([[1, 2]]), kv_cache=eager_cache)
        engine._model_forward(torch.tensor([[1, 2]]), kv_cache=compiled_cache)
        expected = eager_model(torch.tensor([[3]]), kv_cache=eager_cache)
        actual = engine._execute_decode(torch.tensor([[3]]), compiled_cache)

    torch.testing.assert_close(actual, expected)
    assert engine.get_compile_diagnostics()["status"] == "runtime_observed"
    assert engine.decode_execution_mode == "torch_compile"


def _tiny_incremental_benchmark():
    benchmark = incremental.IncrementalBenchmark(device=torch.device("cpu"), warmup=1, iterations=1)
    benchmark.batch_size = 1
    benchmark.prompt_len = 3
    benchmark.decode_len = 4
    benchmark.vocab_size = 64
    flags = {
        "n_layer": 1,
        "n_head": 2,
        "n_kv_head": 2,
        "n_embd": 16,
        "use_flash_attn_varlen": False,
        "use_flash_sdp": False,
    }
    return benchmark, flags


def test_incremental_benchmark_records_compile_held_off(monkeypatch) -> None:
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    benchmark, flags = _tiny_incremental_benchmark()

    prefill, decode, elapsed = benchmark.benchmark_inference(flags)

    assert prefill > 0 and decode > 0 and elapsed > 0
    assert benchmark.last_decode_execution == "eager"
    assert benchmark.last_compile_status == "disabled"
    assert incremental.BenchmarkResult(
        name="row", prefill_tok_s=1.0, decode_tok_s=1.0, total_time=1.0,
        category="c", description="d",
    ).compile_status == "not_run"


def test_incremental_benchmark_fails_when_a_row_decodes_compiled(monkeypatch, compile_reset) -> None:
    # Stands in for Engine's Blackwell default, which compiles unless told not to.
    class _AlwaysCompilingEngine(Engine):
        def __init__(self, *args, **kwargs):
            kwargs.pop("allow_compile", None)
            super().__init__(*args, **kwargs)
            _compile_with_eager_backend(self)

    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    monkeypatch.setattr(incremental, "Engine", _AlwaysCompilingEngine)
    benchmark, flags = _tiny_incremental_benchmark()

    with pytest.raises(RuntimeError, match="observed torch_compile"):
        benchmark.benchmark_inference(flags)
    assert benchmark.last_compile_status == "runtime_observed"


def test_b200_sweep_engine_holds_compile_off_and_reports_it(monkeypatch) -> None:
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    model = _small_model()
    engine = bench_b200_flags.build_sweep_engine(model, tokenizer=None)
    assert engine.model is model

    kv_cache = _cache(engine)
    with torch.inference_mode():
        model(torch.tensor([[1, 2]]), kv_cache=kv_cache)
        engine._execute_decode(torch.tensor([[3]]), kv_cache)

    diagnostics = bench_b200_flags.require_isolated_decode(engine, model.config)
    assert diagnostics["status"] == "disabled"
    assert diagnostics["reason"] == "Engine(allow_compile=False)"


def test_b200_baseline_forces_math_and_restores_attention_policy() -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model = _small_model()
    attention = model.transformer.h[0].attn
    original_factory = attention.sdpa_ctx_factory
    backends = [
        SDPBackend.MATH,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    ]
    with sdpa_kernel(backends):
        bench_b200_flags.configure_model_mode("baseline", model)
        with attention.sdpa_ctx_factory([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            assert torch.backends.cuda.math_sdp_enabled()
            assert not torch.backends.cuda.flash_sdp_enabled()
            assert not torch.backends.cuda.mem_efficient_sdp_enabled()
            assert not torch.backends.cuda.cudnn_sdp_enabled()

        bench_b200_flags.configure_model_mode("flash_sdp", model)
        assert attention.sdpa_ctx_factory is original_factory
        assert attention.use_flash_sdp
        assert not attention.use_flash_attn_varlen

        bench_b200_flags.configure_model_mode("baseline", model)
        bench_b200_flags.configure_model_mode("flash_sdp", model)
        assert attention.sdpa_ctx_factory is original_factory


def test_b200_sweep_uses_truthful_fa2_varlen_mode_names_and_cache_metadata() -> None:
    config_fields = GPTConfig.__dataclass_fields__
    assert "use_flash_attn_varlen" in config_fields
    assert "use_flash3" not in config_fields
    assert "flash3_block_size" not in config_fields
    assert "use_cta_clustering" not in config_fields
    assert bench_b200_flags.BENCHMARK_MODES == (
        "baseline",
        "flash_sdp",
        "flash_attn_varlen",
        "flash_attn_varlen_aligned_cache",
        "persistent",
    )
    config = GPTConfig(use_flash_attn_varlen=False)

    enabled = bench_b200_flags.configure_mode("flash_attn_varlen", config)
    assert config.use_flash_attn_varlen
    assert enabled == {"use_flash_attn_varlen": True}

    enabled = bench_b200_flags.configure_mode("flash_attn_varlen_aligned_cache", config)
    assert config.use_flash_attn_varlen
    assert enabled == {
        "use_flash_attn_varlen": True,
        "kv_block_size": 32,
        "kv_page_size": 1024,
        "kv_cache_layout": "contiguous",
    }

    enabled = bench_b200_flags.configure_mode("persistent", config)
    assert config.use_flash_attn_varlen
    assert config.enable_persistent_decode
    assert config.kv_block_size is None
    assert config.kv_page_size is None
    assert enabled == {
        "enable_persistent_decode": True,
        "use_flash_attn_varlen": True,
    }


def test_b200_sweep_rejects_compiled_decode(monkeypatch, compile_reset) -> None:
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    model = _small_model()
    engine = Engine(model, tokenizer=None)
    _compile_with_eager_backend(engine)

    kv_cache = _cache(engine)
    with torch.inference_mode():
        engine._model_forward(torch.tensor([[1, 2]]), kv_cache=kv_cache)
        engine._execute_decode(torch.tensor([[3]]), kv_cache)

    with pytest.raises(RuntimeError, match="observed torch_compile"):
        bench_b200_flags.require_isolated_decode(engine, model.config)


def test_b200_sweep_rejects_compile_configured_engine_before_decode(monkeypatch, compile_reset) -> None:
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    model = _small_model()
    engine = Engine(model, tokenizer=None)
    _compile_with_eager_backend(engine)
    # Even with a matching execution label, a compiled Engine must not pass.
    engine.decode_execution_mode = "eager"

    with pytest.raises(RuntimeError, match="observed compile status configured"):
        bench_b200_flags.require_isolated_decode(engine, model.config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SKIPPED: requires a real CUDA device and side stream")
@pytest.mark.parametrize("persistent", [False, True])
def test_cuda_b200_sweep_rows_decode_uncompiled_through_engine(monkeypatch, persistent) -> None:
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    model = _small_model("cuda", enable_persistent_decode=persistent)
    engine = bench_b200_flags.build_sweep_engine(model, tokenizer=None)

    prefill_tok_s, decode_tok_s = bench_b200_flags.bench_once(
        model, batch_size=2, prompt_len=4, decode_len=4, device=torch.device("cuda"), engine=engine,
    )

    assert prefill_tok_s > 0 and decode_tok_s > 0
    diagnostics = bench_b200_flags.require_isolated_decode(engine, model.config)
    assert diagnostics["status"] == "disabled"
    assert engine.decode_execution_mode == ("side_stream" if persistent else "eager")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SKIPPED: requires a real CUDA device and side stream")
def test_cuda_compiled_side_stream_decode_is_labeled(monkeypatch, compile_reset) -> None:
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    model = _small_model("cuda", enable_persistent_decode=True)
    # Skip the device-dependent default and compile the same way on every GPU.
    engine = Engine(model, tokenizer=None, allow_compile=False)
    _compile_with_eager_backend(engine)

    kv_cache = _cache(engine)
    with torch.inference_mode():
        engine._model_forward(torch.tensor([[1, 2]], device="cuda"), kv_cache=kv_cache)
        engine._execute_decode(torch.tensor([[3]], device="cuda"), kv_cache)
    torch.cuda.synchronize()

    assert engine.decode_execution_mode == "side_stream_torch_compile"
