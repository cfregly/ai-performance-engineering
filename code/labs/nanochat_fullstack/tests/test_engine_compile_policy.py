"""Compile-lifecycle regressions for the NanoChat inference Engine."""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

import labs.nanochat_fullstack  # noqa: F401 - register the nanochat package alias
from nanochat.engine import Engine, KVCache
from nanochat.gpt import GPT, GPTConfig


def _small_model(*, device: torch.device | str = "cpu", **config_overrides) -> GPT:
    config_kwargs = dict(
        sequence_len=32,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        use_flash_sdp=False,
        use_flash3=False,
        use_cta_clustering=False,
    )
    config_kwargs.update(config_overrides)
    config = GPTConfig(**config_kwargs)
    model = GPT(config).to(device).eval()
    if torch.device(device).type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    return model


def _cache(engine: Engine, *, batch_size: int = 1, seq_len: int = 16) -> KVCache:
    return KVCache(**engine._kv_cache_params(batch_size=batch_size, seq_len=seq_len))


def _assert_cache_prefix_matches(
    eager_cache: KVCache,
    compiled_cache: KVCache,
    *,
    expected_pos: int,
    expected_data_ptrs: tuple[int, int] | None = None,
) -> tuple[int, int]:
    assert eager_cache.pos == compiled_cache.pos == expected_pos
    assert eager_cache.kv_cache is not None
    assert compiled_cache.kv_cache is not None
    data_ptrs = (eager_cache.kv_cache.data_ptr(), compiled_cache.kv_cache.data_ptr())
    if expected_data_ptrs is not None:
        assert data_ptrs == expected_data_ptrs
    eager_prefix = eager_cache.kv_cache[..., :expected_pos, :]
    compiled_prefix = compiled_cache.kv_cache[..., :expected_pos, :]
    assert torch.isfinite(compiled_prefix).all()
    torch.testing.assert_close(compiled_prefix, eager_prefix, atol=2e-2, rtol=2e-2)
    return data_ptrs


def test_cpu_engine_reports_compile_as_unsupported_and_still_runs_decode(monkeypatch) -> None:
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    engine = Engine(_small_model(), tokenizer=None)

    assert engine.compile_status == "unsupported_hardware"
    assert engine.compile_device == "cpu"
    assert engine.get_compile_diagnostics() == {
        "status": "unsupported_hardware",
        "reason": "model device cpu is not an available CUDA device",
        "mode": None,
        "device": "cpu",
        "fullgraph": None,
        "dynamic": None,
        "runtime_observed": False,
        "dynamo_process_unique_graphs_since_config": None,
        "error": None,
    }

    kv_cache = _cache(engine)
    with torch.inference_mode():
        prefill = engine._model_forward(torch.tensor([[1, 2]]), kv_cache=kv_cache)
        decode = engine._execute_decode(torch.tensor([[3]]), kv_cache)

    assert prefill.shape == (1, 2, 64)
    assert decode.shape == (1, 1, 64)
    assert torch.isfinite(decode).all()
    assert kv_cache.pos == 3
    assert engine.compile_status == "unsupported_hardware"


@pytest.mark.parametrize(
    ("config_overrides", "environment_disabled", "expected_status"),
    [
        ({}, True, "disabled"),
        ({"use_cuda_graphs": True}, False, "manual_graphs"),
    ],
)
def test_compile_skip_reason_is_explicit(
    monkeypatch,
    config_overrides,
    environment_disabled,
    expected_status,
) -> None:
    if environment_disabled:
        monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    else:
        monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)

    engine = Engine(_small_model(**config_overrides), tokenizer=None)

    assert engine.compile_status == expected_status
    assert engine.get_compile_diagnostics()["runtime_observed"] is False


def test_compile_constructor_failure_is_not_swallowed() -> None:
    engine = Engine(_small_model(), tokenizer=None)
    original_model = engine.model
    engine._COMPILE_MODE = "invalid-nanochat-test-mode"

    with pytest.raises(RuntimeError, match="NanoChat torch.compile configuration failed"):
        engine._configure_compiled_model()

    assert engine.model is original_model
    assert engine.compile_status == "configuration_failed"
    assert "Unrecognized mode" in engine.get_compile_diagnostics()["error"]


def test_lazy_compile_failure_is_reported_without_eager_fallback() -> None:
    class _LazyFailure(nn.Module):
        def forward(self, value):
            torch._dynamo.graph_break()
            return value + 1

    engine = Engine(_small_model(), tokenizer=None)
    engine.model = torch.compile(_LazyFailure(), backend="eager", fullgraph=True)
    engine._compile_active = True
    engine.compile_mode = "test-eager"
    engine._compile_graph_count_at_config = engine._read_dynamo_unique_graphs()
    engine._set_compile_status(
        "configured",
        "test wrapper created; compiled execution has not yet been observed",
    )

    with pytest.raises(RuntimeError, match="failed during runtime execution"):
        engine._model_forward(torch.ones(2))

    diagnostics = engine.get_compile_diagnostics()
    assert diagnostics["status"] == "runtime_failed"
    assert diagnostics["runtime_observed"] is False
    assert diagnostics["error"]


def test_cpu_counting_backend_reports_observed_execution_and_graph_delta() -> None:
    backend_calls = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal backend_calls
        backend_calls += 1
        return graph_module.forward

    torch.compiler.reset()
    torch._dynamo.utils.counters.clear()
    try:
        engine = Engine(_small_model(), tokenizer=None)
        engine.model = nn.Linear(2, 2).eval()
        engine._COMPILE_BACKEND = counting_backend
        engine._COMPILE_MODE = "default"
        engine._configure_compiled_model()

        assert engine.compile_status == "configured"
        result = engine._model_forward(torch.ones(1, 2))
        diagnostics = engine.get_compile_diagnostics()

        assert result.shape == (1, 2)
        assert backend_calls == 1
        assert diagnostics["status"] == "runtime_observed"
        assert diagnostics["runtime_observed"] is True
        assert diagnostics["dynamo_process_unique_graphs_since_config"] >= 1
    finally:
        torch.compiler.reset()
        torch._dynamo.utils.counters.clear()


def _require_opt_in_blackwell() -> torch.device:
    if os.getenv("NANOCHAT_RUN_BLACKWELL_COMPILE_TESTS", "0") != "1":
        pytest.skip("set NANOCHAT_RUN_BLACKWELL_COMPILE_TESTS=1 for real Blackwell qualification")
    if not torch.cuda.is_available():
        pytest.skip("real Blackwell CUDA device unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] < 10:
        pytest.skip("Blackwell or newer GPU required")
    return device


def test_cpu_fullgraph_model_uses_functional_rotary_without_eager_cache_mutation() -> None:
    torch.compiler.reset()
    eager_model = _small_model()
    compiled_source = _small_model()
    compiled_source.load_state_dict(eager_model.state_dict())
    compiled_model = torch.compile(
        compiled_source,
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    prompt = torch.tensor([[1, 2, 3]])

    try:
        with torch.inference_mode():
            expected = eager_model(prompt)
            actual = compiled_model(prompt)

        torch.testing.assert_close(actual, expected)
        eager_attention = eager_model.transformer.h[0].attn
        compiled_attention = compiled_source.transformer.h[0].attn
        assert eager_attention._rotary_q_cache is not None
        assert eager_attention._rotary_k_cache is not None
        assert compiled_attention._rotary_q_cache is None
        assert compiled_attention._rotary_k_cache is None
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("n_layer", (1, 2))
def test_blackwell_compiled_multitoken_decode_matches_eager_and_reports_graphs(
    monkeypatch,
    n_layer,
) -> None:
    device = _require_opt_in_blackwell()
    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    torch.compiler.reset()
    torch._dynamo.utils.counters.clear()
    try:
        eager_model = _small_model(device=device, n_layer=n_layer)
        compiled_source = _small_model(device=device, n_layer=n_layer)
        compiled_source.load_state_dict(eager_model.state_dict())
        engine = Engine(compiled_source, tokenizer=None)
        assert engine.compile_status == "configured"

        eager_cache = KVCache(**engine._kv_cache_params(batch_size=1, seq_len=16))
        compiled_cache = _cache(engine)
        prompt = torch.tensor([[1, 2, 3]], device=device)
        first_decode = torch.tensor([[4]], device=device)
        stable_decode_steps = [
            torch.tensor([[token]], device=device) for token in range(5, 11)
        ]

        with torch.inference_mode():
            with torch.compiler.set_stance("default"):
                expected = eager_model(prompt, kv_cache=eager_cache)
                actual = engine._model_forward(prompt, kv_cache=compiled_cache)
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
                cache_data_ptrs = _assert_cache_prefix_matches(
                    eager_cache,
                    compiled_cache,
                    expected_pos=3,
                )

                expected = eager_model(first_decode, kv_cache=eager_cache)
                actual = engine._execute_decode(first_decode, compiled_cache)
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
                _assert_cache_prefix_matches(
                    eager_cache,
                    compiled_cache,
                    expected_pos=4,
                    expected_data_ptrs=cache_data_ptrs,
                )

            warmed_graphs = engine.get_compile_diagnostics()[
                "dynamo_process_unique_graphs_since_config"
            ]
            assert warmed_graphs is not None
            assert warmed_graphs >= 2

            with torch.compiler.set_stance("fail_on_recompile"):
                for expected_pos, step in enumerate(stable_decode_steps, start=5):
                    expected = eager_model(step, kv_cache=eager_cache)
                    actual = engine._execute_decode(step, compiled_cache)
                    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
                    _assert_cache_prefix_matches(
                        eager_cache,
                        compiled_cache,
                        expected_pos=expected_pos,
                        expected_data_ptrs=cache_data_ptrs,
                    )

        diagnostics = engine.get_compile_diagnostics()
        assert diagnostics["status"] == "runtime_observed"
        assert diagnostics["dynamo_process_unique_graphs_since_config"] == warmed_graphs
    finally:
        torch.compiler.reset()
        torch._dynamo.utils.counters.clear()
