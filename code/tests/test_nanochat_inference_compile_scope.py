from __future__ import annotations

from pathlib import Path

import torch

from labs.nanochat_fullstack import optimized_nanochat_inference as optimized_module
from labs.nanochat_fullstack.baseline_nanochat_inference import (
    BaselineNanochatInferenceBenchmark,
)
from labs.nanochat_fullstack.optimized_nanochat_inference import (
    OptimizedNanochatInferenceBenchmark,
)

REPO_CODE = Path(__file__).resolve().parents[1]


def test_nanochat_pair_rejects_degenerate_logits() -> None:
    for benchmark_type in (
        BaselineNanochatInferenceBenchmark,
        OptimizedNanochatInferenceBenchmark,
    ):
        benchmark = benchmark_type()
        benchmark.output = torch.zeros(2, 1, 8)
        assert benchmark.validate_result() == "benchmark_fn() produced degenerate all-zero logits"
        benchmark.output[0, 0, 0] = 1
        assert benchmark.validate_result() is None


def test_nanochat_pair_reinitializes_zeroed_training_projections() -> None:
    for filename in (
        "baseline_nanochat_inference.py",
        "optimized_nanochat_inference.py",
    ):
        source = (REPO_CODE / "labs" / "nanochat_fullstack" / filename).read_text(
            encoding="utf-8"
        )
        assert "model.init_weights()" in source
        assert "model.apply(model._init_weights)" in source


def test_nanochat_compiles_only_the_fixed_prefill_shape(monkeypatch) -> None:
    model = object()
    compiled = object()
    compile_call = {}

    def capture_compile(target, **kwargs):
        compile_call["target"] = target
        compile_call["kwargs"] = kwargs
        return compiled

    monkeypatch.setattr(torch, "compile", capture_compile)

    assert optimized_module._compile_prefill(model) is compiled
    assert compile_call == {
        "target": model,
        "kwargs": {
            "mode": "max-autotune-no-cudagraphs",
            "fullgraph": False,
            "dynamic": False,
        },
    }


def test_nanochat_uses_compiled_prefill_and_eager_decode() -> None:
    class FakeCache:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

    class Recorder:
        def __init__(self) -> None:
            self.inputs: list[torch.Tensor] = []

        def __call__(self, token_ids, *, kv_cache):
            self.inputs.append(token_ids)
            return token_ids.float()

    benchmark = OptimizedNanochatInferenceBenchmark()
    benchmark.prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    benchmark.decode_tokens = torch.tensor([[7, 8], [9, 10]])
    benchmark.decode_len = benchmark.decode_tokens.size(1)
    benchmark.decode_token_steps = tuple(
        benchmark.decode_tokens[:, step : step + 1]
        for step in range(benchmark.decode_len)
    )
    benchmark.kv_cache = FakeCache()
    benchmark.prefill_model = Recorder()
    benchmark.model = Recorder()

    benchmark.benchmark_fn()

    assert benchmark.kv_cache.reset_count == 1
    assert len(benchmark.prefill_model.inputs) == 1
    assert benchmark.prefill_model.inputs[0] is benchmark.prompt
    assert len(benchmark.model.inputs) == benchmark.decode_len
    assert all(
        actual is expected
        for actual, expected in zip(
            benchmark.model.inputs,
            benchmark.decode_token_steps,
            strict=True,
        )
    )
    assert torch.equal(benchmark.output, benchmark.decode_token_steps[-1].float())
