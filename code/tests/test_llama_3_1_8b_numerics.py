from __future__ import annotations

import torch

from labs.real_world_models.llama_3_1_8b_optimization import (
    LLAMA_RMS_NORM_EPS,
    Llama31_8B_Optimization,
)


def test_llama_layers_use_the_declared_rms_norm_epsilon(monkeypatch) -> None:
    monkeypatch.setattr(Llama31_8B_Optimization, "HIDDEN_SIZE", 16)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_HEADS", 2)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_LAYERS", 2)
    monkeypatch.setattr(Llama31_8B_Optimization, "INTERMEDIATE_SIZE", 32)

    benchmark = Llama31_8B_Optimization(
        batch_size=1,
        seq_length=4,
        use_compile=False,
        use_fp8=False,
        use_flex_attention=False,
        prefer_sdpa=False,
    )
    benchmark.device = torch.device("cpu")
    benchmark.setup()

    assert LLAMA_RMS_NORM_EPS == 1e-5
    assert len(benchmark.layers) == 2
    for layer in benchmark.layers:
        assert layer.input_layernorm.eps == LLAMA_RMS_NORM_EPS
        assert layer.post_attention_layernorm.eps == LLAMA_RMS_NORM_EPS

    benchmark.teardown()
