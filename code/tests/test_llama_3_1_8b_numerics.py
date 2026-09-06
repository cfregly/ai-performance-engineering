from __future__ import annotations

import torch
import torch._inductor.config as inductor_config

import labs.real_world_models.llama_3_1_8b_optimization as llama_module
from labs.real_world_models.llama_3_1_8b_optimization import (
    LLAMA_EMULATE_EAGER_PRECISION_CASTS,
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


def test_compiled_llama_scopes_eager_bf16_rounding_to_compile_options(monkeypatch) -> None:
    monkeypatch.setattr(Llama31_8B_Optimization, "HIDDEN_SIZE", 16)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_HEADS", 2)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_LAYERS", 2)
    monkeypatch.setattr(Llama31_8B_Optimization, "INTERMEDIATE_SIZE", 32)
    monkeypatch.setattr(inductor_config, "emulate_precision_casts", False)
    mode_options = {
        "max_autotune": True,
        "triton.cudagraphs": True,
        "coordinate_descent_tuning": True,
    }
    monkeypatch.setattr(
        torch._inductor,
        "list_mode_options",
        lambda mode: mode_options,
    )
    compile_kwargs = {}

    def capture_compile(model, **kwargs):
        compile_kwargs.update(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", capture_compile)
    benchmark = Llama31_8B_Optimization(
        batch_size=1,
        seq_length=4,
        use_compile=True,
        use_fp8=False,
        use_flex_attention=True,
        prefer_sdpa=False,
    )
    benchmark.device = torch.device("cpu")
    benchmark.setup()

    assert LLAMA_EMULATE_EAGER_PRECISION_CASTS is True
    assert inductor_config.emulate_precision_casts is False
    assert compile_kwargs == {
        "mode": None,
        "options": {
            **mode_options,
            "emulate_precision_casts": True,
        },
        "fullgraph": False,
        "dynamic": False,
    }
    assert benchmark.compile_options == compile_kwargs["options"]
    assert benchmark.compile_options is not compile_kwargs["options"]
    assert "emulate_precision_casts" not in mode_options
    assert benchmark.model._is_compiled_benchmark_module is True

    benchmark.teardown()
    assert inductor_config.emulate_precision_casts is False
    assert benchmark.compile_options is None


def test_compiled_llama_skips_without_max_autotune_option_profile(monkeypatch) -> None:
    monkeypatch.setattr(torch._inductor, "list_mode_options", lambda mode: {})

    try:
        llama_module._max_autotune_compile_options()
    except RuntimeError as exc:
        assert str(exc).startswith("SKIPPED:")
        assert "max-autotune" in str(exc)
    else:
        raise AssertionError("missing max-autotune options must skip the benchmark")
