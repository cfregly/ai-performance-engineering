from __future__ import annotations

import pytest
import torch
import torch._inductor.config as inductor_config

import labs.real_world_models.llama_3_1_8b_optimization as llama_module
from labs.real_world_models.llama_3_1_8b_optimization import (
    LLAMA_BF16_OUTPUT_TOLERANCE,
    LLAMA_EMULATE_EAGER_PRECISION_CASTS,
    LLAMA_RMS_NORM_EPS,
    Llama31_8B_Optimization,
    StableLlamaRMSNorm,
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
        attention_mode="preferred_sdpa",
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
        attention_mode="sdpa",
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


def test_stable_llama_rms_norm_matches_explicit_fp32_math() -> None:
    norm = StableLlamaRMSNorm(8)
    value = torch.linspace(-2.0, 2.0, 24, dtype=torch.bfloat16).reshape(3, 8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))

    actual = norm(value)
    values = value.float()
    expected = (
        values
        * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + LLAMA_RMS_NORM_EPS)
        * norm.weight.float()
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_eager_sdpa_full_output_changes_with_input_and_caller_seed(monkeypatch) -> None:
    monkeypatch.setattr(Llama31_8B_Optimization, "HIDDEN_SIZE", 32)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_HEADS", 4)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_LAYERS", 2)
    monkeypatch.setattr(Llama31_8B_Optimization, "INTERMEDIATE_SIZE", 64)

    def run_seed(seed: int) -> tuple[Llama31_8B_Optimization, torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        wrapper = Llama31_8B_Optimization(
            batch_size=1,
            seq_length=8,
            use_compile=False,
            attention_mode="preferred_sdpa",
        )
        wrapper.device = torch.device("cpu")
        wrapper.setup()
        wrapper.run()
        assert wrapper.input is not None
        assert wrapper.output is not None
        return wrapper, wrapper.input.clone(), wrapper.output.clone()

    first, first_input, first_output = run_seed(42)
    repeated, repeated_input, repeated_output = run_seed(42)
    changed_seed, changed_seed_input, changed_seed_output = run_seed(1042)

    try:
        assert torch.equal(first_input, repeated_input)
        assert torch.equal(first_output, repeated_output)
        assert not torch.equal(first_input, changed_seed_input)
        assert not torch.equal(first_output, changed_seed_output)
        assert first_output.shape == (1, 8, 32)
        assert first_output.dtype == torch.float32
        assert bool(torch.isfinite(first_output).all())

        before_change = first_output.clone()
        first.input.add_(torch.full_like(first.input, 0.25))
        first.run()
        assert first.output is not None
        assert first.output.shape == before_change.shape
        assert bool(torch.isfinite(first.output).all())
        assert not torch.equal(first.output, before_change)
    finally:
        first.teardown()
        repeated.teardown()
        changed_seed.teardown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA compile path required")
def test_cuda_compiled_sdpa_matches_eager_for_two_full_changed_outputs(monkeypatch) -> None:
    monkeypatch.setattr(Llama31_8B_Optimization, "HIDDEN_SIZE", 128)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_HEADS", 4)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_LAYERS", 2)
    monkeypatch.setattr(Llama31_8B_Optimization, "INTERMEDIATE_SIZE", 256)
    device = torch.device("cuda")
    wrappers: list[Llama31_8B_Optimization] = []
    torch.compiler.reset()

    try:
        for use_compile in (False, True):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            wrapper = Llama31_8B_Optimization(
                batch_size=1,
                seq_length=8,
                use_compile=use_compile,
                attention_mode="preferred_sdpa",
            )
            wrapper.device = device
            wrapper.setup()
            wrappers.append(wrapper)

        eager, compiled = wrappers
        eager_parameters = dict(eager.layers.named_parameters())
        compiled_parameters = dict(compiled.layers.named_parameters())
        assert eager_parameters.keys() == compiled_parameters.keys()
        assert all(
            torch.equal(eager_parameters[name], compiled_parameters[name])
            for name in eager_parameters
        )

        previous_outputs: list[torch.Tensor] | None = None
        for input_seed in (20_000, 20_001):
            generator = torch.Generator(device=device).manual_seed(input_seed)
            source = torch.randn(
                1,
                8,
                128,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            outputs: list[torch.Tensor] = []
            for wrapper in wrappers:
                wrapper.input.copy_(source)
                wrapper.run()
                assert wrapper.output is not None
                output = wrapper.output.detach().clone()
                assert output.shape == (1, 8, 128)
                assert output.dtype == torch.float32
                assert bool(torch.isfinite(output).all())
                outputs.append(output)

            rtol, atol = LLAMA_BF16_OUTPUT_TOLERANCE
            torch.testing.assert_close(outputs[1], outputs[0], rtol=rtol, atol=atol)
            if previous_outputs is not None:
                assert not torch.equal(outputs[0], previous_outputs[0])
                assert not torch.equal(outputs[1], previous_outputs[1])
            previous_outputs = outputs
    finally:
        for wrapper in wrappers:
            wrapper.teardown()
        torch.compiler.reset()
