from __future__ import annotations

import pytest
import torch

from labs.train_distributed.training_utils.fsdp_training import (
    fsdp1_mixed_precision_policy,
    move_fsdp_model_to_device,
)


def _tiny_llama_with_production_dtypes() -> torch.nn.Module:
    transformers = pytest.importorskip(
        "transformers",
        reason="the FSDP rotary precision regression requires a real Hugging Face Llama",
    )
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=2048,
        rope_theta=10_000.0,
        use_cache=False,
    )
    return transformers.AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )


def _rotary_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules = [
        module
        for module in model.modules()
        if isinstance(getattr(module, "inv_freq", None), torch.Tensor)
    ]
    assert modules, "real Hugging Face Llama did not expose a rotary inv_freq buffer"
    return modules


def _manual_rope(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    attention_scaling: object = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq.float().reshape(1, 1, -1)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return angles.cos() * attention_scaling, angles.sin() * attention_scaling


def test_fsdp_device_move_preserves_real_llama_rotary_precision() -> None:
    model = _tiny_llama_with_production_dtypes()
    parameters = [parameter for parameter in model.parameters() if parameter.is_floating_point()]
    assert parameters
    assert {parameter.dtype for parameter in parameters} == {torch.bfloat16}

    rotary_before = _rotary_modules(model)
    inv_freq_before = [module.inv_freq.detach().clone() for module in rotary_before]
    assert {value.dtype for value in inv_freq_before} == {torch.float32}

    position_ids = torch.tensor([[0, 1, 511, 1024]], dtype=torch.long)
    reference_cos, reference_sin = _manual_rope(inv_freq_before[0], position_ids)
    rounded_cos, rounded_sin = _manual_rope(
        inv_freq_before[0].to(torch.bfloat16), position_ids
    )
    assert (reference_cos - rounded_cos).abs().max().item() > 0.1
    assert (reference_sin - rounded_sin).abs().max().item() > 0.1

    moved = move_fsdp_model_to_device(model, torch.device("cpu"))

    assert moved is model
    moved_parameter_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    assert moved_parameter_dtypes == {torch.bfloat16}
    rotary_after = _rotary_modules(model)
    assert len(rotary_after) == len(rotary_before)
    for before, module in zip(inv_freq_before, rotary_after, strict=True):
        assert module.inv_freq.dtype == torch.float32
        torch.testing.assert_close(module.inv_freq, before, rtol=0, atol=0)

    preserved_cos, preserved_sin = _manual_rope(rotary_after[0].inv_freq, position_ids)
    torch.testing.assert_close(preserved_cos, reference_cos, rtol=0, atol=0)
    torch.testing.assert_close(preserved_sin, reference_sin, rtol=0, atol=0)

    rotary = rotary_after[0]
    head_dim = int(model.config.head_dim)
    hidden = torch.zeros(1, position_ids.shape[1], head_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        actual_cos, actual_sin = rotary(hidden, position_ids)
    expected_cos, expected_sin = _manual_rope(
        rotary.inv_freq,
        position_ids,
        getattr(rotary, "attention_scaling", 1.0),
    )
    one_bfloat16_ulp = torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(
        actual_cos,
        expected_cos.to(dtype=actual_cos.dtype),
        rtol=0,
        atol=one_bfloat16_ulp,
    )
    torch.testing.assert_close(
        actual_sin,
        expected_sin.to(dtype=actual_sin.dtype),
        rtol=0,
        atol=one_bfloat16_ulp,
    )


def test_fsdp1_policy_keeps_real_llama_parameters_bfloat16_and_buffers_float32() -> None:
    model = _tiny_llama_with_production_dtypes()
    parameter_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    assert parameter_dtypes == {torch.bfloat16}
    assert {module.inv_freq.dtype for module in _rotary_modules(model)} == {torch.float32}

    policy = fsdp1_mixed_precision_policy()

    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.bfloat16
    assert policy.buffer_dtype == torch.float32
