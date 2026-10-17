"""Regressions for W2-089 and W2-090 in the Llama 3.1 8B lab."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.nn.functional import linear

from core.benchmark.expectations import ExpectationsStore
from labs.real_world_models.baseline_llama_3_1_8b import BaselineLlama31_8B
from labs.real_world_models.llama_3_1_8b_optimization import Llama31_8B_Optimization

LAB_DIR = Path(__file__).resolve().parents[1] / "labs" / "real_world_models"

# Commit whose B200 run measured the 4-layer stack against the FP32-upcast
# materialized-attention baseline.
RETIRED_LLAMA_EXPECTATION_COMMIT = "e1af4ba1ac65cde755e836586cdb44cad06a0010"

# Published Llama 3.1 8B configuration.
LLAMA_3_1_8B_TOTAL_PARAMETERS = 8_030_261_248
LLAMA_3_1_8B_VOCAB_SIZE = 128_256
LLAMA_3_1_8B_HIDDEN_SIZE = 4096
LLAMA_3_1_8B_KV_PROJECTION_SIZE = 8 * 128


def _small_gqa_dims(monkeypatch: pytest.MonkeyPatch, *, hidden: int = 32) -> None:
    monkeypatch.setattr(Llama31_8B_Optimization, "HIDDEN_SIZE", hidden)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_HEADS", 4)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_KEY_VALUE_HEADS", 2)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_LAYERS", 2)
    monkeypatch.setattr(Llama31_8B_Optimization, "INTERMEDIATE_SIZE", 2 * hidden)


def _attention_reference(
    attention: torch.nn.Module,
    x: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    kv_head_for: str,
) -> torch.Tensor:
    batch, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    group = num_heads // num_kv_heads
    weights = {
        name: getattr(attention, name).weight.float()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    q = linear(x, weights["q_proj"]).view(batch, seq_len, num_heads, head_dim)
    k = linear(x, weights["k_proj"]).view(batch, seq_len, num_kv_heads, head_dim)
    v = linear(x, weights["v_proj"]).view(batch, seq_len, num_kv_heads, head_dim)
    allowed = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
    heads = []
    for head in range(num_heads):
        kv_head = head // group if kv_head_for == "grouped" else head % num_kv_heads
        scores = q[:, :, head] @ k[:, :, kv_head].transpose(-2, -1) / math.sqrt(head_dim)
        scores = scores.masked_fill(~allowed, float("-inf"))
        heads.append(torch.softmax(scores, dim=-1) @ v[:, :, kv_head])
    return linear(torch.cat(heads, dim=-1), weights["o_proj"])


def test_b200_llama_expectation_is_not_the_retired_four_layer_fp32_measurement() -> None:
    store = ExpectationsStore(LAB_DIR, "b200")

    assert store.path == LAB_DIR / "expectations_b200.json"
    assert store.data["hardware_key"] == "b200"
    assert store.data["schema_version"] == 2
    assert not store.validate_all().has_issues
    entry = store.get_entry("llama_3_1_8b")
    assert entry is None or entry.provenance.git_commit != RETIRED_LLAMA_EXPECTATION_COMMIT


def test_full_decoder_stack_matches_published_llama_3_1_8b_layer_shapes() -> None:
    wrapper = Llama31_8B_Optimization(batch_size=1, seq_length=2048, use_compile=False)
    wrapper.device = torch.device("meta")
    with torch.device("meta"):
        wrapper.setup()
    try:
        assert len(wrapper.layers) == 32
        for layer in wrapper.layers:
            attention = layer.attention
            assert attention.q_proj.weight.shape == (4096, 4096)
            assert attention.k_proj.weight.shape == (LLAMA_3_1_8B_KV_PROJECTION_SIZE, 4096)
            assert attention.v_proj.weight.shape == (LLAMA_3_1_8B_KV_PROJECTION_SIZE, 4096)
            assert attention.o_proj.weight.shape == (4096, 4096)

        decoder_parameters = sum(p.numel() for p in wrapper.layers.parameters())
        embedding_and_lm_head = 2 * LLAMA_3_1_8B_VOCAB_SIZE * LLAMA_3_1_8B_HIDDEN_SIZE
        final_norm = LLAMA_3_1_8B_HIDDEN_SIZE
        assert decoder_parameters == 6_979_584_000
        assert (
            decoder_parameters + embedding_and_lm_head + final_norm
            == LLAMA_3_1_8B_TOTAL_PARAMETERS
        )
        assert not hasattr(Llama31_8B_Optimization, "VOCAB_SIZE")
    finally:
        wrapper.teardown()


@pytest.mark.parametrize("attention_mode", ["preferred_sdpa", "sdpa", "manual"])
def test_every_attention_mode_shares_each_kv_head_with_its_query_group(
    monkeypatch: pytest.MonkeyPatch,
    attention_mode: str,
) -> None:
    _small_gqa_dims(monkeypatch)
    torch.manual_seed(7)
    wrapper = Llama31_8B_Optimization(
        batch_size=2,
        seq_length=6,
        use_compile=False,
        attention_mode=attention_mode,
    )
    attention = wrapper._create_attention_layer().eval()
    x = torch.randn(2, 6, 32)

    with torch.inference_mode():
        actual = attention(x)
        grouped = _attention_reference(
            attention, x, num_heads=4, num_kv_heads=2, kv_head_for="grouped"
        )
        interleaved = _attention_reference(
            attention, x, num_heads=4, num_kv_heads=2, kv_head_for="interleaved"
        )

    assert attention.k_proj.weight.shape == (16, 32)
    assert attention.v_proj.weight.shape == (16, 32)
    torch.testing.assert_close(actual, grouped, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(actual, interleaved, rtol=1e-4, atol=1e-5)


def test_attention_rejects_query_heads_that_do_not_group_evenly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_gqa_dims(monkeypatch)
    monkeypatch.setattr(Llama31_8B_Optimization, "NUM_KEY_VALUE_HEADS", 3)
    wrapper = Llama31_8B_Optimization(batch_size=1, seq_length=4, use_compile=False)

    with pytest.raises(ValueError, match="multiple of num_kv_heads=3"):
        wrapper._create_attention_layer()


def test_baseline_factory_records_the_grouped_query_parameter_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_gqa_dims(monkeypatch)
    torch.manual_seed(11)
    benchmark = BaselineLlama31_8B(batch_size=1, seq_length=8)
    try:
        benchmark.setup()
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()

        hidden, kv_size, intermediate = 32, 16, 64
        attention = 2 * hidden * hidden + 2 * hidden * kv_size
        mlp = 3 * hidden * intermediate
        norms = 2 * hidden
        assert benchmark.parameter_count == 2 * (attention + mlp + norms)
        assert benchmark.output is not None
        assert benchmark.output.shape == (1, 8, hidden)
        assert bool(torch.isfinite(benchmark.output).all())
    finally:
        benchmark.teardown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA SDPA backends required")
def test_cuda_preferred_sdpa_accepts_grouped_query_bf16_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _small_gqa_dims(monkeypatch, hidden=256)
    torch.manual_seed(13)
    wrapper = Llama31_8B_Optimization(
        batch_size=1,
        seq_length=128,
        use_compile=False,
        attention_mode="preferred_sdpa",
    )
    attention = wrapper._create_attention_layer().to("cuda", torch.bfloat16).eval()
    x = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        actual = attention(x).float()
        grouped = _attention_reference(
            attention, x.float(), num_heads=4, num_kv_heads=2, kv_head_for="grouped"
        )

    torch.testing.assert_close(actual, grouped, rtol=0.02, atol=0.02)
