"""Regressions for truthful packed INT6 storage and matched-model comparison."""

from __future__ import annotations

import math
import re

import pytest
import torch

import ch19.native_fp6_quantization as native_fp6
from ch19.native_fp6_quantization import PackedInt6Linear, PackedInt6MLP, PackedInt6Tensor

STALE_CLAIMS = (
    "50% memory",
    "2x model capacity",
    "2x larger models",
    "tensor cores active",
    "speedup from reduced memory bandwidth",
    "uses native FP6 tensor cores",
    "~12.5% relative error",
    "acceptable for most tasks",
    "FP6 Quantization Benchmark (Blackwell B200)",
)


def _assert_no_stale_claims(text: str) -> None:
    for claim in STALE_CLAIMS:
        assert claim not in text


def test_pack_writes_documented_three_byte_layout() -> None:
    int6 = PackedInt6Tensor(torch.zeros(4))
    codes = torch.tensor([0b101010, 0b010101, 0b111000, 0b000111], dtype=torch.uint8)

    packed = int6._pack_int6(codes)

    # AAAAAABB, BBBBCCCC, CCDDDDDD
    assert packed.dtype == torch.uint8
    assert packed.tolist() == [0b10101001, 0b01011110, 0b00000111]


def test_pack_rejects_codes_that_are_not_uint8() -> None:
    int6 = PackedInt6Tensor(torch.zeros(4))
    with pytest.raises(TypeError, match="uint8"):
        int6._pack_int6(torch.arange(4, dtype=torch.int64))


def test_pack_unpack_round_trips_every_code_through_padding() -> None:
    int6 = PackedInt6Tensor(torch.zeros(4))
    codes = torch.cat(
        (torch.arange(64, dtype=torch.uint8), torch.arange(63, 56, -1, dtype=torch.uint8))
    )
    assert codes.numel() % 4 != 0

    packed = int6._pack_int6(codes)

    assert packed.numel() == math.ceil(codes.numel() / 4) * 3
    torch.testing.assert_close(int6._unpack_int6(packed, codes.numel()), codes, rtol=0, atol=0)


def test_memory_usage_reports_the_stored_packed_bytes() -> None:
    torch.manual_seed(0)
    data = torch.randn(37, 11, dtype=torch.float16)

    int6 = PackedInt6Tensor(data)

    packed_bytes = math.ceil(data.numel() / 4) * 3
    scale_bytes = int6.scales.numel() * int6.scales.element_size()
    assert int6.quantized_data.dtype == torch.uint8
    assert int6.quantized_data.numel() == packed_bytes
    assert int6.memory_usage() == packed_bytes + scale_bytes

    codes = int6._round_to_int6(data / int6.scales)
    expected = (codes.to(torch.int16) - 32).to(torch.float32) * int6.scales
    torch.testing.assert_close(int6.dequantize(), expected.to(torch.float16), rtol=0, atol=0)


def test_symmetric_int6_preserves_zero_and_nonzero_signs() -> None:
    data = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=torch.float32)
    int6 = PackedInt6Tensor(data, dtype=torch.float32)
    reconstructed = int6.dequantize()

    assert reconstructed[2].item() == 0.0
    torch.testing.assert_close(reconstructed, -reconstructed.flip(0), rtol=0, atol=0)
    assert (data - reconstructed).abs().max() <= int6.scales / 2


def test_int6_linear_stores_six_bits_per_weight() -> None:
    layer = PackedInt6Linear(256, 128, bias=False)
    fp16_bytes = layer.memory_usage()["total_bytes"]

    layer.quantize()

    packed = layer.weight_int6.quantized_data
    assert packed.numel() * packed.element_size() == 256 * 128 * 6 // 8
    savings = 1.0 - layer.memory_usage()["total_bytes"] / fp16_bytes
    assert savings == pytest.approx(0.625, abs=1e-3)


def test_benchmark_prints_measured_weight_savings_and_honest_throughput(
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.manual_seed(0)
    native_fp6.benchmark_int6_vs_fp16(batch_size=2, seq_len=4, d_model=256, d_ff=512)
    out = capsys.readouterr().out

    match = re.search(r"Weight\+bias memory savings: (-?[0-9.]+)%", out)
    assert match is not None, out
    # The FP16 biases are not quantized, so savings land just under 62.5%.
    assert 62.0 < float(match.group(1)) <= 62.5

    assert "Six-bit codes use 62.5% fewer bits per weight than FP16" in out
    assert "This proxy does not implement E3M2 or issue native FP6 instructions" in out
    _assert_no_stale_claims(out)


def test_matched_pair_uses_identical_source_weights_and_deterministic_inference() -> None:
    torch.manual_seed(7)
    baseline, quantized = native_fp6._build_matched_mlp_pair(
        16,
        32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert not baseline.training
    assert not quantized.training
    for baseline_layer, quantized_layer in (
        (baseline.fc1, quantized.fc1),
        (baseline.fc2, quantized.fc2),
    ):
        expected = PackedInt6Tensor(
            baseline_layer.weight.detach(), dtype=torch.float32
        ).dequantize()
        torch.testing.assert_close(quantized_layer.weight_int6.dequantize(), expected)
        torch.testing.assert_close(quantized_layer.bias, baseline_layer.bias)

    inputs = torch.randn(2, 3, 16)
    with torch.inference_mode():
        first = quantized(inputs)
        second = quantized(inputs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_takeaways_and_docstrings_use_six_bit_math(capsys: pytest.CaptureFixture[str]) -> None:
    native_fp6.print_key_takeaways()
    out = capsys.readouterr().out

    assert "INT6 codes use 62.5% fewer bits per weight than FP16" in out
    assert "preserve zero exactly" in out
    _assert_no_stale_claims(out)

    for doc in (
        PackedInt6Linear.__doc__,
        PackedInt6MLP.__doc__,
        PackedInt6Linear.forward.__doc__,
    ):
        assert doc is not None
        _assert_no_stale_claims(doc)
    assert "62.5%" in PackedInt6Linear.__doc__
    assert "six bits" in PackedInt6MLP.__doc__
    assert "does not issue FP6 instructions" in PackedInt6Linear.forward.__doc__
