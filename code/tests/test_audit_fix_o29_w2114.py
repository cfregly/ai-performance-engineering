"""Regressions for W2-114: FP6 memory-savings math, packed storage, and throughput claims."""

from __future__ import annotations

import math
import re

import pytest
import torch

import ch19.native_fp6_quantization as native_fp6
from ch19.native_fp6_quantization import FP6MLP, FP6Linear, FP6Tensor

STALE_CLAIMS = (
    "50% memory",
    "2x model capacity",
    "2x larger models",
    "tensor cores active",
    "speedup from reduced memory bandwidth",
    "uses native FP6 tensor cores",
)


def _assert_no_stale_claims(text: str) -> None:
    for claim in STALE_CLAIMS:
        assert claim not in text


def test_pack_writes_documented_three_byte_layout() -> None:
    fp6 = FP6Tensor(torch.zeros(4))
    codes = torch.tensor([0b101010, 0b010101, 0b111000, 0b000111], dtype=torch.uint8)

    packed = fp6._pack_fp6(codes)

    # AAAAAABB, BBBBCCCC, CCDDDDDD
    assert packed.dtype == torch.uint8
    assert packed.tolist() == [0b10101001, 0b01011110, 0b00000111]


def test_pack_rejects_codes_that_are_not_uint8() -> None:
    fp6 = FP6Tensor(torch.zeros(4))
    with pytest.raises(TypeError, match="uint8"):
        fp6._pack_fp6(torch.arange(4, dtype=torch.int64))


def test_pack_unpack_round_trips_every_code_through_padding() -> None:
    fp6 = FP6Tensor(torch.zeros(4))
    codes = torch.cat(
        (torch.arange(64, dtype=torch.uint8), torch.arange(63, 56, -1, dtype=torch.uint8))
    )
    assert codes.numel() % 4 != 0

    packed = fp6._pack_fp6(codes)

    assert packed.numel() == math.ceil(codes.numel() / 4) * 3
    torch.testing.assert_close(fp6._unpack_fp6(packed, codes.numel()), codes, rtol=0, atol=0)


def test_memory_usage_reports_the_stored_packed_bytes() -> None:
    torch.manual_seed(0)
    data = torch.randn(37, 11, dtype=torch.float16)

    fp6 = FP6Tensor(data)

    packed_bytes = math.ceil(data.numel() / 4) * 3
    scale_bytes = fp6.scales.numel() * fp6.scales.element_size()
    assert fp6.quantized_data.dtype == torch.uint8
    assert fp6.quantized_data.numel() == packed_bytes
    assert fp6.memory_usage() == packed_bytes + scale_bytes

    codes = fp6._round_to_fp6(data / fp6.scales)
    expected = ((codes.float() * 32.0 / 63.0) - 16.0) * fp6.scales
    torch.testing.assert_close(fp6.dequantize(), expected.to(torch.float16), rtol=0, atol=0)


def test_fp6_linear_stores_six_bits_per_weight() -> None:
    layer = FP6Linear(256, 128, bias=False)
    fp16_bytes = layer.memory_usage()["total_bytes"]

    layer.quantize()

    packed = layer.weight_fp6.quantized_data
    assert packed.numel() * packed.element_size() == 256 * 128 * 6 // 8
    savings = 1.0 - layer.memory_usage()["total_bytes"] / fp16_bytes
    assert savings == pytest.approx(0.625, abs=1e-3)


def test_benchmark_prints_measured_weight_savings_and_honest_throughput(
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.manual_seed(0)
    native_fp6.benchmark_fp6_vs_fp16(batch_size=2, seq_len=4, d_model=256, d_ff=512)
    out = capsys.readouterr().out

    match = re.search(r"Weight\+bias memory savings: (-?[0-9.]+)%", out)
    assert match is not None, out
    # The FP16 biases are not quantized, so savings land just under 62.5%.
    assert 62.0 < float(match.group(1)) <= 62.5

    assert "62.5% less weight memory than FP16 (up to 2.67x model capacity)" in out
    assert "does not measure FP6 tensor core throughput" in out
    _assert_no_stale_claims(out)


def test_takeaways_and_docstrings_use_six_bit_math(capsys: pytest.CaptureFixture[str]) -> None:
    native_fp6.print_key_takeaways()
    out = capsys.readouterr().out

    assert "62.5% less memory than FP16" in out
    assert "up to 2.67x as many weights as FP16" in out
    _assert_no_stale_claims(out)

    for doc in (FP6Linear.__doc__, FP6MLP.__doc__, FP6Linear.forward.__doc__):
        assert doc is not None
        _assert_no_stale_claims(doc)
    assert "62.5%" in FP6Linear.__doc__
    assert "62.5%" in FP6MLP.__doc__
    assert "No FP6 tensor core instructions" in FP6Linear.forward.__doc__
