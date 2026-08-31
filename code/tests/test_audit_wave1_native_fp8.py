"""Actual FP8 encoding and independent logical-output checks; GPU gates explicit."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from labs.moe_optimization_journey.level6_native_fp8 import NativeFP8MoE
from labs.moe_optimization_journey.native_fp8_math import (
    AccuracyLimits,
    combine_sorted_routes,
    full_output_errors,
    load_accuracy_limits,
    quantize_e4m3,
    reference_moe,
)


@pytest.mark.parametrize("values", [
    [0., 0., 0.], [448., -448., 0.], [2048., -4096., 16384.],
    [2.**-30, -(2.**-28), 0.], [-16384., -4096., -2048.],
])
def test_actual_fp8_encode_roundtrip(values):
    value = torch.tensor(values)
    out = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scale = quantize_e4m3(value, out)
    assert scale.dtype == torch.float32 and scale.numel() == 1
    assert torch.isfinite(out.float()).all()
    # Every value in this dyadic family maps exactly to representable E4M3.
    torch.testing.assert_close(out.float() * scale, value, rtol=0, atol=0)


def test_original_unscaled_cast_produces_nan_but_scaled_encoding_is_finite():
    hidden = torch.tensor([500., 2000., -2000., 32000.])
    assert torch.isnan(hidden.to(torch.float8_e4m3fn).float()).all()
    encoded = torch.empty_like(hidden, dtype=torch.float8_e4m3fn)
    scale = quantize_e4m3(hidden, encoded)
    assert torch.isfinite(encoded.float()).all()
    assert float(scale) > 1
    # Bound comes from the 3-bit E4M3 mantissa, not a MoE quality policy.
    torch.testing.assert_close(encoded.float() * scale, hidden, rtol=0.0625, atol=0)


def test_quantization_writes_noncontiguous_destination_and_preserves_input():
    value = torch.arange(32.).reshape(4, 8).T
    original = value.clone()
    encoded = torch.empty((4, 8), dtype=torch.float8_e4m3fn).T
    scale = quantize_e4m3(value, encoded)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    torch.testing.assert_close(encoded.float() * scale, value, rtol=0.0625, atol=0.0625)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_cannot_become_accepted_output(invalid):
    value = torch.tensor([invalid, 1.])
    encoded = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scale = quantize_e4m3(value, encoded)
    with pytest.raises(AssertionError, match="Nonfinite"):
        full_output_errors(encoded.float() * scale, torch.ones(2))


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_complete_unsort_and_weighted_route_sum(top_k):
    logical = torch.arange(5 * top_k * 16, dtype=torch.float32).reshape(5 * top_k, 16)
    order = torch.randperm(5 * top_k, generator=torch.Generator().manual_seed(52))
    actual = torch.full((5, 16), float("nan"))
    combine_sorted_routes(logical[order], torch.argsort(order), top_k,
                          torch.empty_like(logical), actual)
    expected = torch.stack([sum(logical[row * top_k + route] for route in range(top_k))
                            for row in range(5)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reference_uses_distinct_experts_empty_expert_and_duplicate_routes():
    rng = torch.Generator().manual_seed(51)
    x = torch.randn(5, 16, generator=rng, dtype=torch.bfloat16)
    weights = (torch.randn(4, 16, 32, generator=rng, dtype=torch.bfloat16),
               torch.randn(4, 16, 32, generator=rng, dtype=torch.bfloat16),
               torch.randn(4, 32, 16, generator=rng, dtype=torch.bfloat16))
    ids = torch.tensor([[2, 0], [0, 0], [3, 2], [3, 0], [2, 3]])
    routing = torch.tensor([[.25, .75]], dtype=torch.bfloat16).repeat(5, 1)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for row in range(5):
        for route in range(2):
            e = int(ids[row, route])
            gate = F.silu(x[row:row + 1] @ weights[0][e])
            hidden = gate * (x[row:row + 1] @ weights[1][e])
            expected[row] += ((hidden @ weights[2][e]) * routing[row, route]).float()[0]
    actual = reference_moe(x, weights, ids, routing)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["zero", "last", "cancel", "nan", "alias", "short"])
def test_full_output_checker_rejects_invalid_candidate(corruption):
    ref = torch.arange(1., 65.).reshape(4, 16)
    actual = ref.clone()
    if corruption == "zero":
        actual.zero_()
    elif corruption == "last":
        actual[-1, -1] += 100
    elif corruption == "cancel":
        actual[0, 0] += 100
        actual[-1, -1] -= 100
    elif corruption == "nan":
        actual[-1, -1] = float("nan")
    elif corruption == "alias":
        actual = ref
    elif corruption == "short":
        actual = actual[:1, :8]
    with pytest.raises(AssertionError):
        AccuracyLimits(0, 0, 0, 0).check(full_output_errors(actual, ref))


def test_accuracy_requires_explicit_policy_and_rejects_loose_or_nonfinite_bounds(monkeypatch, tmp_path):
    monkeypatch.delenv("AISP_NATIVE_FP8_ACCURACY_POLICY", raising=False)
    with pytest.raises(RuntimeError, match="uncalibrated"):
        load_accuracy_limits()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema_version": 1, "native_fp8": {
        "relative_l2": 0, "normalized_max_abs": 0, "pairwise_rtol": 0, "pairwise_atol": 0}}))
    monkeypatch.setenv("AISP_NATIVE_FP8_ACCURACY_POLICY", str(policy))
    assert load_accuracy_limits() == AccuracyLimits(0, 0, 0, 0)
    for bad in (1, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            AccuracyLimits(bad, 0, 0, 0)


def test_calibration_records_unavailable_or_invalid_actual_environment(tmp_path):
    output = tmp_path / "attempt.json"
    command = [sys.executable, "-m", "labs.moe_optimization_journey.calibrate_native_fp8",
               "--output", str(output), "--hidden-size", "17"]
    cwd = Path(__file__).resolve().parents[1]
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=45)
    assert output.exists(), completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    if torch.cuda.is_available():
        assert completed.returncode == 1
        assert report["status"] == "FAILURE_NOT_ACCEPTED"
        assert report["error_type"] == "ValueError"
    else:
        assert completed.returncode == 3
        assert report["status"] == "HOLD"
    assert report["config"]["HIDDEN_SIZE"] == 17
    assert report["errors"] == [] and len(report["source_sha256"]) == 3
    original = output.read_bytes()
    repeated = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=45)
    assert repeated.returncode != 0 and output.read_bytes() == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual native FP8 CUDA execution required")
def test_native_fp8_complete_production_path_with_reviewed_policy():
    if not os.environ.get("AISP_NATIVE_FP8_ACCURACY_POLICY"):
        pytest.skip("Reviewed native FP8 accuracy policy required; calibration is not acceptance")
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("Native FP8 CUDA requires supported compute capability")
    bench = NativeFP8MoE()
    bench.HIDDEN_SIZE, bench.INTERMEDIATE_SIZE = 32, 48
    bench.NUM_EXPERTS, bench.TOP_K = 4, 3
    bench.BATCH_SIZE, bench.SEQ_LEN = 1, 7
    try:
        bench.setup()
        with torch.inference_mode():
            base = bench.x.clone()
            for amplitude in (1, 4, 16):
                bench.x.copy_(base * amplitude)
                bench.benchmark_fn()
                assert bench.output.shape == (7, 32)
                bench.capture_verification_payload()
                assert bench.validate_result() is None
                torch.testing.assert_close(bench.get_verify_output(), bench.output, rtol=0, atol=0)
                # Last element was outside the original eight-element payload.
                bench.output[-1, -1] = float("nan")
                with pytest.raises(AssertionError, match="Nonfinite"):
                    bench.capture_verification_payload()
    finally:
        bench.teardown()
