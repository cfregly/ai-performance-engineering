"""Equivalence checks for the Chapter 19 dynamic-quantized-cache benchmark pair."""

from __future__ import annotations

import pytest
import torch

from ch19.baseline_dynamic_quantized_cache import get_benchmark as get_baseline_benchmark
from ch19.baseline_dynamic_quantized_cache_coalesced import (
    get_benchmark as get_coalesced_baseline_benchmark,
)
from ch19.optimized_dynamic_quantized_cache import get_benchmark as get_optimized_benchmark
from ch19.optimized_dynamic_quantized_cache_coalesced import (
    get_benchmark as get_coalesced_optimized_benchmark,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for dynamic_quantized_cache benchmarks")
def test_dynamic_quantized_cache_pair_outputs_match_within_contract_tolerance() -> None:
    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()
    try:
        baseline.setup()
        optimized.setup()
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        baseline_pair = baseline._timing_pair
        optimized_pair = optimized._timing_pair
        assert baseline_pair is not None
        assert optimized_pair is not None
        assert baseline._pending_timing_pair is baseline_pair
        assert optimized._pending_timing_pair is optimized_pair

        baseline.capture_verification_payload()
        optimized.capture_verification_payload()

        b_out = baseline.get_verify_output()
        o_out = optimized.get_verify_output()
        assert b_out.shape == o_out.shape
        rtol, atol = baseline.get_output_tolerance()
        assert torch.allclose(b_out.detach().cpu(), o_out.detach().cpu(), rtol=rtol, atol=atol)

        baseline.benchmark_fn()
        optimized.benchmark_fn()
        assert baseline._timing_pair is baseline_pair
        assert optimized._timing_pair is optimized_pair
        baseline.finalize_iteration_metrics()
        optimized.finalize_iteration_metrics()
    finally:
        baseline.teardown()
        optimized.teardown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for dynamic_quantized_cache benchmarks")
def test_dynamic_quantized_cache_coalesced_pair_outputs_match_within_contract_tolerance() -> None:
    baseline = get_coalesced_baseline_benchmark()
    optimized = get_coalesced_optimized_benchmark()
    try:
        assert baseline._refresh_schedule_bits == baseline.schedule_bits
        assert optimized._refresh_schedule_bits == [8, 6, 4]
        baseline.setup()
        optimized.setup()
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        baseline.capture_verification_payload()
        optimized.capture_verification_payload()

        b_out = baseline.get_verify_output()
        o_out = optimized.get_verify_output()
        assert b_out.shape == o_out.shape
        rtol, atol = baseline.get_output_tolerance()
        assert torch.allclose(b_out.detach().cpu(), o_out.detach().cpu(), rtol=rtol, atol=atol)
    finally:
        baseline.teardown()
        optimized.teardown()
