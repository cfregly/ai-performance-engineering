"""Regression tests for the W1-053 and W1-066 residual fixes."""
from __future__ import annotations

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

_TFLOPS_MEASUREMENTS = (
    "measure_fp16_compute",
    "measure_bf16_compute",
    "measure_fp4_compute",
    "measure_fp6_compute",
    "measure_fp8_compute",
)


@pytest.mark.parametrize("measurement", _TFLOPS_MEASUREMENTS)
@pytest.mark.parametrize("iterations", [0, -1, True, 1.5])
def test_tflops_measurements_reject_invalid_iterations_before_any_work(measurement, iterations):
    # Zero timed calls used to report 0 TFLOPS, or fail FP4 on an unbound output.
    from core.benchmark import benchmark_peak

    measure = getattr(benchmark_peak, measurement)
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        measure(matrix_size=64, iterations=iterations)


def test_sdpa_auto_falls_back_but_pinned_cudnn_refuses_cpu_tensors():
    from labs.cudnn_sdpa_bench.baseline_flash_sdp import _sdpa_context

    q = torch.arange(64, dtype=torch.float32).reshape(1, 2, 4, 8) / 64
    with _sdpa_context("math"):
        reference = sdpa(q, q.flip(-2), q)
    with _sdpa_context("auto"):
        auto = sdpa(q, q.flip(-2), q)
    torch.testing.assert_close(auto, reference)
    with pytest.raises(RuntimeError, match="No viable backend"), _sdpa_context("cudnn"):
        sdpa(q, q.flip(-2), q)
