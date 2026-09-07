"""Verification input binding for the naive KV-cache benchmark pair."""

from contextlib import nullcontext

import pytest
import torch

import ch13.baseline_kv_cache_naive as baseline_module
import ch13.optimized_kv_cache_naive_flash_blockwise as optimized_module
from ch13.kv_cache_workload import KVCacheWorkload


@pytest.mark.parametrize(
    ("benchmark_module", "benchmark_cls", "view_attribute"),
    (
        (
            baseline_module,
            baseline_module.BaselineKVCacheNaiveBenchmark,
            "_input_token_views",
        ),
        (
            optimized_module,
            optimized_module.OptimizedKVCacheNaiveFlashBlockwiseBenchmark,
            "_input_block_views",
        ),
    ),
)
def test_verification_input_is_live_final_request_storage(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_module,
    benchmark_cls,
    view_attribute: str,
) -> None:
    workload = KVCacheWorkload(
        batch_size=1,
        num_layers=1,
        num_heads=1,
        head_dim=4,
        sequence_lengths=(2, 3),
        dtype=torch.float32,
        page_size=4,
        block_size=2,
    )
    monkeypatch.setattr(benchmark_module, "WORKLOAD", workload)
    if benchmark_module is optimized_module:
        monkeypatch.setattr(optimized_module, "_flash_sdp_context", nullcontext)

    benchmark = benchmark_cls()
    benchmark.device = torch.device("cpu")
    benchmark.setup()
    try:
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        declared_input = benchmark.get_verify_inputs()["input"]
        original_output = benchmark.get_verify_output()

        assert declared_input.data_ptr() == benchmark.inputs[-1].data_ptr()
        request_views = getattr(benchmark, view_attribute)[-1]
        if benchmark_module is optimized_module:
            request_views = request_views[1]
            first_view = request_views[0][1]
        else:
            first_view = request_views[0]
        assert first_view.data_ptr() == declared_input.data_ptr()

        with torch.no_grad():
            declared_input.add_(0.25)
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        perturbed_output = benchmark.get_verify_output()

        assert not torch.allclose(original_output, perturbed_output, rtol=1e-5, atol=1e-5)
    finally:
        benchmark.teardown()
