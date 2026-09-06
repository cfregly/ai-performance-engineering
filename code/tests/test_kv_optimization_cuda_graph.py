"""CUDA graph regressions for the standard optimized FP8 KV workload."""

from __future__ import annotations

import inspect

import pytest
import torch

from labs.kv_optimization.optimized_kv_standard import (
    OptimizedKVFP8Compressed,
    get_benchmark,
    run_benchmark,
)


def test_canonical_factory_selects_graph_and_eager_remains_explicit() -> None:
    eager = OptimizedKVFP8Compressed()
    candidate = get_benchmark()

    assert eager.use_cuda_graph is False
    assert candidate.use_cuda_graph is True
    parameters = list(inspect.signature(run_benchmark).parameters)
    assert parameters.index("use_cuda_graph") > parameters.index("profile")


def test_cuda_graph_variant_rejects_the_unvalidated_fp4_path() -> None:
    with pytest.raises(ValueError, match="only for the FP8 KV path"):
        OptimizedKVFP8Compressed(use_fp8=False, use_fp4=True, use_cuda_graph=True)


def test_graph_replay_has_no_eager_exception_fallback() -> None:
    benchmark_source = inspect.getsource(OptimizedKVFP8Compressed.benchmark_fn)
    capture_source = inspect.getsource(OptimizedKVFP8Compressed._configure_cuda_graph)

    assert "self._cuda_graph.replay()" in benchmark_source
    assert "except" not in benchmark_source
    assert "except" not in capture_source
    assert "_run_cuda_graph_body()" in capture_source


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"),
    reason="CUDA with torch.float8_e4m3fn required for graph replay parity",
)
def test_cuda_graph_replay_matches_the_full_eager_fp8_body() -> None:
    common = {
        "batch_size": 2,
        "num_layers": 3,
        "num_heads": 2,
        "head_dim": 16,
        "max_seq_length": 8,
        "active_layers": 2,
        "num_decode_steps": 2,
    }
    eager = OptimizedKVFP8Compressed(**common, use_cuda_graph=False)
    graph = OptimizedKVFP8Compressed(**common, use_cuda_graph=True)
    eager.setup()
    graph.setup()
    try:
        torch.testing.assert_close(eager._generated_k_steps, graph._generated_k_steps)
        torch.testing.assert_close(eager._generated_v_steps, graph._generated_v_steps)

        eager.benchmark_fn()
        graph.benchmark_fn()
        torch.cuda.synchronize()

        eager_cache = eager._written_cache_view().contiguous().view(torch.uint8)
        graph_cache = graph._written_cache_view().contiguous().view(torch.uint8)
        assert torch.equal(eager_cache, graph_cache)
        assert torch.equal(
            eager.k_scales[: eager.active_layers, : eager.num_decode_steps],
            graph.k_scales[: graph.active_layers, : graph.num_decode_steps],
        )
        assert torch.equal(
            eager.v_scales[: eager.active_layers, : eager.num_decode_steps],
            graph.v_scales[: graph.active_layers, : graph.num_decode_steps],
        )

        eager.capture_verification_payload()
        graph.capture_verification_payload()
        assert torch.equal(eager.get_verify_output(), graph.get_verify_output())
        assert graph.get_custom_metrics()["cuda_graph_enabled"] == 1.0
        assert eager.get_custom_metrics()["cuda_graph_enabled"] == 0.0
    finally:
        graph.teardown()
        eager.teardown()
