from __future__ import annotations

import inspect

import pytest
import torch

from ch19.adaptive_parallelism_benchmark_common import (
    AdaptiveParallelismBenchmarkConfig,
    STRATEGY_TO_ID,
    build_workload,
    classify_baseline,
    classify_vectorized,
)
from ch19.adaptive_parallelism_strategy import ParallelismStrategy
from ch19.baseline_adaptive_parallelism import BaselineAdaptiveParallelismBenchmark
from ch19.optimized_adaptive_parallelism import OptimizedAdaptiveParallelismBenchmark


def test_adaptive_parallelism_common_logic_covers_all_strategy_branches() -> None:
    cfg = AdaptiveParallelismBenchmarkConfig(num_requests=16)
    workload = build_workload(cfg, torch.device("cpu"))
    result = classify_vectorized(workload).cpu()

    expected = torch.tensor(
        [
            STRATEGY_TO_ID[ParallelismStrategy.TENSOR],
            STRATEGY_TO_ID[ParallelismStrategy.PIPELINE],
            STRATEGY_TO_ID[ParallelismStrategy.HYBRID],
            STRATEGY_TO_ID[ParallelismStrategy.DATA],
        ]
        * 4,
        dtype=torch.int64,
    )
    assert torch.equal(result, expected)


def test_adaptive_parallelism_baseline_and_vectorized_paths_match_on_cpu() -> None:
    cfg = AdaptiveParallelismBenchmarkConfig(num_requests=64)
    workload = build_workload(cfg, torch.device("cpu"))

    baseline = classify_baseline(workload, device=torch.device("cpu")).cpu()
    optimized = classify_vectorized(workload).cpu()

    assert torch.equal(baseline, optimized)


def test_adaptive_parallelism_baseline_materializes_feature_rows_once() -> None:
    source = inspect.getsource(classify_baseline)

    assert "feature_rows = torch.stack(" in source
    assert ").detach().cpu().tolist()" in source
    assert "for seq_len, gpu_mem_util, concurrent_reqs, batch_size, prefill_tokens, decode_tokens in feature_rows" in source
    assert "[idx].item()" not in source
    assert 'workload["seq_len"].detach().cpu()' not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for chapter 19 adaptive-parallelism benchmark pair")
def test_adaptive_parallelism_benchmark_pair_matches_on_gpu() -> None:
    cfg = AdaptiveParallelismBenchmarkConfig(num_requests=64)
    baseline = BaselineAdaptiveParallelismBenchmark(cfg=cfg)
    optimized = OptimizedAdaptiveParallelismBenchmark(cfg=cfg)

    baseline.setup()
    optimized.setup()
    try:
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        assert baseline.output is not None
        assert optimized.output is not None
        assert torch.equal(baseline.output.cpu(), optimized.output.cpu())
    finally:
        baseline.teardown()
        optimized.teardown()
