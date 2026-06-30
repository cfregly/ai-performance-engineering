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
    classify_vectorized_out,
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


def test_adaptive_parallelism_baseline_reuses_result_buffers_on_cpu() -> None:
    cfg = AdaptiveParallelismBenchmarkConfig(num_requests=64)
    workload = build_workload(cfg, torch.device("cpu"))
    result = torch.empty(cfg.num_requests, dtype=torch.int64)
    strategy_ids_cpu = torch.empty(cfg.num_requests, dtype=torch.int64)

    baseline = classify_baseline(
        workload,
        device=torch.device("cpu"),
        strategy_ids_cpu=strategy_ids_cpu,
        result=result,
    )

    assert baseline is result
    assert torch.equal(baseline, classify_vectorized(workload))


def test_adaptive_parallelism_vectorized_out_reuses_buffers_on_cpu() -> None:
    cfg = AdaptiveParallelismBenchmarkConfig(num_requests=64)
    workload = build_workload(cfg, torch.device("cpu"))
    result = torch.empty(cfg.num_requests, dtype=torch.int64)
    steady_decode = torch.empty(cfg.num_requests, dtype=torch.bool)
    data_mask = torch.empty_like(steady_decode)
    long_prefill = torch.empty_like(steady_decode)
    heavy_context = torch.empty_like(steady_decode)
    pipeline_mask = torch.empty_like(steady_decode)
    hybrid_mask = torch.empty_like(steady_decode)
    doubled_decode_tokens = torch.empty_like(workload["decode_tokens"])

    optimized = classify_vectorized_out(
        workload,
        result=result,
        steady_decode=steady_decode,
        data_mask=data_mask,
        long_prefill=long_prefill,
        heavy_context=heavy_context,
        pipeline_mask=pipeline_mask,
        hybrid_mask=hybrid_mask,
        doubled_decode_tokens=doubled_decode_tokens,
    )

    assert optimized is result
    assert torch.equal(optimized, classify_vectorized(workload))


def test_adaptive_parallelism_workload_reuses_slot_masks_without_where_fallbacks() -> None:
    source = inspect.getsource(build_workload)

    assert "slot1 = slots == 1" in source
    assert "slot2 = slots == 2" in source
    assert "slot3 = slots == 3" in source
    assert "torch.where(" not in source
    assert "torch.full_like(" not in source
    assert "masked_fill_" in source


def test_adaptive_parallelism_baseline_materializes_feature_rows_once() -> None:
    source = inspect.getsource(classify_baseline)

    assert "feature_rows = torch.stack(" in source
    assert ").detach().cpu()" in source
    assert ").detach().cpu().tolist()" not in source
    assert "strategy_ids_cpu = torch.empty(feature_rows.size(0), dtype=torch.int64)" in source
    assert "for row_idx in range(feature_rows.size(0)):" in source
    assert "feature_row = feature_rows[row_idx]" in source
    assert "seq_len=int(feature_row[0])" in source
    assert "gpu_mem_util=float(feature_row[1])" in source
    assert "strategy_ids_cpu[row_idx] = STRATEGY_TO_ID[config.strategy]" in source
    assert "result.copy_(strategy_ids_cpu, non_blocking=result.is_cuda)" in source
    assert "return result" in source
    assert "return strategy_ids_cpu.to(device=device)" in source
    assert "strategy_ids: list[int]" not in source
    assert "strategy_ids.append(" not in source
    assert "torch.tensor(strategy_ids" not in source
    assert "[idx].item()" not in source
    assert 'workload["seq_len"].detach().cpu()' not in source


def test_adaptive_parallelism_baseline_benchmark_reuses_result_buffers() -> None:
    source = inspect.getsource(BaselineAdaptiveParallelismBenchmark)

    assert "self._result_buffer: Optional[torch.Tensor] = None" in source
    assert "self._strategy_ids_cpu: Optional[torch.Tensor] = None" in source
    assert "self._result_buffer = torch.empty(" in source
    assert "self._strategy_ids_cpu = torch.empty(" in source
    assert "pin_memory=True" in source
    assert "strategy_ids_cpu=self._strategy_ids_cpu" in source
    assert "result=self._result_buffer" in source
    assert "self.output = classify_baseline(self.workload, device=self.device)" not in source


def test_adaptive_parallelism_optimized_benchmark_reuses_mask_buffers() -> None:
    source = inspect.getsource(OptimizedAdaptiveParallelismBenchmark)

    assert "self._result_buffer: Optional[torch.Tensor] = None" in source
    assert "self._steady_decode_mask: Optional[torch.Tensor] = None" in source
    assert "self._data_mask: Optional[torch.Tensor] = None" in source
    assert "self._doubled_decode_tokens: Optional[torch.Tensor] = None" in source
    assert "self._result_buffer = torch.empty(" in source
    assert "self._steady_decode_mask = torch.empty(" in source
    assert "self._data_mask = torch.empty_like(self._steady_decode_mask)" in source
    assert "self._doubled_decode_tokens = torch.empty_like(self.workload[\"decode_tokens\"])" in source
    assert "self.output = classify_vectorized_out(" in source
    assert "self.output = classify_vectorized(self.workload)" not in source


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
