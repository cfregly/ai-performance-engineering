from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ch04.baseline_nixl_tier_handoff import get_benchmark as get_baseline_benchmark
from ch04.optimized_nixl_tier_handoff import get_benchmark as get_optimized_benchmark
from labs.nccl_nixl_nvshmem.comm_stack_common import TierHandoffBenchmark

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ch04_nixl_tier_handoff_wrappers_surface_real_chapter_pair() -> None:
    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()

    assert isinstance(baseline, TierHandoffBenchmark)
    assert isinstance(optimized, TierHandoffBenchmark)
    assert baseline.optimized is False
    assert optimized.optimized is True


def test_ch04_nixl_tier_handoff_optimized_reuses_pack_buffer() -> None:
    source = (
        REPO_ROOT / "labs" / "nccl_nixl_nvshmem" / "comm_stack_common.py"
    ).read_text(encoding="utf-8")
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload", maxsplit=1
    )[0]

    assert "self.packed_stage = torch.empty_like(self.gpu_stage)" in source
    assert "self._output_buffer = torch.empty_like(self.gpu_stage)" in source
    assert "self.dst = torch.empty_like(self.src)" in source
    assert "self.dst = torch.zeros_like(self.src)" not in source
    assert "self.dst.zero_()" not in benchmark_section
    assert "packed = self.src.index_select(0, self.selected_idx)" not in benchmark_section
    assert "torch.index_select(self.src, 0, self.selected_idx, out=self.packed_stage)" in benchmark_section
    assert "self.output = self.dst.index_select(0, self.selected_idx)" not in benchmark_section
    assert "torch.index_select(self.dst, 0, self.selected_idx, out=self._output_buffer)" in benchmark_section
    assert "self.selected_idx.cpu().tolist()" not in source
    assert ".cpu().tolist()" not in benchmark_section
    assert "selected_cpu = self.selected_cpu" in benchmark_section
    assert "self.selected_cpu = [int(idx) for idx in selected_cpu.tolist()] if not self.optimized else None" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ch04 nixl tier handoff benchmark")
def test_ch04_nixl_tier_handoff_pair_matches_selected_blocks() -> None:
    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()

    baseline.setup()
    optimized.setup()
    try:
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        assert baseline.output is not None
        assert optimized.output is not None
        assert torch.equal(baseline.output, optimized.output)
    finally:
        baseline.teardown()
        optimized.teardown()
