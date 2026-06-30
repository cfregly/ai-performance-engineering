from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from ch04.baseline_nixl_tier_handoff import get_benchmark as get_baseline_benchmark
from ch04.optimized_nixl_tier_handoff import get_benchmark as get_optimized_benchmark
from labs.nccl_nixl_nvshmem.comm_stack_common import TierHandoffBenchmark
from labs.nccl_nixl_nvshmem.run_lab_nccl_nixl_nvshmem import _measure

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
    capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]
    validate_section = source.split("def validate_result", maxsplit=1)[1].split(
        "def apply_target_overrides", maxsplit=1
    )[0]

    assert "self.packed_stage = torch.empty_like(self.gpu_stage)" in source
    assert "self._output_buffer = torch.empty_like(self.gpu_stage)" in source
    assert "self._expected_buffer = torch.empty_like(self.gpu_stage)" in source
    assert "self.dst = torch.empty_like(self.src)" in source
    assert "self.dst = torch.zeros_like(self.src)" not in source
    assert "self.dst.zero_()" not in benchmark_section
    assert "packed = self.src.index_select(0, self.selected_idx)" not in benchmark_section
    assert "src = self.src" in benchmark_section
    assert "dst = self.dst" in benchmark_section
    assert "selected_idx = self.selected_idx" in benchmark_section
    assert "packed_stage = self.packed_stage" in benchmark_section
    assert "copy_stream = self.copy_stream" in benchmark_section
    assert "copy_ready = self.copy_ready" in benchmark_section
    assert "current_stream = torch.cuda.current_stream(self.device)" in benchmark_section
    assert "torch.index_select(src, 0, selected_idx, out=packed_stage)" in benchmark_section
    assert "copy_stream.wait_stream(current_stream)" in benchmark_section
    assert "current_stream.wait_event(copy_ready)" in benchmark_section
    assert "torch.cuda.current_stream().wait_event(self.copy_ready)" not in benchmark_section
    assert "self.output = self.dst.index_select(0, self.selected_idx)" not in benchmark_section
    assert "torch.index_select(self.dst, 0, self.selected_idx, out=self._output_buffer)" in benchmark_section
    assert "selected_source = self.src.index_select(0, self.selected_idx)" not in capture_section
    assert "expected = self.src.index_select(0, self.selected_idx)" not in validate_section
    assert "torch.index_select(self.src, 0, self.selected_idx, out=self._expected_buffer)" in capture_section
    assert "torch.index_select(self.src, 0, self.selected_idx, out=self._expected_buffer)" in validate_section
    assert "self.selected_idx.cpu().tolist()" not in source
    assert ".cpu().tolist()" not in benchmark_section
    assert "selected_copy_pairs = self.selected_copy_pairs" in benchmark_section
    assert "self.selected_cpu = [int(idx) for idx in selected_cpu.tolist()] if not self.optimized else None" in source
    assert "self.selected_copy_pairs = list(enumerate(self.selected_cpu)) if not self.optimized else None" in source
    assert "for slot, block_idx in selected_copy_pairs:" in benchmark_section
    assert "for slot, block_idx in enumerate(selected_cpu):" not in benchmark_section
    assert "self.baseline_copy_ready: Optional[torch.cuda.Event] = None" in source
    assert "self.baseline_copy_ready = torch.cuda.Event() if not self.optimized else None" in source
    assert "copy_ready.synchronize()" in benchmark_section
    assert "torch.cuda.synchronize(self.device)" not in benchmark_section


def test_nccl_nixl_runner_measure_cuda_path_uses_single_event_bracket() -> None:
    source = inspect.getsource(_measure)
    cuda_section = source.split("if torch.cuda.is_available():", maxsplit=1)[1].split(
        "sample_count = max(iterations, 1)",
        maxsplit=1,
    )[0]

    assert cuda_section.count("torch.cuda.synchronize()") == 1
    assert cuda_section.count("current_stream = torch.cuda.current_stream()") == 1
    assert cuda_section.count("start.record(current_stream)") == 1
    assert cuda_section.count("end.record(current_stream)") == 1
    assert "start.record()" not in cuda_section
    assert "end.record()" not in cuda_section
    assert cuda_section.count("end.synchronize()") == 1
    assert "timings.append(start.elapsed_time(end))" not in cuda_section
    assert "timings = []" not in source
    assert "timings.append(" not in source
    assert "sum(timings)" not in source
    assert "sample_count = max(iterations, 1)" in source
    assert "total_ms += (time.perf_counter() - t0) * 1000.0" in source
    assert "return float(total_ms / sample_count)" in source


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
