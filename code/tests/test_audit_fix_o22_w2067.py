"""Regressions for W2-067: ch17 pipeline stage timings across warmup and timed runs.

Harness warmup and the standalone runner call benchmark_fn() several times
without finalize_iteration_metrics(). On two or more GPUs the pipeline must
accept those calls, re-record every stage timing event, and report per-stage
GPU time from the latest iteration only.
"""
from __future__ import annotations

import contextlib
import math

import pytest
import torch
import torch.nn as nn

from ch17.optimized_pipeline_parallelism import OptimizedPipelineParallelismBenchmark
from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness

_STAGE_FEATURES = ((16, 64), (64, 64), (64, 32), (32, 16))
_MICRO_BATCHES = 4


class _StageTimingEvent:
    """Timing event stand-in whose elapsed time encodes stage and iteration.

    Each benchmark_fn() call records every pair once, so the record count is
    the 1-based iteration number of the latest record.
    """

    def __init__(self, stage_weight_ms: float) -> None:
        self.stage_weight_ms = stage_weight_ms
        self.record_count = 0
        self.synchronize_calls = 0

    def record(self, stream=None) -> None:
        self.record_count += 1

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def elapsed_time(self, end: _StageTimingEvent) -> float:
        if self.record_count != end.record_count:
            raise AssertionError("start and end events were recorded in different iterations")
        return end.stage_weight_ms * end.record_count


class _OrderingEvent:
    def __init__(self) -> None:
        self.record_count = 0

    def record(self, stream=None) -> None:
        self.record_count += 1


class _Stream:
    def __init__(self) -> None:
        self.waited_events: list[_OrderingEvent] = []
        self.waited_streams: list[_Stream] = []

    def wait_event(self, event: _OrderingEvent) -> None:
        self.waited_events.append(event)

    def wait_stream(self, stream: _Stream) -> None:
        self.waited_streams.append(stream)


@pytest.fixture
def cpu_cuda_seams(monkeypatch) -> _Stream:
    """Replace the CUDA stream and device context seams so the pipeline runs on CPU."""
    current_stream = _Stream()
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current_stream)
    return current_stream


def _cpu_pipeline() -> OptimizedPipelineParallelismBenchmark:
    """Build the multi-GPU pipeline state from setup() with CPU stages and event stand-ins."""
    torch.manual_seed(2067)
    bench = OptimizedPipelineParallelismBenchmark(micro_batches=_MICRO_BATCHES)
    stages = []
    for stage_idx, (in_features, out_features) in enumerate(_STAGE_FEATURES):
        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        if stage_idx < len(_STAGE_FEATURES) - 1:
            layers.append(nn.GELU())
        stages.append(nn.Sequential(*layers).to(dtype=torch.bfloat16).eval())
    stage_count = len(stages)
    micro_batches = bench.micro_batches

    bench._single_gpu_mode = False
    bench.pipeline_stages = stages
    bench._pipeline_stage_groups = list(enumerate(stages))
    bench._input_data = torch.randn(8, _STAGE_FEATURES[0][0], dtype=torch.bfloat16)
    bench.microbatch_inputs = list(bench._input_data.chunk(micro_batches, dim=0))
    bench._stage_devices = [torch.device("cpu") for _ in stages]
    bench.stage_streams = [_Stream() for _ in stages]
    bench.stage_events = [[_OrderingEvent() for _ in range(micro_batches)] for _ in stages]
    bench._stage_timing_events = [
        [
            (_StageTimingEvent(float(stage_idx + 1)), _StageTimingEvent(float(stage_idx + 1)))
            for _ in range(micro_batches)
        ]
        for stage_idx in range(stage_count)
    ]
    bench._stage_buffers = [[None for _ in range(micro_batches)] for _ in range(stage_count + 1)]
    bench._stage_buffers[0] = list(bench.microbatch_inputs)
    bench._stage_transfer_buffers = [[None for _ in range(micro_batches)] for _ in stages]
    bench._last_stage_durations_ms = [0.0 for _ in stages]
    bench._final_output_slots = [torch.empty(0, dtype=torch.bfloat16) for _ in range(micro_batches)]
    bench._pipeline_stage_count = stage_count
    bench._pipeline_stage_range = range(stage_count)
    bench._pipeline_micro_step_range = range(micro_batches + stage_count - 1)
    bench._stage_reuse_counts = (
        len(bench._last_stage_durations_ms),
        len(bench._pipeline_stage_groups),
        len(bench._stage_buffers),
        len(bench._stage_transfer_buffers),
        len(bench._stage_devices),
        len(bench._final_output_slots),
        len(bench._stage_timing_events),
    )
    bench._expected_stage_reuse_counts = (
        stage_count,
        stage_count,
        stage_count + 1,
        stage_count,
        stage_count,
        micro_batches,
        stage_count,
    )
    bench._stage_buffer_row_counts = tuple(len(row) for row in bench._stage_buffers)
    bench._expected_stage_buffer_row_counts = (micro_batches,) * (stage_count + 1)
    bench._transfer_buffer_row_counts = tuple(len(row) for row in bench._stage_transfer_buffers)
    bench._expected_transfer_buffer_row_counts = (micro_batches,) * stage_count
    return bench


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_w2_067_harness_warmup_reruns_pipeline_before_finalize(cpu_cuda_seams) -> None:
    bench = _cpu_pipeline()
    warmup = bench.get_config().warmup
    assert warmup >= 2
    config = BenchmarkConfig(device=torch.device("cpu"), warmup=warmup)
    harness = BenchmarkHarness(config=config)

    # The real harness warmup loop never calls finalize_iteration_metrics().
    harness._warmup(bench.benchmark_fn, config.warmup, config)
    assert bench._pending_stage_timings is True

    # The timed loop runs one more iteration, then finalizes after its sync.
    bench.benchmark_fn()
    bench.finalize_iteration_metrics()

    iterations = warmup + 1
    expected = [
        _MICRO_BATCHES * float(stage_idx + 1) * iterations
        for stage_idx in range(len(_STAGE_FEATURES))
    ]
    assert bench._last_stage_durations_ms == expected
    assert bench._pending_stage_timings is False
    for stage_pairs in bench._stage_timing_events:
        for start, end in stage_pairs:
            assert start.record_count == iterations
            assert end.record_count == iterations
            assert end.synchronize_calls == 1

    metrics = bench.get_custom_metrics()
    assert metrics["mode"] == "multi_gpu_pipeline"
    assert metrics["stage_time_min_ms"] == expected[0]
    assert metrics["stage_time_max_ms"] == expected[-1]
    assert metrics["stage_imbalance_ratio"] == pytest.approx(expected[-1] / expected[0])
    assert cpu_cuda_seams.waited_streams == [bench.stage_streams[-1]] * iterations

    with torch.inference_mode():
        for micro_idx, micro_input in enumerate(bench.microbatch_inputs):
            reference = micro_input
            for stage in bench.pipeline_stages:
                reference = stage(reference)
            assert torch.equal(bench._last_final_outputs[micro_idx], reference)


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_w2_067_missing_stage_input_fails_instead_of_timing_a_stale_pair(cpu_cuda_seams) -> None:
    bench = _cpu_pipeline()
    bench.benchmark_fn()
    bench.finalize_iteration_metrics()
    stale_pair = bench._stage_timing_events[0][1]

    bench._stage_buffers[0][1] = None
    with pytest.raises(RuntimeError, match="stage 0 microbatch 1 has no input"):
        bench.benchmark_fn()
    assert stale_pair[0].record_count == 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="SKIPPED: the multi-GPU pipeline path needs at least 2 CUDA devices",
)
def test_w2_067_cuda_pipeline_survives_harness_warmup_and_times_each_stage() -> None:
    bench = OptimizedPipelineParallelismBenchmark()
    bench.setup()
    try:
        assert bench._single_gpu_mode is False
        warmup = bench.get_config().warmup
        config = BenchmarkConfig(device=torch.device("cuda", 0), warmup=warmup)
        harness = BenchmarkHarness(config=config)
        harness._warmup(bench.benchmark_fn, config.warmup, config)

        bench.benchmark_fn()
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_idx)
        bench.finalize_iteration_metrics()
        metrics = bench.get_custom_metrics()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for micro_idx, micro_input in enumerate(bench.microbatch_inputs):
                reference = micro_input
                for stage, stage_device in zip(bench.pipeline_stages, bench._stage_devices, strict=True):
                    reference = stage(reference.to(stage_device))
                torch.testing.assert_close(
                    bench._last_final_outputs[micro_idx], reference, rtol=1e-3, atol=1e-3
                )
    finally:
        bench.teardown()

    assert metrics["mode"] == "multi_gpu_pipeline"
    assert metrics["pipeline_stages"] == 4
    assert metrics["stage_time_min_ms"] > 0.0
    assert math.isfinite(metrics["stage_imbalance_ratio"])
