from __future__ import annotations

import pytest
import torch

from ch17.baseline_prefill_decode_disagg import BaselinePrefillDecodeMonolithicBenchmark
from ch17.optimized_prefill_decode_disagg import OptimizedDisaggregatedBenchmark

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _FakeLLM:
    def __init__(self, *, device: torch.device) -> None:
        self.prefill_calls = 0
        self.decode_calls = 0
        self.prefill_output = torch.randn(1, 4, 8, device=device, dtype=torch.bfloat16)
        self.decode_output = torch.randn(1, 1, 8, device=device, dtype=torch.bfloat16)

    def prefill(self, prompt: torch.Tensor) -> torch.Tensor:
        self.prefill_calls += 1
        assert prompt.is_cuda
        return self.prefill_output

    def decode(self, token: torch.Tensor, *, num_tokens: int) -> torch.Tensor:
        self.decode_calls += num_tokens
        assert token.is_cuda
        return self.decode_output

    def decode_step(self, token: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        assert token.is_cuda
        return self.decode_output


@CUDA_REQUIRED
def test_ch17_monolithic_prefill_decode_reuses_timing_events() -> None:
    bench = BaselinePrefillDecodeMonolithicBenchmark()
    bench.decode_seq = 3
    fake_model = _FakeLLM(device=bench.device)
    bench.model = fake_model
    bench.prompt = torch.zeros(1, 4, device=bench.device, dtype=torch.long)

    bench.benchmark_fn()
    torch.cuda.synchronize(bench.device)

    ttft_events = bench._ttft_events
    tpot_events = bench._tpot_events
    assert ttft_events is not None
    assert bench._pending_ttft_pair is ttft_events
    assert bench._pending_tpot_pairs is tpot_events
    assert len(tpot_events) == bench.decode_seq
    assert fake_model.prefill_calls == 1
    assert fake_model.decode_calls == bench.decode_seq

    metrics_payload = bench.finalize_iteration_metrics()
    assert metrics_payload is not None
    ttft_times = metrics_payload["ttft_times_ms"]
    tpot_times = metrics_payload["tpot_times_ms"]
    assert bench._pending_tpot_pairs is bench._empty_tpot_pairs

    bench.benchmark_fn()
    torch.cuda.synchronize(bench.device)
    next_metrics_payload = bench.finalize_iteration_metrics()

    assert bench._ttft_events is ttft_events
    assert bench._tpot_events is tpot_events
    assert next_metrics_payload is metrics_payload
    assert next_metrics_payload is not None
    assert next_metrics_payload["ttft_times_ms"] is ttft_times
    assert next_metrics_payload["tpot_times_ms"] is tpot_times
    assert bench._pending_tpot_pairs is bench._empty_tpot_pairs


@CUDA_REQUIRED
def test_ch17_disaggregated_prefill_decode_reuses_timing_events() -> None:
    bench = OptimizedDisaggregatedBenchmark()
    bench.decode_seq = 3
    fake_model = _FakeLLM(device=bench.device)
    bench.model = fake_model
    bench.prompt = torch.zeros(1, 4, device=bench.device, dtype=torch.long)
    bench.prefill_stream = torch.cuda.Stream(device=bench.device)
    bench.decode_stream = torch.cuda.Stream(device=bench.device)
    bench._prefill_done = torch.cuda.Event()
    bench._ttft_events = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    bench._tpot_events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(bench.decode_seq)
    ]
    bench._tpot_event_count = bench.decode_seq

    bench.benchmark_fn()
    torch.cuda.synchronize(bench.device)

    ttft_events = bench._ttft_events
    tpot_events = bench._tpot_events
    assert ttft_events is not None
    assert bench._pending_ttft_pair is ttft_events
    assert bench._pending_tpot_pairs is tpot_events
    assert bench._pending_tpot_count == bench.decode_seq
    assert len(tpot_events) == bench.decode_seq
    assert fake_model.prefill_calls == 1
    assert fake_model.decode_calls == bench.decode_seq

    metrics_payload = bench.finalize_iteration_metrics()
    assert metrics_payload is not None
    ttft_times = metrics_payload["ttft_times_ms"]
    tpot_times = metrics_payload["tpot_times_ms"]
    assert bench._pending_tpot_pairs is bench._empty_tpot_pairs
    assert bench._pending_tpot_count == 0

    bench.benchmark_fn()
    torch.cuda.synchronize(bench.device)
    next_metrics_payload = bench.finalize_iteration_metrics()

    assert bench._ttft_events is ttft_events
    assert bench._tpot_events is tpot_events
    assert next_metrics_payload is metrics_payload
    assert next_metrics_payload is not None
    assert next_metrics_payload["ttft_times_ms"] is ttft_times
    assert next_metrics_payload["tpot_times_ms"] is tpot_times
    assert bench._pending_tpot_pairs is bench._empty_tpot_pairs
    assert bench._pending_tpot_count == 0
