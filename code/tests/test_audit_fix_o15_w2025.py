"""W2-025 regressions: ch19 double buffering must not race across calls.

The optimized benchmark copies microbatches into two device slots on a copy
stream while a compute stream reads them. Within one call each H2D waited for
the compute that last read its slot. Across calls it did not. The preload into
slot 0 skipped the wait, the copy of batch 1 into slot 1 skipped it behind a
``batch_idx > 0`` guard, and neither side stream waited for the caller's
stream. A back-to-back call could therefore overwrite a slot that the previous
call's last microbatches were still reading, and the compute stream could
overwrite the output while the caller was still reading it.

The CPU tests run the real ``benchmark_fn`` with stream, event, buffer and
model stand-ins that track happens-before with vector clocks. They flag any
pair of accesses to the same resource from different streams where one is a
write and the two are not ordered. The CUDA tests run the real benchmark on a
GPU with ``torch.cuda._sleep`` delays that turn each race into a wrong value.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import ch19.optimized_memory_double_buffering as double_buffer


def _merge(target: dict[str, int], source: dict[str, int]) -> None:
    for name, tick in source.items():
        target[name] = max(target.get(name, 0), tick)


class _Simulator:
    def __init__(self) -> None:
        self.active_stream: _FakeStream | None = None
        self.current_stream: _FakeStream | None = None
        self.accesses: list[tuple[str, str, _FakeStream, int, dict[str, int]]] = []

    def access(self, kind: str, resource: str, stream: _FakeStream | None = None) -> None:
        if stream is None:
            stream = self.active_stream or self.current_stream
        if stream is None:
            raise AssertionError(f"{kind} of {resource} issued with no stream")
        stream.clock[stream.name] += 1
        self.accesses.append(
            (kind, resource, stream, stream.clock[stream.name], dict(stream.clock))
        )

    def races(self) -> list[str]:
        found = []
        for index, (kind_a, resource_a, stream_a, tick_a, _) in enumerate(self.accesses):
            for kind_b, resource_b, stream_b, _, clock_b in self.accesses[index + 1 :]:
                if resource_a != resource_b or stream_a is stream_b:
                    continue
                if kind_a == "read" and kind_b == "read":
                    continue
                if clock_b.get(stream_a.name, 0) < tick_a:
                    found.append(
                        f"{kind_a} of {resource_a} on {stream_a.name} is not ordered "
                        f"before {kind_b} on {stream_b.name}"
                    )
        return found


class _FakeStream:
    def __init__(self, sim: _Simulator, name: str) -> None:
        self.sim = sim
        self.name = name
        self.clock = {name: 0}

    def wait_event(self, event: _FakeEvent) -> None:
        # torch skips the wait when the event was never recorded.
        if event.clock is not None:
            _merge(self.clock, event.clock)

    def wait_stream(self, stream: _FakeStream) -> None:
        _merge(self.clock, stream.clock)


class _FakeEvent:
    def __init__(self) -> None:
        self.clock: dict[str, int] | None = None

    def record(self, stream: _FakeStream) -> None:
        self.clock = dict(stream.clock)


class _FakeBuffer:
    def __init__(self, sim: _Simulator, name: str) -> None:
        self.sim = sim
        self.name = name

    def copy_(self, source: str, *, non_blocking: bool) -> _FakeBuffer:
        assert non_blocking
        self.sim.access("write", self.name)
        return self


class _FakeModel:
    def __init__(self, sim: _Simulator) -> None:
        self.sim = sim

    def forward_prepared(self, buffer: _FakeBuffer) -> str:
        self.sim.access("read", buffer.name)
        self.sim.access("write", "output")
        return "output"


def _schedule(micro_batches: int) -> list[tuple[int, int, int | None, int | None]]:
    return [
        (
            batch_idx,
            batch_idx & 1,
            batch_idx + 1 if batch_idx + 1 < micro_batches else None,
            (batch_idx + 1) & 1 if batch_idx + 1 < micro_batches else None,
        )
        for batch_idx in range(micro_batches)
    ]


def _simulated_benchmark(
    monkeypatch: pytest.MonkeyPatch, micro_batches: int
) -> tuple[double_buffer.OptimizedMemoryDoubleBufferingBenchmark, _Simulator]:
    sim = _Simulator()
    sim.current_stream = _FakeStream(sim, "caller-0")

    @contextmanager
    def use_stream(stream: _FakeStream):
        previous = sim.active_stream
        sim.active_stream = stream
        try:
            yield
        finally:
            sim.active_stream = previous

    monkeypatch.setattr(double_buffer.torch.cuda, "stream", use_stream)
    monkeypatch.setattr(
        double_buffer.torch.cuda,
        "current_stream",
        lambda device=None: sim.current_stream,
    )

    benchmark = object.__new__(double_buffer.OptimizedMemoryDoubleBufferingBenchmark)
    buffers = [_FakeBuffer(sim, "slot-0"), _FakeBuffer(sim, "slot-1")]
    benchmark.copy_stream = _FakeStream(sim, "copy")
    benchmark.compute_stream = _FakeStream(sim, "compute")
    benchmark.copy_events = [_FakeEvent(), _FakeEvent()]
    benchmark.compute_events = [_FakeEvent(), _FakeEvent()]
    benchmark.buffers = buffers
    benchmark.buffer_a, benchmark.buffer_b = buffers
    benchmark.host_batches = [f"batch-{index}" for index in range(micro_batches)]
    benchmark.model = _FakeModel(sim)
    benchmark.output = None
    benchmark._buffer_event_counts = (2, 2)
    benchmark._expected_buffer_event_counts = (2, 2)
    benchmark._compute_event_count = 2
    benchmark._expected_compute_event_count = 2
    benchmark._micro_batch_schedule = _schedule(micro_batches)
    benchmark._micro_batch_schedule_count = micro_batches
    benchmark._expected_micro_batch_schedule_count = micro_batches
    benchmark._enable_nvtx = False
    benchmark.device = torch.device("cpu")
    return benchmark, sim


def _caller_reads(sim: _Simulator) -> None:
    for resource in ("output", "slot-0", "slot-1"):
        sim.access("read", resource, stream=sim.current_stream)


@pytest.mark.parametrize("micro_batches", [2, 3, 4, 5, 16])
def test_back_to_back_calls_have_no_cross_stream_races(
    monkeypatch: pytest.MonkeyPatch, micro_batches: int
) -> None:
    benchmark, sim = _simulated_benchmark(monkeypatch, micro_batches)

    for _ in range(3):
        benchmark.benchmark_fn()
        _caller_reads(sim)

    assert sim.races() == []


@pytest.mark.parametrize("micro_batches", [2, 3, 4, 5, 16])
def test_slot_reuse_waits_for_previous_call_compute_on_any_caller_stream(
    monkeypatch: pytest.MonkeyPatch, micro_batches: int
) -> None:
    # Each call runs under a new caller stream that has no ordering with the
    # previous one, so only the per-slot compute events can protect the slots.
    benchmark, sim = _simulated_benchmark(monkeypatch, micro_batches)

    for call_index in range(3):
        sim.current_stream = _FakeStream(sim, f"caller-{call_index}")
        benchmark.benchmark_fn()

    assert sim.races() == []


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SKIPPED: CUDA GPU required to run real copy and compute streams",
)

# About 1 ms or more per delay on current GPUs, far longer than one H2D copy
# of a small microbatch, so an unordered copy always lands inside the delay.
_SLEEP_CYCLES = 2_000_000


def _small_cuda_benchmark(
    micro_batches: int,
) -> double_buffer.OptimizedMemoryDoubleBufferingBenchmark:
    benchmark = double_buffer.OptimizedMemoryDoubleBufferingBenchmark()
    benchmark.batch_size = 2
    benchmark.seq_len = 32
    benchmark.hidden_dim = 64
    benchmark.micro_batches = micro_batches
    benchmark.setup()
    return benchmark


def _reference_output(
    model: double_buffer.BufferedMicrobatchMlp, host_batch: torch.Tensor
) -> torch.Tensor:
    x = host_batch.to(model.fc1.weight.device)
    hidden = torch.relu(x @ model.fc1.weight.t() + model.fc1.bias)
    return hidden @ model.fc2.weight.t() + model.fc2.bias


@CUDA_REQUIRED
def test_cuda_back_to_back_calls_read_the_intended_microbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    micro_batches = 6
    benchmark = _small_cuda_benchmark(micro_batches)
    try:
        original_forward = benchmark.model.forward_prepared
        seen: list[torch.Tensor] = []

        def delayed_forward(buffer: torch.Tensor) -> torch.Tensor:
            # The delay keeps the compute stream behind the copy stream, so an
            # H2D that does not wait for this read overwrites the slot first.
            torch.cuda._sleep(_SLEEP_CYCLES)
            seen.append(buffer.clone())
            return original_forward(buffer)

        monkeypatch.setattr(benchmark.model, "forward_prepared", delayed_forward)

        benchmark.benchmark_fn()
        benchmark.benchmark_fn()
        torch.cuda.synchronize()

        assert len(seen) == 2 * micro_batches
        for index, observed in enumerate(seen):
            expected = benchmark.host_batches[index % micro_batches].to(observed.device)
            assert torch.equal(observed, expected), (
                f"call {index // micro_batches} microbatch {index % micro_batches} "
                "read a slot that a later H2D had already overwritten"
            )
        with torch.inference_mode():
            reference = _reference_output(benchmark.model, benchmark.host_batches[-1])
        torch.testing.assert_close(benchmark.output, reference, rtol=1e-2, atol=1e-2)
    finally:
        benchmark.teardown()


@CUDA_REQUIRED
def test_cuda_next_call_waits_for_caller_reads_of_the_output() -> None:
    micro_batches = 4
    benchmark = _small_cuda_benchmark(micro_batches)
    try:
        first_batches = benchmark.host_batches
        benchmark.benchmark_fn()

        # The caller reads the output on its own stream after a delay. The next
        # call uses different inputs, so an unordered overwrite changes the read.
        torch.cuda._sleep(10 * _SLEEP_CYCLES)
        snapshot = benchmark.output.clone()
        benchmark.host_batches = [
            torch.randn_like(batch).pin_memory() for batch in first_batches
        ]
        benchmark.benchmark_fn()
        torch.cuda.synchronize()

        with torch.inference_mode():
            first_reference = _reference_output(benchmark.model, first_batches[-1])
            second_reference = _reference_output(benchmark.model, benchmark.host_batches[-1])
        torch.testing.assert_close(snapshot, first_reference, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(benchmark.output, second_reference, rtol=1e-2, atol=1e-2)
    finally:
        benchmark.teardown()
