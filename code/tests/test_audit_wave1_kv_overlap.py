"""Stream-lifetime regression that requires actual CUDA for the race workload."""

import pytest
import torch

from labs.moe_cuda.optimized_kv_transfer import OptimizedKVTransferBenchmark


def test_kv_overlap_constructor_does_not_allocate_cuda_resources():
    benchmark = OptimizedKVTransferBenchmark()
    assert benchmark.compute_stream is None and benchmark.copy_stream is None
    assert benchmark.compute_done_events == []
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="requires CUDA"):
            benchmark.setup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA streams and matmul/copy overlap required")
@pytest.mark.parametrize("chunks,rows", [(1, 1), (3, 7), (9, 33)])
def test_repeated_kv_transfer_preserves_every_changed_chunk(chunks, rows):
    # Values and binary scale are exactly representable, giving an exact CPU
    # reference without choosing a loose tolerance that could hide a torn copy.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        benchmark = OptimizedKVTransferBenchmark()
        benchmark.num_chunks, benchmark.chunk_size, benchmark.hidden_size = chunks, rows, 32
        benchmark.setup()
        try:
            benchmark.weight.fill_(0.125)
            saved, references = [], []
            base = torch.arange(chunks * rows * 32, dtype=torch.float32).reshape(chunks, rows, 32) % 7
            for iteration in range(32):
                host_input = base + iteration
                benchmark.input_chunks.copy_(host_input)
                # Real device work delays the consumer. No streams, events,
                # CUDA calls or tensor results are mocked in this acceptance gate.
                with torch.cuda.stream(benchmark.copy_stream):
                    torch.cuda._sleep(100000)
                benchmark.benchmark_fn()
                saved.append(benchmark.kv_dest.clone())
                references.append((host_input.sum(-1, keepdim=True) * 0.125).expand(-1, -1, 32).half())
            torch.cuda.synchronize()
            for actual, expected in zip(saved, references, strict=True):
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        finally:
            benchmark.teardown()
