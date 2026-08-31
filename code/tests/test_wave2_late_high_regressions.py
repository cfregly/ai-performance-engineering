from __future__ import annotations

import importlib
import inspect
from contextlib import nullcontext

import pytest
import torch


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    (
        ("ch15.baseline_kv_cache_nvlink_pool", "BaselineKVCacheLocalOnlyBenchmark"),
        ("ch15.optimized_kv_cache_nvlink_pool", "OptimizedKVCacheNvlinkPoolBenchmark"),
        (
            "ch15.baseline_kv_cache_nvlink_pool_multigpu",
            "BaselineKVCacheLocalOnlyBenchmark",
        ),
        (
            "ch15.optimized_kv_cache_nvlink_pool_multigpu",
            "OptimizedKVCacheNvlinkPoolBenchmark",
        ),
    ),
)
def test_w2_019_kv_pool_verification_rejects_garbage_outputs(
    module_name: str,
    class_name: str,
) -> None:
    benchmark_type = getattr(importlib.import_module(module_name), class_name)
    benchmark = benchmark_type()
    benchmark.model = torch.nn.Identity()
    benchmark.output = torch.ones(1, 4)
    benchmark._verify_q = torch.ones(1, 4)
    benchmark._synchronize = lambda: None

    benchmark.capture_verification_payload()
    rtol, atol = benchmark.get_output_tolerance()

    assert (rtol, atol) == (1e-5, 1e-6)
    assert not torch.allclose(
        benchmark.output,
        torch.zeros_like(benchmark.output),
        rtol=rtol,
        atol=atol,
    )


def test_w2_020_fp8_kv_cache_preserves_scale_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ch16.inference_optimizations_blackwell as blackwell

    # Float16 provides a CPU-supported stand-in storage dtype so this test can
    # exercise the FP8 scale bookkeeping without claiming FP8 hardware coverage.
    monkeypatch.setattr(blackwell, "FP8_AVAILABLE", True)
    monkeypatch.setattr(blackwell, "FP8_E4M3", torch.float16)
    cache = blackwell.DynamicQuantizedKVCache(
        num_layers=1,
        max_batch_size=1,
        max_seq_len=4,
        num_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    first = torch.tensor([[[[1.0, -1.0]]]])
    second = torch.tensor([[[[100.0, -100.0]]]])

    cache.update(0, first, first)
    cached_key, cached_value = cache.update(0, second, second)

    assert cache.scales.shape == (1, 2, 1, 4)
    torch.testing.assert_close(cache.scales[0, 0, 0, :2], torch.tensor([1.0, 100.0]))
    torch.testing.assert_close(cached_key[0, 0, :, 0], torch.tensor([1.0, 100.0]))
    torch.testing.assert_close(cached_value[0, 0, :, 0], torch.tensor([1.0, 100.0]))


def test_w2_021_sliding_window_mask_is_causal_with_cache_offset() -> None:
    from ch16.inference_optimizations_blackwell import (
        _dense_sliding_window_causal_mask,
    )

    mask = _dense_sliding_window_causal_mask(
        seq_len=2,
        total_len=4,
        window_size=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [False, True, True, False],
                [False, False, True, True],
            ]
        ),
    )


def test_w2_021_sdpa_fallback_blocks_future_token_influence() -> None:
    from ch16.inference_optimizations_blackwell import OptimizedDecoderLayer

    torch.manual_seed(20260831)
    layer = OptimizedDecoderLayer(
        d_model=32,
        num_heads=2,
        window_size=2,
        device="cpu",
        use_flex_attention=False,
    ).eval()
    original = torch.randn(1, 2, 32)
    changed_future = original.clone()
    changed_future[:, 1].mul_(1000.0)

    with torch.inference_mode():
        original_first = layer(original)[:, 0].clone()
        changed_first = layer(changed_future)[:, 0].clone()

    torch.testing.assert_close(original_first, changed_first, rtol=1e-5, atol=1e-6)


def test_w2_022_aggregate_counts_replicated_tensor_parallel_work_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ch16.inference_server_load_test as load_test

    completion = {
        "request_id": "request-0",
        "prompt_tokens": 8,
        "generated_tokens": 4,
        "latency_ms": 12.0,
    }
    local = {
        "rank": 0,
        "elapsed": 2.0,
        "stats": {
            "total_requests": 3,
            "completed_requests": 1,
            "rejected_requests": 0,
            "total_tokens_generated": 4,
        },
        "completions": [completion],
        "world_size": 2,
    }
    replica = {
        **local,
        "rank": 1,
        "elapsed": 1.5,
        "completions": [{**completion, "latency_ms": 20.0}],
    }
    monkeypatch.setattr(load_test.dist, "get_world_size", lambda: 2)

    def _all_gather_object(gathered, _local_result) -> None:
        gathered[:] = [local, replica]

    monkeypatch.setattr(load_test.dist, "all_gather_object", _all_gather_object)

    result = load_test.aggregate_results(local)

    assert result["total_requests"] == 3
    assert result["completed_requests"] == 1
    assert result["tokens_generated"] == 4
    assert result["throughput_tok_per_s"] == 2.0
    assert result["samples_collected"] == 1
    assert result["latency_p50_ms"] == 12.0


def test_w2_022_aggregate_rejects_divergent_rank_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ch16.inference_server_load_test as load_test

    local = {
        "rank": 0,
        "elapsed": 1.0,
        "stats": {
            "total_requests": 1,
            "completed_requests": 1,
            "rejected_requests": 0,
            "total_tokens_generated": 1,
        },
        "completions": [],
        "world_size": 2,
    }
    divergent = {**local, "rank": 1, "stats": {**local["stats"], "total_requests": 2}}
    monkeypatch.setattr(load_test.dist, "get_world_size", lambda: 2)

    def _all_gather_object(gathered, _local_result) -> None:
        gathered[:] = [local, divergent]

    monkeypatch.setattr(load_test.dist, "all_gather_object", _all_gather_object)

    with pytest.raises(RuntimeError, match="rank 1 diverged for total_requests"):
        load_test.aggregate_results(local)


def test_w2_023_demo_lm_samples_each_sequence_last_real_token() -> None:
    from ch16.inference_serving_multigpu import DemoCausalLM

    torch.manual_seed(20260831)
    model = DemoCausalLM(
        vocab_size=32,
        d_model=32,
        num_layers=1,
        num_heads=2,
        num_gpus=1,
        max_batch_size=2,
        max_seq_len=8,
    ).eval()
    padded_batch = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [1, 2, 3, 4, 5],
        ]
    )

    with torch.inference_mode():
        padded_logits, _, _ = model(padded_batch, input_lengths=[3, 5])
        unpadded_logits, _, _ = model(padded_batch[:1, :3])

    torch.testing.assert_close(
        padded_logits[0],
        unpadded_logits[0],
        rtol=1e-5,
        atol=1e-6,
    )


def test_w2_024_compiled_fp8_path_dispatches_scaled_mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ch19.fp8_compiled_matmul as fp8_matmul

    recorded = {}

    def _scaled_mm(a, b, **kwargs):
        recorded.update(a=a, b=b, **kwargs)
        return torch.full((a.shape[0], b.shape[1]), 7.0)

    monkeypatch.setattr(torch, "_scaled_mm", _scaled_mm)
    uncompiled = inspect.unwrap(fp8_matmul.fp8_matmul_compiled)
    a = torch.ones(2, 3)
    b = torch.ones(3, 4)
    scale_a = torch.tensor(0.5)
    scale_b = torch.tensor(0.25)

    result = uncompiled(a, b, scale_a, scale_b)
    source = inspect.getsource(uncompiled)

    assert recorded["a"] is a
    assert recorded["b"] is b
    assert recorded["scale_a"] is scale_a
    assert recorded["scale_b"] is scale_b
    assert recorded["out_dtype"] is torch.float16
    assert recorded["use_fast_accum"] is False
    assert torch.equal(result, torch.full((2, 4), 7.0))
    assert "torch._scaled_mm(" in source
    assert ".to(torch.float16)" not in source


def test_w2_024_native_fp8_entrypoint_fails_closed_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ch19.fp8_compiled_matmul as fp8_matmul

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert fp8_matmul.main() == 3
    assert "SKIPPED: Native FP8 benchmark unavailable" in capsys.readouterr().out


def test_w2_025_double_buffer_reuse_waits_for_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ch19.optimized_memory_double_buffering as double_buffer

    log = []

    class FakeStream:
        def __init__(self, name: str):
            self.name = name

        def wait_event(self, event) -> None:
            log.append(("wait_event", self.name, event.name))

        def wait_stream(self, stream) -> None:
            log.append(("wait_stream", self.name, stream.name))

    class FakeEvent:
        def __init__(self, name: str):
            self.name = name

        def record(self, stream) -> None:
            log.append(("record", self.name, stream.name))

    class FakeBuffer:
        def __init__(self, name: str):
            self.name = name

        def copy_(self, source, *, non_blocking: bool):
            log.append(("copy", self.name, source, non_blocking))
            return self

    class FakeModel:
        def forward_prepared(self, buffer):
            log.append(("compute", buffer.name))
            return buffer

    copy_stream = FakeStream("copy")
    compute_stream = FakeStream("compute")
    current_stream = FakeStream("current")
    buffers = [FakeBuffer("slot-0"), FakeBuffer("slot-1")]
    benchmark = object.__new__(double_buffer.OptimizedMemoryDoubleBufferingBenchmark)
    benchmark.copy_stream = copy_stream
    benchmark.compute_stream = compute_stream
    benchmark.copy_events = [FakeEvent("copy-0"), FakeEvent("copy-1")]
    benchmark.compute_events = [FakeEvent("compute-0"), FakeEvent("compute-1")]
    benchmark.buffers = buffers
    benchmark.buffer_a, benchmark.buffer_b = buffers
    benchmark.host_batches = ["batch-0", "batch-1", "batch-2", "batch-3"]
    benchmark.model = FakeModel()
    benchmark.output = None
    benchmark._buffer_event_counts = (2, 2)
    benchmark._expected_buffer_event_counts = (2, 2)
    benchmark._compute_event_count = 2
    benchmark._expected_compute_event_count = 2
    benchmark._micro_batch_schedule = [
        (0, 0, 1, 1),
        (1, 1, 2, 0),
        (2, 0, 3, 1),
        (3, 1, None, None),
    ]
    benchmark._micro_batch_schedule_count = 4
    benchmark._expected_micro_batch_schedule_count = 4
    benchmark._enable_nvtx = False
    benchmark.device = torch.device("cpu")

    monkeypatch.setattr(double_buffer, "nvtx_range", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(double_buffer.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        double_buffer.torch.cuda,
        "current_stream",
        lambda device=None: current_stream,
    )

    benchmark.benchmark_fn()

    wait_slot_0 = log.index(("wait_event", "copy", "compute-0"))
    overwrite_slot_0 = log.index(("copy", "slot-0", "batch-2", True))
    wait_slot_1 = log.index(("wait_event", "copy", "compute-1"))
    overwrite_slot_1 = log.index(("copy", "slot-1", "batch-3", True))
    assert wait_slot_0 < overwrite_slot_0
    assert wait_slot_1 < overwrite_slot_1
