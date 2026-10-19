"""Regression tests for audit work unit r02 residual checks (W2-014, W2-022)."""

from __future__ import annotations

import json
import time
import types
import weakref
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

_LOAD_TEST_DURATION_S = 1.0
_RANK1_CLOCK_RATE = 4.0


def test_inference_server_close_releases_prefill_graph_idempotently() -> None:
    from ch16.inference_serving_multigpu import InferenceServerMultiGPU

    class GraphHandle:
        pass

    server = InferenceServerMultiGPU.__new__(InferenceServerMultiGPU)
    server.device = torch.device("cpu")
    graph = GraphHandle()
    graph_ref = weakref.ref(graph)
    server._prefill_graph = graph
    server._prefill_graph_available = True
    server._prefill_graph_input = object()
    server._prefill_graph_logits = object()
    server._prefill_graph_keys = object()
    server._prefill_graph_values = object()
    server._prefill_logits_buffer = object()
    server._prefill_key_buffer = object()
    server._prefill_value_buffer = object()
    del graph

    server.close()
    server.close()

    assert graph_ref() is None
    assert not server._prefill_graph_available
    assert server._prefill_graph is None
    assert server._prefill_graph_input is None
    assert server._prefill_graph_logits is None
    assert server._prefill_graph_keys is None
    assert server._prefill_graph_values is None
    assert server._prefill_logits_buffer is None
    assert server._prefill_key_buffer is None
    assert server._prefill_value_buffer is None


def _load_test_worker(rank: int, rendezvous: str, output_dir: str) -> None:
    import numpy as np

    import ch16.inference_server_load_test as load_test
    from ch16.inference_serving_multigpu import DemoCausalLM, InferenceServerMultiGPU

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    report: dict[str, object] = {"rank": rank, "status": "FAIL"}
    try:
        assert not torch.cuda.is_available()
        assert dist.get_backend() == "gloo"
        if rank == 1:
            # Rank 1's clock runs fast, so its own deadline passes after a
            # quarter of rank 0's run. Only the load-test module sees this clock.
            real_perf_counter = time.perf_counter
            load_test.time = types.SimpleNamespace(
                perf_counter=lambda: _RANK1_CLOCK_RATE * real_perf_counter(),
                time=time.time,
                sleep=time.sleep,
            )

        broadcast_calls = 0
        real_broadcast_requests = load_test._broadcast_requests

        def counting_broadcast_requests(requests):
            nonlocal broadcast_calls
            broadcast_calls += 1
            return real_broadcast_requests(requests)

        load_test._broadcast_requests = counting_broadcast_requests

        torch.manual_seed(20261012 + rank)
        model = DemoCausalLM(
            vocab_size=64,
            d_model=32,
            num_layers=1,
            num_heads=2,
            num_gpus=2,
            max_batch_size=4,
            max_seq_len=32,
        )
        server = InferenceServerMultiGPU(
            model=model,
            num_layers=1,
            d_model=32,
            num_heads=2,
            max_batch_size=4,
            max_seq_len=32,
            enable_compile=False,
        )
        local = load_test.run_load_test(
            server,
            duration=_LOAD_TEST_DURATION_S,
            target_qps=40.0,
            interval=0.05,
            prompt_range=(2, 8),
            max_new_tokens=4,
            temperature=0.8,
            rng=np.random.default_rng(42 + rank),
        )
        aggregate = load_test.aggregate_results(local)
        server.close()
        server.close()
        report.update(
            status="PASS",
            backend=dist.get_backend(),
            device=str(server.device),
            broadcast_calls=broadcast_calls,
            local_elapsed=local["elapsed"],
            stats={
                name: local["stats"][name]
                for name in (
                    "total_requests",
                    "completed_requests",
                    "rejected_requests",
                    "total_tokens_generated",
                )
            },
            aggregate_total_requests=aggregate["total_requests"],
            aggregate_tokens_generated=aggregate["tokens_generated"],
        )
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Actual Gloo backend required",
)
def test_w2_022_load_test_ranks_stop_on_rank0_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This exercises deadline coordination with real CPU Gloo ranks. Spawned
    # workers must keep that device selection even when the parent has CUDA.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = torch.multiprocessing.spawn(
        _load_test_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Two-rank Gloo load test did not finish within 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert [report["status"] for report in reports] == ["PASS", "PASS"], reports
    assert [report["backend"] for report in reports] == ["gloo", "gloo"]
    assert [report["device"] for report in reports] == ["cpu", "cpu"]
    rank0, rank1 = reports
    # Rank 1's own clock passed the deadline long before the loop ended, so it
    # stayed in the loop only because it followed rank 0's stop decision.
    assert rank1["local_elapsed"] > 2.0 * _LOAD_TEST_DURATION_S
    assert rank0["broadcast_calls"] == rank1["broadcast_calls"]
    assert rank0["stats"] == rank1["stats"]
    assert rank0["stats"]["total_requests"] > 0
    assert rank0["stats"]["total_tokens_generated"] > 0
    assert rank0["aggregate_total_requests"] == rank0["stats"]["total_requests"]
    assert rank0["aggregate_tokens_generated"] == rank0["stats"]["total_tokens_generated"]


@pytest.mark.parametrize("profile_name", ["BLACKWELL_B200", "HOPPER_H100"])
def test_w2_014_compute_bound_docs_match_roofline_classification(
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
) -> None:
    import ch09.baseline_compute_bound as baseline_module
    import ch09.baseline_memory_bound as baseline_memory_module
    import ch09.optimized_memory_bound as optimized_memory_module
    from core.benchmark import metrics as benchmark_metrics

    specs = getattr(benchmark_metrics, profile_name)
    monkeypatch.setattr(benchmark_metrics, "detect_hardware_specs", lambda: specs)
    benchmark_type = baseline_module.BaselineComputeBoundBenchmark

    result = benchmark_type().get_custom_metrics()
    baseline_memory = baseline_memory_module.BaselineMemoryBoundBenchmark().get_custom_metrics()
    optimized_memory = optimized_memory_module.OptimizedMemoryBoundBenchmark().get_custom_metrics()

    assert result is not None
    assert baseline_memory is not None
    assert optimized_memory is not None
    intensity = result["roofline.arithmetic_intensity"]
    baseline_memory_intensity = baseline_memory["roofline.arithmetic_intensity"]
    optimized_memory_intensity = optimized_memory["roofline.arithmetic_intensity"]
    assert 0.99 < intensity < 1.0
    assert intensity < result["roofline.ridge_point"]
    assert result["roofline.is_compute_bound"] == 0.0

    docs = "\n".join(
        (
            baseline_module.__doc__ or "",
            benchmark_type.__doc__ or "",
            benchmark_type.get_custom_metrics.__doc__ or "",
        )
    )
    docs_flat = " ".join(docs.split())
    assert "roofline.is_compute_bound = 0" in docs_flat
    assert "below the ridge point" in docs_flat
    assert "WHY the kernel is compute-bound" not in docs_flat
    assert "Compute-heavy kernel" not in docs_flat
    assert "tensor core utilization" not in docs_flat

    # The two memory_bound arms model very different intensities, so the
    # docstring must name the arm it compares against and quote its real value.
    ratio = intensity / baseline_memory_intensity
    assert abs(ratio - round(ratio)) < 0.01
    assert (
        f"{round(ratio)}x baseline_memory_bound's {baseline_memory_intensity:g} FLOP/byte"
        in docs_flat
    )
    assert optimized_memory_intensity > intensity
    assert (
        f"optimized_memory_bound arm models {optimized_memory_intensity:g} FLOP/byte, "
        "which is higher than this chain"
    ) in docs_flat
    assert "memory_bound pair's" not in docs_flat
    assert "than memory_bound" not in docs_flat


def test_fp8_compiled_matmul_does_not_swallow_fp16_comparison_failure() -> None:
    import ast
    import inspect

    import ch19.fp8_compiled_matmul as fp8_matmul

    main_def = ast.parse(inspect.getsource(fp8_matmul.main)).body[0]
    fp16_calls = [
        node
        for node in ast.walk(main_def)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "benchmark_matmul"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "fp16_matmul_compiled"
    ]
    assert len(fp16_calls) == 1
    guarded_calls = [
        call
        for try_node in ast.walk(main_def)
        if isinstance(try_node, ast.Try)
        for statement in try_node.body
        for call in ast.walk(statement)
        if call is fp16_calls[0]
    ]
    # The FP8 speedup is reported against the FP16 result, so an FP16 failure
    # must propagate instead of being caught and turned into a missing row.
    assert guarded_calls == []
    assert "Failed to compile" not in inspect.getsource(fp8_matmul.main)


@pytest.mark.parametrize("profile_name", ["BLACKWELL_B200", "HOPPER_H100"])
def test_w2_014_optimized_compute_bound_docs_match_roofline_classification(
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
) -> None:
    import ch09.optimized_compute_bound as optimized_module
    from core.benchmark import metrics as benchmark_metrics

    specs = getattr(benchmark_metrics, profile_name)
    monkeypatch.setattr(benchmark_metrics, "detect_hardware_specs", lambda: specs)
    benchmark_type = optimized_module.OptimizedComputeBoundBenchmark

    result = benchmark_type().get_custom_metrics()
    assert result is not None
    intensity = result["roofline.arithmetic_intensity"]
    assert 0.99 < intensity < 1.0
    assert intensity < result["roofline.ridge_point"]
    assert result["roofline.is_compute_bound"] == 0.0

    module_summary = (optimized_module.__doc__ or "").splitlines()[0]
    assert module_summary == (
        "optimized_compute_bound.py - CUDA-graph replay of the compute_bound vector MLP chain."
    )
    assert benchmark_type.__doc__ == (
        "Vector MLP chain at about 1 FLOP/byte, below the ridge point, replayed from a "
        "CUDA graph to cut launch overhead."
    )
    assert benchmark_type.get_custom_metrics.__doc__ == (
        "Return roofline metrics using the centralized helper."
    )
    docs_flat = " ".join(
        " ".join(
            (
                optimized_module.__doc__ or "",
                benchmark_type.__doc__ or "",
                benchmark_type.get_custom_metrics.__doc__ or "",
            )
        ).split()
    ).lower()
    assert "compute-bound kernel" not in docs_flat
    assert "compute-bound analysis" not in docs_flat
    assert "memory_bound" not in docs_flat
    source = (Path(optimized_module.__file__)).read_text(encoding="utf-8")
    assert "compute-bound" not in source
