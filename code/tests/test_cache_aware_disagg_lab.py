"""Tests for the cache-aware disaggregated inference lab."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from ch17.prefill_decode_disagg_multigpu_common import TinyPrefillDecode
from core.harness.benchmark_harness import BaseBenchmark
from labs.cache_aware_disagg_inference.cache_aware_disagg_common import (
    CacheAwareDisaggBenchmark,
    CacheAwareDisaggConfig,
    _extend_cache_buffer as _extend_cache_buffer_single,
)
from labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_common import (
    CacheAwareDisaggMultiGPUBenchmark,
    CacheAwareDisaggMultiGPUConfig,
    _extend_cache_buffer as _extend_cache_buffer_multi,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = REPO_ROOT / "labs" / "cache_aware_disagg_inference"


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_aware_disagg_single_gpu_surface_is_locality_comparison_contract() -> None:
    bench = CacheAwareDisaggBenchmark(
        optimized=True,
        label="optimized_cache_aware_disagg_test",
    )
    payload = json.loads((LAB_DIR / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = payload["examples"]["cache_aware_disagg"]

    assert bench.get_optimization_goal() == "comparison"
    assert entry["metadata"]["optimization_goal"] == "comparison"
    assert "minimum_required_speedup" not in entry["metadata"]


@pytest.mark.parametrize(
    "relative_path",
    [
        "labs/cache_aware_disagg_inference/baseline_cache_aware_disagg.py",
        "labs/cache_aware_disagg_inference/optimized_cache_aware_disagg.py",
        "labs/cache_aware_disagg_inference/baseline_cache_aware_disagg_multigpu.py",
        "labs/cache_aware_disagg_inference/optimized_cache_aware_disagg_multigpu.py",
    ],
)
def test_cache_aware_disagg_wrappers_attach_metadata(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()

    assert isinstance(bench, BaseBenchmark)
    assert getattr(bench, "_module_file_override", None) == str(module_path)
    assert getattr(bench, "_factory_name_override", None) == "get_benchmark"
    assert Path(bench.script_path) == module_path


def test_cache_aware_disagg_single_gpu_extend_cache_buffer_reuses_storage() -> None:
    cfg = CacheAwareDisaggConfig(
        hidden_size=4,
        batch_size=1,
        context_window=6,
        dtype=torch.float32,
    )
    kv_buffers = {}
    empty = torch.empty((1, 0, 4), dtype=torch.float32)
    first_chunk = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    second_chunk = torch.arange(8, 16, dtype=torch.float32).view(1, 2, 4)

    first = _extend_cache_buffer_single(
        cfg,
        request_id=7,
        cache=empty,
        chunk_kv=first_chunk,
        kv_buffers=kv_buffers,
    )
    second = _extend_cache_buffer_single(
        cfg,
        request_id=7,
        cache=first,
        chunk_kv=second_chunk,
        kv_buffers=kv_buffers,
    )

    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(second[:, :2], first_chunk)
    torch.testing.assert_close(second[:, 2:4], second_chunk)


def test_tiny_prefill_decode_prefill_into_writes_kv_destination() -> None:
    torch.manual_seed(123)
    model = TinyPrefillDecode(
        hidden_size=4,
        num_layers=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).eval()
    prompt = torch.randn(2, 3, 4)
    expected_kv, expected_seed = model.prefill(prompt)
    destination = torch.full((2, 7, 4), -123.0)
    kv_view = destination[:, 2:5]
    seed_buffer = torch.empty(2, 4)

    actual_kv, actual_seed = model.prefill_into(prompt, kv_view, seed_buffer)

    assert actual_kv.data_ptr() == kv_view.data_ptr()
    assert actual_seed.data_ptr() == seed_buffer.data_ptr()
    torch.testing.assert_close(actual_kv, expected_kv)
    torch.testing.assert_close(actual_seed, expected_seed)
    torch.testing.assert_close(destination[:, :2], torch.full((2, 2, 4), -123.0))
    torch.testing.assert_close(destination[:, 5:], torch.full((2, 2, 4), -123.0))


def test_cache_aware_disagg_single_gpu_prefill_appends_directly_to_kv_cache() -> None:
    source = (LAB_DIR / "cache_aware_disagg_common.py").read_text(encoding="utf-8")
    setup_section = source.split("def setup", maxsplit=1)[1].split("def _empty_kv", maxsplit=1)[0]
    benchmark_section = source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def capture_verification_payload",
        maxsplit=1,
    )[0]

    assert "def _reserve_cache_append_buffer(" in source
    assert "self._prefill_seed_buffer = torch.empty(" in setup_section
    assert "self.prefill_model.prefill_into(" in setup_section
    assert "prefix_buffer[:, offset:next_offset]" in setup_section
    assert "prefix_buffer[:, offset:next_offset].copy_" not in setup_section
    assert "accumulated_kv, append_kv = _reserve_cache_append_buffer(" in benchmark_section
    assert "self.prefill_model.prefill_into(" in benchmark_section
    assert "append_kv" in benchmark_section
    assert "prefill_seed_buffer" in benchmark_section
    assert "chunk_kv, seed = self.prefill_model.prefill(chunk)" not in benchmark_section
    assert "_extend_cache_buffer(" not in benchmark_section


def test_cache_aware_disagg_multigpu_warm_prefix_writes_directly_to_kv_cache() -> None:
    source = (LAB_DIR / "cache_aware_disagg_multigpu_common.py").read_text(encoding="utf-8")
    distributed_warm_section = source.split("warm_cache_store: Dict[int, torch.Tensor]", maxsplit=1)[1].split(
        "recv_chunk_buffers: Dict[int, torch.Tensor]",
        maxsplit=1,
    )[0]
    setup_warm_section = source.split("with torch.inference_mode():", maxsplit=1)[1].split(
        "self.parameter_count = total_params",
        maxsplit=1,
    )[0]

    assert "model.prefill_into(" in distributed_warm_section
    assert "prefill_model.prefill_into(" in setup_warm_section
    assert "next_offset = offset + int(chunk.size(1))" in distributed_warm_section
    assert "next_offset = offset + int(chunk.size(1))" in setup_warm_section
    assert "prefix_cache[:, offset:next_offset].copy_(chunk_kv)" not in source
    assert "prefix_buffer[:, offset:next_offset].copy_(chunk_kv)" not in source


def test_cache_aware_disagg_single_gpu_reload_materialization_reuses_storage() -> None:
    cfg = CacheAwareDisaggConfig(
        hidden_size=4,
        batch_size=1,
        context_window=6,
        dtype=torch.float32,
    )
    bench = CacheAwareDisaggBenchmark(
        optimized=False,
        label="baseline_cache_aware_disagg_test",
        cfg=cfg,
    )
    bench.reload_materialization_passes = 2
    first_source = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    second_source = first_source + 100.0

    first = bench._materialize_reload_cache(7, first_source)
    first_storage = first.data_ptr()
    second = bench._materialize_reload_cache(7, second_source)

    assert second.data_ptr() == first_storage
    assert second.data_ptr() != second_source.data_ptr()
    torch.testing.assert_close(second, second_source)


def test_cache_aware_disagg_multigpu_extend_cache_buffer_reuses_storage() -> None:
    cfg = CacheAwareDisaggMultiGPUConfig(
        hidden_size=4,
        batch_size=1,
        context_window=6,
        dtype=torch.float32,
    )
    kv_buffers = {}
    empty = torch.empty((1, 0, 4), dtype=torch.float32)
    first_chunk = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    second_chunk = torch.arange(8, 16, dtype=torch.float32).view(1, 2, 4)

    first = _extend_cache_buffer_multi(
        cfg,
        request_id=7,
        cache=empty,
        chunk_kv=first_chunk,
        kv_buffers=kv_buffers,
    )
    second = _extend_cache_buffer_multi(
        cfg,
        request_id=7,
        cache=first,
        chunk_kv=second_chunk,
        kv_buffers=kv_buffers,
    )

    assert first.data_ptr() == second.data_ptr()
    torch.testing.assert_close(second[:, :2], first_chunk)
    torch.testing.assert_close(second[:, 2:4], second_chunk)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2+ GPUs required for multi-GPU wrapper spec")
@pytest.mark.parametrize(
    "relative_path,prefill_ranks",
    [
        ("labs/cache_aware_disagg_inference/baseline_cache_aware_disagg_multigpu.py", 1),
        ("labs/cache_aware_disagg_inference/optimized_cache_aware_disagg_multigpu.py", 1),
    ],
)
def test_cache_aware_disagg_multigpu_wrappers_expose_torchrun_specs(
    relative_path: str,
    prefill_ranks: int,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    module = _load_module(module_path)

    bench = module.get_benchmark()
    bench.cfg = CacheAwareDisaggMultiGPUConfig(prefill_ranks=prefill_ranks)
    spec = bench.get_torchrun_spec()

    assert spec.script_path == module_path
    assert spec.parse_rank0_only is True
    assert spec.multi_gpu_required is True
    assert METRICS_ENV_KEY in spec.env
    assert "--prefill-ranks" in spec.script_args
    assert str(prefill_ranks) in spec.script_args


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cache-aware lab metrics")
def test_cache_aware_disagg_optimized_path_improves_locality_without_changing_output() -> None:
    cfg = CacheAwareDisaggConfig(
        hidden_size=128,
        num_layers=2,
        batch_size=1,
        requests_per_iteration=6,
        context_window=384,
        chunk_size=96,
        decode_tokens=24,
        logical_decode_workers=3,
        warm_request_ratio=0.5,
        warm_prefix_ratio=0.5,
    )

    baseline = CacheAwareDisaggBenchmark(
        optimized=False,
        label="baseline_cache_aware_disagg_test",
        cfg=cfg,
    )
    optimized = CacheAwareDisaggBenchmark(
        optimized=True,
        label="optimized_cache_aware_disagg_test",
        cfg=cfg,
    )

    baseline.setup()
    try:
        baseline.benchmark_fn()
        baseline.capture_verification_payload()
        baseline_metrics = baseline.get_custom_metrics()
        baseline_output = baseline.output.detach().cpu() if baseline.output is not None else None
    finally:
        baseline.teardown()

    optimized.setup()
    try:
        optimized.benchmark_fn()
        optimized.capture_verification_payload()
        optimized_metrics = optimized.get_custom_metrics()
        optimized_output = optimized.output.detach().cpu() if optimized.output is not None else None
    finally:
        optimized.teardown()

    assert baseline_metrics is not None
    assert optimized_metrics is not None
    assert baseline_output is not None
    assert optimized_output is not None
    assert torch.allclose(baseline_output, optimized_output, atol=0.0, rtol=0.0)
    assert optimized_metrics["cache_aware.cache_hit_rate"] > baseline_metrics["cache_aware.cache_hit_rate"]
    assert optimized_metrics["cache_aware.kv_transfer_mb"] < baseline_metrics["cache_aware.kv_transfer_mb"]
    assert optimized_metrics["cache_aware.worker_switches_per_request"] < baseline_metrics["cache_aware.worker_switches_per_request"]


METRICS_ENV_KEY = "AISP_CACHE_AWARE_DISAGG_METRICS_PATH"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2+ GPUs required for cache-aware multi-GPU lab")
def test_cache_aware_disagg_multigpu_optimized_path_improves_transfer_locality() -> None:
    prefill_ranks = 1 if torch.cuda.device_count() == 2 else 2
    cfg = CacheAwareDisaggMultiGPUConfig(
        hidden_size=64,
        num_layers=2,
        batch_size=1,
        requests_per_rank=2,
        context_window=192,
        chunk_size=64,
        decode_tokens=8,
        warm_request_ratio=0.5,
        warm_prefix_ratio=0.5,
        prefill_ranks=prefill_ranks,
    )

    baseline = CacheAwareDisaggMultiGPUBenchmark(
        optimized=False,
        label="baseline_cache_aware_disagg_multigpu_test",
        cfg=cfg,
    )
    optimized = CacheAwareDisaggMultiGPUBenchmark(
        optimized=True,
        label="optimized_cache_aware_disagg_multigpu_test",
        cfg=cfg,
    )

    baseline.setup()
    try:
        baseline.benchmark_fn()
        baseline.capture_verification_payload()
        baseline_metrics = baseline.get_custom_metrics()
        baseline_output = baseline.output.detach().cpu() if baseline.output is not None else None
    finally:
        baseline.teardown()

    optimized.setup()
    try:
        optimized.benchmark_fn()
        optimized.capture_verification_payload()
        optimized_metrics = optimized.get_custom_metrics()
        optimized_output = optimized.output.detach().cpu() if optimized.output is not None else None
    finally:
        optimized.teardown()

    assert baseline_metrics is not None
    assert optimized_metrics is not None
    assert baseline_output is not None
    assert optimized_output is not None
    assert torch.allclose(baseline_output, optimized_output, atol=0.0, rtol=0.0)
    assert optimized_metrics["cache_aware.kv_transfer_mb"] < baseline_metrics["cache_aware.kv_transfer_mb"]
    assert optimized_metrics["cache_aware.worker_switches_per_request"] < baseline_metrics["cache_aware.worker_switches_per_request"]
