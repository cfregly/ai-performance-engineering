"""Tests for the training-hotpath supporting lab."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch

from core.discovery import discover_benchmarks
from core.harness.benchmark_harness import BaseBenchmark
from labs.training_hotpath.compare import _measure
from labs.training_hotpath.training_hotpath_common import (
    MetricReductionCudaBenchmark,
    MetricReductionVectorizedBenchmark,
    PaddingAwareTransformerBenchmark,
    PaddingAwareWorkload,
    active_mask_and_rows,
    baseline_segment_abs_mean,
    build_padding_inputs,
    build_segment_metadata,
    scalar_metric_reduction,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = REPO_ROOT / "labs" / "training_hotpath"


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_benchmark_once(bench: BaseBenchmark) -> tuple[torch.Tensor, dict]:
    bench.setup()
    try:
        bench.benchmark_fn()
        bench.capture_verification_payload()
        result = getattr(bench, "output", None)
        metrics = bench.get_custom_metrics()
        assert isinstance(result, torch.Tensor)
        assert isinstance(metrics, dict)
        return result.detach().cpu(), metrics
    finally:
        bench.teardown()


def test_training_hotpath_discovery_finds_all_three_pairs() -> None:
    pairs = discover_benchmarks(LAB_DIR)
    discovered = {example_name for _, _, example_name in pairs}
    assert discovered == {
        "metric_reduction_vectorized",
        "metric_reduction_cuda",
        "padding_aware_transformer",
    }


def test_training_hotpath_compare_measure_cuda_path_uses_single_event_bracket() -> None:
    source = inspect.getsource(_measure)
    cuda_section = source.split("if torch.cuda.is_available():", maxsplit=1)[1].split(
        "timings = []",
        maxsplit=1,
    )[0]

    assert cuda_section.count("torch.cuda.synchronize()") == 1
    assert cuda_section.count("start.record()") == 1
    assert cuda_section.count("end.record()") == 1
    assert cuda_section.count("end.synchronize()") == 1
    assert "timings.append(start.elapsed_time(end))" not in cuda_section


@pytest.mark.parametrize(
    "relative_path",
    [
        "labs/training_hotpath/baseline_metric_reduction_vectorized.py",
        "labs/training_hotpath/optimized_metric_reduction_vectorized.py",
        "labs/training_hotpath/baseline_metric_reduction_cuda.py",
        "labs/training_hotpath/optimized_metric_reduction_cuda.py",
        "labs/training_hotpath/baseline_padding_aware_transformer.py",
        "labs/training_hotpath/optimized_padding_aware_transformer.py",
    ],
)
def test_training_hotpath_wrappers_expose_get_benchmark(relative_path: str) -> None:
    module_path = REPO_ROOT / relative_path
    module = _load_module(module_path)
    bench = module.get_benchmark()
    assert isinstance(bench, BaseBenchmark)
    assert getattr(bench, "_module_file_override", None) == str(module_path)
    assert getattr(bench, "_factory_name_override", None) == "get_benchmark"


def test_padding_aware_transformer_is_memory_goal_with_memory_tracking_enabled() -> None:
    baseline = PaddingAwareTransformerBenchmark(
        optimized=False,
        label="baseline_padding_aware_transformer_test",
    )
    optimized = PaddingAwareTransformerBenchmark(
        optimized=True,
        label="optimized_padding_aware_transformer_test",
    )

    assert baseline.get_optimization_goal() == "memory"
    assert optimized.get_optimization_goal() == "memory"
    assert baseline.get_config().enable_memory_tracking is True
    assert optimized.get_config().enable_memory_tracking is True


def test_padding_aware_transformer_expectation_entry_is_memory_goal() -> None:
    payload = json.loads((LAB_DIR / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = payload["examples"]["padding_aware_transformer"]

    assert entry["metadata"]["optimization_goal"] == "memory"
    assert entry["metrics"]["is_regression"] is False


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only guard only matters without CUDA")
@pytest.mark.parametrize(
    "factory",
    [
        lambda: MetricReductionVectorizedBenchmark(optimized=False, label="baseline_metric_reduction_vectorized_test"),
        lambda: MetricReductionCudaBenchmark(optimized=True, label="optimized_metric_reduction_cuda_test"),
        lambda: PaddingAwareTransformerBenchmark(optimized=True, label="optimized_padding_aware_transformer_test"),
    ],
)
def test_training_hotpath_setup_requires_cuda(factory) -> None:
    bench = factory()
    with pytest.raises(RuntimeError, match="require CUDA"):
        bench.setup()


def test_baseline_segment_abs_mean_reuses_abs_buffer_on_cpu() -> None:
    flat = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=torch.float32)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    segment_ids, segment_lengths = build_segment_metadata(offsets)
    out = torch.empty(2, dtype=torch.float32)
    abs_buf = torch.empty_like(flat)

    result = baseline_segment_abs_mean(flat, segment_ids, segment_lengths, out, abs_buf)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(abs_buf, flat.abs())
    torch.testing.assert_close(result, torch.tensor([1.5, 4.0], dtype=torch.float32))

    source = (LAB_DIR / "training_hotpath_common.py").read_text(encoding="utf-8")
    helper_source = source.split("def baseline_segment_abs_mean", maxsplit=1)[1].split(
        "def active_mask_and_rows", maxsplit=1
    )[0]
    assert "out.zero_()" not in helper_source
    assert 'out.scatter_reduce_(0, segment_ids, values, reduce="sum", include_self=False)' in helper_source


def test_scalar_metric_reduction_reuses_output_buffer_on_cpu() -> None:
    preds = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    targets = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)
    out = torch.empty(6, dtype=torch.float32)

    result = scalar_metric_reduction(preds, targets, out)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        result,
        torch.tensor([10.0, 20.0, 74.0, 100.0, 26.0, 44.0], dtype=torch.float32),
    )

    source = (LAB_DIR / "training_hotpath_common.py").read_text(encoding="utf-8")
    helper_source = source.split("def scalar_metric_reduction", maxsplit=1)[1].split(
        "def vectorized_metric_reduction",
        maxsplit=1,
    )[0]
    assert "result = out if out is not None else preds.new_empty(responders * 3)" in helper_source
    assert "torch.stack(" not in helper_source
    assert "torch.cat(" not in helper_source


def test_padding_inputs_return_mask_and_host_active_token_count() -> None:
    common_source = (LAB_DIR / "training_hotpath_common.py").read_text(encoding="utf-8")

    assert "offsets = torch.empty(workload.num_segments + 1, dtype=torch.int64)" in common_source
    assert "offsets[0] = 0" in common_source
    assert "total = int(offsets[-1].item())" in common_source
    assert "active_tokens = int(seq_lens_cpu.sum().item())" in common_source
    assert "active_tokens = sum(int(seq_len) for seq_len in seq_lens_cpu.tolist())" not in common_source
    assert "total = sum(int(length) for length in lengths.tolist())" not in common_source
    assert "active_tokens = int(self.seq_lens.sum().item())" not in common_source
    assert "self.flat, self.offsets, total = build_gradient_inputs" in common_source
    assert "self.offsets[-1].item()" not in common_source
    assert "self._active_mask" in common_source

    workload = PaddingAwareWorkload(
        batch_size=3,
        max_num_tokens=8,
        min_num_tokens=2,
        input_size=4,
    )
    inputs, seq_lens, active_mask, active_rows, active_tokens = build_padding_inputs(
        workload,
        torch.device("cpu"),
    )
    expected_mask, expected_rows = active_mask_and_rows(seq_lens, workload.max_num_tokens)

    assert inputs.shape == (workload.batch_size, workload.max_num_tokens, workload.input_size)
    assert active_tokens == int(seq_lens.sum().item())
    assert active_rows.numel() == active_tokens
    torch.testing.assert_close(active_mask, expected_mask)
    torch.testing.assert_close(active_rows, expected_rows)


def test_metric_reduction_fused_optimized_path_reuses_output_buffer_source() -> None:
    common_source = (LAB_DIR / "training_hotpath_common.py").read_text(encoding="utf-8")
    kernel_source = (LAB_DIR / "training_hotpath_kernels.cu").read_text(encoding="utf-8")

    assert "metric_reduction_fused_out" in kernel_source
    assert "torch::Tensor reusable_out" in kernel_source
    assert "auto out = reuse_output ? reusable_out : torch::empty" in kernel_source
    assert "self.output = torch.empty(self.workload.responders * 3" in common_source
    assert "metric_reduction_fused_out(self.preds, self.targets, self.output)" in common_source
    assert "scalar_metric_reduction(self.preds, self.targets, self.output)" in common_source
    assert "scalar_metric_reduction(self.preds, self.targets)" not in common_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused metric output reuse check")
def test_metric_reduction_vectorized_optimized_reuses_output_buffer() -> None:
    bench = MetricReductionVectorizedBenchmark(
        optimized=True,
        label="optimized_metric_reduction_vectorized_reuse_test",
    )
    bench.apply_target_overrides(["--batch-size", "2", "--max-num-tokens", "16", "--responders", "16"])
    bench.setup()
    try:
        assert bench.output is not None
        data_ptr = bench.output.data_ptr()

        bench.benchmark_fn()
        assert bench.output is not None
        assert bench.output.data_ptr() == data_ptr

        bench.benchmark_fn()
        assert bench.output is not None
        assert bench.output.data_ptr() == data_ptr
    finally:
        bench.teardown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for training-hotpath lab parity checks")
def test_metric_reduction_vectorized_pair_matches_output_and_metrics() -> None:
    baseline = MetricReductionVectorizedBenchmark(
        optimized=False,
        label="baseline_metric_reduction_vectorized_test",
    )
    optimized = MetricReductionVectorizedBenchmark(
        optimized=True,
        label="optimized_metric_reduction_vectorized_test",
    )
    overrides = ["--batch-size", "4", "--max-num-tokens", "64", "--responders", "32"]
    baseline.apply_target_overrides(overrides)
    optimized.apply_target_overrides(overrides)

    baseline_output, baseline_metrics = _run_benchmark_once(baseline)
    optimized_output, optimized_metrics = _run_benchmark_once(optimized)

    assert torch.allclose(baseline_output, optimized_output, atol=1e-4, rtol=1e-4)
    assert baseline_metrics["metric_reduction.is_vectorized"] == 0.0
    assert optimized_metrics["metric_reduction.is_vectorized"] == 1.0
    assert baseline_metrics["metric_reduction.uses_cuda_extension"] == 0.0
    assert optimized_metrics["metric_reduction.uses_cuda_extension"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for training-hotpath lab parity checks")
def test_metric_reduction_cuda_pair_matches_output_and_metrics() -> None:
    baseline = MetricReductionCudaBenchmark(
        optimized=False,
        label="baseline_metric_reduction_cuda_test",
    )
    optimized = MetricReductionCudaBenchmark(
        optimized=True,
        label="optimized_metric_reduction_cuda_test",
    )
    overrides = ["--num-segments", "8", "--min-segment-length", "256", "--max-segment-length", "512"]
    baseline.apply_target_overrides(overrides)
    optimized.apply_target_overrides(overrides)

    baseline_output, baseline_metrics = _run_benchmark_once(baseline)
    optimized_output, optimized_metrics = _run_benchmark_once(optimized)

    assert torch.allclose(baseline_output, optimized_output, atol=1e-5, rtol=1e-5)
    assert baseline_metrics["metric_reduction.is_fused_cuda"] == 0.0
    assert optimized_metrics["metric_reduction.is_fused_cuda"] == 1.0
    assert baseline_metrics["metric_reduction.uses_cuda_extension"] == 0.0
    assert optimized_metrics["metric_reduction.uses_cuda_extension"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for training-hotpath lab parity checks")
def test_padding_aware_transformer_pair_matches_output_and_metrics() -> None:
    baseline = PaddingAwareTransformerBenchmark(
        optimized=False,
        label="baseline_padding_aware_transformer_test",
    )
    optimized = PaddingAwareTransformerBenchmark(
        optimized=True,
        label="optimized_padding_aware_transformer_test",
    )
    overrides = [
        "--batch-size", "2",
        "--max-num-tokens", "64",
        "--min-num-tokens", "8",
        "--input-size", "32",
        "--hidden-size", "64",
        "--projection-size", "128",
        "--num-heads", "4",
        "--num-blocks", "2",
        "--output-size", "32",
    ]
    baseline.apply_target_overrides(overrides)
    optimized.apply_target_overrides(overrides)

    baseline_output, baseline_metrics = _run_benchmark_once(baseline)
    optimized_output, optimized_metrics = _run_benchmark_once(optimized)

    assert torch.allclose(baseline_output, optimized_output, atol=1e-5, rtol=1e-5)
    assert baseline_metrics["padding_aware.enabled"] == 0.0
    assert optimized_metrics["padding_aware.enabled"] == 1.0
    assert baseline_metrics["padding_aware.uses_cuda_extension"] == 0.0
    assert optimized_metrics["padding_aware.uses_cuda_extension"] == 1.0
