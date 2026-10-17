from __future__ import annotations

from pathlib import Path

from ch07.baseline_tma_copy import BaselineTMACopyBenchmark
from ch07.optimized_tma_copy import OptimizedTMACopyBenchmark


def test_tma_copy_wrappers_surface_the_strict_tma_story() -> None:
    baseline = BaselineTMACopyBenchmark()
    optimized = OptimizedTMACopyBenchmark()

    assert baseline.friendly_name == "Scalar Neighbor Gather Copy"
    assert optimized.friendly_name == "Descriptor-Backed 2D TMA Tiled-Neighbor Transform"

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()
    assert baseline_metrics["copy.tensor_map_2d_requested"] == 0.0
    assert optimized_metrics["copy.tensor_map_2d_requested"] == 1.0
    assert optimized_metrics["copy.tensor_map_2d_required"] == 1.0


def test_ch07_readme_describes_tma_copy_as_strict_tma_only() -> None:
    readme = (Path(__file__).resolve().parents[1] / "ch07" / "README.md").read_text(encoding="utf-8")

    assert "`tma_copy` now means a strict tensor-map/TMA-capable run only" in readme
    assert "legacy async-neighbor demo" not in readme
    assert "descriptor-backed TMA" in readme


def test_tma_copy_cuda_binary_skips_before_timing_without_cuda13_descriptor_support() -> None:
    source = (Path(__file__).resolve().parents[1] / "ch07" / "optimized_tma_copy.cu").read_text(encoding="utf-8")
    benchmark_section = source.split("bool benchmark_tma_2d", 1)[1].split("int main()", 1)[0]

    assert "#if !TMA_CUDA13_AVAILABLE" in benchmark_section
    assert "SKIPPED: optimized_tma_copy requires usable tensor-map/TMA support" in benchmark_section
    assert benchmark_section.index("#if !TMA_CUDA13_AVAILABLE") < benchmark_section.index("CUDA_CHECK(cudaMalloc(&d_mat_src, matrix_bytes));")
