from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import ch04.baseline_nvshmem_vs_nccl_benchmark_multigpu as baseline_module
import ch04.optimized_nvshmem_vs_nccl_benchmark_multigpu as optimized_module
from ch04.baseline_nvshmem_vs_nccl_benchmark_multigpu import (
    NVSHMEMVsNCCLBenchmarkMultiGPU,
)
from ch04.nvshmem_vs_nccl_benchmark import BenchmarkResult
from ch04.optimized_nvshmem_vs_nccl_benchmark_multigpu import (
    OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"


def test_repo_performance_skill_is_routed_and_source_bounded() -> None:
    skill = (REPO_ROOT / ".agents/skills/dean-performance-review/SKILL.md").read_text(
        encoding="utf-8"
    )
    agents = (CODE_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    readme_generator = (CODE_ROOT / "core/scripts/refresh_readmes.py").read_text(
        encoding="utf-8"
    )

    assert "https://abseil.io/fast/hints.html" in skill
    assert "single-binary performance" in skill
    assert "Static review identifies\n  hypotheses, not measured wins." in skill
    assert ".agents/skills/dean-performance-review/SKILL.md" in agents
    assert ".agents/skills/dean-performance-review/SKILL.md" in readme_generator


def test_performance_intake_requires_cost_model_and_evidence() -> None:
    intake = (CODE_ROOT / "templates/performance_intake.yaml").read_text(encoding="utf-8")
    methodology = (CODE_ROOT / "docs/benchmark_methodology.md").read_text(encoding="utf-8")

    assert "hot_path_model:" in intake
    assert "invocation_frequency:" in intake
    assert "best_case_primary_kpi_improvement_pct:" in intake
    assert "baseline_artifact:" in intake
    assert "## Dean/Ghemawat Optimization Loop" in methodology


def test_nvshmem_wrappers_report_measured_collective_results_without_hot_path_prints() -> None:
    cases = (
        (NVSHMEMVsNCCLBenchmarkMultiGPU(), "nccl", 11.0, 101.0),
        (OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU(), "nvshmem", 7.0, 151.0),
    )

    for benchmark, transport, latency_us, bandwidth_gbps in cases:
        benchmark._benchmark_results = {
            "nccl": [],
            "nvshmem": [],
            transport: [
                BenchmarkResult(
                    bytes=1_048_576,
                    latency_us=latency_us,
                    bandwidth_gbps=bandwidth_gbps,
                )
            ],
        }
        metrics = benchmark.get_custom_metrics()

        assert metrics is not None
        assert metrics["collective.message_bytes"] == 1_048_576.0
        assert metrics["collective.latency_us"] == latency_us
        assert metrics["collective.bandwidth_gbps"] == bandwidth_gbps

    optimized_source = (
        CODE_ROOT / "ch04/optimized_nvshmem_vs_nccl_benchmark_multigpu.py"
    ).read_text(encoding="utf-8")
    benchmark_section = optimized_source.split("def benchmark_fn", maxsplit=1)[1].split(
        "def teardown", maxsplit=1
    )[0]
    assert "print(" not in benchmark_section
    assert "self._benchmark_results = benchmark(self._benchmark_args)" in benchmark_section


def test_nvshmem_wrappers_capture_benchmark_results_and_handle_empty_results() -> None:
    cases = (
        (
            NVSHMEMVsNCCLBenchmarkMultiGPU(),
            baseline_module,
            "nccl",
        ),
        (
            OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU(),
            optimized_module,
            "nvshmem",
        ),
    )

    for benchmark, module, transport in cases:
        assert benchmark.get_custom_metrics() is None
        benchmark._benchmark_args = argparse.Namespace()
        result = BenchmarkResult(bytes=4096, latency_us=5.0, bandwidth_gbps=25.0)
        captured = {"nccl": [], "nvshmem": [], transport: [result]}

        with (
            mock.patch.object(module, "init_distributed", return_value=0),
            mock.patch.object(module, "benchmark", return_value=captured),
            mock.patch.object(module.dist, "is_initialized", return_value=False),
        ):
            benchmark.benchmark_fn()

        assert benchmark._benchmark_results is captured
        assert benchmark.get_custom_metrics() == {
            "collective.message_bytes": 4096.0,
            "collective.latency_us": 5.0,
            "collective.bandwidth_gbps": 25.0,
        }


def test_baseline_wrapper_keeps_single_environment_restoring_teardown() -> None:
    source = (
        CODE_ROOT / "ch04/baseline_nvshmem_vs_nccl_benchmark_multigpu.py"
    ).read_text(encoding="utf-8")

    assert source.count("    def teardown(self) -> None:") == 1
    assert "self._benchmark_results = None" in source
    assert "os.environ.pop(key, None)" in source
