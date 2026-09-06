"""Integration test for bench_commands manifest persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.env import apply_env_defaults
apply_env_defaults()

from core.benchmark.bench_commands import _execute_benchmarks


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available"
)


def _write_benchmark(
    path: Path,
    *,
    emit_runtime_limitation: bool = False,
    use_subprocess: bool = True,
) -> None:
    setup_limitation = ""
    if emit_runtime_limitation:
        setup_limitation = """
        from core.harness import validity_checks
        validity_checks._emit_validity_limitation_once(
            "cuda_allocator_reset_setting",
            "This PyTorch runtime does not support the reset_allocator allocator setting; memory-pool cleanup will proceed without forcing allocator-setting resets and allocator reuse checks are partially degraded",
            exc=RuntimeError("Unrecognized CachingAllocator option: reset_allocator"),
            component="cuda_memory_pool_reset",
            category="allocator_cleanup",
        )
        validity_checks._emit_validity_limitation_once(
            "cuda_release_pool_signature",
            "This PyTorch runtime exposes torch._C._cuda_releasePool without a zero-argument reset entrypoint; memory-pool reset will fall back to cache/IPC/stat cleanup and allocator reuse checks are partially degraded",
            exc=TypeError("_cuda_releasePool(): incompatible function arguments"),
            component="cuda_memory_pool_reset",
            category="allocator_cleanup",
        )
"""
    code = """\
import torch
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class SimpleMatmulBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = False

    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.weight = None
        self.output = None

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
__SETUP_LIMITATION__
        # Keep the synthetic workload large enough that timing cross-validation
        # remains stable under the real harness.
        self.input = torch.randn(1024, 1024, device=self.device, dtype=torch.float16)
        self.weight = torch.randn(1024, 1024, device=self.device, dtype=torch.float16)

    def benchmark_fn(self) -> None:
        if self.input is None or self.weight is None:
            raise RuntimeError("Benchmark not initialized")
        self.output = self.input @ self.weight

    def capture_verification_payload(self) -> None:
        if self.input is None or self.weight is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self.input, "weight": self.weight},
            output=self.output,
            batch_size=self.input.shape[0],
            parameter_count=0,
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(1e-3, 1e-3),
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            full_device_sync=True,
            timing_method="wall_clock",
            allow_foreign_gpu_processes=True,
            use_subprocess=__USE_SUBPROCESS__,
        )


def get_benchmark() -> BaseBenchmark:
    return SimpleMatmulBenchmark()
"""
    code = code.replace("__SETUP_LIMITATION__", setup_limitation.rstrip())
    code = code.replace("__USE_SUBPROCESS__", "True" if use_subprocess else "False")
    path.write_text(code, encoding="utf-8")


def test_bench_commands_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench_root = tmp_path / "bench_root"
    chapter_dir = bench_root / "ch01"
    chapter_dir.mkdir(parents=True)

    _write_benchmark(chapter_dir / "baseline_manifest_demo.py")
    _write_benchmark(chapter_dir / "optimized_manifest_demo.py")

    artifacts_dir = tmp_path / "artifacts"
    _execute_benchmarks(
        targets=["ch01:manifest_demo"],
        bench_root=bench_root,
        output_format="json",
        profile_type="none",
        validity_profile="portable",
        allow_foreign_gpu_processes=True,
        iterations=5,
        warmup=5,
        single_gpu=True,
        artifacts_dir=str(artifacts_dir),
        # These two fixtures execute identical work. A truthful no-speedup
        # verdict must not prevent checking the manifest-persistence contract.
        exit_on_failure=False,
    )

    manifest_files = list(artifacts_dir.rglob("manifest.json"))
    assert manifest_files, "manifest.json not written to artifacts directory"

    data = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    manifests = data.get("manifests", [])
    assert manifests, "manifest.json missing per-run entries"

    variants = {entry.get("variant") for entry in manifests}
    assert "baseline" in variants
    assert "optimized" in variants

    hardware = manifests[0]["manifest"].get("hardware", {})
    assert hardware.get("gpu_app_clock_mhz") is not None
    assert hardware.get("memory_app_clock_mhz") is not None


def test_bench_commands_persists_runtime_capability_limitations_in_results_json(
    tmp_path: Path,
) -> None:
    bench_root = tmp_path / "bench_root"
    chapter_dir = bench_root / "ch01"
    chapter_dir.mkdir(parents=True)

    _write_benchmark(
        chapter_dir / "baseline_manifest_runtime_limitations.py",
        emit_runtime_limitation=True,
        use_subprocess=False,
    )
    _write_benchmark(
        chapter_dir / "optimized_manifest_runtime_limitations.py",
        emit_runtime_limitation=True,
        use_subprocess=False,
    )

    artifacts_dir = tmp_path / "artifacts"
    _execute_benchmarks(
        targets=["ch01:manifest_runtime_limitations"],
        bench_root=bench_root,
        output_format="json",
        profile_type="none",
        validity_profile="portable",
        allow_foreign_gpu_processes=True,
        iterations=5,
        warmup=5,
        single_gpu=True,
        artifacts_dir=str(artifacts_dir),
        exit_on_failure=False,
    )

    results_files = list(artifacts_dir.rglob("benchmark_test_results.json"))
    assert results_files, "benchmark_test_results.json not written to artifacts directory"

    payload = json.loads(results_files[0].read_text(encoding="utf-8"))
    results = payload.get("results", [])
    assert results, "benchmark_test_results.json missing results entries"

    manifests = results[0].get("manifests", [])
    assert manifests, "benchmark_test_results.json missing embedded manifests"

    baseline_manifest = next(
        entry["manifest"]
        for entry in manifests
        if entry.get("variant") == "baseline"
    )
    runtime_limitations = baseline_manifest.get("runtime_capability_limitations", [])
    assert runtime_limitations, "runtime capability limitations missing from persisted manifest"

    limitation_map = {entry["key"]: entry for entry in runtime_limitations}
    assert "cuda_allocator_reset_setting" in limitation_map
    assert "cuda_release_pool_signature" in limitation_map

    reset_allocator = limitation_map["cuda_allocator_reset_setting"]
    assert reset_allocator["category"] == "allocator_cleanup"
    assert reset_allocator["component"] == "cuda_memory_pool_reset"
    assert (
        reset_allocator["summary"]
        == "This PyTorch runtime does not support the reset_allocator allocator setting; memory-pool cleanup will proceed without forcing allocator-setting resets and allocator reuse checks are partially degraded"
    )
    assert reset_allocator["detail"] == "Unrecognized CachingAllocator option: reset_allocator"

    release_pool = limitation_map["cuda_release_pool_signature"]
    assert release_pool["category"] == "allocator_cleanup"
    assert release_pool["component"] == "cuda_memory_pool_reset"
    assert (
        release_pool["summary"]
        == "This PyTorch runtime exposes torch._C._cuda_releasePool without a zero-argument reset entrypoint; memory-pool reset will fall back to cache/IPC/stat cleanup and allocator reuse checks are partially degraded"
    )
    assert release_pool["detail"].startswith("_cuda_releasePool(): incompatible function arguments")
