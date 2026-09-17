"""Static, CPU, and opt-in GB300 coverage for the native fast.cu NVFP4 pair."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.baseline_nvfp4_gemm import FastCuNvfp4CublasLtBenchmark
from labs.fast_cu.nvfp4 import (
    COMPUTE_CAPABILITY,
    CUDA_VERSION,
    DEFAULT_K,
    DEFAULT_M,
    DEFAULT_N,
    DEFAULT_RUNG,
    Nvfp4Workload,
    require_nvfp4_runtime,
    validate_rung,
    workload_bytes,
    workload_flops,
)
from labs.fast_cu.optimized_nvfp4_gemm import FastCuNvfp4KernelBenchmark, get_benchmark
from tests.protection_test_utils import preserve_rng_state

LAB_DIR = Path(__file__).resolve().parents[1] / "labs" / "fast_cu"
NATIVE_SOURCE = LAB_DIR / "nvfp4_native.cu"
PYTHON_SOURCE = LAB_DIR / "nvfp4.py"


def test_default_workload_and_rung_are_explicit_and_frozen() -> None:
    workload = Nvfp4Workload()

    assert workload == Nvfp4Workload(DEFAULT_M, DEFAULT_N, DEFAULT_K)
    assert (DEFAULT_M, DEFAULT_N, DEFAULT_K) == (8192, 8192, 8192)
    assert DEFAULT_RUNG == 9
    assert CUDA_VERSION == (13, 1)
    assert COMPUTE_CAPABILITY == (10, 3)
    workload.validate()
    assert workload_bytes(workload) == 209_715_200
    assert workload_flops(workload) == 1_099_511_627_776
    with pytest.raises(FrozenInstanceError):
        workload.m = 4096  # type: ignore[misc]


@pytest.mark.parametrize(
    ("workload", "message"),
    [
        (Nvfp4Workload(0, 128, 128), "m must be"),
        (Nvfp4Workload(128, -1, 128), "n must be"),
        (Nvfp4Workload(128, 128, True), "k must be"),
        (Nvfp4Workload(128, 128, 48), "divisible by 32"),
    ],
)
def test_workload_rejects_invalid_or_cublaslt_unsupported_shapes(
    workload: Nvfp4Workload,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        workload.validate()


@pytest.mark.parametrize("rung", range(10))
def test_every_upstream_optimization_rung_is_selectable(rung: int) -> None:
    assert validate_rung(rung) == rung
    benchmark = FastCuNvfp4KernelBenchmark(Nvfp4Workload(256, 256, 256), rung=rung)
    assert benchmark.rung == rung


@pytest.mark.parametrize("rung", [-1, 10, True, 1.5, "9"])
def test_invalid_rung_is_rejected(rung: object) -> None:
    with pytest.raises(ValueError, match=r"integer in \[0, 9\]"):
        validate_rung(rung)  # type: ignore[arg-type]


def test_benchmark_factories_are_import_safe_and_harness_native() -> None:
    baseline = FastCuNvfp4CublasLtBenchmark(Nvfp4Workload(256, 256, 256))
    optimized = get_benchmark()

    assert isinstance(baseline, BaseBenchmark)
    assert isinstance(optimized, FastCuNvfp4KernelBenchmark)
    assert baseline.optimized is False
    assert optimized.optimized is True
    assert optimized.rung == 9
    assert baseline.output is None and optimized.output is None
    assert baseline._output_buffer is None and optimized._output_buffer is None
    metadata = optimized.get_workload_metadata()
    assert metadata is not None
    assert metadata.bytes_per_iteration == float(workload_bytes(optimized.workload))
    assert metadata.custom_units_per_iteration == float(workload_flops(optimized.workload))
    assert metadata.custom_unit_name == "nvfp4_flops"
    config = optimized.get_config()
    assert config.iterations == 25
    assert config.warmup == 5
    assert config.single_gpu is True


def test_verification_capture_rejects_pre_execution_state() -> None:
    benchmark = FastCuNvfp4CublasLtBenchmark(Nvfp4Workload(256, 256, 256))

    with pytest.raises(RuntimeError, match=r"benchmark_fn\(\) must run"):
        benchmark.capture_verification_payload()


def test_teardown_releases_the_full_verification_payload() -> None:
    benchmark = FastCuNvfp4CublasLtBenchmark(Nvfp4Workload(256, 256, 256))
    benchmark._native_context = object()
    benchmark._inputs = {"packed": torch.ones(4, dtype=torch.uint8)}
    benchmark.output = torch.ones((2, 2), dtype=torch.float16)
    benchmark._setup_gate_passed = True
    benchmark.capture_verification_payload()
    assert benchmark._verification_payload is not None

    benchmark.teardown()

    assert benchmark._verification_payload is None
    assert benchmark._inputs == {}
    assert benchmark.output is None


def test_native_source_preserves_the_equivalent_workload_contract() -> None:
    source = NATIVE_SOURCE.read_text(encoding="utf-8")
    python_source = PYTHON_SOURCE.read_text(encoding="utf-8")

    assert '#include "main.cu"' in source
    assert "host::make_fixture(M, N, K, seed" in source
    assert "fixture.inputs.A == fixture.A_logical" in source
    assert "CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3" in source
    assert "CUDA_R_4F_E2M1" in (LAB_DIR / "upstream" / "gb300" / "nvfp4" / "main.cu").read_text(
        encoding="utf-8"
    )
    assert "output=self.output" in python_source
    assert "output.sum" not in python_source
    assert "checksum" not in python_source.lower()


def test_native_source_uses_current_stream_and_rejects_r9_degradation() -> None:
    source = NATIVE_SOURCE.read_text(encoding="utf-8")

    assert "at::cuda::getCurrentCUDAStream(device)" in source
    assert "cublasLtMatmul(" in source and "stream)," in source
    assert "<<<grid_, nvfp4::TB_SIZE, sizeof(nvfp4::SmemCD), stream>>>" in source
    assert "strict_setup_r9_schedule" in source
    assert "sched::build_schedule" in source
    assert "table.empty()" in source
    assert "raster fallback is forbidden" in source
    assert "l2a_placement_errors" in source
    assert "bench::small_correctness_gates()" in source
    assert "small host-reference, guard, or determinism gate failed" in source
    assert "bench::setup_schedule" not in source


def test_build_contract_is_exact_sm103a_cuda131_and_setup_only() -> None:
    source = PYTHON_SOURCE.read_text(encoding="utf-8")

    assert "verify_upstream()" in source
    assert "require_nvfp4_runtime()" in source
    assert "require_nvfp4_toolkit()" in source
    assert '"-gencode=arch=compute_103a,code=sm_103a"' in source
    assert "minimum_cuda=CUDA_VERSION" in source
    benchmark_body = source.split("def benchmark_fn(self)", 1)[1].split(
        "def capture_verification_payload", 1
    )[0]
    assert "load_nvfp4_extension" not in benchmark_body
    assert "subprocess" not in benchmark_body


def test_runtime_gate_reports_the_real_host_without_mocking_success() -> None:
    supported = (
        torch.cuda.is_available()
        and torch.version.cuda is not None
        and tuple(map(int, torch.version.cuda.split(".")[:2])) >= (13, 1)
        and torch.cuda.get_device_capability(torch.cuda.current_device()) == (10, 3)
    )
    if supported:
        assert require_nvfp4_runtime() == torch.cuda.current_device()
    else:
        with pytest.raises(RuntimeError, match=r"SKIPPED:.*(?:CUDA|SM103)"):
            require_nvfp4_runtime()


@pytest.mark.skipif(
    os.environ.get("AISP_RUN_FAST_CU_NVFP4_GPU_TEST") != "1",
    reason="set AISP_RUN_FAST_CU_NVFP4_GPU_TEST=1 on CUDA 13.1 SM103 hardware",
)
def test_real_sm103_native_pair_matches_full_output_on_current_stream() -> None:
    """Compile and run both native arms; no mocked GPU/runtime success is allowed."""
    require_nvfp4_runtime()
    workload = Nvfp4Workload(4096, 4096, 1024)
    seed = 20260916

    with preserve_rng_state():
        torch.manual_seed(seed)
        baseline = FastCuNvfp4CublasLtBenchmark(workload)
        optimized = FastCuNvfp4KernelBenchmark(workload, rung=9)
        stream = torch.cuda.Stream()
        try:
            with torch.cuda.stream(stream):
                baseline.setup()
                optimized.setup()
            assert baseline.output is None and optimized.output is None
            assert baseline._setup_gate_passed and optimized._setup_gate_passed
            with pytest.raises(RuntimeError, match=r"benchmark_fn\(\) must run"):
                baseline.capture_verification_payload()
            with pytest.raises(RuntimeError, match=r"benchmark_fn\(\) must run"):
                optimized.capture_verification_payload()
            assert all(
                torch.equal(baseline._inputs[name], optimized._inputs[name])
                for name in baseline._inputs
            )
            with torch.cuda.stream(stream):
                baseline.benchmark_fn()
                optimized.benchmark_fn()
            stream.synchronize()
            baseline.capture_verification_payload()
            optimized.capture_verification_payload()
            baseline_output = baseline.get_verify_output()
            optimized_output = optimized.get_verify_output()

            assert baseline_output.numel() == workload.m * workload.n
            assert torch.isfinite(baseline_output).all()
            assert torch.isfinite(optimized_output).all()
            torch.testing.assert_close(
                optimized_output,
                baseline_output,
                rtol=2e-3,
                atol=0.5,
            )
        finally:
            optimized.teardown()
            baseline.teardown()
