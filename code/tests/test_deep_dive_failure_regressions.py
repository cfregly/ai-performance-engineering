from __future__ import annotations

import pytest
import torch

from ch09.baseline_cublas_gemm_fp4_perchannel import BaselineCublasGemmFp4PerchannelBenchmark
from ch09.optimized_compute_bound import OptimizedComputeBoundBenchmark
from ch10.baseline_dsmem_reduction import BaselineDSMEMReductionBenchmark
from ch10.optimized_dsmem_reduction import OptimizedDSMEMReductionBenchmark
from ch10.optimized_dsmem_reduction_cluster_atomic import OptimizedDSMEMClusterAtomicBenchmark
from ch10.optimized_dsmem_reduction_v3 import OptimizedDSMEMReductionV3Benchmark
from ch10.optimized_dsmem_reduction_warp_specialized import OptimizedDSMEMWarpSpecializedBenchmark
from ch12.optimized_cuda_graphs import OptimizedCudaGraphsBenchmark
from ch12.optimized_cuda_graphs_router import CUDAGraphRouterBenchmark
from ch12.optimized_graph_bandwidth import OptimizedGraphBandwidthBenchmark
from ch12.optimized_graph_conditional_runtime import OptimizedGraphBenchmark
from ch12.optimized_kernel_launches import OptimizedKernelLaunchesBenchmark
from ch13.baseline_autograd_standard import BaselineAutogradStandardBenchmark
from ch13.optimized_autograd_standard import OptimizedAutogradCompiledBenchmark
from ch13.optimized_fp8_static import StaticFP8Benchmark
from ch13.optimized_matmul_pytorch import OptimizedMatmulPyTorchBenchmark
from ch13.optimized_memory_profiling import OptimizedMemoryProfilingBenchmark
from ch13.optimized_precisionfp8_te import OptimizedTEFP8Benchmark
from ch16.baseline_regional_compilation import DummyTransformer
from ch16.optimized_regional_compilation import RegionalCompilationTransformer
from ch16.optimized_regional_compilation import OptimizedRegionalCompilationBenchmark
from ch17.optimized_memory import OptimizedMemoryBenchmark
from ch20.optimized_end_to_end_bandwidth import OptimizedEndToEndBandwidthBenchmark
from core.harness.benchmark_harness import BenchmarkConfig
from core.harness.run_benchmarks import _apply_profile_env_overrides

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for deep-dive regression coverage")


@pytest.mark.parametrize(
    ("factory", "label"),
    [
        (OptimizedComputeBoundBenchmark, "ch09.compute_bound.optimized"),
        (BaselineCublasGemmFp4PerchannelBenchmark, "ch09.cublas_gemm_fp4_perchannel.baseline"),
        (OptimizedCudaGraphsBenchmark, "ch12.cuda_graphs.optimized"),
        (CUDAGraphRouterBenchmark, "ch12.cuda_graphs_router.optimized"),
        (OptimizedGraphBandwidthBenchmark, "ch12.graph_bandwidth.optimized"),
        (OptimizedGraphBenchmark, "ch12.graph_conditional_runtime.optimized"),
        (OptimizedKernelLaunchesBenchmark, "ch12.kernel_launches.optimized"),
        (OptimizedMatmulPyTorchBenchmark, "ch13.matmul_pytorch.optimized"),
        (OptimizedMemoryProfilingBenchmark, "ch13.memory_profiling.optimized"),
        (OptimizedTEFP8Benchmark, "ch13.precisionfp8_te.optimized"),
        (OptimizedRegionalCompilationBenchmark, "ch16.regional_compilation.optimized"),
        (OptimizedMemoryBenchmark, "ch17.memory.optimized"),
        (OptimizedEndToEndBandwidthBenchmark, "ch20.end_to_end_bandwidth.optimized"),
    ],
    ids=[
        "compute_bound",
        "cublas_gemm_fp4_perchannel",
        "cuda_graphs",
        "cuda_graphs_router",
        "graph_bandwidth",
        "graph_conditional_runtime",
        "kernel_launches",
        "matmul_pytorch",
        "memory_profiling",
        "precisionfp8_te",
        "regional_compilation",
        "memory",
        "end_to_end_bandwidth",
    ],
)
def test_deep_dive_regressions_raise_nsys_timeout(factory, label: str) -> None:
    bench = factory()
    config = bench.get_config()
    assert config.nsys_timeout_seconds == 1200, label


def test_regional_compilation_uses_light_nsys_capture() -> None:
    bench = OptimizedRegionalCompilationBenchmark()
    config = bench.get_config()

    assert config.nsys_preset_override == "light"


def test_regional_compilation_optimized_only_uses_compiled_layer_helper(monkeypatch) -> None:
    call_count = {"value": 0}

    def _wrapped_layer(layer, x):
        call_count["value"] += 1
        return layer(x)

    monkeypatch.setattr("ch16.optimized_regional_compilation._run_compiled_layer", _wrapped_layer)

    baseline = DummyTransformer(n_layers=2, d_model=8, d_ff=16)
    optimized = RegionalCompilationTransformer(n_layers=2, d_model=8, d_ff=16)
    x = torch.randn(1, 4, 8)

    baseline(x)
    assert call_count["value"] == 0

    optimized(x)
    assert call_count["value"] == 2


@pytest.mark.parametrize(
    ("factory", "label"),
    [
        (OptimizedComputeBoundBenchmark, "ch09.compute_bound.optimized"),
        (OptimizedCudaGraphsBenchmark, "ch12.cuda_graphs.optimized"),
        (CUDAGraphRouterBenchmark, "ch12.cuda_graphs_router.optimized"),
        (OptimizedGraphBandwidthBenchmark, "ch12.graph_bandwidth.optimized"),
        (OptimizedGraphBenchmark, "ch12.graph_conditional_runtime.optimized"),
        (OptimizedKernelLaunchesBenchmark, "ch12.kernel_launches.optimized"),
        (StaticFP8Benchmark, "ch13.fp8_static.optimized"),
        (OptimizedMatmulPyTorchBenchmark, "ch13.matmul_pytorch.optimized"),
        (OptimizedMemoryProfilingBenchmark, "ch13.memory_profiling.optimized"),
        (OptimizedTEFP8Benchmark, "ch13.precisionfp8_te.optimized"),
        (OptimizedMemoryBenchmark, "ch17.memory.optimized"),
        (OptimizedEndToEndBandwidthBenchmark, "ch20.end_to_end_bandwidth.optimized"),
        (OptimizedRegionalCompilationBenchmark, "ch16.regional_compilation.optimized"),
    ],
    ids=[
        "compute_bound_light_nsys",
        "cuda_graphs_light_nsys",
        "cuda_graphs_router_light_nsys",
        "graph_bandwidth_light_nsys",
        "graph_conditional_runtime_light_nsys",
        "kernel_launches_light_nsys",
        "fp8_static_light_nsys",
        "matmul_pytorch_light_nsys",
        "memory_profiling_light_nsys",
        "precisionfp8_te_light_nsys",
        "memory_light_nsys",
        "end_to_end_bandwidth_light_nsys",
        "regional_compilation_light_nsys",
    ],
)
def test_python_deep_dive_problem_cases_use_light_nsys(factory, label: str) -> None:
    bench = factory()
    config = bench.get_config()
    assert config.nsys_preset_override == "light", label


@pytest.mark.parametrize(
    ("factory", "label"),
    [
        (BaselineDSMEMReductionBenchmark, "ch10.dsmem_reduction.baseline"),
        (OptimizedDSMEMReductionBenchmark, "ch10.dsmem_reduction.optimized"),
        (OptimizedDSMEMClusterAtomicBenchmark, "ch10.dsmem_reduction_cluster_atomic.optimized"),
        (OptimizedDSMEMReductionV3Benchmark, "ch10.dsmem_reduction_v3.optimized"),
        (OptimizedDSMEMWarpSpecializedBenchmark, "ch10.dsmem_reduction_warp_specialized.optimized"),
    ],
    ids=[
        "baseline_dsmem_reduction",
        "optimized_dsmem_reduction",
        "optimized_dsmem_reduction_cluster_atomic",
        "optimized_dsmem_reduction_v3",
        "optimized_dsmem_reduction_warp_specialized",
    ],
)
def test_dsmem_wrappers_allow_slow_profiled_binary_runs(factory, label: str) -> None:
    bench = factory()
    config = bench.get_config()
    assert bench.timeout_seconds == 600, label
    assert config.nsys_timeout_seconds == 300, label
    assert config.nsys_preset_override == "light", label
    assert config.profiling_warmup == 0, label
    assert config.profiling_iterations == 1, label
    assert config.profile_env_overrides == {
        "AISP_CUDA_BINARY_PROFILE_WARMUP": "0",
        "AISP_CUDA_BINARY_PROFILE_ITERATIONS": "1",
    }, label


def test_profile_env_overrides_merge_into_profiler_env() -> None:
    config = BenchmarkConfig()
    config.profile_env_overrides = {
        "AISP_CUDA_BINARY_PROFILE_WARMUP": "0",
        "AISP_CUDA_BINARY_PROFILE_ITERATIONS": "1",
    }

    merged = _apply_profile_env_overrides({"PYTHONPATH": "/tmp"}, config=config)

    assert merged["PYTHONPATH"] == "/tmp"
    assert merged["AISP_CUDA_BINARY_PROFILE_WARMUP"] == "0"
    assert merged["AISP_CUDA_BINARY_PROFILE_ITERATIONS"] == "1"


@pytest.mark.parametrize(
    "factory",
    [BaselineAutogradStandardBenchmark, OptimizedAutogradCompiledBenchmark],
    ids=["baseline", "optimized"],
)
def test_autograd_standard_verification_payload_uses_float_output(factory) -> None:
    bench = factory()
    bench.inputs = torch.randn(2, 4, dtype=torch.float16)
    bench.targets = torch.randn(2, 4, dtype=torch.float16)
    bench.output = torch.randn(2, 4, dtype=torch.float16)
    # setup() preallocates this transport buffer; keep the focused fixture's
    # lifecycle equivalent while avoiding model and optimizer setup.
    bench._verify_output_buffer = torch.empty_like(bench.output, dtype=torch.float32)
    bench.capture_verification_payload()

    verify_output = bench.get_verify_output()

    assert verify_output.dtype == torch.float32
