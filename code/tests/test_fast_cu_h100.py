"""Contracts and real-GPU coverage for the fast.cu Hopper adapters."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from labs.fast_cu.baseline_h100_bf16_gemm import (
    BaselineH100Bf16GemmBenchmark,
)
from labs.fast_cu.baseline_int32_reduction import (
    BaselineInt32ReductionBenchmark,
)
from labs.fast_cu.h100_common import (
    GEMM_K,
    GEMM_M,
    GEMM_N,
    H100_CAPABILITY,
    REDUCTION_CAPABILITIES,
    REDUCTION_ELEMENTS,
    _gemm_cuda_flags,
    _reduction_build_contract,
    ensure_h100_capability_supported,
    ensure_reduction_capability_supported,
    load_h100_gemm_extension,
    load_int32_reduction_extension,
)
from labs.fast_cu.optimized_h100_bf16_gemm import (
    OptimizedH100Bf16GemmBenchmark,
)
from labs.fast_cu.optimized_int32_reduction import (
    OptimizedInt32ReductionBenchmark,
)

LAB_DIR = Path(__file__).resolve().parents[1] / "labs" / "fast_cu"


def _require_cuda_capability(supported: set[tuple[int, int]] | frozenset[tuple[int, int]]):
    if not torch.cuda.is_available():
        pytest.skip("real CUDA GPU is unavailable")
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in supported:
        pytest.skip(f"real GPU has unsupported capability {capability}")
    return capability


def test_h100_gate_is_exact_and_reports_skip():
    ensure_h100_capability_supported((9, 0))
    for capability in ((8, 0), (8, 9), (9, 1), (10, 0), (10, 3), (12, 0)):
        with pytest.raises(RuntimeError, match=r"SKIPPED:.*exact SM 9\.0"):
            ensure_h100_capability_supported(capability)


def test_reduction_gate_and_build_contract_cover_h100_and_b200_only():
    assert {(9, 0), (10, 0)} == REDUCTION_CAPABILITIES
    for capability, arch, minimum in (
        ((9, 0), "90a", (12, 0)),
        ((10, 0), "100a", (12, 8)),
    ):
        ensure_reduction_capability_supported(capability)
        flags, minimum_cuda = _reduction_build_contract(capability)
        assert f"-gencode=arch=compute_{arch},code=sm_{arch}" in flags
        assert minimum_cuda == minimum
    for capability in ((8, 0), (9, 1), (10, 3), (12, 0)):
        with pytest.raises(RuntimeError, match=r"SKIPPED:.*SM 9\.0.*SM 10\.0"):
            ensure_reduction_capability_supported(capability)


def test_h100_gemm_build_targets_arch_specific_wgmma():
    flags = _gemm_cuda_flags(H100_CAPABILITY)
    assert "-gencode=arch=compute_90a,code=sm_90a" in flags
    assert "-DNDEBUG" in flags
    assert "-U__CUDA_NO_BFLOAT16_CONVERSIONS__" in flags
    assert not any("compute_100" in flag for flag in flags)
    reduction_flags, _ = _reduction_build_contract(H100_CAPABILITY)
    assert "-U__CUDA_NO_BFLOAT16_CONVERSIONS__" not in reduction_flags


def test_benchmark_contracts_are_equivalent_and_full_output():
    baseline_gemm = BaselineH100Bf16GemmBenchmark()
    optimized_gemm = OptimizedH100Bf16GemmBenchmark()
    assert baseline_gemm.get_input_signature() == optimized_gemm.get_input_signature()
    gemm_signature = baseline_gemm.get_input_signature()
    assert gemm_signature.shapes == {
        "matrix_a": (GEMM_M, GEMM_K),
        "matrix_b": (GEMM_N, GEMM_K),
        "output": (GEMM_M, GEMM_N),
    }
    assert gemm_signature.dtypes["output"] == "torch.bfloat16"
    assert baseline_gemm.get_config().warmup == 5
    assert optimized_gemm.get_config().iterations == 20

    baseline_reduction = BaselineInt32ReductionBenchmark()
    optimized_reduction = OptimizedInt32ReductionBenchmark()
    assert baseline_reduction.get_input_signature() == optimized_reduction.get_input_signature()
    reduction_signature = baseline_reduction.get_input_signature()
    assert reduction_signature.shapes == {
        "input": (REDUCTION_ELEMENTS,),
        "output": (1,),
    }
    assert reduction_signature.dtypes == {
        "input": "torch.int32",
        "output": "torch.int32",
    }
    assert baseline_reduction.get_config().warmup == 10
    assert optimized_reduction.get_config().iterations == 50


def test_teardown_releases_full_verification_payloads_on_cpu():
    gemm = BaselineH100Bf16GemmBenchmark()
    gemm.matrix_a = torch.ones((2, 3), dtype=torch.bfloat16)
    gemm.matrix_b = torch.ones((4, 3), dtype=torch.bfloat16)
    gemm._physical_output = torch.ones((4, 2), dtype=torch.bfloat16)
    gemm._logical_output = gemm._physical_output.transpose(0, 1)
    gemm.output = gemm._logical_output
    gemm.capture_verification_payload()
    assert gemm._verification_payload is not None

    gemm.teardown()

    assert gemm._verification_payload is None
    assert gemm.matrix_a is None and gemm.matrix_b is None
    assert gemm._physical_output is None and gemm._logical_output is None
    with pytest.raises(RuntimeError, match="must be called before verification"):
        gemm.get_verify_output()

    reduction = BaselineInt32ReductionBenchmark()
    reduction.input = torch.tensor([-1, 0, 1], dtype=torch.int32)
    reduction._output_buffer = torch.tensor([0], dtype=torch.int32)
    reduction.output = reduction._output_buffer
    reduction.capture_verification_payload()
    assert reduction._verification_payload is not None

    reduction.teardown()

    assert reduction._verification_payload is None
    assert reduction.input is None and reduction._output_buffer is None
    with pytest.raises(RuntimeError, match="must be called before verification"):
        reduction.get_verify_output()


def test_safe_reduction_source_orders_reset_before_launch_on_current_stream():
    source = (LAB_DIR / "h100_reduction_kernels.cu").read_text(encoding="utf-8")
    adapter = source[source.index("void run_safe_vectorized_reduction") :]
    reset = adapter.index("cudaMemsetAsync")
    launch = adapter.index("safe_vectorized_sum_kernel<<<")
    assert reset < launch
    assert "at::cuda::getCurrentCUDAStream" in adapter[:launch]
    kernel = source[
        source.index("__global__ __launch_bounds__") : source.index(
            "int64_t cub_temp_storage_bytes"
        )
    ]
    assert "*output = 0" not in kernel
    assert "int4* input" in kernel
    assert "atomicAdd(output, sum)" in kernel


def test_gemm_source_uses_pinned_latest_kernel_and_current_stream():
    source = (LAB_DIR / "h100_gemm_extension.cu").read_text(encoding="utf-8")
    assert '#include "upstream/h100/matmul/matmul_12.cuh"' in source
    assert "M12::matmulKernel12" in source
    launch = source[source.index("void run_upstream_gemm") :]
    assert "at::cuda::getCurrentCUDAStream" in launch
    assert "<<<kGridCtas, kThreads, shared_bytes, stream>>>" in launch
    assert "physical output must have transposed [N,M] layout" in source


def test_real_supported_gpu_int32_reduction_matches_full_reference():
    _require_cuda_capability(REDUCTION_CAPABILITIES)
    extension = load_int32_reduction_extension()
    count = 1 << 20
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = (torch.arange(count, device="cuda", dtype=torch.int32) % 3) - 1
        cub_output = torch.empty((1,), device="cuda", dtype=torch.int32)
        optimized_output = torch.empty_like(cub_output)
        storage_bytes = extension.cub_temp_storage_bytes(values, cub_output)
        temp_storage = torch.empty((storage_bytes,), device="cuda", dtype=torch.uint8)
        extension.run_cub_reduction(values, cub_output, temp_storage)
        extension.run_safe_vectorized_reduction(values, optimized_output)
        reference = values.sum(dtype=torch.int64).to(torch.int32).reshape(1)
    stream.synchronize()
    assert torch.equal(cub_output, reference)
    assert torch.equal(optimized_output, reference)


def test_real_h100_gemm_matches_cublas_full_logical_output():
    _require_cuda_capability({H100_CAPABILITY})
    extension = load_h100_gemm_extension()
    # Two M-cluster tiles avoid the pinned host schedule's one-tile clz edge.
    m, n, k = 512, 256, 64
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    cublas_physical = torch.empty((n, m), device="cuda", dtype=torch.bfloat16)
    upstream_physical = torch.empty_like(cublas_physical)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        extension.prepare_cublas_gemm(a, b, cublas_physical)
        extension.prepare_upstream_gemm(a, b, upstream_physical)
        extension.run_cublas_gemm(a, b, cublas_physical)
        extension.run_upstream_gemm(a, b, upstream_physical)
        fp32_reference = a.float() @ b.float().transpose(0, 1)
    stream.synchronize()
    cublas_logical = cublas_physical.transpose(0, 1)
    upstream_logical = upstream_physical.transpose(0, 1)
    assert torch.allclose(upstream_logical, cublas_logical, rtol=1e-2, atol=1e-1)
    assert torch.allclose(cublas_logical.float(), fp32_reference, rtol=1e-2, atol=1e-1)
    assert torch.allclose(upstream_logical.float(), fp32_reference, rtol=1e-2, atol=1e-1)
