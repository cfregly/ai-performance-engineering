"""CPU contracts and opt-in SM100 coverage for the fast.cu NVFP4 port."""

from __future__ import annotations

import os

import pytest
import torch

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.baseline_nvfp4_sm100 import (
    FastCuNvfp4Sm100CublasLtBenchmark,
)
from labs.fast_cu.baseline_nvfp4_sm100 import (
    get_benchmark as get_baseline_benchmark,
)
from labs.fast_cu.nvfp4 import (
    DEFAULT_K,
    DEFAULT_M,
    DEFAULT_N,
    Nvfp4Workload,
    workload_bytes,
    workload_flops,
)
from labs.fast_cu.nvfp4_sm100 import (
    COMPUTE_CAPABILITY,
    CUDA_VERSION,
    ORIGIN_RUNG,
    FastCuNvfp4Sm100Benchmark,
    load_sm100_extension,
    require_sm100_runtime,
)
from labs.fast_cu.optimized_nvfp4_sm100 import (
    FastCuNvfp4Sm100KernelBenchmark,
)
from labs.fast_cu.optimized_nvfp4_sm100 import (
    get_benchmark as get_optimized_benchmark,
)
from tests.protection_test_utils import preserve_rng_state

SMALL_GPU_ENV = "AISP_RUN_FAST_CU_NVFP4_SM100_GPU_TEST"
FULL_GPU_ENV = "AISP_RUN_FAST_CU_NVFP4_SM100_FULL_GPU_TEST"
GUARD_ELEMENTS = 128
GUARD_VALUE = -1234.0


def test_sm100_pair_is_import_safe_and_dispatches_the_expected_arms() -> None:
    workload = Nvfp4Workload(256, 256, 256)
    baseline = FastCuNvfp4Sm100CublasLtBenchmark(workload)
    optimized = FastCuNvfp4Sm100KernelBenchmark(workload)

    assert isinstance(baseline, FastCuNvfp4Sm100Benchmark)
    assert isinstance(optimized, FastCuNvfp4Sm100Benchmark)
    assert isinstance(get_baseline_benchmark(), BaseBenchmark)
    assert isinstance(get_optimized_benchmark(), BaseBenchmark)
    assert baseline.optimized is False
    assert optimized.optimized is True
    assert baseline.rung == optimized.rung == ORIGIN_RUNG == 5
    assert baseline._native_module is None and optimized._native_module is None
    assert baseline._native_context is None and optimized._native_context is None

    baseline_metadata = baseline.get_workload_metadata()
    optimized_metadata = optimized.get_workload_metadata()
    assert baseline_metadata == optimized_metadata
    assert optimized_metadata is not None
    assert optimized_metadata.bytes_per_iteration == float(workload_bytes(workload))
    assert optimized_metadata.custom_units_per_iteration == float(workload_flops(workload))


def test_sm100_default_workload_and_runtime_contract_are_explicit() -> None:
    benchmark = get_optimized_benchmark()

    assert benchmark.workload == Nvfp4Workload(DEFAULT_M, DEFAULT_N, DEFAULT_K)
    assert (DEFAULT_M, DEFAULT_N, DEFAULT_K) == (8192, 8192, 8192)
    assert CUDA_VERSION == (13, 0)
    assert COMPUTE_CAPABILITY == (10, 0)


def test_sm100_runtime_gate_reports_the_real_host_without_mocking_success() -> None:
    runtime = None
    if isinstance(torch.version.cuda, str):
        fields = torch.version.cuda.split(".")
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            runtime = (int(fields[0]), int(fields[1]))
    supported = (
        torch.cuda.is_available()
        and runtime is not None
        and runtime >= CUDA_VERSION
        and torch.cuda.get_device_capability(torch.cuda.current_device()) == COMPUTE_CAPABILITY
    )

    if supported:
        assert require_sm100_runtime() == torch.cuda.current_device()
    else:
        with pytest.raises(RuntimeError, match=r"SKIPPED:.*(?:CUDA|B200|SM100)"):
            require_sm100_runtime()


def _require_opt_in_sm100(env_name: str) -> None:
    if os.environ.get(env_name) != "1":
        pytest.skip(f"set {env_name}=1 on a CUDA 13.0+ B200 host")
    require_sm100_runtime()


def _guarded_poisoned_output(
    m: int, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    storage = torch.full(
        (m * n + 2 * GUARD_ELEMENTS,),
        GUARD_VALUE,
        device="cuda",
        dtype=torch.float16,
    )
    output = storage[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(m, n)
    output.fill_(float("nan"))
    prefix = storage[:GUARD_ELEMENTS].clone()
    suffix = storage[-GUARD_ELEMENTS:].clone()
    return storage, output, prefix, suffix


def test_real_sm100_tail_classes_match_independent_host_oracles() -> None:
    """Exercise both ragged K classes through the native host-reference gate."""
    _require_opt_in_sm100(SMALL_GPU_ENV)
    extension = load_sm100_extension()

    for k, seed in ((1136, 20260916), (1152, 20260917)):
        assert extension.validate_against_host(129, 257, k, seed) is True


@pytest.mark.parametrize("k", [32, 64, 96, 128, 160])
def test_real_sm100_k64_dispatch_overwrites_output_and_is_deterministic(k: int) -> None:
    """Compare ragged GPU output with an independent CPU oracle on another stream."""
    _require_opt_in_sm100(SMALL_GPU_ENV)
    extension = load_sm100_extension()
    m, n, seed = 259, 513, 20261000 + k
    context = extension.Nvfp4Context(m, n, k, seed, True)
    reference = context.host_reference()

    assert reference.device.type == "cpu"
    assert reference.dtype == torch.float32
    assert tuple(reference.shape) == (m, n)

    first_storage, first, first_prefix, first_suffix = _guarded_poisoned_output(m, n)
    second_storage, second, second_prefix, second_suffix = _guarded_poisoned_output(m, n)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        context.launch_fast(first)
        context.launch_fast(second)
    stream.synchronize()

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert torch.equal(first_storage[:GUARD_ELEMENTS], first_prefix)
    assert torch.equal(first_storage[-GUARD_ELEMENTS:], first_suffix)
    assert torch.equal(second_storage[:GUARD_ELEMENTS], second_prefix)
    assert torch.equal(second_storage[-GUARD_ELEMENTS:], second_suffix)
    assert torch.equal(first, second)
    torch.testing.assert_close(
        first.float().cpu(),
        reference,
        rtol=2e-3,
        atol=0.5,
    )


def test_real_sm100_default_cube_matches_full_cublaslt_output() -> None:
    """Run the complete 8192-cube pair and compare every FP16 output element."""
    _require_opt_in_sm100(FULL_GPU_ENV)
    workload = Nvfp4Workload()
    seed = 20260916

    with preserve_rng_state():
        torch.manual_seed(seed)
        baseline = FastCuNvfp4Sm100CublasLtBenchmark(workload)
        optimized = FastCuNvfp4Sm100KernelBenchmark(workload)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream):
                baseline.setup()
                optimized.setup()
            assert baseline._setup_gate_passed
            assert optimized._setup_gate_passed
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

            assert baseline_output.numel() == DEFAULT_M * DEFAULT_N
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
