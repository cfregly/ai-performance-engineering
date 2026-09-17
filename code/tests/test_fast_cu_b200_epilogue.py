"""CPU contracts plus optional exact-SM100 execution for the B200 epilogue pair."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
import torch

from core.discovery import discover_benchmarks
from core.harness.validity_checks import check_setup_precomputation
from labs.fast_cu import b200_epilogue
from labs.fast_cu.b200_epilogue import (
    DEFAULT_NUM_ELEMENTS,
    ELEMENTS_PER_THREAD,
    EXPECTED_COMPUTE_CAPABILITY,
    MINIMUM_CUDA,
    B200EpilogueBenchmark,
    ensure_b200_capability_supported,
    ensure_cuda_build_supported,
    parse_cuda_build,
    require_b200_runtime,
    validate_epilogue_tensors,
    validate_num_elements,
)

LAB_DIR = Path(__file__).resolve().parents[1] / "labs" / "fast_cu"
CUDA_SOURCE = LAB_DIR / "b200_epilogue.cu"


def test_default_workload_is_representative_and_aligned() -> None:
    assert DEFAULT_NUM_ELEMENTS == 1 << 24
    assert ELEMENTS_PER_THREAD == 16
    assert DEFAULT_NUM_ELEMENTS % ELEMENTS_PER_THREAD == 0
    assert MINIMUM_CUDA == (12, 9)
    assert EXPECTED_COMPUTE_CAPABILITY == (10, 0)


@pytest.mark.parametrize("value", [True, 0, -16, 17, 31, 1.5, "16"])
def test_num_elements_contract_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        validate_num_elements(value)


def test_cuda_build_parser_is_explicit() -> None:
    assert parse_cuda_build("12.9") == (12, 9)
    assert parse_cuda_build("13.0") == (13, 0)
    assert parse_cuda_build("13.0.1") == (13, 0)
    with pytest.raises(RuntimeError, match="CUDA-enabled"):
        parse_cuda_build(None)
    with pytest.raises(RuntimeError, match="Cannot parse"):
        parse_cuda_build("development")


def test_benchmark_construction_does_not_resolve_cuda() -> None:
    baseline = B200EpilogueBenchmark(optimized=False, num_elements=32)
    optimized = B200EpilogueBenchmark(optimized=True, num_elements=32)

    assert baseline._device is None
    assert optimized._device is None
    assert baseline.get_config() == optimized.get_config()
    assert baseline.get_workload_metadata() == optimized.get_workload_metadata()
    assert baseline.get_custom_metrics()["fast_cu.store_instructions_per_thread"] == 2.0
    assert optimized.get_custom_metrics()["fast_cu.store_instructions_per_thread"] == 1.0


def test_real_discovery_finds_the_pair() -> None:
    pairs = {
        example: (baseline.name, [path.name for path in optimized])
        for baseline, optimized, example in discover_benchmarks(LAB_DIR, warn_missing=False)
        if example == "b200_epilogue"
    }
    assert pairs == {
        "b200_epilogue": (
            "baseline_b200_epilogue.py",
            ["optimized_b200_epilogue.py"],
        )
    }


@pytest.mark.parametrize(
    ("module_name", "optimized"),
    [
        ("baseline_b200_epilogue", False),
        ("optimized_b200_epilogue", True),
    ],
)
def test_wrapper_factories_construct_without_cuda(module_name: str, optimized: bool) -> None:
    module = importlib.import_module(f"labs.fast_cu.{module_name}")
    benchmark = module.get_benchmark()
    assert isinstance(benchmark, B200EpilogueBenchmark)
    assert benchmark.optimized is optimized
    assert benchmark._device is None


def test_tensor_contract_rejects_dtype_layout_shape_alignment_and_cpu() -> None:
    accumulators = torch.empty(16, dtype=torch.float32)
    output = torch.empty(16, dtype=torch.float16)

    with pytest.raises(TypeError, match="float32"):
        validate_epilogue_tensors(accumulators.to(torch.float64), output)
    with pytest.raises(TypeError, match="float16"):
        validate_epilogue_tensors(accumulators, output.to(torch.float32))
    noncontiguous = torch.empty((16, 2), dtype=torch.float32)[:, 0]
    with pytest.raises(ValueError, match="contiguous"):
        validate_epilogue_tensors(noncontiguous, output)
    with pytest.raises(ValueError, match="matching shapes"):
        validate_epilogue_tensors(accumulators, torch.empty(32, dtype=torch.float16))

    output_storage = torch.empty(17, dtype=torch.float16)
    misaligned_output = output_storage[1:17]
    assert misaligned_output.is_contiguous()
    assert misaligned_output.data_ptr() % 32 != 0
    with pytest.raises(ValueError, match="32-byte aligned"):
        validate_epilogue_tensors(accumulators, misaligned_output)

    with pytest.raises(ValueError, match="CUDA tensors"):
        validate_epilogue_tensors(accumulators, output)


def test_pure_runtime_contract_requires_cuda_12_9_and_exact_sm100() -> None:
    ensure_b200_capability_supported((10, 0))
    ensure_cuda_build_supported((12, 9))
    ensure_cuda_build_supported((13, 0))
    with pytest.raises(RuntimeError, match="exact SM100"):
        ensure_b200_capability_supported((10, 3))
    with pytest.raises(RuntimeError, match=r"12\.9\+"):
        ensure_cuda_build_supported((12, 8))


def test_runtime_gate_reports_real_host_without_mocking_success() -> None:
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cuda")
    )
    supported = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(device) == EXPECTED_COMPUTE_CAPABILITY
        and parse_cuda_build(torch.version.cuda) >= MINIMUM_CUDA
    )
    if supported:
        require_b200_runtime(device)
    else:
        with pytest.raises(RuntimeError, match=r"SKIPPED:"):
            require_b200_runtime(device)


def test_extension_build_is_pinned_to_sm100_and_cuda_12_9(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_load(name, sources, **kwargs):
        captured.update(name=name, sources=sources, **kwargs)
        return sentinel

    b200_epilogue.load_b200_epilogue_extension.cache_clear()
    monkeypatch.setattr(b200_epilogue, "load_cuda_extension", fake_load)
    try:
        assert b200_epilogue.load_b200_epilogue_extension() is sentinel
    finally:
        b200_epilogue.load_b200_epilogue_extension.cache_clear()

    assert captured["name"] == "fast_cu_b200_epilogue"
    assert captured["sources"] == [CUDA_SOURCE]
    assert captured["minimum_cuda"] == (12, 9)
    assert "-gencode=arch=compute_100,code=sm_100" in captured["extra_cuda_cflags"]


def test_cuda_source_keeps_conversion_policy_and_store_width_isolated() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")

    assert source.count("cvt.rn.f16x2.f32") == 1
    assert ': "f"(hi), "f"(lo)' in source
    assert source.count("createpolicy.fractional.L2::evict_first") == 1
    assert source.count("st.global.L1::no_allocate.L2::cache_hint.v4.b32") == 1
    assert source.count("st.global.L1::no_allocate.L2::cache_hint.v8.b32") == 1
    assert source.count("pack_f16x2(accumulators[") == 8

    branch = re.search(
        r"if constexpr \(WideStore\) \{(?P<wide>.*?)\} else \{(?P<narrow>.*?)\n  \}",
        source,
        re.DOTALL,
    )
    assert branch is not None
    assert branch.group("wide").count("store_b256(") == 1
    assert "store_b128(" not in branch.group("wide")
    assert branch.group("narrow").count("store_b128(") == 2
    assert "store_b256(" not in branch.group("narrow")


def _require_actual_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires exact SM100")
    if parse_cuda_build(torch.version.cuda) < MINIMUM_CUDA:
        pytest.skip("requires a CUDA 12.9+ PyTorch build")


def test_actual_sm100_kernels_match_full_reference_on_current_stream() -> None:
    _require_actual_sm100()
    extension = b200_epilogue.load_b200_epilogue_extension()
    extension.require_exact_sm100()

    count = 1 << 20
    accumulators = torch.randn(count, device="cuda", dtype=torch.float32)
    expected = accumulators.to(torch.float16)
    baseline = torch.empty_like(expected)
    optimized = torch.empty_like(expected)
    validate_epilogue_tensors(accumulators, baseline)
    validate_epilogue_tensors(accumulators, optimized)

    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        extension.baseline(accumulators, baseline)
        extension.optimized(accumulators, optimized)
    stream.synchronize()

    torch.testing.assert_close(baseline, expected, rtol=0, atol=0)
    torch.testing.assert_close(optimized, expected, rtol=0, atol=0)
    torch.testing.assert_close(optimized, baseline, rtol=0, atol=0)


def test_actual_sm100_setup_uses_caller_seed_and_resets_verification_state() -> None:
    _require_actual_sm100()
    b200_epilogue.load_b200_epilogue_extension()
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)

    with torch.random.fork_rng(devices=[device_index]):
        snapshots = []
        for seed in (42, 42, 1042):
            torch.manual_seed(seed)
            benchmark = B200EpilogueBenchmark(optimized=False, num_elements=32)
            benchmark.device = device
            benchmark._ran = True
            benchmark._verification_output = torch.ones(1, device=device)
            benchmark._verification_payload = object()
            setup_valid, setup_error = check_setup_precomputation(
                lambda benchmark=benchmark: {"output": benchmark.output},
                benchmark.setup,
            )
            assert setup_valid, setup_error
            snapshots.append(
                (
                    benchmark.accumulators.detach().clone(),
                    benchmark._caller_seed,
                    int(torch.initial_seed()),
                )
            )
            assert benchmark._ran is False
            assert benchmark.output is None
            assert benchmark._output_buffer is not None
            assert benchmark._verification_output is None
            assert benchmark._verification_payload is None
            benchmark.teardown()

    assert torch.equal(snapshots[0][0], snapshots[1][0])
    assert not torch.equal(snapshots[0][0], snapshots[2][0])
    assert snapshots[0][1:] == snapshots[1][1:] == (42, 42)
    assert snapshots[2][1:] == (1042, 1042)
