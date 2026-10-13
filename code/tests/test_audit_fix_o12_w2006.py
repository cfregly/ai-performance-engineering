"""Regression tests for audit finding W2-006 (ch02 grace_coherent_memory).

The optimized variant used to keep its buffer in GPU memory and skip the host
transfers that the baseline times, so its speedup came from removed work. These
tests check that it now moves the same bytes and produces the same host result.
"""

from __future__ import annotations

import platform
from types import SimpleNamespace

import pytest
import torch

import ch02.baseline_grace_coherent_memory as baseline_grace
import ch02.optimized_grace_coherent_memory as optimized_grace
from core.common import grace_platform
from core.utils.pinned_host_view import pinned_host_cuda_view

_MIB = 1024 * 1024
_CUDA_SKIP = "SKIPPED: requires CUDA to map pinned host memory into the GPU address space"


def _pass_grace_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CUDA driver report an NVLink-C2C host link on an Arm host, as on GB200."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx: SimpleNamespace(name="NVIDIA GB200", major=10, minor=0),
    )
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", lambda _index, _attribute: 1)


def _impl_without_grace_gate(cls: type, size_mb: int):
    """Skip only the Grace host check so the real kernels also run on PCIe hosts such as x86 B200."""
    impl = cls.__new__(cls)
    impl.size_mb = size_mb
    impl.iterations = 1
    impl.device = torch.device("cuda")
    # False skips NUMA binding, which would change this process's CPU affinity.
    impl.is_grace_blackwell = False
    return impl


def test_optimized_step_counts_the_same_host_link_bytes_as_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pass_grace_gate(monkeypatch)
    optimized = optimized_grace.OptimizedGraceCoherentMemoryBenchmark(size_mb=8, iterations=1)
    baseline = baseline_grace.GraceCoherentMemoryBenchmark(size_mb=8, iterations=1)

    expected = float(8 * _MIB * 2)
    assert optimized.get_workload_metadata().bytes_per_iteration == expected
    assert baseline.get_workload_metadata().bytes_per_iteration == expected


def test_optimized_metrics_report_moved_bytes_and_no_invented_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pass_grace_gate(monkeypatch)
    benchmark = optimized_grace.OptimizedGraceCoherentMemoryBenchmark(size_mb=8, iterations=1)
    expected = float(8 * _MIB * 2)

    assert benchmark.get_custom_metrics() == {"transfer.bytes": expected}

    benchmark.elapsed_s = 0.004
    metrics = benchmark.get_custom_metrics()
    assert metrics["transfer.bytes"] == expected
    assert metrics["transfer.achieved_gbps"] == pytest.approx(expected / 0.004 / 1e9)
    assert not any(key.startswith("coherent_memory.") for key in metrics)


def test_zero_copy_mapping_rejects_pageable_host_memory() -> None:
    with pytest.raises(RuntimeError, match="requires a pinned host tensor"):
        pinned_host_cuda_view(torch.zeros(16), torch.device("cuda", 0))


def test_optimized_step_maps_host_memory_through_the_shared_view() -> None:
    assert optimized_grace.pinned_host_cuda_view is pinned_host_cuda_view
    assert not hasattr(optimized_grace, "_map_pinned_host_tensor")


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_SKIP)
def test_mapped_view_writes_land_in_pinned_host_memory() -> None:
    host = torch.arange(4096, dtype=torch.float32).pin_memory()
    mapped = pinned_host_cuda_view(host, torch.device("cuda", torch.cuda.current_device()))

    assert mapped.is_cuda
    assert mapped.data_ptr() == host.data_ptr()
    mapped.mul_(2.0)
    torch.cuda.synchronize()
    assert torch.equal(host, torch.arange(4096, dtype=torch.float32) * 2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_SKIP)
def test_optimized_step_matches_baseline_step_in_host_memory() -> None:
    baseline = _impl_without_grace_gate(baseline_grace.BaselineGraceCoherentMemory, 1)
    optimized = _impl_without_grace_gate(optimized_grace.OptimizedGraceCoherentMemory, 1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        baseline.setup()
        torch.manual_seed(1234)
        optimized.setup()
    try:
        assert optimized.cpu_data.device.type == "cpu"
        assert optimized.cpu_data.is_pinned()
        assert optimized.gpu_data.is_cuda
        assert optimized.gpu_data.data_ptr() == optimized.cpu_data.data_ptr()
        reference = optimized.cpu_data.clone()
        assert torch.equal(reference, baseline.cpu_data)

        for _ in range(3):
            baseline.run_step()
            optimized.run_step()
            reference.mul_(2.0).add_(1.0)

        assert torch.equal(optimized.cpu_data, reference)
        assert torch.equal(optimized.cpu_data, baseline.cpu_data)
    finally:
        baseline.cleanup()
        optimized.cleanup()


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_SKIP)
def test_grace_pair_wrappers_produce_identical_host_results() -> None:
    try:
        baseline = baseline_grace.GraceCoherentMemoryBenchmark(size_mb=8, iterations=3)
        optimized = optimized_grace.OptimizedGraceCoherentMemoryBenchmark(size_mb=8, iterations=3)
    except RuntimeError as exc:
        if str(exc).startswith("SKIPPED:"):
            pytest.skip(str(exc))
        raise

    outputs = []
    for benchmark in (baseline, optimized):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            benchmark.setup()
        try:
            for _ in range(3):
                benchmark.benchmark_fn()
            benchmark.capture_verification_payload()
            outputs.append(benchmark.get_verify_output())
        finally:
            benchmark.teardown()

    assert optimized.bandwidth_gb_s is not None and optimized.bandwidth_gb_s > 0.0
    assert outputs[1].device.type == "cpu"
    assert torch.equal(outputs[0], outputs[1])
