"""Regression tests for audit finding W2-040 (GB10 bandwidth claims).

GB10 (DGX Spark) has no separate GPU memory. Its CPU and GPU share LPDDR5X
with a published 273 GB/s peak, so the old 800 to 900 GB/s GB10 figures in
ch02 were about 3x too high. The ch02 bandwidth benchmark also labeled a DMA
copy as a zero-copy read, treated every SM 12.x GPU as GB10, and compared
every SM 10.x GPU with PCIe, including GB200 and GB300 GPUs that reach host
memory over NVLink-C2C.

The CPU tests run the benchmark's real classification, comparison, and
input-check code. The classification tests replace only the host machine
name and the CUDA driver attribute query in core.common.grace_platform, which
the benchmark uses for the host link. The zero-copy view comes from
core.utils.pinned_host_view, which the benchmark shares with the
grace_coherent_memory pair. The CUDA tests run the real
zero-copy kernel and the module entrypoint. They need a CUDA GPU such as a
B200. The CUDA demo cannot compile on a host without nvcc, so its comment fix
is checked as a source contract and needs the B200 build listed with this fix.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.autograd import DeviceType

import ch02.nvlink_c2c_bandwidth_benchmark as nvlink
from core.common import grace_platform
from core.utils.pinned_host_view import pinned_host_cuda_view

CODE_ROOT = Path(__file__).resolve().parents[1]
HOST_ATOMICS = grace_platform.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED
HOST_PAGE_TABLES = grace_platform.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SKIPPED: the zero-copy read path needs a CUDA GPU",
)


def _capability(major: int, minor: int) -> SimpleNamespace:
    return SimpleNamespace(major=major, minor=minor)


def _driver_reporting(values: dict[int, int]):
    def _query(device_index: int, attribute: int) -> int:
        return values[attribute]

    return _query


def _refuse_driver_query(device_index: int, attribute: int) -> int:
    raise AssertionError(f"driver attribute {attribute} must not be queried for device {device_index}")


def _host(monkeypatch: pytest.MonkeyPatch, machine: str, driver_query) -> None:
    monkeypatch.setattr(grace_platform.platform, "machine", lambda: machine)
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", driver_query)


def _comparison_text(
    arch_type: str,
    *,
    h2d: float = 60.0,
    d2h: float = 50.0,
    bidir: float = 100.0,
    zero_copy: float = 120.0,
) -> str:
    return "\n".join(
        nvlink.theoretical_comparison_lines(
            arch_type,
            h2d_gbps=h2d,
            d2h_gbps=d2h,
            bidir_gbps=bidir,
            zero_copy_gbps=zero_copy,
        )
    )


def _percentages(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"\((\d+\.\d)%\)", text)]


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((12, 1), "gb10"), ((12, 0), "other"), ((11, 0), "other"), ((9, 0), "other")],
)
def test_capability_alone_classifies_only_gb10_and_unknown_gpus(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: str
) -> None:
    # SM 12.0 is the RTX 50 and RTX PRO family, which are discrete PCIe GPUs.
    _host(monkeypatch, "aarch64", _refuse_driver_query)

    assert nvlink._classify_platform(_capability(*capability), 0) == expected


@pytest.mark.parametrize(("capability", "expected"), [((10, 0), "b200"), ((10, 3), "b300")])
def test_sm10_on_x86_host_compares_against_pcie(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: str
) -> None:
    _host(monkeypatch, "x86_64", _refuse_driver_query)

    arch_type = nvlink._classify_platform(_capability(*capability), 0)
    text = _comparison_text(arch_type)

    assert arch_type == expected
    assert "PCIe 5.0 x16 spec peak:    64 GB/s per direction, 128 GB/s both directions" in text
    assert "NVLink-C2C" not in text


@pytest.mark.parametrize(("capability", "expected"), [((10, 0), "gb200"), ((10, 3), "gb300")])
def test_sm10_with_driver_reported_c2c_link_has_no_pcie_reference(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: str
) -> None:
    _host(monkeypatch, "aarch64", _driver_reporting({HOST_ATOMICS: 1, HOST_PAGE_TABLES: 1}))

    arch_type = nvlink._classify_platform(_capability(*capability), 3)
    # These rates exceed the PCIe 5.0 x16 peak but fit within NVLink-C2C.
    text = _comparison_text(arch_type, h2d=380.0, d2h=360.0, bidir=600.0, zero_copy=300.0)
    lines = text.splitlines()

    assert arch_type == expected
    assert "PCIe" not in text
    assert "NVLink-C2C spec peak:      450 GB/s per direction, 900 GB/s both directions" in text
    assert next(line for line in lines if line.startswith("H2D:")).endswith("380.00 GB/s (84.4%)")
    assert next(line for line in lines if line.startswith("Bidirectional:")).endswith(
        "600.00 GB/s (66.7%)"
    )
    assert f"NVLink-C2C of a {expected.upper()} superchip, which joins one Grace CPU to two GPUs" in text
    assert _percentages(text)
    assert max(_percentages(text)) <= 100.0


def test_sm10_on_arm_host_without_host_atomics_compares_against_pcie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A PCIe GPU in an Arm server reports no native host atomics.
    _host(monkeypatch, "aarch64", _driver_reporting({HOST_ATOMICS: 0}))

    assert nvlink._classify_platform(_capability(10, 0), 0) == "b200"


def test_unanswered_host_link_query_stops_instead_of_assuming_pcie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _driver_fails(device_index: int, attribute: int) -> int:
        raise RuntimeError("cuInit failed with CUresult 100")

    _host(monkeypatch, "aarch64", _driver_fails)

    with pytest.raises(RuntimeError, match="over NVLink-C2C or PCIe") as excinfo:
        nvlink._classify_platform(_capability(10, 3), 0)
    assert "CUresult 100" in str(excinfo.value.__cause__)


def test_gb10_comparison_uses_unified_memory_peak_and_copy_traffic() -> None:
    text = _comparison_text("gb10")

    assert "273 GB/s, shared by CPU and GPU" in text
    assert "900" not in text
    assert "800" not in text
    # Each payload byte is read and written in the same LPDDR5X, so the
    # percentage uses twice the payload against the 273 GB/s peak.
    assert "60.00 GB/s payload, 120.00 GB/s traffic (44.0% of peak)" in text
    assert "100.00 GB/s payload, 200.00 GB/s traffic (73.3% of peak)" in text
    assert "120.00 GB/s payload, 240.00 GB/s traffic (87.9% of peak)" in text
    assert "is 4.3x the 64 GB/s per-direction rate of PCIe 5.0 x16" in text


@pytest.mark.parametrize("arch_type", ["b200", "b300"])
def test_pcie_comparison_states_gb10_peak_and_per_direction_pcie(arch_type: str) -> None:
    text = _comparison_text(arch_type)
    lines = text.splitlines()

    assert "900" not in text
    assert f"GB10 vs {arch_type.upper()} comparison" in text
    assert "GB10 unified LPDDR5X:  273 GB/s peak, shared by CPU and GPU (4.3x" in text
    assert "payload ceiling is 136.5 GB/s (2.1x PCIe 5.0 x16 per direction)" in text
    h2d = next(line for line in lines if line.startswith("H2D:"))
    bidir = next(line for line in lines if line.startswith("Bidirectional:"))
    # Bidirectional payload counts both directions, so it is compared with 128 GB/s.
    assert h2d.endswith("60.00 GB/s (93.8%)")
    assert bidir.endswith("100.00 GB/s (78.1%)")


def test_unknown_platform_prints_no_reference_peak() -> None:
    assert _comparison_text("other") == ""


def test_zero_copy_view_rejects_pageable_host_memory() -> None:
    with pytest.raises(RuntimeError, match="pinned host tensor"):
        pinned_host_cuda_view(torch.zeros(16, dtype=torch.float32), torch.device("cuda", 0))


def test_benchmark_reads_through_the_shared_pinned_host_view() -> None:
    # The zero-copy read must use the core helper, which keeps the pinned,
    # float32 and address-equality checks, not a private copy of it.
    assert nvlink.pinned_host_cuda_view is pinned_host_cuda_view
    assert not hasattr(nvlink, "_pinned_host_cuda_view")
    assert nvlink.host_link_flags is grace_platform.host_link_flags


def test_zero_copy_demo_kernel_comment_drops_nvlink_c2c_claim() -> None:
    source = (CODE_ROOT / "ch02" / "memory_transfer_zero_copy_demo.cu").read_text(
        encoding="utf-8"
    )

    assert "via NVLink-C2C" not in source
    assert "const float* __restrict__ cpu_input,   // Pinned system memory, read in place" in source
    assert "273 GB/s peak unified system-memory bandwidth" in source


@requires_cuda
def test_zero_copy_view_aliases_pinned_host_memory() -> None:
    host = torch.zeros(1024, dtype=torch.float32).pin_memory()
    view = pinned_host_cuda_view(host, torch.device("cuda", torch.cuda.current_device()))

    assert view.is_cuda
    assert view.data_ptr() == host.data_ptr()
    host[7] = 42.0
    assert view[7].item() == 42.0


@requires_cuda
def test_zero_copy_read_runs_a_kernel_without_host_to_device_copies() -> None:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as prof:
        bandwidth_gbs, latency_ms = nvlink.measure_zero_copy_read_bandwidth(
            size_mb=8, iterations=4
        )
    events = list(prof.events())
    # The CPU op aten::neg also matches "neg", so only GPU-side events count.
    gpu_names = [event.name for event in events if event.device_type == DeviceType.CUDA]

    assert bandwidth_gbs > 0.0
    assert latency_ms > 0.0
    assert [name for name in gpu_names if "neg" in name], gpu_names
    assert [event.name for event in events if "HtoD" in event.name] == []


@requires_cuda
def test_module_entrypoint_reports_corrected_reference_peaks() -> None:
    env = dict(os.environ, BENCHMARK_QUICK="1")
    completed = subprocess.run(
        [sys.executable, "-m", "ch02.nvlink_c2c_bandwidth_benchmark"],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stderr
    out = completed.stdout
    assert "Test 4: Zero-Copy Reads of Pinned Host Memory" in out
    comparison = out.split("Theoretical Bandwidth Comparison", 1)[1]
    arch_type, _, _ = nvlink.detect_architecture()
    if arch_type in ("gb10", "b200", "b300", "gb200", "gb300"):
        assert "273 GB/s" in comparison
    if arch_type in ("gb10", "b200", "b300"):
        assert "900 GB/s" not in out
    if arch_type in ("b200", "b300"):
        assert "Host link: PCIe" in out
        assert "PCIe 5.0 x16 spec peak:" in comparison
    if arch_type in ("gb200", "gb300"):
        assert "Host link: NVLink-C2C" in out
        assert "NVLink-C2C spec peak:" in comparison
        assert "PCIe" not in comparison
    if arch_type in ("b200", "b300", "gb200", "gb300"):
        # A measured payload rate cannot exceed the spec peak of the link it crosses.
        assert _percentages(comparison)
        assert max(_percentages(comparison)) <= 100.0
