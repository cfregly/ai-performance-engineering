from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ch03.baseline_pinned_prefetch_mlp import (
    PINNED_PREFETCH_MLP_OUTPUT_TOLERANCE as BASELINE_MLP_TOLERANCE,
)
from ch03.optimized_pinned_prefetch_mlp import (
    PINNED_PREFETCH_MLP_OUTPUT_TOLERANCE as OPTIMIZED_MLP_TOLERANCE,
)

CODE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def hardware_info(monkeypatch: pytest.MonkeyPatch):
    """Import hardware_info without requiring its optional GPUtil dependency."""
    module_name = "ch02.hardware_info"
    sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "GPUtil", SimpleNamespace(getGPUs=lambda: []))
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_w2_035_datacenter_blackwell_check_includes_sm_103() -> None:
    source = (CODE_ROOT / "ch02" / "cpu_gpu_grace_blackwell_coherency.cu").read_text(
        encoding="utf-8"
    )

    assert "bool is_blackwell = (prop.major == 10);" in source
    assert "prop.major == 10 && prop.minor == 0" not in source


def test_w2_036_b200_b300_branch_uses_sm_capability(
    hardware_info,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert hardware_info._is_b200_or_b300({"sm_version": "sm_100"})
    assert hardware_info._is_b200_or_b300({"sm_version": "sm_103"})
    assert not hardware_info._is_b200_or_b300({"sm_version": "sm_121"})

    b300_info = {
        "sm_version": "sm_103",
        "memory_bandwidth_gbps": 8_000.0,
        "nvlink_c2c": False,
        "max_unified_memory_tb": None,
    }
    monkeypatch.setattr(hardware_info.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware_info, "get_gpu_info", lambda: b300_info)

    hardware_info.demonstrate_blackwell_optimizations()

    output = capsys.readouterr().out
    assert "Blackwell B200/B300 Optimizations:" in output
    assert "Detected memory bandwidth: 8000.0 GB/s" in output
    assert "NVLink-C2C was not detected on this host" in output
    assert "does not support Blackwell" not in output


def test_w2_037_copy_and_gemm_metrics_use_correct_work_models(hardware_info) -> None:
    # Ten 1 GB copies in one second move 10 GB of reads plus 10 GB of writes.
    assert hardware_info._effective_copy_bandwidth_gbps(
        1_000_000_000,
        10,
        1_000.0,
    ) == pytest.approx(20.0)
    # A 1000^3 GEMM performs 2e9 operations; 2 ms is 1 TFLOP/s.
    assert hardware_info._gemm_throughput_tflops(1_000, 0.002) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="must be positive"):
        hardware_info._effective_copy_bandwidth_gbps(1, 1, 0.0)
    with pytest.raises(ValueError, match="must be positive"):
        hardware_info._gemm_throughput_tflops(0, 1.0)

    source = (CODE_ROOT / "ch02" / "hardware_info.py").read_text(encoding="utf-8")
    gemm_section = source.split("for size in sizes:", maxsplit=1)[1].split(
        "def benchmark_tensor_operations",
        maxsplit=1,
    )[0]
    assert "TFLOP/s" in gemm_section
    assert "GB/s" not in gemm_section


def test_w2_038_b300_detection_uses_sm_103_not_capacity_window() -> None:
    source = (CODE_ROOT / "ch02" / "memory_transfer_nvlink_demo.cu").read_text(
        encoding="utf-8"
    )
    detection = source.split("SystemInfo detect_system_capabilities()", maxsplit=1)[1].split(
        "double benchmark_host_to_device",
        maxsplit=1,
    )[0]

    assert "const bool is_b300 = prop.major == 10 && prop.minor == 3;" in detection
    assert "if (is_b300)" in detection
    assert "mem_gib >= 270.0" not in detection
    assert "mem_gib <= 300.0" not in detection
    assert "GiB per GPU" in detection


def test_w2_039_no_hint_path_faults_pages_instead_of_prefetching() -> None:
    source = (CODE_ROOT / "ch02" / "memory_transfer_nvlink_demo.cu").read_text(
        encoding="utf-8"
    )
    migration = source.split("void demonstrate_page_migration()", maxsplit=1)[1].split(
        "void benchmark_multigpu_p2p_bandwidth",
        maxsplit=1,
    )[0]
    no_hint = migration.split("// Strategy 1:", maxsplit=1)[1].split(
        "// Reset to CPU",
        maxsplit=1,
    )[0]
    prefetch = migration.split("// Strategy 2:", maxsplit=1)[1].split(
        "// Strategy 3:",
        maxsplit=1,
    )[0]

    kernel_launch = "touch_managed_pages<<<grid, block>>>(managed_data, elements);"
    assert kernel_launch in no_hint
    assert "cudaMemPrefetchAsync" not in no_hint
    assert kernel_launch in prefetch
    assert "cudaMemPrefetchAsync(managed_data, size, gpuLoc" in prefetch


def test_w2_040_gb10_claims_respect_unified_memory_peak() -> None:
    source = (CODE_ROOT / "ch02" / "memory_transfer_zero_copy_demo.cu").read_text(
        encoding="utf-8"
    )

    assert "prop.major == 12 && prop.minor == 1" in source
    assert "273 GB/s peak unified system-memory bandwidth" in source
    assert "SKIPPED: this coherent unified-memory benchmark requires" in source
    assert "800 GB/s" not in source
    assert "900 GB/s" not in source


def test_w2_041_pinned_prefetch_pair_rejects_grossly_wrong_outputs() -> None:
    assert BASELINE_MLP_TOLERANCE == (1e-3, 1e-3)
    assert OPTIMIZED_MLP_TOLERANCE == BASELINE_MLP_TOLERANCE

    for filename in (
        "baseline_pinned_prefetch_mlp.py",
        "optimized_pinned_prefetch_mlp.py",
    ):
        source = (CODE_ROOT / "ch03" / filename).read_text(encoding="utf-8")
        assert "output_tolerance=PINNED_PREFETCH_MLP_OUTPUT_TOLERANCE" in source
        assert "output_tolerance=(1.0, 10.0)" not in source

    reference = torch.tensor([1.0, -3.0, 10.0], dtype=torch.float32)
    near = reference + torch.tensor([1e-4, -1e-4, 5e-4])
    garbage = torch.zeros_like(reference)
    rtol, atol = BASELINE_MLP_TOLERANCE

    assert torch.allclose(near, reference, rtol=rtol, atol=atol)
    assert not torch.allclose(garbage, reference, rtol=rtol, atol=atol)
