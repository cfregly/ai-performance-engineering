"""Regression tests for public chapter profiling claims and units."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest


def test_ch08_gpu_roofline_examples_refuse_cpu_fallback(
    monkeypatch,
    capsys,
) -> None:
    chapter = importlib.import_module("ch08.roofline")
    monkeypatch.setattr(chapter.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CPU timing is not a reviewed GPU benchmark"):
        chapter.benchmark_example_kernels()
    output = capsys.readouterr().out
    assert "GPU Roofline Analysis" in output
    assert "Blackwell" not in output


def test_core_roofline_default_constructor_refuses_cpu_assumptions(monkeypatch) -> None:
    roofline = importlib.import_module("core.analysis.kernel_roofline")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="No CUDA GPU is available"):
        roofline.RooflineAnalyzer()


def test_h100_roofline_output_is_not_labeled_blackwell(capsys) -> None:
    roofline = importlib.import_module("core.analysis.kernel_roofline")
    analyzer = roofline.RooflineAnalyzer(roofline.get_architecture_specs_for_profile("h100-sxm"))
    results = analyzer.analyze_kernel(
        kernel_time_ms=1.0,
        flops=1_000_000.0,
        bytes_transferred=1_000_000.0,
    )

    analyzer.print_analysis(results, "fixture")
    output = capsys.readouterr().out

    assert "NVIDIA H100" in output
    assert "Profile source:" in output
    assert "Blackwell" not in output


def test_ch17_hbm_analyzer_keeps_dram_bytes_and_l2_sectors_separate() -> None:
    chapter = importlib.import_module("ch17.blackwell_profiling_guide")

    result = chapter.HBMMemoryAnalyzer.analyze_memory_pattern(
        dram_read_throughput=1.0,
        dram_write_throughput=2.0,
        l2_read_sectors=1_000_000,
        l2_write_sectors=2_000_000,
        kernel_duration_ns=1_000_000_000.0,
    )

    assert result["dram_read_bytes"] == pytest.approx(1_000_000_000.0)
    assert result["dram_write_bytes"] == pytest.approx(2_000_000_000.0)
    assert result["l2_read_sector_bytes"] == 32_000_000
    assert result["l2_write_sector_bytes"] == 64_000_000
    assert result["coalescing_efficiency_pct"] is None
    assert result["coalescing_status"].startswith("not_computable")
    assert "avg_bytes_per_read_sector" not in result
    assert "read_burst_efficiency_pct" not in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dram_read_throughput": -1.0},
        {"dram_write_throughput": -1.0},
        {"l2_read_sectors": -1},
        {"l2_write_sectors": -1},
        {"kernel_duration_ns": -1.0},
    ],
)
def test_ch17_hbm_analyzer_rejects_negative_measurements(kwargs) -> None:
    chapter = importlib.import_module("ch17.blackwell_profiling_guide")
    inputs = {
        "dram_read_throughput": 1.0,
        "dram_write_throughput": 1.0,
        "l2_read_sectors": 1,
        "l2_write_sectors": 1,
        "kernel_duration_ns": 1.0,
    }
    inputs.update(kwargs)

    with pytest.raises(ValueError):
        chapter.HBMMemoryAnalyzer.analyze_memory_pattern(**inputs)


def test_ch17_b200_workflow_refuses_cpu_fallback(monkeypatch) -> None:
    chapter = importlib.import_module("ch17.blackwell_profiling_guide")
    monkeypatch.setattr(chapter.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"SKIPPED:.*requires.*NVIDIA B200"):
        chapter.require_reviewed_b200_device()


@pytest.mark.parametrize(
    ("name", "major", "minor"),
    [
        ("NVIDIA H100", 9, 0),
        ("NVIDIA GB200", 10, 0),
        ("NVIDIA B300", 10, 3),
        ("NVIDIA B200", 10, 3),
    ],
)
def test_ch17_b200_workflow_refuses_unreviewed_cuda_device(
    monkeypatch,
    name,
    major,
    minor,
) -> None:
    chapter = importlib.import_module("ch17.blackwell_profiling_guide")
    monkeypatch.setattr(chapter.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(chapter.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        chapter.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name=name, major=major, minor=minor),
    )

    with pytest.raises(RuntimeError, match="does not match the reviewed NVIDIA B200"):
        chapter.require_reviewed_b200_device()


def test_ch17_b200_device_gate_accepts_exact_reviewed_profile(monkeypatch) -> None:
    chapter = importlib.import_module("ch17.blackwell_profiling_guide")
    acquired = []
    synchronized = []
    monkeypatch.setattr(chapter.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(chapter.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        chapter.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(name="NVIDIA B200", major=10, minor=0),
    )
    monkeypatch.setattr(
        chapter.torch,
        "empty",
        lambda size, *, device: acquired.append((size, device)),
    )
    monkeypatch.setattr(
        chapter.torch.cuda,
        "synchronize",
        lambda device: synchronized.append(device),
    )

    device = chapter.require_reviewed_b200_device()

    assert device == chapter.torch.device("cuda:2")
    assert acquired == [(1, device)]
    assert synchronized == [device]


def test_ch17_b200_workflow_rejects_cpu_workload_before_writing_artifacts(
    monkeypatch,
    tmp_path,
) -> None:
    chapter = importlib.import_module("ch17.blackwell_profiling_guide")
    expected_device = chapter.torch.device("cuda:0")
    monkeypatch.setattr(
        chapter,
        "require_reviewed_b200_device",
        lambda: expected_device,
    )
    output_dir = tmp_path / "profiling"
    model = chapter.torch.nn.Linear(2, 2)
    input_tensor = chapter.torch.ones(1, 2)

    with pytest.raises(RuntimeError, match="requires input_tensor on cuda:0"):
        chapter.complete_profiling_workflow(
            model,
            input_tensor,
            output_dir=str(output_dir),
        )
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("kernel_time_ms", "flops", "bytes_transferred"),
    [
        (math.nan, 1.0, 1.0),
        (math.inf, 1.0, 1.0),
        (1.0, math.nan, 1.0),
        (1.0, math.inf, 1.0),
        (1.0, 1.0, math.nan),
        (1.0, 1.0, math.inf),
    ],
)
def test_roofline_rejects_nonfinite_measurements(
    kernel_time_ms,
    flops,
    bytes_transferred,
) -> None:
    roofline = importlib.import_module("core.analysis.kernel_roofline")
    analyzer = roofline.RooflineAnalyzer(roofline.get_architecture_specs_for_profile("b200"))

    with pytest.raises(ValueError, match="finite"):
        analyzer.analyze_kernel(kernel_time_ms, flops, bytes_transferred)


def test_roofline_rejects_unknown_precision() -> None:
    roofline = importlib.import_module("core.analysis.kernel_roofline")
    analyzer = roofline.RooflineAnalyzer(roofline.get_architecture_specs_for_profile("b200"))

    with pytest.raises(ValueError, match="Unknown precision"):
        analyzer.analyze_kernel(1.0, 1.0, 1.0, precision="fp64")


@pytest.mark.parametrize("invalid_peak", [math.nan, math.inf, 0.0, -1.0])
def test_roofline_rejects_invalid_selected_peak(invalid_peak) -> None:
    roofline = importlib.import_module("core.analysis.kernel_roofline")
    specs = roofline.ArchitectureSpecs(
        name="invalid fixture",
        peak_fp32_tflops=invalid_peak,
        peak_fp16_tflops=1.0,
        peak_fp8_tflops=1.0,
        peak_tf32_tflops=1.0,
        memory_bandwidth_gbs=1.0,
    )

    with pytest.raises(ValueError, match="positive peak"):
        roofline.RooflineAnalyzer(specs).analyze_kernel(1.0, 1.0, 1.0)


@pytest.mark.parametrize("invalid_bandwidth", [math.nan, math.inf, 0.0, -1.0])
def test_roofline_rejects_invalid_peak_bandwidth(invalid_bandwidth) -> None:
    roofline = importlib.import_module("core.analysis.kernel_roofline")
    specs = roofline.ArchitectureSpecs(
        name="invalid fixture",
        peak_fp32_tflops=1.0,
        peak_fp16_tflops=1.0,
        peak_fp8_tflops=1.0,
        peak_tf32_tflops=1.0,
        memory_bandwidth_gbs=invalid_bandwidth,
    )

    with pytest.raises(ValueError, match="positive and finite"):
        roofline.RooflineAnalyzer(specs).analyze_kernel(1.0, 1.0, 1.0)
