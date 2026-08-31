"""Real arithmetic, artifact compatibility, and explicitly gated GPU execution.

CPU payload fixtures never certify bandwidth, device identity, or GPU kernels.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os

import pytest
import torch

from core.benchmark.metrics import (
    BLACKWELL_B200, HOPPER_H100, DEFAULT_SPECS, compute_copy_bandwidth_metrics,
    compute_roofline_metrics, hardware_specs_for_device,
)
from core.benchmark.performance_targets import _build_targets, _get_peak_values
from core.harness.arch_config import ArchitectureConfig


def test_local_copy_counts_reads_and_writes_and_actual_iterations():
    result = compute_copy_bandwidth_metrics(2**30, 5, 5000)
    assert result["peak_bandwidth_gbs"] == pytest.approx(2.147483648)
    assert result["bytes_per_iteration"] == 2**31
    assert result["total_bytes"] == 5 * 2**31
    assert result["peak_bandwidth_tbs"] == pytest.approx(0.002147483648)


def test_peer_copy_counts_payload_once_and_uses_decimal_gigabytes():
    result = compute_copy_bandwidth_metrics(2**30, 20, 20000, traffic="one_way_payload")
    assert result["peak_bandwidth_gbs"] == pytest.approx(1.073741824)
    assert result["bytes_per_iteration"] == 2**30
    assert result["byte_accounting"] == "one_way_payload"


@pytest.mark.parametrize("kwargs", [
    {"payload_bytes": 0}, {"payload_bytes": -1}, {"payload_bytes": True},
    {"iterations": 0}, {"iterations": -1}, {"iterations": 1.5},
    {"elapsed_ms": 0}, {"elapsed_ms": -1}, {"elapsed_ms": float("nan")},
    {"elapsed_ms": True}, {"elapsed_ms": "100"},
    {"elapsed_ms": float("inf")}, {"traffic": "ambiguous"},
])
def test_invalid_bandwidth_inputs_cannot_generate_targets(kwargs):
    inputs = {"payload_bytes": 1024, "iterations": 10, "elapsed_ms": 100}
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        compute_copy_bandwidth_metrics(**inputs)


def test_corrected_fp8_peak_changes_roofline_classification():
    result = compute_roofline_metrics(4e12, 1e10, None, precision="fp8", specs=BLACKWELL_B200)
    assert result["roofline.arithmetic_intensity"] == 400
    assert result["roofline.ridge_point"] == 562.5
    assert result["roofline.is_compute_bound"] == 0
    result = compute_roofline_metrics(2.25e12, 2.25e9, 1, precision="fp8", specs=BLACKWELL_B200)
    assert result["roofline.is_compute_bound"] == 1
    assert result["roofline.efficiency_pct"] == 50


def test_memory_roofline_is_a_hardware_ceiling_not_measured_throughput():
    first = compute_roofline_metrics(4e9, 1e9, 1, precision="fp8", specs=BLACKWELL_B200)
    slower = compute_roofline_metrics(4e9, 1e9, 2, precision="fp8", specs=BLACKWELL_B200)
    assert first["roofline.memory_ceiling_tflops"] == slower["roofline.memory_ceiling_tflops"] == 32
    assert first["roofline.achieved_tflops"] == 4
    assert slower["roofline.achieved_tflops"] == 2


def _write_peak(tmp_path, hbm):
    path = tmp_path / "benchmark_peak_results_fixture.json"
    path.write_text(json.dumps({"hbm": hbm}), encoding="utf-8")
    return path


def test_legacy_half_counted_artifact_is_preserved_and_rejected(tmp_path):
    path = _write_peak(tmp_path, {"peak_bandwidth_tbs": 3.5})
    original = path.read_bytes()
    targets, warnings, artifact, source = _build_targets(tmp_path)
    assert path.read_bytes() == original
    assert artifact == str(path)
    assert source == "defaults_due_to_peak_results_warning"
    assert any("Rejected HBM" in warning and "provenance" in warning for warning in warnings)
    assert targets["overall"]["hbm3e_bandwidth_tbs"]["target"] == 7
    # This chapter target's old provenance is not established; do not double it.
    assert targets["ch02"]["metrics"]["hbm3e_bandwidth_tbs"]["target"] == 5500


def test_new_artifact_uses_read_write_units_through_real_target_loader(tmp_path):
    # Fixture arithmetic: 1e9 bytes read + written, 7 iterations, 2ms -> 7 TB/s.
    payload = compute_copy_bandwidth_metrics(10**9, 7, 2)
    path = _write_peak(tmp_path, payload)
    original = path.read_bytes()
    targets, warnings, artifact, source = _build_targets(tmp_path)
    assert not warnings
    assert source == "measured_peak_results"
    assert artifact == str(path) and path.read_bytes() == original
    target = targets["overall"]["hbm3e_bandwidth_tbs"]
    assert target["target"] == 7000 and target["min"] == 5950
    assert target["unit"] == "GB/s"


@pytest.mark.parametrize("key,value", [
    ("accounting_version", 1), ("byte_accounting", "one_way_payload"),
    ("bandwidth_unit", "GiB/s"), ("payload_bytes", 0),
    ("total_bytes", 1), ("peak_bandwidth_tbs", 3.5),
    ("peak_bandwidth_gbs", float("nan")), ("elapsed_ms", float("inf")),
])
def test_relabeling_old_or_inconsistent_values_does_not_bypass_provenance(tmp_path, key, value):
    payload = compute_copy_bandwidth_metrics(10**9, 7, 2)
    payload[key] = value
    _write_peak(tmp_path, payload)
    peaks, warnings, _, _ = _get_peak_values(tmp_path)
    assert "hbm_bandwidth_tbs" not in peaks
    assert any("Rejected HBM" in warning for warning in warnings)


@pytest.mark.parametrize("name,cc,expected", [
    ("NVIDIA B200", (10, 0), BLACKWELL_B200),
    ("NVIDIA H100 80GB HBM3", (9, 0), HOPPER_H100),
    ("NVIDIA H100-SXM5-80GB", (9, 0), HOPPER_H100),
])
def test_only_known_observed_sku_and_capability_select_profile(name, cc, expected):
    assert hardware_specs_for_device(name, cc) is expected


@pytest.mark.parametrize("name,cc", [
    ("NVIDIA GB200", (10, 0)), ("NVIDIA GB300", (10, 3)),
    ("NVIDIA B300", (10, 3)), ("NVIDIA GB10", (12, 1)),
    ("NVIDIA GeForce RTX 5090", (12, 0)), ("NVIDIA H200", (9, 0)),
    ("NVIDIA H100 PCIe", (9, 0)), ("NVIDIA H100 NVL", (9, 0)),
    ("NVIDIA B200", (12, 2)), ("Future GPU", (13, 0)),
])
def test_unknown_sku_never_inherits_b200_or_h100_peaks(name, cc):
    with pytest.raises(ValueError, match="supply explicit HardwareSpecs"):
        hardware_specs_for_device(name, cc)


def test_cpu_default_is_labeled_as_assumed():
    assert DEFAULT_SPECS.profile_source == "assumed_static_no_cuda"
    assert "assumed static" in DEFAULT_SPECS.name


@pytest.mark.parametrize("cc,arch,sm,tcgen", [
    ((10, 0), "blackwell", "sm_100", True),
    ((10, 3), "blackwell_ultra", "sm_103", True),
    ((12, 0), "blackwell_consumer", "sm_120", False),
    ((12, 1), "grace_blackwell", "sm_121", False),
    ((11, 0), "other", "sm_110", False),
    ((12, 2), "other", "sm_122", False),
    ((13, 0), "other", "sm_130", False),
])
def test_architecture_metadata_preserves_exact_capability(cc, arch, sm, tcgen):
    metadata = ArchitectureConfig.metadata_for_capability(cc)
    assert metadata["architecture"] == arch
    assert metadata["sm_version"] == sm
    assert metadata["compute_capability"] == f"{cc[0]}.{cc[1]}"
    assert metadata["tcgen05_supported"] is tcgen
    assert metadata["runtime_qualified"] is False
    assert not any("HBM" in feature or "C2C" in feature or "coherence" in feature for feature in metadata["features"])


def test_target_configuration_preserves_121_and_explicit_user_values(monkeypatch):
    # Control-plane configuration fixture only; no CUDA execution/detection mocked.
    config = ArchitectureConfig.__new__(ArchitectureConfig)
    config.compute_capability, config.arch = (12, 1), "grace_blackwell"
    config.config = ArchitectureConfig.metadata_for_capability((12, 1))
    for key in ("TORCH_CUDA_ARCH_LIST", "CMAKE_CUDA_ARCHITECTURES", "CUDAARCHS"):
        monkeypatch.delenv(key, raising=False)
    config._configure_arch_environment()
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "12.1"
    assert os.environ["CMAKE_CUDA_ARCHITECTURES"] == "121"
    assert os.environ["CUDAARCHS"] == "121"
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "12.1a")
    config._configure_arch_environment()
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "12.1a"
    assert config._sanitize_arch_value("sm_121a") == "sm_121a"
    with pytest.raises(RuntimeError, match="tcgen05 is unsupported for sm_121"):
        config.require_tcgen05()


def test_peak_import_does_not_require_execution_dependency_on_cpu():
    from core.benchmark import benchmark_peak
    if importlib.util.find_spec("transformer_engine") is not None:
        pytest.skip("Negative dependency control requires actually absent Transformer Engine")
    with pytest.raises(RuntimeError, match="Transformer Engine is REQUIRED"):
        benchmark_peak._make_low_precision_recipe("fp4")
    assert benchmark_peak.TE_AVAILABLE is False


def test_peak_never_installs_ptx_target_rewriting_shim():
    from core.benchmark import benchmark_peak
    with pytest.raises(RuntimeError, match="PTX target rewriting is unsupported"):
        benchmark_peak._install_ptxas_wrapper()


def test_real_cpu_fp8_operand_preparation_uses_cast_and_column_major_b():
    from core.diagnostics.microbench import _prepare_fp8_matmul
    a, b, _ = _prepare_fp8_matmul(32, torch.device("cpu"))
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a.stride() == (32, 1) and b.stride() == (1, 32)
    assert torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()
    assert torch.count_nonzero(a.float()) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA GPU required for FP8 scaled-mm and complete output comparison")
def test_real_gpu_fp8_scaled_mm_and_diagnostic_entrypoint():
    from core.diagnostics.microbench import _prepare_fp8_matmul, tensor_core_bench
    from core.utils.compile_utils import tf32_override
    a, b, run = _prepare_fp8_matmul(64, torch.device("cuda"))
    actual = run()
    with tf32_override(enable_matmul=False):
        expected = (a.float() @ b.float()).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.125)
    result = tensor_core_bench(size=64, precision="fp8", iters=2, warmup=1)
    assert "error" not in result
    assert result["matmul_api"] == "torch._scaled_mm"
    assert result["operand_dtype"] == "torch.float8_e4m3fn"
    assert result["placeholder_used"] is False
    assert result["tflops"] > 0 and math.isfinite(result["tflops"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA GPU and Transformer Engine NVFP4 kernel required")
def test_real_gpu_fp4_recipe_and_measurement():
    from core.benchmark import benchmark_peak
    recipe = benchmark_peak._make_low_precision_recipe("fp4")
    assert type(recipe).__name__ == "NVFP4BlockScaling"
    result = benchmark_peak.measure_fp4_compute(matrix_size=256, iterations=2)
    assert result["precision"] == "fp4" and result["recipe"] == "NVFP4BlockScaling"
    assert result["peak_tflops"] > 0 and math.isfinite(result["peak_tflops"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA GPU required for copy timing")
def test_real_gpu_copy_measurements_emit_consistent_provenance():
    from core.benchmark import benchmark_peak
    result = benchmark_peak.measure_hbm_bandwidth(size_gb=0.0625, iterations=2)
    assert result["payload_bytes"] == 64 * 1024**2
    assert result["bytes_per_iteration"] == 128 * 1024**2
    assert result["peak_bandwidth_gbs"] == pytest.approx(result["total_bytes"] / (result["elapsed_ms"] / 1000) / 1e9)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two actual CUDA GPUs required to test a noncurrent measurement device")
def test_real_gpu_noncurrent_device_copy_and_compute_timing():
    from core.benchmark import benchmark_peak
    with torch.cuda.device(0):
        target = torch.device("cuda:1")
        hbm = benchmark_peak.measure_hbm_bandwidth(device=target, size_gb=0.0625, iterations=2)
        compute = benchmark_peak.measure_fp16_compute(device=target, matrix_size=256, iterations=2)
        assert hbm["gpu_name"] == torch.cuda.get_device_name(1)
        assert hbm["elapsed_ms"] > 0 and math.isfinite(hbm["elapsed_ms"])
        assert compute["peak_tflops"] > 0 and math.isfinite(compute["peak_tflops"])
        l2 = benchmark_peak.measure_l2_cache_bandwidth(device=target, size_mb=8, iterations=2)
        if getattr(torch.cuda.get_device_properties(1), "l2_cache_size", 0):
            assert l2["elapsed_ms"] > 0
        else:
            assert l2["peak_bandwidth_gbs"] is None and "size unknown" in l2["error"]
        assert torch.cuda.current_device() == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two actual CUDA GPUs required for peer-copy timing and byte accounting")
def test_real_gpu_peer_copy_reports_payload_bytes_not_nvlink_qualification():
    from core.benchmark import benchmark_peak
    if not torch.cuda.can_device_access_peer(0, 1):
        pytest.skip("Actual GPU topology does not support peer access from GPU0 to GPU1")
    with torch.cuda.device(1):
        result = benchmark_peak.measure_nvlink_bandwidth(iterations=2, size_mb=16)
        assert result["payload_bytes"] == 16 * 1024**2
        assert result["bytes_per_iteration"] == result["payload_bytes"]
        assert result["total_bytes"] == 2 * result["payload_bytes"]
        assert result["transport"] == "cuda_peer_copy" and result["nvlink_verified"] is False
        assert result["peak_bandwidth_gbs"] == pytest.approx(result["total_bytes"] / (result["elapsed_ms"] / 1000) / 1e9)
        assert torch.cuda.current_device() == 1
