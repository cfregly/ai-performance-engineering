from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from core.harness import hardware_capabilities
from core.scripts.utilities import probe_hardware_capabilities


def _b200_probe_payload() -> dict:
    return {
        "version": 1,
        "timestamp": 0.0,
        "devices": [
            {
                "device_index": 0,
                "name": "NVIDIA B200",
                "key": "sm_100",
                "architecture": "blackwell",
                "compute_capability": "10.0",
                "total_memory_gb": 180.0,
                "num_sms": 148,
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "max_threads_per_sm": 2048,
                "max_shared_mem_per_block": 49152,
                "max_shared_mem_per_sm": 233472,
                "l2_cache_bytes": 100663296,
                "features": ["HBM3e"],
                "tensor_cores": "5th Gen",
                "memory_bandwidth_tbps": None,
                "max_unified_memory_tb": None,
                "nvlink_c2c": False,
                "grace_coherence": False,
                "tma": {
                    "supported": True,
                    "compiler_support": True,
                    "max_1d": 1024,
                    "max_2d_width": 128,
                    "max_2d_height": 128,
                },
                "cluster": {
                    "supports_clusters": True,
                    "has_dsmem": True,
                    "max_cluster_size": 8,
                    "notes": None,
                },
                "notes": [],
                "driver_version": "580.126.09",
                "cuda_runtime_version": "13.0",
            }
        ],
    }


def _point_hardware_cache(monkeypatch, tmp_path: Path) -> Path:
    cache_path = tmp_path / "hardware_capabilities.json"
    probe_script = tmp_path / "probe_hardware_capabilities.py"
    probe_script.write_text("# fake probe script\n", encoding="utf-8")
    monkeypatch.setattr(hardware_capabilities, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(hardware_capabilities, "PROBE_FILE", cache_path)
    monkeypatch.setattr(hardware_capabilities, "PROBE_SCRIPT", probe_script)
    hardware_capabilities.refresh_capability_cache()
    return cache_path


def test_detect_capabilities_does_not_run_strict_probe_without_cuda(monkeypatch, tmp_path):
    cache_path = _point_hardware_cache(monkeypatch, tmp_path)
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(hardware_capabilities, "torch", fake_torch)

    def _unexpected_probe(*_args, **_kwargs):
        raise AssertionError("portable capability detection should not spawn the strict probe")

    monkeypatch.setattr(hardware_capabilities.subprocess, "run", _unexpected_probe)

    assert hardware_capabilities.detect_capabilities() is None
    assert not cache_path.exists()


def test_detect_capabilities_refreshes_empty_cache_when_cuda_is_available(monkeypatch, tmp_path):
    cache_path = _point_hardware_cache(monkeypatch, tmp_path)
    cache_path.write_text(json.dumps({"version": 1, "devices": []}), encoding="utf-8")
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setattr(hardware_capabilities, "torch", fake_torch)

    def _write_probe_cache(cmd, **kwargs):
        cache_path.write_text(json.dumps(_b200_probe_payload()), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(hardware_capabilities.subprocess, "run", _write_probe_cache)

    cap = hardware_capabilities.detect_capabilities()

    assert cap is not None
    assert cap.device_name == "NVIDIA B200"
    assert cap.sm_version == "sm_100"


def test_probe_allow_unavailable_skips_without_writing_cache(monkeypatch, tmp_path, capsys):
    cache_path = tmp_path / "hardware_capabilities.json"
    monkeypatch.setattr(probe_hardware_capabilities, "CACHE_PATH", cache_path)
    monkeypatch.setattr(probe_hardware_capabilities, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(probe_hardware_capabilities, "torch", None)

    probe_hardware_capabilities.main(["--allow-unavailable"])

    captured = capsys.readouterr()
    assert "SKIPPED: PyTorch is not installed" in captured.out
    assert not cache_path.exists()


def test_probe_strict_mode_fails_without_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(probe_hardware_capabilities, "CACHE_PATH", tmp_path / "hardware.json")
    monkeypatch.setattr(probe_hardware_capabilities, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(probe_hardware_capabilities, "torch", None)

    try:
        probe_hardware_capabilities.main([])
    except RuntimeError as exc:
        assert "PyTorch is not installed" in str(exc)
    else:
        raise AssertionError("strict probe should fail when PyTorch is unavailable")
