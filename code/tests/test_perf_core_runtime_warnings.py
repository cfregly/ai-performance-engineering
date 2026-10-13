from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.perf_core_base import PerformanceCoreBase


def _core(tmp_path: Path) -> PerformanceCoreBase:
    return PerformanceCoreBase(bench_root=tmp_path)


def test_get_gpu_info_surfaces_torch_runtime_warning(tmp_path: Path) -> None:
    csv_row = (
        "Fake GPU, 45, 50, 200, 300, 1000, 80000, 30, 40, 1230, 1593, 50, "
        "Enabled, P0, Enabled, 550.54"
    )
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.4"),
        cuda=SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(RuntimeError("torch boom"))),
    )

    with patch(
        "core.perf_core_base.subprocess.run",
        return_value=subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout=csv_row, stderr=""),
    ), patch.dict(sys.modules, {"torch": fake_torch}):
        result = _core(tmp_path).get_gpu_info()

    assert result["live"] is True
    assert result["cuda_version"] == "12.4"
    assert result["compute_capability"] is None
    assert any("Failed to query torch CUDA runtime" in warning for warning in result["warnings"])


def test_get_gpu_info_surfaces_explicit_nvidia_smi_failure(tmp_path: Path) -> None:
    with patch("core.perf_core_base.subprocess.run", side_effect=OSError("nvidia-smi missing")):
        result = _core(tmp_path).get_gpu_info()

    assert result["live"] is False
    assert "nvidia-smi missing" in result["error"]


def test_get_gpu_topology_surfaces_torch_warning(tmp_path: Path) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: (_ for _ in ()).throw(RuntimeError("device count failed")),
        )
    )

    def _fake_run(cmd, *args, **kwargs):
        if "--query-gpu=index,name,uuid,pci.bus_id" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="0, Fake GPU 0, UUID0, 0000:01:00.0\n1, Fake GPU 1, UUID1, 0000:02:00.0\n",
                stderr="",
            )
        if cmd == ["nvidia-smi", "topo", "-m"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="GPU0\tX\tNV1\nGPU1\tNV1\tX\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("core.perf_core_base.subprocess.run", side_effect=_fake_run), patch.dict(
        sys.modules,
        {"torch": fake_torch},
    ):
        result = _core(tmp_path).get_gpu_topology()

    assert result["gpu_count"] == 2
    assert any("Failed to query torch peer access matrix" in warning for warning in result["warnings"])


def test_get_nvlink_status_surfaces_warning(tmp_path: Path) -> None:
    with patch("core.perf_core_base.subprocess.run", side_effect=OSError("nvlink unavailable")):
        result = _core(tmp_path).get_nvlink_status()

    assert result["available"] is False
    assert any("Failed to query NVLink status" in warning for warning in result["warnings"])


def test_get_software_info_surfaces_runtime_warnings(tmp_path: Path) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.7.0",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(RuntimeError("cuda unavailable"))),
    )

    with patch("core.perf_core_base.subprocess.run", side_effect=OSError("nvidia-smi missing")), patch.dict(
        sys.modules,
        {"torch": fake_torch},
    ):
        result = _core(tmp_path).get_software_info()

    assert result["python"] == sys.version.split()[0]
    assert any("Failed to query torch software details" in warning for warning in result["warnings"])
    assert any("Failed to query NVIDIA driver details" in warning for warning in result["warnings"])


def test_get_dependency_health_surfaces_parse_and_symlink_warnings(tmp_path: Path) -> None:
    cutlass_root = tmp_path / "third_party" / "cutlass" / "include" / "cutlass"
    cutlass_root.mkdir(parents=True)
    (cutlass_root / "version.h").write_text("#define CUTLASS_MAJOR ???\n", encoding="utf-8")

    te_root = tmp_path / "transformer_engine"
    cutlass_link = te_root / "csrc" / "cutlass"
    cutlass_link.parent.mkdir(parents=True)
    cutlass_link.mkdir()

    original_resolve = Path.resolve

    def _resolve_with_failure(path_self: Path, *args, **kwargs):
        if path_self == cutlass_link:
            raise RuntimeError("symlink loop")
        return original_resolve(path_self, *args, **kwargs)

    with patch("core.perf_core_base.CODE_ROOT", tmp_path), patch(
        "core.perf_core_base._package_root",
        return_value=te_root,
    ), patch("core.perf_core_base._safe_package_version", return_value="1.0.0"), patch.object(
        Path,
        "resolve",
        _resolve_with_failure,
    ):
        result = _core(tmp_path).get_dependency_health()

    assert any("Failed to parse CUTLASS version" in warning for warning in result["warnings"])
    assert any("Failed to resolve transformer_engine CUTLASS symlink" in warning for warning in result["warnings"])


def test_get_cpu_memory_analysis_surfaces_optional_tool_warnings(tmp_path: Path) -> None:
    def _fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] in {"numactl", "cpuid"}:
            raise FileNotFoundError(cmd[0])
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("core.perf_core_base.subprocess.run", side_effect=_fake_run):
        result = _core(tmp_path).get_cpu_memory_analysis()

    assert any("numactl unavailable or failed" in warning for warning in result["warnings"])
    assert any("cpuid unavailable or failed" in warning for warning in result["warnings"])
