from __future__ import annotations

import io
import inspect
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import pytest

import ch02.baseline_grace_coherent_memory as baseline_grace_coherent_memory
import ch02.cpu_gpu_topology_aware as cpu_gpu_topology_aware
import ch02.optimized_grace_coherent_memory as optimized_grace_coherent_memory


def test_cpu_gpu_topology_detect_interconnect_uses_regex_for_link_speed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cpu_gpu_topology_aware.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="LnkCap: Port #0, Speed 32GT/s, Width x16"),
    )

    interconnect = cpu_gpu_topology_aware.detect_interconnect_type(
        {"is_grace": False},
        {"family": "Ampere", "nvlink_capable": False},
    )

    assert interconnect == "PCIe Gen5 (~128 GB/s)"


def test_cpu_gpu_topology_does_not_guess_numa_mapping_when_platform_does_not_expose_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpu_gpu_topology_aware, "detect_cpu_info", lambda: {"cpu_memory_gb": 512.0, "numa_nodes": 4, "is_grace": False, "cpu_type": "x86_64"})
    monkeypatch.setattr(cpu_gpu_topology_aware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpu_gpu_topology_aware.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        cpu_gpu_topology_aware.torch.cuda,
        "get_device_properties",
        lambda idx: SimpleNamespace(name=f"GPU{idx}", major=10, minor=0, total_memory=80 * 1024**3),
    )
    monkeypatch.setattr(cpu_gpu_topology_aware, "_get_gpu_numa_node", lambda idx: None)
    monkeypatch.setattr(cpu_gpu_topology_aware, "detect_interconnect_type", lambda cpu, gpu: "PCIe Gen5 (~128 GB/s)")

    topology = cpu_gpu_topology_aware.detect_system_topology()

    assert topology["numa_gpu_mapping"] == {}


def test_cpu_gpu_topology_leaves_cpu_numa_unknown_when_numactl_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpu_gpu_topology_aware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cpu_gpu_topology_aware.psutil, "cpu_count", lambda logical=True: 16 if logical else 8)
    monkeypatch.setattr(
        cpu_gpu_topology_aware.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=512 * 1024**3),
    )
    monkeypatch.setattr(
        cpu_gpu_topology_aware.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    with mock.patch("builtins.open", lambda *args, **kwargs: io.StringIO("Intel")):
        info = cpu_gpu_topology_aware.detect_cpu_info()

    assert info["numa_nodes"] is None


def test_optimized_grace_coherent_memory_skips_binding_when_gpu_numa_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    affinity_calls: list[tuple[int, list[int]]] = []
    instance = optimized_grace_coherent_memory.OptimizedGraceCoherentMemory.__new__(
        optimized_grace_coherent_memory.OptimizedGraceCoherentMemory
    )
    instance.is_grace_blackwell = True

    monkeypatch.setattr(optimized_grace_coherent_memory.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(optimized_grace_coherent_memory, "_gpu_numa_node_from_sysfs", lambda gpu_id: None)
    monkeypatch.setattr(
        optimized_grace_coherent_memory.os,
        "sched_setaffinity",
        lambda pid, cpus: affinity_calls.append((pid, list(cpus))),
    )

    instance._bind_numa_node()

    assert affinity_calls == []


def test_optimized_grace_coherent_memory_reads_negative_sysfs_numa_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    numa_node = tmp_path / "numa_node"
    numa_node.write_text("-1", encoding="utf-8")

    class _FakePath:
        def __init__(self, path: str) -> None:
            self.path = path

        def read_text(self, encoding: str = "utf-8") -> str:
            return numa_node.read_text(encoding=encoding)

    monkeypatch.setattr(optimized_grace_coherent_memory, "_query_gpu_pci_bus_id", lambda gpu_id: "0000:17:00.0")
    monkeypatch.setattr(optimized_grace_coherent_memory, "Path", _FakePath)

    assert optimized_grace_coherent_memory._gpu_numa_node_from_sysfs(0) is None


def test_grace_coherent_memory_baseline_fails_fast_without_grace_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import platform

    monkeypatch.setattr(baseline_grace_coherent_memory.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        baseline_grace_coherent_memory.torch.cuda,
        "get_device_properties",
        lambda idx: SimpleNamespace(major=12, minor=1),
    )
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match="SKIPPED: grace_coherent_memory requires Grace-Blackwell coherent memory support"):
        baseline_grace_coherent_memory.BaselineGraceCoherentMemory()


def test_grace_coherent_memory_optimized_fails_fast_without_grace_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import platform

    monkeypatch.setattr(optimized_grace_coherent_memory.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        optimized_grace_coherent_memory.torch.cuda,
        "get_device_properties",
        lambda idx: SimpleNamespace(major=12, minor=1),
    )
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match="SKIPPED: grace_coherent_memory requires Grace-Blackwell coherent memory support"):
        optimized_grace_coherent_memory.OptimizedGraceCoherentMemory()


def test_grace_coherent_memory_async_pinned_reuses_current_stream() -> None:
    source = inspect.getsource(optimized_grace_coherent_memory.OptimizedGraceCoherentMemory.run_step)
    async_section = source.split('elif self.strategy == "async_pinned":', maxsplit=1)[1].split(
        "torch.cuda.synchronize()",
        maxsplit=1,
    )[0]

    assert "current_stream = torch.cuda.current_stream()" in async_section
    assert "current_stream.wait_stream(self.stream)" in async_section
    assert "self.stream.wait_stream(current_stream)" in async_section
    assert "torch.cuda.current_stream().wait_stream(self.stream)" not in async_section
    assert "self.stream.wait_stream(torch.cuda.current_stream())" not in async_section
