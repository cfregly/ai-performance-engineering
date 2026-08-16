from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import ch03.bind_numa_affinity as bind_numa_affinity
import ch04.gb200_grace_numa_optimization as grace_numa
import core.optimization.parallelism_planner.topology_detector as planner_topology_detector
import core.optimization.parallelism_planner.advisor as planner_advisor
import core.optimization.parallelism_planner.cluster_config as planner_cluster_config
import labs.dynamic_router.topology as dynamic_router_topology
import labs.dynamic_router.topology_probe as dynamic_router_topology_probe
import labs.dynamic_router.driver as dynamic_router_driver


def _configure_bind_nvml(monkeypatch: pytest.MonkeyPatch, fake_nvml: object) -> None:
    monkeypatch.setattr(bind_numa_affinity, "_HAS_NVML", True)
    monkeypatch.setattr(bind_numa_affinity, "_NVML_INIT_DONE", True)
    monkeypatch.setattr(bind_numa_affinity, "_ensure_nvml_initialized", lambda: None)
    monkeypatch.setattr(bind_numa_affinity, "_gpu_pci_bus", lambda _: "0000:17:00.0")
    monkeypatch.setattr(bind_numa_affinity, "nvml", fake_nvml, raising=False)


def test_bind_numa_affinity_prefers_explicit_nvml_numa_node(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nvml = SimpleNamespace(
        nvmlDeviceGetHandleByPciBusId_v2=lambda pci: "handle",
        nvmlDeviceGetNumaNodeId=lambda handle: 3,
    )

    _configure_bind_nvml(monkeypatch, fake_nvml)

    assert bind_numa_affinity._gpu_node_from_nvml(0) == 3


def test_bind_numa_affinity_formats_pci_bus_id_when_torch_exposes_integer_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bind_numa_affinity, "_libcudart", None)
    monkeypatch.setattr(
        bind_numa_affinity.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(pci_bus_id=5, pci_domain_id=0, pci_device_id=0),
    )

    assert bind_numa_affinity._gpu_pci_bus(0) == "0000:05:00.0"


def test_bind_numa_affinity_returns_none_when_explicit_nvml_numa_query_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NotSupported(RuntimeError):
        pass

    fake_nvml = SimpleNamespace(
        nvmlDeviceGetHandleByPciBusId_v2=lambda pci: "handle",
        nvmlDeviceGetNumaNodeId=lambda handle: (_ for _ in ()).throw(_NotSupported("not supported")),
        NVMLError_NotSupported=_NotSupported,
    )

    _configure_bind_nvml(monkeypatch, fake_nvml)

    assert bind_numa_affinity._gpu_node_from_nvml(0) is None


def test_bind_numa_affinity_does_not_guess_numa_node_from_process_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bind_numa_affinity, "_gpu_node_from_nvml", lambda _: None)
    monkeypatch.setattr(bind_numa_affinity, "_gpu_node_from_sysfs", lambda _: None)

    assert bind_numa_affinity.get_gpu_numa_node(0) is None


def test_dynamic_router_topology_uses_nvml_numa_node_api(monkeypatch: pytest.MonkeyPatch) -> None:
    shutdown_calls: list[bool] = []
    fake_pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda idx: "handle",
        nvmlDeviceGetPciInfo=lambda handle: SimpleNamespace(busId=b"00000000:17:00.0"),
        nvmlDeviceGetNumaNodeId=lambda handle: 4,
        nvmlShutdown=lambda: shutdown_calls.append(True),
    )

    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)

    assert dynamic_router_topology._nvml_gpu_bus_and_numa(max_gpus=1) == {
        0: {"bus_id": "00000000:17:00.0", "numa_node": 4}
    }
    assert shutdown_calls == [True]


def test_dynamic_router_topology_shuts_down_nvml_once_for_multiple_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls: list[bool] = []
    fake_pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda idx: f"handle{idx}",
        nvmlDeviceGetPciInfo=lambda handle: SimpleNamespace(busId=f"00000000:{17 + int(handle[-1]):02d}:00.0".encode()),
        nvmlDeviceGetNumaNodeId=lambda handle: None,
        nvmlShutdown=lambda: shutdown_calls.append(True),
    )

    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_pynvml)

    assert dynamic_router_topology._nvml_gpu_bus_and_numa(max_gpus=2) == {
        0: {"bus_id": "00000000:17:00.0", "numa_node": None},
        1: {"bus_id": "00000000:18:00.0", "numa_node": None},
    }
    assert shutdown_calls == [True]


def test_dynamic_router_read_int_normalizes_negative_numa_to_none(tmp_path: Path) -> None:
    numa_node = tmp_path / "numa_node"
    numa_node.write_text("-1", encoding="utf-8")

    assert dynamic_router_topology._read_int(numa_node) is None


def test_dynamic_router_detect_topology_preserves_gpu_slots_without_nvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dynamic_router_topology, "_nvml_gpu_bus_and_numa", lambda max_gpus=None: {})
    monkeypatch.setattr(dynamic_router_topology, "_distance_matrix", lambda: {})

    snapshot = dynamic_router_topology.detect_topology(max_gpus=2)

    assert snapshot.gpu_numa == {0: None, 1: None}
    assert snapshot.gpu_numa_status == "unknown"


def test_dynamic_router_topology_probe_reports_explicit_gpu_numa_status_metrics() -> None:
    bench = dynamic_router_topology_probe.TopologyProbeBenchmark.__new__(
        dynamic_router_topology_probe.TopologyProbeBenchmark
    )
    bench.snapshot = dynamic_router_topology.TopologySnapshot(
        gpu_numa={0: None, 1: 2},
        distance={0: [10, 20], 1: [20, 10]},
        timestamp=0.0,
        gpu_numa_status="partial",
    )

    metrics = bench.get_custom_metrics()

    assert metrics is not None
    assert metrics["num_gpus_detected"] == 2.0
    assert metrics["gpus_with_known_numa"] == 1.0
    assert metrics["host_numa_nodes_detected"] == 2.0
    assert metrics["gpu_numa_status_partial"] == 1.0
    assert metrics["gpu_numa_status_unknown"] == 0.0
    assert "numa_nodes_known" not in metrics


def test_planner_topology_detector_ignores_negative_sysfs_numa_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = planner_topology_detector.TopologyDetector()

    monkeypatch.setattr(
        planner_topology_detector.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="00000000:17:00.0\n"),
    )

    class _FakePath:
        def __init__(self, path: str) -> None:
            self.path = path

        def exists(self) -> bool:
            return True

        def read_text(self) -> str:
            return "-1"

    monkeypatch.setattr(planner_topology_detector, "Path", _FakePath)

    assert detector._get_gpu_numa_node(0) is None


def test_planner_topology_detector_reports_unknown_gpu_numa_status_when_no_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = planner_topology_detector.TopologyDetector()
    monkeypatch.setattr(detector, "_get_gpu_numa_node", lambda gpu_index: None)

    numa_nodes, gpu_numa_mapping, numa_distance, gpu_numa_status = detector._detect_numa(2)

    assert numa_nodes == 0
    assert gpu_numa_mapping == {}
    assert numa_distance == []
    assert gpu_numa_status == "unknown"


def test_planner_mock_topology_marks_gpu_numa_as_synthetic() -> None:
    preset = planner_advisor.create_mock_topology_b200_multigpu(num_gpus=4)
    topology = preset.materialize()

    assert isinstance(preset, planner_topology_detector.SyntheticTopologyPreset)
    assert topology.gpu_numa_status == "synthetic"
    assert topology.topology_source == "synthetic_preset"
    assert topology.topology_preset_name == "mock_b200_multigpu"


def test_cluster_detector_preserves_unknown_numa_for_detected_node() -> None:
    detector = planner_cluster_config.ClusterDetector()
    topology = planner_advisor.create_mock_topology_b200_multigpu(num_gpus=4).materialize()
    topology.numa_nodes = 0
    topology.gpu_numa_mapping = {}
    topology.gpu_numa_status = "unknown"

    nodes = detector._build_node_specs(num_nodes=1, gpus_per_node=4, topology=topology)

    assert len(nodes) == 1
    assert nodes[0].numa_nodes is None


def test_parallelism_advisor_materializes_synthetic_preset_before_use() -> None:
    advisor = planner_advisor.ParallelismAdvisor(auto_detect_topology=False)
    preset = planner_advisor.create_mock_topology_h100_multigpu(num_gpus=4)

    advisor.set_topology(preset)

    assert advisor.topology is not None
    assert advisor.topology.topology_source == "synthetic_preset"
    assert advisor.topology.topology_preset_name == "mock_h100_multigpu"


def test_setup_grace_affinity_does_not_guess_numa_node_when_mapping_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sched_calls: list[tuple[int, list[int]]] = []

    monkeypatch.setattr(
        grace_numa,
        "detect_grace_cpu",
        lambda: {"is_grace": True, "cpu_arch": "aarch64", "cpu_count": 72, "cpu_threads": 144, "gpus": 1},
    )
    monkeypatch.setattr(
        grace_numa,
        "get_numa_topology",
        lambda: {0: {"cpus": [0, 1, 2, 3], "size_gb": 240, "gpus": []}},
    )
    monkeypatch.setattr(
        grace_numa.os,
        "sched_getaffinity",
        lambda pid: {4, 5, 6, 7},
        raising=False,
    )
    monkeypatch.setattr(
        grace_numa.os,
        "sched_setaffinity",
        lambda pid, cpus: sched_calls.append((pid, list(cpus))),
        raising=False,
    )

    cpu_list, numa_node = grace_numa.setup_grace_affinity(gpu_id=0, num_workers=4, verbose=False)

    assert cpu_list == [4, 5, 6, 7]
    assert numa_node is None
    assert sched_calls == []


def test_dynamic_router_virtual_gpu_defaults_numa_to_unknown() -> None:
    gpu = dynamic_router_driver.VirtualGPU(
        "gpu0",
        is_prefill=True,
        is_decode=False,
        prefill_rate=1000.0,
        decode_rate=500.0,
        hourly_cost=1.0,
    )

    assert gpu.numa_node is None
