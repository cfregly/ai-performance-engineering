from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.diagnostics import microbench
from core.scripts.utilities import probe_hardware_capabilities as capability_probe
from monitoring.cluster_monitor import ClusterAggregator, NodeMetrics, SystemCollector

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_gpu_memory_diagnostic_binds_state_and_work_to_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocated_devices: list[torch.device] = []
    captured_devices: list[int] = []

    class FakeTensor:
        def copy_(self, _source: object) -> FakeTensor:
            return self

    class FakeEvent:
        def record(self, _stream: object) -> None:
            return None

        def elapsed_time(self, _end: object) -> float:
            return 1.0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: FakeEvent())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(major=10, minor=0, name=f"device-{device}"),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: f"device-{device}")

    def fake_empty(*_args: object, device: torch.device, **_kwargs: object) -> FakeTensor:
        allocated_devices.append(device)
        return FakeTensor()

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch, "empty_like", lambda _source: FakeTensor())

    from core.benchmark import metrics as benchmark_metrics
    from core.harness import validity_checks

    monkeypatch.setattr(
        benchmark_metrics,
        "hardware_specs_for_device",
        lambda _name, _capability: SimpleNamespace(hbm_bandwidth_gbps=8_000.0),
    )
    monkeypatch.setattr(
        validity_checks,
        "capture_gpu_state",
        lambda device_index: captured_devices.append(device_index) or object(),
    )
    monkeypatch.setattr(microbench, "_gpu_state_payload", lambda _before, _after: {})

    result = microbench.gpu_memory_bandwidth_test(size_mb=1, iters=1, warmup=0)

    assert allocated_devices == [torch.device("cuda", 3)]
    assert captured_devices == [3, 3]
    assert result["gpu_name"] == "device-cuda:3"


def test_grace_host_detection_requires_explicit_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capability_probe.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        capability_probe,
        "_read_grace_host_identity",
        lambda: "arm neoverse v2 server",
    )
    assert capability_probe._is_grace_host() is False

    monkeypatch.setattr(
        capability_probe,
        "_read_grace_host_identity",
        lambda: "nvidia grace cpu superchip",
    )
    assert capability_probe._is_grace_host() is True

    monkeypatch.setattr(capability_probe.platform, "machine", lambda: "x86_64")
    assert capability_probe._is_grace_host() is False


def test_generated_arch_fallback_does_not_infer_grace_from_compute_capability() -> None:
    checked_in = (CODE_ROOT / "core/common/headers/arch_detection.cuh").read_text(
        encoding="utf-8"
    )
    generator = (
        CODE_ROOT / "core/scripts/utilities/generate_arch_detection_header.py"
    ).read_text(encoding="utf-8")

    for source in (checked_in, generator):
        assert "cached.has_grace_coherence = false;" in source
        assert "cached.has_grace_coherence = (props.major" not in source


def test_cpu_sample_read_and_prior_update_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = threading.Lock()
    active = 0
    max_active = 0
    sample_index = 0
    samples = [(100, 1_000), (140, 1_200)]

    def fake_read() -> tuple[int, int]:
        nonlocal active, max_active, sample_index
        with guard:
            active += 1
            max_active = max(max_active, active)
            index = sample_index
            sample_index += 1
        time.sleep(0.02)
        with guard:
            active -= 1
        return samples[index]

    monkeypatch.setattr(SystemCollector, "_previous_cpu_sample", None)
    monkeypatch.setattr(SystemCollector, "_read_cpu_sample", staticmethod(fake_read))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: SystemCollector.collect(), range(2)))

    assert max_active == 1
    assert {result["cpu_telemetry_status"] for result in results} == {"warming_up", "ok"}


def test_unknown_legacy_ib_zero_does_not_drive_cluster_recommendation() -> None:
    aggregator = ClusterAggregator()
    for node_id in range(5):
        aggregator.report_node_metrics(
            NodeMetrics(
                hostname=f"node-{node_id}",
                node_id=node_id,
                timestamp=1.0,
                ib_rx_gbps=0.0,
                ib_telemetry_status="unavailable",
            )
        )

    assert "Consider enabling InfiniBand for better scaling" not in (
        aggregator.aggregate().recommendations
    )


def test_grafana_memory_panel_uses_exported_device_wide_series() -> None:
    dashboard = json.loads(
        (CODE_ROOT / "monitoring/grafana_dashboard_template.json").read_text(
            encoding="utf-8"
        )
    )
    memory_panel = next(panel for panel in dashboard["panels"] if panel["id"] == 2)
    expressions = {target["expr"] for target in memory_panel["targets"]}

    assert expressions == {"gpu_memory_used_gb", "gpu_memory_total_gb"}
    assert "gpu_memory_allocated_gb" not in expressions
    assert "gpu_memory_reserved_gb" not in expressions
