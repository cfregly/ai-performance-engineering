"""W2-004 regressions: SM 12.x must not be labeled a Grace-Blackwell GPU.

The planner topology detector used to map every compute capability with
major >= 12 to ``grace_blackwell``. That labeled x86 RTX PRO and GeForce
RTX 50 GPUs (SM 12.0) and every future major as GB200/GB300-class parts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.optimization.parallelism_planner.topology_detector as topology_detector


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 0), "ampere"),
        ((9, 0), "hopper"),
        ((10, 0), "blackwell"),
        ((10, 3), "blackwell"),
        ((11, 0), "sm_110"),
        ((12, 0), "blackwell_sm12x"),
        ((12, 1), "blackwell_sm12x"),
        ((13, 0), "sm_130"),
        ((7, 5), "sm_75"),
    ],
)
def test_gpu_architecture_label_uses_exact_capability(
    capability: tuple[int, int], expected: str
) -> None:
    major, minor = capability
    label = topology_detector.gpu_architecture_for_capability(major, minor)

    assert label == expected
    assert "grace" not in label


def _no_external_tools(*args: object, **kwargs: object) -> object:
    raise FileNotFoundError(args[0][0] if args and args[0] else "tool")


def _detect_with_device(
    monkeypatch: pytest.MonkeyPatch,
    *,
    major: int,
    minor: int,
    name: str,
    num_gpus: int,
) -> tuple[topology_detector.TopologyDetector, topology_detector.TopologyInfo]:
    props = SimpleNamespace(
        major=major,
        minor=minor,
        name=name,
        total_memory=96 * 1024**3,
        multi_processor_count=188,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: num_gpus,
            get_device_properties=lambda index: props,
            can_device_access_peer=lambda a, b: False,
        )
    )
    monkeypatch.setattr(topology_detector, "torch", fake_torch)
    monkeypatch.setattr(topology_detector.subprocess, "run", _no_external_tools)
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    detector = topology_detector.TopologyDetector()
    return detector, detector.detect(force_refresh=True)


def test_detect_reports_sm120_workstation_gpu_without_grace_blackwell_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, topology = _detect_with_device(
        monkeypatch,
        major=12,
        minor=0,
        name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        num_gpus=2,
    )

    assert [gpu.architecture for gpu in topology.gpus] == ["blackwell_sm12x"] * 2
    assert topology.is_grace_cpu is False
    assert topology.has_nvlink_c2c is False

    exported = topology.to_dict()
    assert {gpu["architecture"] for gpu in exported["gpus"]} == {"blackwell_sm12x"}

    report = detector.format_topology_report(topology)
    assert "Architecture: blackwell_sm12x" in report
    assert "grace_blackwell" not in report
    assert "NVLink-C2C: No" in report


def test_detect_keeps_datacenter_blackwell_label_for_b200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, topology = _detect_with_device(
        monkeypatch,
        major=10,
        minor=0,
        name="NVIDIA B200",
        num_gpus=1,
    )

    assert topology.gpus[0].architecture == "blackwell"
    assert topology.gpus[0].compute_capability == "10.0"
    assert topology.is_grace_cpu is False
    assert topology.has_nvlink_c2c is False


def test_detect_does_not_assign_a_family_to_an_unlisted_future_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, topology = _detect_with_device(
        monkeypatch,
        major=13,
        minor=0,
        name="Future NVIDIA GPU",
        num_gpus=1,
    )

    assert topology.gpus[0].architecture == "sm_130"
