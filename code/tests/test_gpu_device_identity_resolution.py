"""CPU control-plane tests for logical CUDA to NVML device identity mapping.

These tests exercise identifier selection only; they do not claim a successful
GPU query or benchmark run.
"""

from types import SimpleNamespace
import uuid

import pytest

from core.profiling.gpu_telemetry import normalize_gpu_uuid, resolve_nvml_device_handle


_DEVICE_UUID = "12345678-1234-5678-9abc-123456789abc"


@pytest.mark.parametrize(
    "value",
    [
        _DEVICE_UUID,
        _DEVICE_UUID.upper(),
        uuid.UUID(_DEVICE_UUID),
        f"GPU-{_DEVICE_UUID}",
        f"GPU-{_DEVICE_UUID}".encode("utf-8"),
    ],
)
def test_bare_cuda_and_prefixed_nvml_uuid_have_identical_identity(value: object) -> None:
    assert normalize_gpu_uuid(value) == f"GPU-{_DEVICE_UUID}"


@pytest.mark.parametrize("value", ["MIG-12345678-1234-5678-9abc-123456789abc", "MIG-GPU-example/1/0"])
def test_mig_identity_is_preserved(value: str) -> None:
    assert normalize_gpu_uuid(value) == value


def test_bare_cuda_uuid_uses_nvml_canonical_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    calls: list[tuple[str, object]] = []
    handle = resolve_nvml_device_handle(
        _recording_nvml(calls), 0, cuda_device_uuid=_DEVICE_UUID
    )
    assert handle == f"uuid:GPU-{_DEVICE_UUID}"
    assert calls == [("uuid", f"GPU-{_DEVICE_UUID}")]


def _recording_nvml(calls: list[tuple[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(
        nvmlDeviceGetHandleByUUID=lambda uuid: calls.append(("uuid", uuid)) or f"uuid:{uuid}",
        nvmlDeviceGetHandleByIndex=lambda index: calls.append(("index", index)) or f"index:{index}",
    )


@pytest.mark.parametrize("logical_index", [0, 1])
def test_exact_cuda_uuid_takes_precedence_over_visible_device_order(
    monkeypatch: pytest.MonkeyPatch,
    logical_index: int,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    calls: list[tuple[str, object]] = []
    nvml = _recording_nvml(calls)
    uuid = f"GPU-physical-{logical_index}"

    handle = resolve_nvml_device_handle(
        nvml,
        logical_index,
        cuda_device_uuid=uuid.encode("utf-8"),
    )

    assert handle == f"uuid:{uuid}"
    assert calls == [("uuid", uuid)]


@pytest.mark.parametrize(
    ("logical_index", "physical_index"),
    [(0, 3), (1, 1)],
)
def test_numeric_visibility_order_is_used_when_cuda_uuid_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    logical_index: int,
    physical_index: int,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    calls: list[tuple[str, object]] = []
    nvml = _recording_nvml(calls)

    handle = resolve_nvml_device_handle(nvml, logical_index)

    assert handle == f"index:{physical_index}"
    assert calls == [("index", physical_index)]


@pytest.mark.parametrize("visible", ["GPU-abbreviated", "MIG-GPU-example/1/0"])
def test_nonnumeric_visibility_without_cuda_uuid_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    visible: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    calls: list[tuple[str, object]] = []

    with pytest.raises(RuntimeError, match="Cannot resolve a nonnumeric"):
        resolve_nvml_device_handle(_recording_nvml(calls), 0)

    assert calls == []
