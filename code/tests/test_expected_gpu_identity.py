"""Expected-versus-observed GPU identity checks for environment validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from core.harness import validity_checks
from core.harness.device_identity_contract import observe_cuda_device_identity
from core.harness.validity_checks import EnvironmentProbe, validate_environment

GPU_UUID_A = "GPU-12345678-1234-5678-1234-567812345678"
GPU_UUID_B = "GPU-87654321-4321-8765-4321-876543218765"


@pytest.fixture(autouse=True)
def _use_linux_environment_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the Linux-only validator while preserving real CUDA behavior."""
    monkeypatch.setattr(validity_checks, "sys", SimpleNamespace(platform="linux"))


def _write_probe_file(root, path: str, content: str) -> None:
    target = root / path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _clean_probe(root) -> EnvironmentProbe:
    _write_probe_file(root, "/proc/swaps", "Filename Type Size Used Priority\n")
    _write_probe_file(root, "/proc/sys/vm/swappiness", "0\n")
    _write_probe_file(root, "/proc/cpuinfo", "processor: 0\n")
    _write_probe_file(root, "/sys/devices/virtual/dmi/id/product_name", "BareMetal\n")
    _write_probe_file(
        root,
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
        "performance\n",
    )
    _write_probe_file(root, "/sys/devices/system/node/node0/cpulist", "0-3\n")
    return EnvironmentProbe(root=root, env={}, cpu_affinity={0, 1, 2, 3})


def test_default_cpu_validation_does_not_require_gpu_identity(tmp_path) -> None:
    result = validate_environment(
        device=torch.device("cpu"),
        probe=_clean_probe(tmp_path),
    )

    assert result.is_valid, result.errors
    assert "expected_device_uuid" not in result.details
    assert "observed_device_uuid" not in result.details


def test_expected_gpu_identity_rejects_configured_cpu_device(tmp_path) -> None:
    result = validate_environment(
        device=torch.device("cpu"),
        probe=_clean_probe(tmp_path),
        expected_device_uuid=GPU_UUID_A,
        expected_compute_capability="10.0",
    )

    assert not result.is_valid
    assert any(
        "cannot be observed for configured device type 'cpu'" in error for error in result.errors
    )
    assert result.details["expected_device_uuid"] == GPU_UUID_A
    assert result.details["expected_compute_capability"] == "10.0"
    assert result.details["observed_device_uuid"] is None
    assert result.details["observed_compute_capability"] is None


def test_partial_expected_gpu_identity_is_rejected(tmp_path) -> None:
    result = validate_environment(
        device=torch.device("cpu"),
        probe=_clean_probe(tmp_path),
        expected_device_uuid=GPU_UUID_A,
    )

    assert not result.is_valid
    assert any("must be configured together" in error for error in result.errors)


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CUDA-unavailable behavior requires a host without CUDA",
)
def test_expected_gpu_identity_is_unknown_when_cuda_is_unavailable(tmp_path) -> None:
    result = validate_environment(
        device=torch.device("cuda"),
        probe=_clean_probe(tmp_path),
        expected_device_uuid=GPU_UUID_A,
        expected_compute_capability="10.0",
    )

    assert not result.is_valid
    assert any(
        "CUDA device requested but CUDA is not available" in error for error in result.errors
    )
    assert any(
        "Expected GPU identity/capability is unavailable" in error for error in result.errors
    )
    assert result.details["observed_device_uuid"] is None
    assert result.details["observed_compute_capability"] is None


def _live_cuda_observation():
    if not torch.cuda.is_available():
        pytest.skip("real GPU identity validation requires a CUDA device")
    pytest.importorskip("pynvml", reason="real GPU identity validation requires pynvml")
    device = torch.device("cuda", torch.cuda.current_device())
    return device, observe_cuda_device_identity(device)


def test_expected_gpu_identity_accepts_live_observed_cuda_device(tmp_path) -> None:
    device, observed = _live_cuda_observation()

    result = validate_environment(
        device=device,
        probe=_clean_probe(tmp_path),
        expected_device_uuid=observed.nvml_uuid,
        expected_compute_capability=observed.compute_capability,
    )

    assert result.is_valid, result.errors
    assert result.details["expected_device_uuid"] == observed.nvml_uuid
    assert result.details["expected_compute_capability"] == observed.compute_capability
    assert result.details["observed_device_uuid"] == observed.nvml_uuid
    assert result.details["observed_cuda_device_uuid"] == observed.cuda_uuid
    assert result.details["observed_compute_capability"] == observed.compute_capability


@pytest.mark.parametrize("mismatch", ["device_uuid", "compute_capability"])
def test_expected_gpu_identity_rejects_live_cuda_mismatch(
    tmp_path,
    mismatch: str,
) -> None:
    device, observed = _live_cuda_observation()
    wrong_uuid = GPU_UUID_A if observed.nvml_uuid != GPU_UUID_A else GPU_UUID_B
    wrong_capability = "9.0" if observed.compute_capability != "9.0" else "10.0"

    result = validate_environment(
        device=device,
        probe=_clean_probe(tmp_path),
        expected_device_uuid=(wrong_uuid if mismatch == "device_uuid" else observed.nvml_uuid),
        expected_compute_capability=(
            wrong_capability if mismatch == "compute_capability" else observed.compute_capability
        ),
    )

    assert not result.is_valid
    expected_message = (
        "Expected GPU UUID mismatch"
        if mismatch == "device_uuid"
        else "Expected compute capability mismatch"
    )
    assert any(expected_message in error for error in result.errors), result.errors
    assert result.details["observed_device_uuid"] == observed.nvml_uuid
    assert result.details["observed_compute_capability"] == observed.compute_capability
