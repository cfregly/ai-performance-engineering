"""W2-120: the hardware probe must not claim Grace or NVLink-C2C from compute capability.

The probe takes the host link from core.common.grace_platform.host_link_flags.
The CPU tests run the real probe and core functions. They replace only the
host machine name, the CUDA driver attribute query in the core module, the
ptxas, nvcc and libcudart toolchain probes, and probe.torch, which supplies the
device properties. The CUDA tests compare the probe with the installed cuda.h
and with nvidia-smi, which reports the host link without the probe's code.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.common import grace_platform
from core.harness.arch_config import ArchitectureConfig
from core.scripts.utilities import probe_hardware_capabilities as probe

HOST_ATOMICS = grace_platform.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED
HOST_PAGE_TABLES = grace_platform.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
GRACE_COHERENCE_FEATURE = "Grace hardware coherence"
ALL_CAPABILITIES = [(9, 0), (10, 0), (10, 3), (11, 0), (12, 0), (12, 1)]
SMI_C2C_LINK_RE = re.compile(r"C2C Link\s+\d+\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*GB/s")
SMI_NO_C2C = "No Devices support C2C"


def _props(major: int, minor: int, name: str = "test-gpu") -> SimpleNamespace:
    return SimpleNamespace(
        major=major,
        minor=minor,
        name=name,
        total_memory=96 * 1024**3,
        multi_processor_count=188,
        warp_size=32,
        max_threads_per_block=1024,
        max_threads_per_multi_processor=1536,
        shared_memory_per_block=48 * 1024,
        shared_memory_per_multiprocessor=100 * 1024,
        L2_cache_size=128 * 1024**2,
    )


def _driver_reporting(values: dict[int, int], calls: list[tuple[int, int]]):
    def _query(device_index: int, attribute: int) -> int:
        calls.append((device_index, attribute))
        return values[attribute]

    return _query


def _refuse_driver_query(device_index: int, attribute: int) -> int:
    raise AssertionError(f"driver attribute {attribute} must not be queried for device {device_index}")


def _probe_entry(monkeypatch: pytest.MonkeyPatch, props: SimpleNamespace, machine: str) -> dict:
    """Run the real _probe_device with the GPU toolchain probes replaced."""
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            set_device=lambda _index: None,
            get_device_properties=lambda _index: props,
        ),
        version=SimpleNamespace(cuda="13.0"),
    )
    monkeypatch.setattr(probe, "torch", fake_torch)
    monkeypatch.setattr(grace_platform.platform, "machine", lambda: machine)
    monkeypatch.setattr(probe, "_probe_tma_compiler_support", lambda _sm: (True, None))
    monkeypatch.setattr(probe, "_query_cluster_launch", lambda _index: 1)
    monkeypatch.setattr(probe, "_probe_dsmem_support", lambda _sm: (True, None))
    return probe._probe_device(0)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((12, 0), "blackwell_consumer"),
        ((12, 1), "grace_blackwell"),
        ((10, 0), "blackwell"),
        ((10, 3), "blackwell_ultra"),
        ((11, 0), "blackwell"),
        ((9, 0), "hopper"),
        ((13, 0), "other"),
        ((8, 0), "other"),
    ],
)
def test_architecture_label_follows_capability_table(capability, expected) -> None:
    assert probe._architecture_label(_props(*capability)) == expected


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0), (12, 1)])
def test_architecture_label_agrees_with_harness_arch_config(capability) -> None:
    harness = ArchitectureConfig.metadata_for_capability(capability)["architecture"]
    assert probe._architecture_label(_props(*capability)) == harness


@pytest.mark.parametrize("capability", ALL_CAPABILITIES)
def test_x86_host_never_reports_c2c(monkeypatch: pytest.MonkeyPatch, capability) -> None:
    monkeypatch.setattr(grace_platform.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", _refuse_driver_query)

    assert probe._host_link_flags(_props(*capability), 0) == (False, False)


@pytest.mark.parametrize("capability", ALL_CAPABILITIES)
def test_arm_host_takes_the_link_from_the_driver_for_every_capability(
    monkeypatch: pytest.MonkeyPatch, capability
) -> None:
    monkeypatch.setattr(grace_platform.platform, "machine", lambda: "aarch64")

    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", _driver_reporting({HOST_ATOMICS: 0}, calls))
    assert probe._host_link_flags(_props(*capability), 2) == (False, False)
    assert calls == [(2, HOST_ATOMICS)]

    calls = []
    monkeypatch.setattr(
        grace_platform, "_cuda_driver_attribute", _driver_reporting({HOST_ATOMICS: 1, HOST_PAGE_TABLES: 1}, calls)
    )
    assert probe._host_link_flags(_props(*capability), 2) == (True, True)
    assert calls == [(2, HOST_ATOMICS), (2, HOST_PAGE_TABLES)]


def test_pcie_gpu_in_grace_server_reports_no_c2c(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(grace_platform.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        grace_platform, "_cuda_driver_attribute", _driver_reporting({HOST_ATOMICS: 0, HOST_PAGE_TABLES: 1}, calls)
    )

    assert probe._host_link_flags(_props(10, 0), 3) == (False, False)
    assert calls == [(3, HOST_ATOMICS)]


def test_c2c_link_without_host_page_tables_is_not_coherent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grace_platform.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        grace_platform, "_cuda_driver_attribute", _driver_reporting({HOST_ATOMICS: 1, HOST_PAGE_TABLES: 0}, [])
    )

    assert probe._host_link_flags(_props(10, 3), 0) == (True, False)


def test_driver_query_failure_is_not_reported_as_absent_link(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        pass
    else:
        pytest.skip("SKIPPED: a CUDA driver is installed, so the missing-driver path cannot run here")

    with pytest.raises(RuntimeError, match="libcuda.so.1"):
        grace_platform._cuda_driver_attribute(0, HOST_ATOMICS)

    monkeypatch.setattr(grace_platform.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="libcuda.so.1"):
        probe._host_link_flags(_props(10, 0), 0)


def test_features_claim_only_what_the_device_has() -> None:
    rtx = probe._build_features(_props(12, 0), nvlink_c2c=False, grace_coherence=False)
    assert "HBM3e" not in rtx
    assert "NVLink-C2C" not in rtx
    assert GRACE_COHERENCE_FEATURE not in rtx

    gb10 = probe._build_features(_props(12, 1), nvlink_c2c=True, grace_coherence=True)
    assert "HBM3e" not in gb10
    assert {"NVLink-C2C", GRACE_COHERENCE_FEATURE} <= set(gb10)

    b200 = probe._build_features(_props(10, 0), nvlink_c2c=False, grace_coherence=False)
    assert "HBM3e" in b200
    assert "NVLink-C2C" not in b200


@pytest.mark.parametrize("capability", [(12, 0), (12, 1), (10, 0)])
def test_x86_probe_entry_makes_no_grace_claims(monkeypatch: pytest.MonkeyPatch, capability) -> None:
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", _refuse_driver_query)
    entry = _probe_entry(monkeypatch, _props(*capability), "x86_64")

    assert entry["architecture"] == probe._architecture_label(_props(*capability))
    assert entry["max_unified_memory_tb"] is None
    assert entry["nvlink_c2c"] is False
    assert entry["grace_coherence"] is False
    assert "NVLink-C2C" not in entry["features"]


def test_rtx_pro_probe_entry_is_not_grace_blackwell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grace_platform, "_cuda_driver_attribute", _refuse_driver_query)
    entry = _probe_entry(monkeypatch, _props(12, 0, "NVIDIA RTX PRO 6000 Blackwell"), "x86_64")

    assert entry["architecture"] == "blackwell_consumer"
    assert entry["max_unified_memory_tb"] is None
    assert "HBM3e" not in entry["features"]


def test_grace_probe_entry_reports_measured_c2c(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        grace_platform, "_cuda_driver_attribute", _driver_reporting({HOST_ATOMICS: 1, HOST_PAGE_TABLES: 1}, [])
    )
    entry = _probe_entry(monkeypatch, _props(10, 0, "NVIDIA GB200"), "aarch64")

    assert entry["architecture"] == "blackwell"
    assert entry["nvlink_c2c"] is True
    assert entry["grace_coherence"] is True
    assert {"NVLink-C2C", GRACE_COHERENCE_FEATURE} <= set(entry["features"])
    assert entry["max_unified_memory_tb"] is None


def test_gh200_probe_entry_reports_driver_reported_c2c(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        grace_platform, "_cuda_driver_attribute", _driver_reporting({HOST_ATOMICS: 1, HOST_PAGE_TABLES: 1}, [])
    )
    entry = _probe_entry(monkeypatch, _props(9, 0, "NVIDIA GH200 480GB"), "aarch64")

    assert entry["architecture"] == "hopper"
    assert entry["nvlink_c2c"] is True
    assert entry["grace_coherence"] is True
    assert {"NVLink-C2C", GRACE_COHERENCE_FEATURE} <= set(entry["features"])
    assert not [feature for feature in entry["features"] if "Blackwell" in feature]


def _installed_cuda_header() -> Path:
    roots = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    nvcc = shutil.which("nvcc")
    if nvcc:
        roots.append(str(Path(nvcc).resolve().parents[1]))
    for root in roots:
        if root and (Path(root) / "include" / "cuda.h").is_file():
            return Path(root) / "include" / "cuda.h"
    pytest.skip("SKIPPED: no CUDA toolkit cuda.h found through CUDA_HOME, CUDA_PATH or nvcc")


@pytest.mark.parametrize(
    ("enum_name", "probe_value"),
    [
        ("CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED", HOST_ATOMICS),
        ("CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES", HOST_PAGE_TABLES),
    ],
)
def test_attribute_ids_match_installed_cuda_header(enum_name: str, probe_value: int) -> None:
    header = _installed_cuda_header().read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"\b{enum_name}\s*=\s*(\d+)", header)

    assert match is not None, f"{enum_name} is missing from cuda.h"
    assert int(match.group(1)) == probe_value


def _nvidia_smi(*args: str) -> subprocess.CompletedProcess:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        pytest.skip("SKIPPED: nvidia-smi is not installed, so no independent host-link report exists")
    return subprocess.run([nvidia_smi, *args], capture_output=True, text=True, check=False, timeout=60)


def _smi_reports_c2c_link_up(gpu: str) -> bool:
    result = _nvidia_smi("c2c", "-i", gpu, "-s")
    if result.returncode != 0:
        if SMI_NO_C2C in result.stdout + result.stderr:
            return False
        pytest.fail(f"nvidia-smi c2c -s failed: {result.stdout} {result.stderr}")
    return any(float(speed) > 0 for speed in SMI_C2C_LINK_RE.findall(result.stdout))


def _smi_addressing_mode(gpu: str) -> str:
    result = _nvidia_smi("-q", "-i", gpu)
    assert result.returncode == 0, result.stderr
    match = re.search(r"^\s*Addressing Mode\s*:\s*(\S+)", result.stdout, flags=re.MULTILINE)
    assert match is not None, "nvidia-smi -q printed no Addressing Mode"
    return match.group(1)


def test_real_device_link_flags_match_nvidia_smi() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SKIPPED: no CUDA device for the real driver link query")

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpu = f"GPU-{props.uuid}"
        c2c_up = _smi_reports_c2c_link_up(gpu)
        # nvidia-smi names hardware page-table sharing ATS. HMM is the software path.
        ats = _smi_addressing_mode(gpu) == "ATS"

        device = f"CUDA device {index}"
        assert (grace_platform._cuda_driver_attribute(index, HOST_ATOMICS) == 1) == c2c_up, device
        assert (grace_platform._cuda_driver_attribute(index, HOST_PAGE_TABLES) == 1) == ats, device
        assert probe._host_link_flags(props, index) == (c2c_up, c2c_up and ats), device
