"""NVIDIA Grace host identity and NVLink-C2C host links.

Each question has one source of evidence.

- Is the host CPU NVIDIA Grace? Linux publishes the Arm SMCCC SoC ID under
  /sys/bus/soc/devices/soc*/soc_id, and an NVIDIA SoC reads ``jep106:036b:``.
  /proc/cpuinfo cannot answer this question. On arm64 it lists only Arm core
  part numbers, and other vendors ship the same Neoverse V2 cores as Grace.
- Does a GPU reach host memory over NVLink-C2C? The CUDA driver reports native
  host atomics for an NVLink-C2C host link and not for PCIe. Compute
  capability cannot answer this question, because the same SM 9.0 or 10.x GPU
  ships on both kinds of link.

``nvidia-smi c2c -s`` lists the C2C links of a GPU and the per-link rate the
driver reports. ``gpu_c2c_links`` returns that list and raises when it
contradicts the CUDA driver.

This module imports no third-party package, so the standalone hardware probe
can use it on a host without PyTorch.
"""

from __future__ import annotations

import ctypes
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100

_ARM_MACHINES = frozenset({"aarch64", "arm64"})
_SOC_BUS_DIR = Path("/sys/bus/soc/devices")
# Arm SMCCC SOC_ID names the SoC vendor with a JEP106 code. Linux prints
# NVIDIA (bank 4, code 0x6b) as "036b".
_NVIDIA_SOC_ID_RE = re.compile(r"jep106:036b:[0-9a-f]{4}", flags=re.IGNORECASE)
_C2C_LINK_RE = re.compile(r"C2C Link\s+(\d+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*GB/s")


def _is_arm_host() -> bool:
    return platform.machine().lower() in _ARM_MACHINES


def read_soc_ids() -> List[str]:
    """Return the SoC IDs that Linux publishes under /sys/bus/soc/devices."""
    if not _SOC_BUS_DIR.is_dir():
        return []
    soc_ids = []
    for soc_id_path in sorted(_SOC_BUS_DIR.glob("soc*/soc_id")):
        soc_id = soc_id_path.read_text(encoding="utf-8").strip()
        if soc_id:
            soc_ids.append(soc_id)
    return soc_ids


def nvidia_grace_soc_id(soc_ids: List[str]) -> Optional[str]:
    """
    Return the SoC ID that names NVIDIA as the Arm SoC vendor, or None.

    On arm64, /proc/cpuinfo lists only Arm core part numbers. Grace uses
    Neoverse V2 cores, which other vendors also ship, so a core name or part
    number cannot identify Grace.
    """
    for soc_id in soc_ids:
        if _NVIDIA_SOC_ID_RE.fullmatch(soc_id):
            return soc_id
    return None


def host_grace_soc_id() -> Optional[str]:
    """
    Return this host's NVIDIA SoC ID, or None when the host CPU is not NVIDIA Grace.

    An x86 host has no Arm SoC ID, so the sysfs read runs only on Arm hosts.
    """
    if not _is_arm_host():
        return None
    return nvidia_grace_soc_id(read_soc_ids())


def query_gpu_c2c_links(gpu_index: int) -> Optional[Dict[int, float]]:
    """
    Return the NVLink-C2C links that ``nvidia-smi c2c -s`` reports as up.

    Each value is the per-link GB/s figure the driver prints, not a measured
    throughput. An empty dict means the driver listed no active C2C link.
    None means nvidia-smi is missing or the query failed, so the C2C status
    is unknown. ``gpu_index`` is the nvidia-smi index, which follows PCI bus
    order.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "c2c", "-i", str(gpu_index), "-s"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    links: Dict[int, float] = {}
    for match in _C2C_LINK_RE.finditer(result.stdout):
        speed_gbps = float(match.group(2))
        if speed_gbps > 0:
            links[int(match.group(1))] = speed_gbps
    return links


def _cuda_driver_attribute(device_index: int, attribute: int) -> int:
    """Read one CUDA driver device attribute. Raise when the driver cannot answer."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise RuntimeError(
            f"Cannot load libcuda.so.1 to read CUDA device attribute {attribute}: {exc}"
        ) from exc
    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    libcuda.cuDeviceGet.restype = ctypes.c_int
    libcuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    libcuda.cuDeviceGetAttribute.restype = ctypes.c_int
    status = libcuda.cuInit(0)
    if status != 0:
        raise RuntimeError(f"cuInit failed with CUresult {status}")
    device = ctypes.c_int()
    status = libcuda.cuDeviceGet(ctypes.byref(device), device_index)
    if status != 0:
        raise RuntimeError(f"cuDeviceGet({device_index}) failed with CUresult {status}")
    value = ctypes.c_int()
    status = libcuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, device)
    if status != 0:
        raise RuntimeError(
            f"cuDeviceGetAttribute({attribute}) failed for device {device_index} with CUresult {status}"
        )
    return int(value.value)


def host_link_flags(props: Any, device_index: int) -> Tuple[bool, bool]:
    """Return (nvlink_c2c, grace_coherence) as the CUDA driver reports the host link.

    ``device_index`` is the CUDA device ordinal and ``props`` holds that
    device's properties. The decision never reads the compute capability in
    ``props``. The same SM 9.0 or 10.x GPU reaches Grace over NVLink-C2C in
    GH200, GB200 and GB300, and reaches the host over PCIe in other servers,
    including Grace servers. NVLink-C2C CPU links exist only on Arm hosts, so
    an x86 host has none and the driver is not asked. On an Arm host the driver
    reports native host atomics for an NVLink-C2C link and not for PCIe.
    Hardware coherence is the GPU reading pageable memory through the host page
    tables. A failed driver query raises instead of reporting that no link
    exists.
    """
    if not _is_arm_host():
        return False, False
    if _cuda_driver_attribute(device_index, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED) != 1:
        return False, False
    uses_host_page_tables = _cuda_driver_attribute(
        device_index, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
    )
    return True, uses_host_page_tables == 1


def gpu_c2c_links(props: Any, device_index: int) -> Optional[Dict[int, float]]:
    """
    Return the NVLink-C2C links of one GPU after checking them against the CUDA driver.

    The driver decides whether the GPU reaches host memory over NVLink-C2C
    (see host_link_flags). nvidia-smi c2c -s lists the links. The result is
    the link list for a GPU on NVLink-C2C, an empty dict for a GPU that the
    driver puts on another host link, or None when the driver reports
    NVLink-C2C but nvidia-smi could not list the links.

    Raises RuntimeError when nvidia-smi lists active links that the driver
    does not report, or lists none for a GPU that the driver puts on
    NVLink-C2C.
    """
    nvlink_c2c, _ = host_link_flags(props, device_index)
    links = query_gpu_c2c_links(device_index)
    if links is None:
        return None if nvlink_c2c else {}
    if bool(links) != nvlink_c2c:
        listed = f"lists {len(links)} active NVLink-C2C links" if links else "lists no active NVLink-C2C link"
        reported = "reports an NVLink-C2C host link" if nvlink_c2c else "reports no NVLink-C2C host link"
        raise RuntimeError(
            f"nvidia-smi c2c -i {device_index} -s {listed}, but the CUDA driver {reported} "
            f"for CUDA device {device_index}. nvidia-smi numbers GPUs in PCI bus order, so "
            "check that CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER keep both numbers on the same GPU."
        )
    return links
