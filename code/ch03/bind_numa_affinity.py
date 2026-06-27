"""NUMA-aware affinity helpers for Chapter 3 examples (CUDA 13 / PyTorch 2.10)."""

from __future__ import annotations

import ctypes
import os

from core.common.device_utils import resolve_local_rank
import re
import subprocess
from functools import partial
from typing import List, Tuple

import psutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

try:
    import pynvml as nvml  # pip install nvidia-ml-py (imports as pynvml)
    _HAS_NVML = True
except Exception:  # pragma: no cover - NVML optional for unit tests
    _HAS_NVML = False

_NVML_INIT_DONE = False

# ---------------------------------------------------------------------------
# libnuma helpers
# ---------------------------------------------------------------------------

_libnuma = None
_HAS_LIBNUMA = False
_libcudart = None

try:
    _libnuma = ctypes.CDLL("libnuma.so")
    if _libnuma.numa_available() >= 0:  # pragma: no cover - hardware guard
        _libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
        _libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
        _HAS_LIBNUMA = True
except (OSError, AttributeError):
    # libnuma.so not found or NUMA not available - degrade gracefully
    pass

try:
    _libcudart = ctypes.CDLL("libcudart.so")
    _libcudart.cudaDeviceGetPCIBusId.restype = ctypes.c_int
    _libcudart.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
except (OSError, AttributeError):
    _libcudart = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_cpu_list(spec: str) -> List[int]:
    """Expand strings like '0-3,8-11' into explicit CPU indices."""
    cpus: List[int] = []
    if not spec:
        return cpus
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-"))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return cpus


def _current_numa_policy() -> Tuple[List[int], int]:
    """Return (cpu_list, preferred_node) from `numactl --show`."""
    try:
        out = subprocess.run(
            ["numactl", "--show"], capture_output=True, text=True, check=True
        ).stdout
    except Exception:  # pragma: no cover - tooling fallback
        cpu_count = psutil.cpu_count() or 1
        return list(range(cpu_count)), 0

    # Handle both formats: "physcpubind: 0 1 2 3" and "physcpubind: 0-3"
    phys_match = re.search(r"physcpubind:\s*([\d,\-\s]+)", out)
    node_match = re.search(r"preferred node:\s*(-?\d+)", out)
    
    if phys_match:
        cpu_str = phys_match.group(1).strip()
        # Check if it's space-separated numbers without ranges
        if " " in cpu_str and "-" not in cpu_str:
            cpus = [int(x) for x in cpu_str.split() if x.strip()]
        else:
            cpus = _parse_cpu_list(cpu_str)
    else:
        cpus = list(range(psutil.cpu_count() or 1))
    
    node = int(node_match.group(1)) if node_match else 0
    if node < 0:
        node = 0
    return cpus, node


def _cpus_for_node(node: int) -> List[int]:
    """Return CPU list for a NUMA node via sysfs (preferred) or policy fallback."""
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist", "r", encoding="utf-8") as f:
            cpus = _parse_cpu_list(f.read().strip())
            if cpus:
                return cpus
    except FileNotFoundError:
        pass
    cpus, _ = _current_numa_policy()
    return cpus


# ---------------------------------------------------------------------------
# Public API (matches book Chapter 3 function names)
# ---------------------------------------------------------------------------

def parse_physical_cpu_list(phys_str: str) -> List[int]:
    """Expand strings like '0-3,8-11' or '0 1 2 3' into explicit CPU indices.
    
    Book reference: Chapter 3, NUMA affinity section.
    
    Args:
        phys_str: CPU specification string (e.g., "0-3,8-11" or "0 1 2 3")
        
    Returns:
        List of CPU indices
        
    Example:
        >>> parse_physical_cpu_list("0-3,8-11")
        [0, 1, 2, 3, 8, 9, 10, 11]
        >>> parse_physical_cpu_list("0 1 2 3")
        [0, 1, 2, 3]
    """
    # Handle space-separated format (e.g., "0 1 2 3")
    if " " in phys_str and "-" not in phys_str and "," not in phys_str:
        return [int(x) for x in phys_str.split() if x.strip()]
    return _parse_cpu_list(phys_str)


def get_numa_cpus_for_node(node: int) -> List[int]:
    """Return the list of CPUs belonging to a specific NUMA node.
    
    Book reference: Chapter 3, NUMA affinity section.
    Reads from /sys/devices/system/node/node{N}/cpulist.
    
    Args:
        node: NUMA node ID
        
    Returns:
        List of CPU indices for that node
        
    Example:
        >>> get_numa_cpus_for_node(0)
        [0, 1, 2, 3, 4, 5, 6, 7]
    """
    return _cpus_for_node(node)


def get_numa_cpus_and_memory() -> Tuple[List[int], int]:
    """Return (current_cpu_mask, preferred_node) from numactl --show.
    
    Book reference: Chapter 3, NUMA affinity section.
    Queries the current process's NUMA policy.
    
    Returns:
        Tuple of (cpu_list, preferred_memory_node)
        
    Example:
        >>> cpus, node = get_numa_cpus_and_memory()
        >>> print(f"Process bound to CPUs {cpus}, memory node {node}")
    """
    return _current_numa_policy()


def _gpu_pci_bus(device_index: int) -> str:
    if _libcudart is not None:
        buffer = ctypes.create_string_buffer(20)
        status = _libcudart.cudaDeviceGetPCIBusId(buffer, len(buffer), device_index)
        if status == 0:
            return buffer.value.decode()
    props = torch.cuda.get_device_properties(device_index)
    if isinstance(props.pci_bus_id, str):
        return props.pci_bus_id  # e.g. '0000:03:00.0'
    return f"{props.pci_domain_id:04x}:{int(props.pci_bus_id):02x}:{int(props.pci_device_id):02x}.0"


def _normalize_pci_for_nvml(pci: str) -> str:
    try:
        domain, bus, devfn = pci.split(":")
        if len(domain) < 8:
            domain = domain.rjust(8, "0")
        return f"{domain}:{bus}:{devfn}"
    except ValueError:
        return pci


def _gpu_node_from_nvml(device_index: int) -> int | None:
    if not _HAS_NVML:
        return None
    try:
        _ensure_nvml_initialized()
        if not _NVML_INIT_DONE:
            return None
        pci = _normalize_pci_for_nvml(_gpu_pci_bus(device_index))
        try:
            handle = nvml.nvmlDeviceGetHandleByPciBusId_v2(pci)
        except AttributeError:
            handle = nvml.nvmlDeviceGetHandleByPciBusId(pci)

        # Prefer explicit NUMA node if driver exposes it
        try:
            numa_id = nvml.nvmlDeviceGetNumaNodeId(handle)
            if isinstance(numa_id, int) and numa_id >= 0:
                return numa_id
        except Exception:
            pass  # explicit NUMA query not available or not supported here
    except Exception:  # pragma: no cover - NVML optional
        return None
    return None


def _gpu_node_from_sysfs(device_index: int) -> int | None:
    try:
        sysfs_path = f"/sys/bus/pci/devices/{_gpu_pci_bus(device_index)}/numa_node"
        with open(sysfs_path, "r", encoding="utf-8") as f:
            val = int(f.read().strip())
            if val >= 0:
                return val
    except Exception:
        return None
    return None


def get_gpu_numa_node(device_index: int) -> int | None:
    """Resolve the authoritative NUMA node for a GPU: NVML -> sysfs."""
    for resolver in (_gpu_node_from_nvml, _gpu_node_from_sysfs):
        node = resolver(device_index)
        if node is not None:
            return node
    return None


def bind_process_to_node(node: int) -> List[int]:
    """Bind current process CPU & memory policy to the specified NUMA node."""
    cpus = _cpus_for_node(node)
    psutil.Process(os.getpid()).cpu_affinity(cpus)
    if _HAS_LIBNUMA and _libnuma is not None:
        _libnuma.numa_run_on_node(node)
        _libnuma.numa_set_preferred(node)
    print(f"PID {os.getpid()} bound to NUMA node {node} (CPUs={cpus})")
    return cpus


def worker_init_fn(worker_id: int, node: int | None, cpus: List[int] | None) -> None:
    """Initialize DataLoader worker without invoking CUDA APIs."""
    if not cpus:
        print(f"Worker {worker_id} (PID={os.getpid()}) leaving CPU affinity unchanged")
        return
    psutil.Process(os.getpid()).cpu_affinity(cpus)
    if node is not None and _HAS_LIBNUMA and _libnuma is not None:
        _libnuma.numa_run_on_node(node)
        _libnuma.numa_set_preferred(node)
    if node is None:
        print(f"Worker {worker_id} (PID={os.getpid()}) applied CPU affinity without NUMA binding")
    else:
        print(f"Worker {worker_id} (PID={os.getpid()}) bound to NUMA node {node}")


def _ensure_nvml_initialized() -> None:
    global _NVML_INIT_DONE
    if not _HAS_NVML or _NVML_INIT_DONE:
        return
    try:
        nvml.nvmlInit()
        _NVML_INIT_DONE = True
    except Exception:
        _NVML_INIT_DONE = False


# ---------------------------------------------------------------------------
# Minimal runnable demo (DDP-friendly)
# ---------------------------------------------------------------------------

class DemoDataset(Dataset):
    def __init__(self, length: int = 1024, feature_dim: int = 224 * 224 * 3) -> None:
        self.length = length
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        del index
        x = torch.randn(self.feature_dim, dtype=torch.float32)
        y = torch.randint(0, 10, (1,), dtype=torch.int64).item()
        return x, y


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the NUMA affinity demo")

    # Check if distributed environment is available
    has_distributed_env = all(
        key in os.environ for key in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    )
    
    if has_distributed_env:
        # Distributed mode
        dist.init_process_group(backend="nccl", init_method="env://")
        try:
            local_rank = resolve_local_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)

            gpu_node = get_gpu_numa_node(local_rank)
            cpus = None
            if gpu_node is None:
                print(f"WARNING: GPU {local_rank} NUMA node unavailable; leaving process affinity unchanged")
            else:
                cpus = bind_process_to_node(gpu_node)

            dataset = DemoDataset()
            loader = DataLoader(
                dataset,
                batch_size=32,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
                worker_init_fn=partial(worker_init_fn, node=gpu_node, cpus=cpus),
            )

            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(dataset.feature_dim, 1024),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(1024, 10),
            ).to(device)

            ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, static_graph=True)
            optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

            ddp_model.train()
            for step, (x_cpu, y_cpu) in enumerate(loader):
                x = x_cpu.to(device, non_blocking=True)
                y = y_cpu.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = ddp_model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                if step % 50 == 0 and dist.get_rank() == 0:
                    print(f"step={step} loss={loss.item():.4f}")
                if step == 100:
                    break
        finally:
            dist.destroy_process_group()
    else:
        # Single-process mode (for testing/verification)
        print("WARNING: Running in single-process mode (distributed environment not detected)")
        local_rank = 0
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        gpu_node = get_gpu_numa_node(local_rank)
        cpus = None
        if gpu_node is None:
            print(f"WARNING: GPU {local_rank} NUMA node unavailable; leaving process affinity unchanged")
        else:
            cpus = bind_process_to_node(gpu_node)

        # Quick sanity test
        dataset = DemoDataset(length=32)
        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=partial(worker_init_fn, node=gpu_node, cpus=cpus),
        )

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dataset.feature_dim, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 10),
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for step, (x_cpu, y_cpu) in enumerate(loader):
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            if step == 0:
                print(f"[OK] NUMA binding sanity test passed (loss={loss.item():.4f})")
            if step == 2:  # Quick test
                break


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
