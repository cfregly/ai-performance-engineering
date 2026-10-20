#!/usr/bin/env python3

"""
GB200/GB300 Grace CPU NUMA Optimization for PyTorch 2.10
=======================================================

Demonstrates optimal NUMA topology configuration and CPU-GPU affinity
for GB200/GB300 Grace-Blackwell superchips.

GB200/GB300 Architecture:
- Grace CPU: 72 ARM Neoverse V2 cores, one thread per core
- CPU Memory: 480GB-1TB LPDDR5X
- NVLink-C2C: 900 GB/s bidirectional (450 GB/s per direction) coherent CPU↔GPU spec peak
- Multiple Blackwell B200 GPUs per system
- Unified memory address space

Key Optimizations:
1. NUMA-aware CPU-GPU affinity binding
2. Optimal thread placement for data loading
3. CPU preprocessing + GPU training pipelines
4. Memory bandwidth optimization

Requirements:
- GB200/GB300 system
- PyTorch 2.10+
- Python 3.10+

Usage:
    python gb200_grace_numa_optimization.py
    
    Or in your training script:
    from extras.ch04.gb200_grace_numa_optimization import setup_grace_affinity
    setup_grace_affinity(gpu_id=0, num_workers=8)
"""
from core.common.grace_platform import gpu_c2c_links, nvidia_grace_soc_id, read_soc_ids
from ch04.distributed_helper import run_main_with_skip_status

import platform
import os
from pathlib import Path
import psutil
import torch
from typing import Any, Dict, List, Optional, Tuple
import subprocess


# NVIDIA specifies NVLink-C2C at 900 GB/s bidirectional, so one copy direction peaks at 450 GB/s.
_C2C_PEAK_PER_DIRECTION_GBPS = 450.0

# A Grace superchip pairs Grace with GPUs of one compute capability. CC 9.0 and
# 12.x also ship as PCIe cards, so these names need C2C links as well.
_GRACE_SUPERCHIP_BY_CAPABILITY = {(9, 0): "GH200", (10, 0): "GB200", (10, 3): "GB300", (12, 1): "GB10"}
# B200 (CC 10.0) and B300 (CC 10.3) ship only with HBM3e.
_HBM3E_CAPABILITIES = {(10, 0), (10, 3)}


def _grace_superchip_name(c2c_by_gpu: Dict[int, Optional[Dict[int, float]]]) -> Optional[str]:
    """
    Name the Grace superchip, or return None.

    A name is returned only when every listed GPU reports NVLink-C2C links and
    all of them share one compute capability from the superchip table.
    """
    if not c2c_by_gpu or not all(c2c_by_gpu.values()) or not torch.cuda.is_available():
        return None
    capabilities = set()
    for gpu_id in c2c_by_gpu:
        props = torch.cuda.get_device_properties(gpu_id)
        capabilities.add((props.major, props.minor))
    if len(capabilities) != 1:
        return None
    return _GRACE_SUPERCHIP_BY_CAPABILITY.get(capabilities.pop())


def detect_grace_cpu() -> Dict[str, Any]:
    """
    Detect Grace CPU and gather system information.

    Returns:
        Dictionary with Grace CPU information
    """
    info = {
        "is_grace": False,
        "soc_id": None,
        "cpu_arch": platform.machine(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "memory_gb": psutil.virtual_memory().total / 1e9,
        "numa_nodes": None,
        "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    # Grace identity comes from the firmware SoC vendor ID, not from Arm core
    # names, because non-NVIDIA CPUs also use Neoverse cores.
    if info["cpu_arch"] in ("aarch64", "arm64"):
        info["soc_id"] = nvidia_grace_soc_id(read_soc_ids())
        if info["soc_id"] is not None:
            info["is_grace"] = True
            info["cpu_model"] = "NVIDIA Grace"

    # Detect NUMA nodes
    try:
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'available:' in line:
                    info["numa_nodes"] = int(line.split(':')[1].split()[0])
    except FileNotFoundError:
        pass  # numactl not installed
    
    return info


def get_numa_topology() -> Dict[int, Dict]:
    """
    Get detailed NUMA topology for Grace CPU.
    
    Returns:
        Dictionary mapping NUMA node to CPUs and GPUs
    """
    topology = {}
    
    try:
        # Get NUMA node information
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return topology
        
        current_node = None
        for line in result.stdout.split('\n'):
            if line.startswith('node') and 'cpus:' in line:
                # Parse "node X cpus: ..." lines
                parts = line.split()
                if len(parts) >= 4:
                    node_id = int(parts[1])
                    current_node = node_id
                    
                    # Initialize node entry if it doesn't exist
                    if node_id not in topology:
                        topology[node_id] = {
                            "cpus": [],
                            "size_gb": 0,
                            "gpus": [],
                        }
                    
                    # Parse CPU list
                    cpu_str = line.split('cpus:')[1].strip()
                    # Parse CPU ranges (e.g., "0-17,36-53" or "0 1 2 3")
                    # Normalize to use commas, then parse
                    if ',' not in cpu_str and ' ' in cpu_str:
                        # Space-separated: "0 1 2 3" -> "0,1,2,3"
                        cpu_str = cpu_str.replace(' ', ',')
                    
                    for part in cpu_str.split(','):
                        part = part.strip()
                        if not part:
                            continue
                        if '-' in part:
                            # Range: "0-17"
                            start, end = map(int, part.split('-'))
                            topology[node_id]["cpus"].extend(range(start, end + 1))
                        else:
                            # Single CPU
                            topology[node_id]["cpus"].append(int(part))
            
            elif line.startswith('node') and 'size:' in line:
                # Parse "node X size: ..." lines
                parts = line.split()
                if len(parts) >= 4 and 'MB' in line:
                    node_id = int(parts[1])
                    current_node = node_id
                    
                    # Initialize node entry if it doesn't exist
                    if node_id not in topology:
                        topology[node_id] = {
                            "cpus": [],
                            "size_gb": 0,
                            "gpus": [],
                        }
                    
                    size_mb = int(parts[3])
                    topology[node_id]["size_gb"] = size_mb / 1024
    except Exception as e:
        print(f"Warning: Could not parse NUMA topology: {e}")
    
    # Map GPUs to NUMA nodes using sysfs when the platform exposes it.
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            node_id = _gpu_numa_node_from_sysfs(gpu_id)
            if node_id is not None and node_id in topology:
                topology[node_id]["gpus"].append(gpu_id)
    
    return topology


def _current_cpu_affinity() -> List[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        cpu_threads = psutil.cpu_count(logical=True) or 1
        return list(range(cpu_threads))


def _gpu_numa_node_from_sysfs(gpu_id: int) -> Optional[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
        pci_id = result.stdout.strip().lower()
        if pci_id.startswith("00000000:"):
            pci_id = pci_id[4:]
        numa_path = Path(f"/sys/bus/pci/devices/{pci_id}/numa_node")
        if not numa_path.exists():
            return None
        value = int(numa_path.read_text().strip())
        return value if value >= 0 else None
    except Exception:
        return None


def setup_grace_affinity(
    gpu_id: int,
    num_workers: int = 4,
    verbose: bool = True
) -> Tuple[List[int], Optional[int]]:
    """
    Setup optimal CPU affinity for a GPU on Grace-Blackwell.
    
    For GB200/GB300, this binds data loading workers to CPUs
    on the same NUMA node as the GPU for optimal NVLink-C2C performance.
    
    Args:
        gpu_id: GPU device ID (0-(num_gpus-1))
        num_workers: Number of data loading workers
        verbose: Print configuration details
        
    Returns:
        (cpu_list, numa_node): List of CPUs and NUMA node ID
    """
    info = detect_grace_cpu()
    
    if not info["is_grace"]:
        current_affinity = _current_cpu_affinity()
        if verbose:
            print(f"Warning: Not running on Grace CPU")
            print(f"  Detected: {info['cpu_arch']}, {info['cpu_count']} cores")
            print(f"  Leaving existing CPU affinity unchanged")
        return current_affinity, None
    
    # Get NUMA topology
    topology = get_numa_topology()
    
    if not topology:
        current_affinity = _current_cpu_affinity()
        if verbose:
            print(f"GPU {gpu_id}: NUMA topology unavailable; leaving existing CPU affinity unchanged")
        return current_affinity, None
    
    # Find NUMA node for this GPU
    numa_node = None
    for node_id, node_info in topology.items():
        if gpu_id in node_info["gpus"]:
            numa_node = node_id
            break
    
    if numa_node is None:
        current_affinity = _current_cpu_affinity()
        if verbose:
            print(f"GPU {gpu_id}: NUMA node unavailable; leaving existing CPU affinity unchanged")
        return current_affinity, None
    
    # Get CPUs from this NUMA node
    available_cpus = topology[numa_node]["cpus"]
    
    if len(available_cpus) < num_workers:
        print(f"Warning: NUMA node {numa_node} has only {len(available_cpus)} CPUs, requested {num_workers}")
        cpu_list = available_cpus
    else:
        # Select first num_workers CPUs from this NUMA node
        cpu_list = available_cpus[:num_workers]
    
    # Set CPU affinity
    try:
        os.sched_setaffinity(0, cpu_list)
        if verbose:
            print(f"GPU {gpu_id} affinity set:")
            print(f"  NUMA node: {numa_node}")
            print(f"  CPUs: {cpu_list}")
            print(f"  Memory: {topology[numa_node]['size_gb']:.1f} GB")
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")
    
    return cpu_list, numa_node


def optimize_data_loading_for_grace(
    gpu_id: int,
    batch_size: int = 32,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Configure optimal DataLoader settings for Grace-Blackwell.
    
    GB200/GB300 benefits from:
    - More workers (72 CPU cores available)
    - Larger prefetch (NVLink-C2C peaks at 450 GB/s per direction)
    - Pinned memory (coherent memory)
    
    Args:
        gpu_id: GPU device ID
        batch_size: Batch size
        num_workers: Number of data loading workers (4-12 recommended)
        prefetch_factor: Prefetch factor (2-4 recommended)
        verbose: Print configuration
        
    Returns:
        Dictionary of DataLoader kwargs
    """
    info = detect_grace_cpu()
    cpu_list, numa_node = setup_grace_affinity(gpu_id, num_workers, verbose=False)

    # Optimal settings for GB200/GB300
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,  # Page-locked buffers allow async copies to the GPU
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,  # Reuse workers
    }

    if verbose:
        c2c_links = (
            gpu_c2c_links(torch.cuda.get_device_properties(gpu_id), gpu_id) if info["is_grace"] else {}
        )
        superchip = _grace_superchip_name({gpu_id: c2c_links})
        print("\n" + "=" * 80)
        if superchip is not None:
            print(f"DataLoader Configuration for {superchip} (GPU {gpu_id})")
        else:
            print(f"DataLoader Configuration (GPU {gpu_id})")
        print("=" * 80)
        if info["is_grace"]:
            print(f"Grace CPU: {info['cpu_count']} cores, {info['cpu_threads']} threads")
        else:
            print(f"CPU: {info['cpu_arch']}, {info['cpu_count']} cores, {info['cpu_threads']} threads (not NVIDIA Grace)")
        if numa_node is None:
            print(f"NUMA node: not bound, existing CPU affinity kept ({len(cpu_list)} CPUs)")
        else:
            print(f"NUMA node: {numa_node}")
            print(f"CPU affinity: {len(cpu_list)} CPUs assigned")
        print(f"\nDataLoader Settings:")
        for key, value in dataloader_kwargs.items():
            print(f"  {key}: {value}")
        print("\nExpected Performance:")
        if c2c_links:
            print(
                f"  CPU→GPU bandwidth: up to {_C2C_PEAK_PER_DIRECTION_GBPS:.0f} GB/s per direction "
                "(NVLink-C2C spec peak, not measured here)"
            )
        elif c2c_links is None:
            print("  CPU→GPU bandwidth: NVLink-C2C status unavailable from nvidia-smi c2c -s")
        elif info["is_grace"]:
            print("  CPU→GPU bandwidth: no NVLink-C2C link reported for this GPU")
        else:
            print("  CPU→GPU bandwidth: no NVLink-C2C path because the host is not NVIDIA Grace")
        print(f"  CPU overhead: <5% vs baseline")
        print(f"  Optimal for: Large batch sizes, CPU preprocessing")
        print("=" * 80)
    
    return dataloader_kwargs


def benchmark_cpu_gpu_bandwidth(
    size_mb: int = 100,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark CPU-GPU transfer bandwidth on GB200/GB300.
    
    Expected performance:
    - Grace with NVLink-C2C: at most 450 GB/s per direction (900 GB/s bidirectional spec)
    - Standard PCIe 5.0: ~64 GB/s
    - Standard PCIe 4.0: ~32 GB/s
    
    Args:
        size_mb: Transfer size in MB
        num_iterations: Number of iterations
        
    Returns:
        Bandwidth measurements in GB/s
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return {}
    
    info = detect_grace_cpu()
    c2c_links = gpu_c2c_links(torch.cuda.get_device_properties(0), 0) if info["is_grace"] else {}
    device = torch.device("cuda:0")

    # Allocate memory
    size_elements = size_mb * 1024 * 1024 // 4  # float32
    cpu_tensor = torch.randn(size_elements, dtype=torch.float32, pin_memory=True)
    gpu_tensor = torch.empty(size_elements, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    
    # Benchmark H2D (Host to Device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iterations):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    end.record()
    end.synchronize()
    
    h2d_time = start.elapsed_time(end) / num_iterations / 1000  # Convert to seconds
    h2d_bandwidth = (size_mb / 1024) / h2d_time  # GB/s
    
    # Benchmark D2H (Device to Host)
    start.record()
    for _ in range(num_iterations):
        cpu_tensor.copy_(gpu_tensor, non_blocking=True)
    end.record()
    end.synchronize()
    
    d2h_time = start.elapsed_time(end) / num_iterations / 1000
    d2h_bandwidth = (size_mb / 1024) / d2h_time
    
    results = {
        "h2d_bandwidth_gbs": h2d_bandwidth,
        "d2h_bandwidth_gbs": d2h_bandwidth,
        "h2d_time_ms": h2d_time * 1000,
        "d2h_time_ms": d2h_time * 1000,
    }
    
    if c2c_links is None:
        system = "NVIDIA Grace (NVLink-C2C status unavailable from nvidia-smi c2c -s)"
    elif c2c_links:
        system = f"NVIDIA Grace with NVLink-C2C ({len(c2c_links)} links up on GPU 0)"
    elif info["is_grace"]:
        system = "NVIDIA Grace (no NVLink-C2C link reported for GPU 0)"
    else:
        system = "Standard"

    print("\n" + "=" * 80)
    print(f"CPU-GPU Bandwidth Benchmark ({size_mb} MB)")
    print("=" * 80)
    print(f"System: {system}")
    print(f"\nResults:")
    print(f"  CPU→GPU: {h2d_bandwidth:.2f} GB/s ({h2d_time*1000:.2f} ms)")
    print(f"  GPU→CPU: {d2h_bandwidth:.2f} GB/s ({d2h_time*1000:.2f} ms)")

    if c2c_links:
        print("\nNVLink-C2C spec peak (not measured here):")
        print(f"  {_C2C_PEAK_PER_DIRECTION_GBPS:.0f} GB/s per direction, 900 GB/s bidirectional")
        print(f"  CPU→GPU share of per-direction peak: {(h2d_bandwidth / _C2C_PEAK_PER_DIRECTION_GBPS) * 100:.1f}%")
    else:
        print(f"\nTypical Performance:")
        print(f"  PCIe 5.0: ~64 GB/s")
        print(f"  PCIe 4.0: ~32 GB/s")
    
    print("=" * 80)
    
    return results


def print_grace_system_info() -> None:
    """
    Print CPU, NUMA, GPU and NVLink-C2C details for this host.

    A superchip name such as GB200 appears only when every GPU reports
    NVLink-C2C links and the GPUs match that superchip's compute capability.
    """
    info = detect_grace_cpu()
    topology = get_numa_topology()
    c2c_by_gpu = (
        {gpu_id: gpu_c2c_links(torch.cuda.get_device_properties(gpu_id), gpu_id) for gpu_id in range(info["gpus"])}
        if info["is_grace"]
        else {}
    )
    superchip = _grace_superchip_name(c2c_by_gpu)

    print("\n" + "=" * 80)
    if superchip is not None:
        print(f"{superchip} Grace Superchip System Information")
    elif info["is_grace"]:
        print("NVIDIA Grace System Information")
    else:
        print("CPU-GPU System Information")
    print("=" * 80)

    # CPU Information
    print("\nCPU:")
    if info["is_grace"]:
        print(f"  NVIDIA Grace CPU detected (SoC ID {info['soc_id']})")
    else:
        print(f"  Architecture: {info['cpu_arch']}")
        print("  Not NVIDIA Grace (no NVIDIA SoC ID under /sys/bus/soc/devices)")

    print(f"  Physical cores: {info['cpu_count']}")
    print(f"  Logical cores: {info['cpu_threads']}")
    print(f"  Memory: {info['memory_gb']:.1f} GB")
    print(f"  NUMA nodes: {info['numa_nodes'] if info['numa_nodes'] is not None else 'unknown'}")

    # NUMA Topology
    if topology:
        print(f"\nNUMA Topology: {len(topology)} nodes")
        for node_id, node_info in sorted(topology.items()):
            print(f"  Node {node_id}:")
            print(f"    CPUs: {len(node_info['cpus'])} ({min(node_info['cpus'])}-{max(node_info['cpus'])})")
            print(f"    Memory: {node_info['size_gb']:.1f} GB")
            if node_info['gpus']:
                print(f"    GPUs: {node_info['gpus']}")
    
    # GPU Information
    if torch.cuda.is_available():
        print(f"\nGPUs: {info['gpus']}")
        for i in range(info["gpus"]):
            props = torch.cuda.get_device_properties(i)
            memory_type = " HBM3e" if (props.major, props.minor) in _HBM3E_CAPABILITIES else ""
            print(f"  GPU {i}: {props.name}")
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    Memory: {props.total_memory / 1e9:.0f} GB{memory_type}")
            print(f"    SMs: {props.multi_processor_count}")

    # NVLink-C2C claims need driver-reported C2C links on every GPU.
    if c2c_by_gpu and not all(c2c_by_gpu.values()):
        print("\nNVLink-C2C: not reported up on every GPU, so no C2C features are claimed")
        for gpu_id, links in sorted(c2c_by_gpu.items()):
            status = "status unavailable from nvidia-smi c2c -s" if links is None else f"{len(links)} links up"
            print(f"  - GPU {gpu_id}: {status}")
    elif c2c_by_gpu:
        print("\nGrace NVLink-C2C Features:")
        for gpu_id, links in sorted(c2c_by_gpu.items()):
            print(f"  - GPU {gpu_id}: {len(links)} NVLink-C2C links up (nvidia-smi c2c -s)")
        print(
            "  - NVLink-C2C spec peak: 900 GB/s bidirectional, "
            f"{_C2C_PEAK_PER_DIRECTION_GBPS:.0f} GB/s per direction (not measured here)"
        )
        print("  - Unified memory address space")
        print("  - Optimal for CPU preprocessing + GPU compute")

        if len(c2c_by_gpu) > 1:
            gpu_props = [torch.cuda.get_device_properties(gpu_id) for gpu_id in c2c_by_gpu]
            print(f"\n{len(gpu_props)}-GPU totals from device properties:")
            print(f"  - SMs: {sum(props.multi_processor_count for props in gpu_props)}")
            print(f"  - Device memory: {sum(props.total_memory for props in gpu_props) / 1e12:.2f} TB")
            if superchip in ("GB200", "GB300"):
                print("  - GPU-GPU NVLink 5 spec peak: 1800 GB/s bidirectional per GPU (not measured here)")
    
    print("=" * 80)


def main():
    """Main demonstration of Grace-Blackwell optimizations."""
    info = detect_grace_cpu()
    if not info["is_grace"]:
        raise RuntimeError(
            "SKIPPED: gb200_grace_numa_optimization requires an NVIDIA Grace CPU "
            f"(host {info['cpu_arch']} reports no NVIDIA SoC ID under /sys/bus/soc/devices)"
        )
    print_grace_system_info()
    
    # Benchmark CPU-GPU bandwidth
    benchmark_cpu_gpu_bandwidth(size_mb=100, num_iterations=100)
    
    # Demonstrate optimal DataLoader configuration
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("DataLoader Optimization Examples")
        print("=" * 80)
        
        for gpu_id in range(min(2, torch.cuda.device_count())):
            optimize_data_loading_for_grace(
                gpu_id=gpu_id,
                num_workers=8,
                verbose=True
            )
            print()
    
    # Usage examples
    print("\n" + "=" * 80)
    print("Usage Examples")
    print("=" * 80)
    print("\n1. Setup CPU affinity for GPU:")
    print("   from extras.ch04.gb200_grace_numa_optimization import setup_grace_affinity")
    print("   cpu_list, numa_node = setup_grace_affinity(gpu_id=0, num_workers=8)")
    
    print("\n2. Configure DataLoader for Grace:")
    print("   from extras.ch04.gb200_grace_numa_optimization import optimize_data_loading_for_grace")
    print("   loader_kwargs = optimize_data_loading_for_grace(gpu_id=0, batch_size=32)")
    print("   dataloader = DataLoader(dataset, **loader_kwargs)")
    
    print("\n3. In multi-GPU training:")
    print("   # In each training process")
    print("   setup_grace_affinity(gpu_id=local_rank, num_workers=8)")
    print("   # Your training code...")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
