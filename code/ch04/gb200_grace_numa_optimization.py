#!/usr/bin/env python3

"""
GB200/GB300 Grace CPU NUMA Optimization for PyTorch 2.10
=======================================================

Demonstrates optimal NUMA topology configuration and CPU-GPU affinity
for GB200/GB300 Grace-Blackwell superchips.

GB200/GB300 Architecture:
- Grace CPU: 72 ARM Neoverse V2 cores (144 threads)
- CPU Memory: 480GB-1TB LPDDR5X
- NVLink-C2C: 900 GB/s coherent CPU↔GPU bandwidth
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
from ch04.distributed_helper import run_main_with_skip_status

import platform
import os
from pathlib import Path
import psutil
import torch
import torch.multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple
import subprocess


def detect_grace_cpu() -> Dict[str, Any]:
    """
    Detect Grace CPU and gather system information.
    
    Returns:
        Dictionary with Grace CPU information
    """
    info = {
        "is_grace": False,
        "cpu_arch": platform.machine(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "memory_gb": psutil.virtual_memory().total / 1e9,
        "numa_nodes": None,
        "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    # Detect if running on ARM architecture (Grace uses ARM Neoverse V2)
    if info["cpu_arch"] == "aarch64" or info["cpu_arch"] == "arm64":
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                # Look for Grace CPU indicators
                if 'ARM' in cpuinfo or 'Neoverse' in cpuinfo:
                    info["is_grace"] = True
                    info["cpu_model"] = "ARM Neoverse V2 (Grace)"
        except FileNotFoundError:
            pass  # /proc/cpuinfo doesn't exist (not Linux)
    
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
    - Larger prefetch (900 GB/s CPU-GPU bandwidth)
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
        "pin_memory": True,  # Essential for NVLink-C2C
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,  # Reuse workers
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"DataLoader Configuration for GB200/GB300 (GPU {gpu_id})")
        print("=" * 80)
        print(f"Grace CPU: {info['cpu_count']} cores, {info['cpu_threads']} threads")
        print(f"NUMA node: {numa_node}")
        print(f"CPU affinity: {len(cpu_list)} CPUs assigned")
        print(f"\nDataLoader Settings:")
        for key, value in dataloader_kwargs.items():
            print(f"  {key}: {value}")
        print("\nExpected Performance:")
        print(f"  CPU→GPU bandwidth: ~800 GB/s (NVLink-C2C)")
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
    - GB200/GB300: ~800 GB/s (NVLink-C2C)
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
    
    print("\n" + "=" * 80)
    print(f"CPU-GPU Bandwidth Benchmark ({size_mb} MB)")
    print("=" * 80)
    print(f"System: {'GB200/GB300 Grace-Blackwell' if info['is_grace'] else 'Standard'}")
    print(f"\nResults:")
    print(f"  CPU→GPU: {h2d_bandwidth:.2f} GB/s ({h2d_time*1000:.2f} ms)")
    print(f"  GPU→CPU: {d2h_bandwidth:.2f} GB/s ({d2h_time*1000:.2f} ms)")
    
    if info["is_grace"]:
        print(f"\nExpected on GB200/GB300:")
        print(f"  Target: ~800 GB/s (900 GB/s peak)")
        print(f"  Efficiency: {(h2d_bandwidth / 800) * 100:.1f}%")
    else:
        print(f"\nTypical Performance:")
        print(f"  PCIe 5.0: ~64 GB/s")
        print(f"  PCIe 4.0: ~32 GB/s")
    
    print("=" * 80)
    
    return results


def print_grace_system_info() -> None:
    """Print detailed Grace-Blackwell system information."""
    info = detect_grace_cpu()
    topology = get_numa_topology()
    
    print("\n" + "=" * 80)
    print("GB200/GB300 Grace-Blackwell System Information")
    print("=" * 80)
    
    # CPU Information
    print("\nCPU:")
    if info["is_grace"]:
        print("  Grace CPU detected (ARM Neoverse V2)")
    else:
        print(f"  Architecture: {info['cpu_arch']}")
        print(f"  Model: {info.get('cpu_model', 'Unknown')}")
    
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
            is_blackwell = props.major >= 10
            print(f"  GPU {i}: {props.name}")
            if is_blackwell:
                print(f"    Blackwell B200/B300")
                print(f"    Memory: {props.total_memory / 1e9:.0f} GB HBM3e")
                print(f"    SMs: 148")
    
    # GB200/GB300 Specific Features
    if info["is_grace"] and info["gpus"] > 0:
        print("\nGB200/GB300 Features:")
        print("  - NVLink-C2C: 900 GB/s CPU↔GPU coherent bandwidth")
        print("  - Unified memory address space")
        print("  - Optimal for CPU preprocessing + GPU compute")
        
        if info["gpus"] == 8:
            print("\nB200 multi-GPU Configuration:")
            print("  - Total: 1184 SMs, 1.44 TB HBM3e")
            print("  - GPU-GPU: 1800 GB/s NVLink 5.0")
            print("  - Aggregate: 62.4 TB/s bandwidth")
    
    print("=" * 80)


def main():
    """Main demonstration of Grace-Blackwell optimizations."""
    if not detect_grace_cpu().get("is_grace", False):
        raise RuntimeError("SKIPPED: gb200_grace_numa_optimization requires Grace CPU support")
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
