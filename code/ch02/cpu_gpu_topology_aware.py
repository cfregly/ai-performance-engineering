"""
CPU-GPU Topology-Aware Tensor Placement
========================================

Demonstrates NUMA-aware tensor placement and CPU-GPU affinity for
various CPU-GPU architectures including:
- GB200/B200 (Grace + Blackwell sm_100)
- GB300/B300 (Grace + Blackwell Ultra sm_103)
- GB10 (Grace + Blackwell Ultra sm_121)
- Any x86_64/ARM + NVIDIA GPU combinations

CPU Architectures Supported:
- ARM Neoverse V2 (Grace)
- x86_64 (Intel Xeon, AMD EPYC)
- Generic ARM aarch64

GPU Architectures Supported:
- Blackwell (sm_100): B200
- Blackwell Ultra (sm_103): B300/GB300
- Grace-Blackwell Ultra (sm_121): GB10
- Hopper (sm_90): H100/H200
- Ada Lovelace (sm_89): RTX 40-series
- Ampere (sm_80, sm_86): A100, RTX 30-series
- Other CUDA-capable GPUs

Key Features:
1. Detect CPU topology and NUMA configuration
2. CPU-GPU affinity binding for optimal data transfer
3. Automatic tensor placement strategies
4. Memory allocation hints for coherent access
5. Support for various interconnect technologies (NVLink-C2C, PCIe, etc.)

Requirements:
- PyTorch 2.10+
- CUDA-capable GPU (graceful fallback on CPU-only systems)
"""
import re
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch


# GPU Architecture mappings
GPU_ARCH_INFO = {
    # Blackwell family
    (10, 0): {"name": "Blackwell", "family": "Blackwell", "sm": "sm_100"},
    (10, 3): {"name": "Blackwell Ultra", "family": "Blackwell", "sm": "sm_103"},
    (12, 1): {"name": "Grace-Blackwell Ultra", "family": "Blackwell", "sm": "sm_121"},
    
    # Hopper family
    (9, 0): {"name": "Hopper", "family": "Hopper", "sm": "sm_90"},
    
    # Ada Lovelace
    (8, 9): {"name": "Ada Lovelace", "family": "Ada", "sm": "sm_89"},
    
    # Ampere family
    (8, 6): {"name": "Ampere", "family": "Ampere", "sm": "sm_86"},
    (8, 0): {"name": "Ampere", "family": "Ampere", "sm": "sm_80"},
    
    # Turing
    (7, 5): {"name": "Turing", "family": "Turing", "sm": "sm_75"},
    
    # Volta
    (7, 0): {"name": "Volta", "family": "Volta", "sm": "sm_70"},
}


def _normalize_pci_bus_id(bus_id: str) -> str:
    value = str(bus_id).strip().lower()
    if value.startswith("00000000:"):
        return value[4:]
    return value


def _read_int(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _query_gpu_pci_bus_id(gpu_index: int) -> Optional[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return _normalize_pci_bus_id(value) if value else None


def _get_gpu_numa_node(gpu_index: int) -> Optional[int]:
    bus_id = _query_gpu_pci_bus_id(gpu_index)
    if not bus_id:
        return None
    return _read_int(Path(f"/sys/bus/pci/devices/{bus_id}/numa_node"))


def _has_known_numa_nodes(cpu_info: Dict[str, Any]) -> bool:
    numa_nodes = cpu_info.get("numa_nodes")
    return isinstance(numa_nodes, int) and numa_nodes > 0


def detect_cpu_info() -> Dict[str, Any]:
    """
    Detect CPU architecture and topology.
    
    Supports ARM (including Grace) and x86_64 architectures.
    
    Returns:
        Dictionary with CPU information
    """
    info = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "num_cores": psutil.cpu_count(logical=False),
        "num_threads": psutil.cpu_count(logical=True),
        "numa_nodes": None,
        "cpu_memory_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_type": "generic",
        "is_grace": False,
    }
    
    # Detect specific CPU types
    if info["architecture"] in ["aarch64", "arm64"]:
        info["cpu_type"] = "ARM"

        # Neoverse is a broad ARM family, not evidence of an NVIDIA Grace CPU.
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
        except OSError:
            cpuinfo = ""
        if re.search(r"\bNeoverse\b", cpuinfo, flags=re.IGNORECASE):
            info["cpu_type"] = "ARM Neoverse"
        if re.search(
            r"\bNVIDIA\s+Grace\b|\bGrace\s+(?:CPU|Processor)\b",
            cpuinfo,
            flags=re.IGNORECASE,
        ):
            info["cpu_type"] = "NVIDIA Grace"
            info["is_grace"] = True
    elif info["architecture"] in ["x86_64", "AMD64"]:
        info["cpu_type"] = "x86_64"
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Intel' in cpuinfo:
                info["cpu_type"] = "Intel x86_64"
            elif 'AMD' in cpuinfo:
                info["cpu_type"] = "AMD x86_64"
    
    # Detect NUMA nodes
    try:
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'available:' in line:
                    info["numa_nodes"] = int(line.split(':')[1].split()[0])
    except FileNotFoundError:
        pass  # NUMA topology unavailable without numactl
    
    return info


def get_gpu_arch_info(major: int, minor: int) -> Dict[str, str]:
    """Get GPU architecture information from compute capability."""
    return GPU_ARCH_INFO.get((major, minor), {
        "name": "Unknown",
        "family": "Unknown",
        "sm": f"sm_{major}{minor}"
    })


def detect_interconnect_type(cpu_info: Dict[str, Any], gpu_info: Dict[str, Any]) -> str:
    """
    Detect the CPU-GPU interconnect type.
    
    Returns:
        Interconnect type string. Bandwidth is labeled as unmeasured unless the
        platform query exposes a negotiated PCIe link speed.
    """
    # NVLink-C2C: Grace CPU + Blackwell/Hopper GPUs
    if cpu_info["is_grace"] and gpu_info["family"] in ["Blackwell", "Hopper"]:
        return "NVLink-C2C (detected; bandwidth unmeasured)"
    
    # NVSwitch: Multi-GPU with NVLink
    if gpu_info.get("nvlink_capable", False):
        return "NVLink (detected; bandwidth unmeasured)"
    
    # PCIe (most common)
    # Try to detect PCIe generation
    try:
        result = subprocess.run(['lspci', '-vv'], capture_output=True, text=True)
        if re.search(r"PCIe Gen5|LnkCap:.*Speed 32GT/s", result.stdout):
            return "PCIe Gen5 (~128 GB/s)"
        elif re.search(r"PCIe Gen4|LnkCap:.*Speed 16GT/s", result.stdout):
            return "PCIe Gen4 (~64 GB/s)"
        elif re.search(r"PCIe Gen3|LnkCap:.*Speed 8GT/s", result.stdout):
            return "PCIe Gen3 (~32 GB/s)"
    except FileNotFoundError:
        pass  # lspci not installed
    
    return "PCIe (~16-128 GB/s)"


def detect_system_topology() -> Dict[str, Any]:
    """
    Detect complete system CPU-GPU topology.
    
    Works with any CPU-GPU combination.
    
    Returns:
        Dictionary with system topology information
    """
    cpu_info = detect_cpu_info()
    
    topology = {
        "cpu_info": cpu_info,
        "num_gpus": 0,
        "gpu_memory_per_device_gb": 0,
        "total_system_memory_gb": cpu_info["cpu_memory_gb"],
        "gpus": [],
        "numa_gpu_mapping": {},
        "interconnect": "None",
        "system_type": "CPU-only",
    }
    
    if not torch.cuda.is_available():
        return topology
    
    topology["num_gpus"] = torch.cuda.device_count()
    
    # Collect GPU information
    gpu_families = set()
    for i in range(topology["num_gpus"]):
        props = torch.cuda.get_device_properties(i)
        arch_info = get_gpu_arch_info(props.major, props.minor)
        
        gpu_families.add(arch_info["family"])
        
        topology["gpus"].append({
            "id": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memory_gb": props.total_memory / (1024**3),
            "arch_family": arch_info["family"],
            "arch_name": arch_info["name"],
            "sm_arch": arch_info["sm"],
        })
        
        if i == 0:
            topology["gpu_memory_per_device_gb"] = props.total_memory / (1024**3)
    
    # Determine system type
    if len(gpu_families) == 1:
        gpu_family = list(gpu_families)[0]
        if cpu_info["is_grace"] and gpu_family == "Blackwell":
            # Check if it's GB10 (Blackwell Ultra) or GB200/GB300 (standard Blackwell)
            if any(g["sm_arch"] == "sm_121" for g in topology["gpus"]):
                topology["system_type"] = "GB10 (Grace + Blackwell Ultra)"
            else:
                topology["system_type"] = "GB200/GB300 (Grace + Blackwell)"
        elif cpu_info["is_grace"] and gpu_family == "Hopper":
            topology["system_type"] = "GH200 (Grace + Hopper)"
        else:
            topology["system_type"] = f"{cpu_info['cpu_type']} + {gpu_family}"
    else:
        topology["system_type"] = f"{cpu_info['cpu_type']} + Mixed GPU"
    
    # Update total system memory
    topology["total_system_memory_gb"] = (
        cpu_info["cpu_memory_gb"] + 
        topology["gpu_memory_per_device_gb"] * topology["num_gpus"]
    )
    
    # Detect interconnect
    if topology["gpus"]:
        topology["interconnect"] = detect_interconnect_type(cpu_info, topology["gpus"][0])
    
    # Detect NUMA-GPU affinity
    if _has_known_numa_nodes(cpu_info):
        for i in range(topology["num_gpus"]):
            numa_node = _get_gpu_numa_node(i)
            if numa_node is not None:
                topology["numa_gpu_mapping"][i] = numa_node
    
    return topology


def print_topology_info(topology: Dict[str, Any]) -> None:
    """Print detailed topology information."""
    print("=" * 80)
    print("CPU-GPU System Topology")
    print("=" * 80)
    
    cpu = topology["cpu_info"]
    print(f"\nCPU Configuration:")
    print(f"  Architecture: {cpu['architecture']}")
    print(f"  CPU Type: {cpu['cpu_type']}")
    print(f"  Physical Cores: {cpu['num_cores']}")
    print(f"  Logical Threads: {cpu['num_threads']}")
    numa_nodes = cpu["numa_nodes"]
    print(f"  NUMA Nodes: {numa_nodes if numa_nodes is not None else 'unknown'}")
    print(f"  CPU Memory: {cpu['cpu_memory_gb']:.1f} GB")
    
    print(f"\nGPU Configuration:")
    print(f"  Number of GPUs: {topology['num_gpus']}")
    for gpu in topology["gpus"]:
        print(f"  GPU {gpu['id']}: {gpu['name']}")
        print(f"    Architecture: {gpu['arch_name']} ({gpu['sm_arch']})")
        print(f"    Compute Capability: {gpu['compute_capability']}")
        print(f"    Memory: {gpu['memory_gb']:.1f} GB")
    
    print(f"\nSystem Type: {topology['system_type']}")
    print(f"Interconnect: {topology['interconnect']}")
    
    if topology['num_gpus'] > 0:
        print(f"\nTotal System Memory: {topology['total_system_memory_gb']:.1f} GB")
        print(f"  CPU Memory: {cpu['cpu_memory_gb']:.1f} GB")
        print(f"  GPU Memory: {topology['gpu_memory_per_device_gb'] * topology['num_gpus']:.1f} GB")
        
        if topology["numa_gpu_mapping"]:
            print(f"\nNUMA-GPU Mapping:")
            for gpu in topology["gpus"]:
                gpu_id = gpu["id"]
                numa_node = topology["numa_gpu_mapping"].get(gpu_id)
                if numa_node is None:
                    print(f"  GPU {gpu_id} → NUMA Node unavailable")
                else:
                    print(f"  GPU {gpu_id} → NUMA Node {numa_node}")
        elif _has_known_numa_nodes(cpu) and cpu["numa_nodes"] > 1:
            print("\nNUMA-GPU Mapping:")
            print("  GPU NUMA locality unavailable from platform APIs")
        elif cpu["numa_nodes"] is None:
            print("\nNUMA-GPU Mapping:")
            print("  CPU NUMA topology unavailable from platform tools")
    
    # System-specific recommendations
    print(f"\nSystem-Specific Optimizations:")
    if "GB200" in topology['system_type'] or "GB300" in topology['system_type'] or "GB10" in topology['system_type']:
        print("  NVLink-C2C enables CPU memory as extended GPU memory")
        print("  Optimal for CPU-GPU hybrid tensor placement")
        print("  Use pinned memory for CPU tensors accessed by GPU")
    elif "GH200" in topology['system_type']:
        print("  NVLink-C2C with Hopper architecture")
        print("  Excellent for unified memory workloads")
    elif _has_known_numa_nodes(cpu) and cpu['numa_nodes'] > 1:
        print("  NUMA-aware process binding recommended")
        print("  Bind each GPU process to nearest NUMA node")
    elif cpu["numa_nodes"] is None:
        print("  CPU NUMA topology unavailable; avoid binding assumptions")
        print("  Use explicit locality only when sysfs or numactl exposes it")
    else:
        print("  Standard CPU-GPU data transfer optimizations")
        print("  Use pinned memory for faster transfers")
    
    print("=" * 80)


def set_cpu_affinity_for_gpu(gpu_id: int, topology: Dict[str, Any]) -> bool:
    """
    Set CPU affinity for current process based on GPU assignment.
    
    Binds process to the NUMA node closest to the GPU.
    
    Args:
        gpu_id: GPU device ID
        topology: System topology dict
        
    Returns:
        True if affinity was set successfully
    """
    numa_nodes = topology["cpu_info"]["numa_nodes"]
    if numa_nodes is None:
        print("⚠ CPU NUMA topology unavailable, skipping CPU affinity")
        return False
    if numa_nodes <= 1:
        print(f"⚠ Single NUMA node system, skipping CPU affinity")
        return False
    
    numa_node = topology["numa_gpu_mapping"].get(gpu_id)
    if numa_node is None:
        print(f"⚠ GPU {gpu_id} NUMA locality unavailable, skipping CPU affinity")
        return False
    
    try:
        # Get CPUs in this NUMA node
        result = subprocess.run(
            ['numactl', '--hardware'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return False
        
        # Parse NUMA node CPU list
        cpu_list = []
        for line in result.stdout.split('\n'):
            if f'node {numa_node} cpus:' in line:
                cpu_str = line.split(':', 1)[1].strip()
                # Parse CPU ranges (e.g., "0-17 36-53")
                for part in cpu_str.split():
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        cpu_list.extend(range(start, end + 1))
                    else:
                        cpu_list.append(int(part))
                break
        
        if cpu_list:
            # Set affinity
            p = psutil.Process()
            p.cpu_affinity(cpu_list)
            print(f"Set CPU affinity for GPU {gpu_id}")
            print(f"  NUMA Node: {numa_node}")
            print(f"  CPUs: {cpu_list[:8]}{'...' if len(cpu_list) > 8 else ''}")
            return True
    except Exception as e:
        print(f"⚠ Failed to set CPU affinity: {e}")
    
    return False


def allocate_tensor_with_numa_hint(
    shape: Tuple[int, ...],
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
    on_cpu_memory: bool = False,
) -> torch.Tensor:
    """
    Allocate tensor with NUMA-aware placement.
    
    Args:
        shape: Tensor shape
        device: Target device
        dtype: Data type
        on_cpu_memory: If True, allocate on CPU memory with GPU access
        
    Returns:
        Allocated tensor
    """
    if on_cpu_memory:
        # Allocate on CPU with GPU access (pinned memory)
        # Optimal on systems with fast CPU-GPU interconnect
        tensor = torch.empty(shape, dtype=dtype, device='cpu', pin_memory=True)
        print(f"  Allocated {tensor.numel() * tensor.element_size() / 1e6:.1f} MB on CPU (pinned)")
    else:
        # Standard GPU allocation
        tensor = torch.empty(shape, dtype=dtype, device=device)
        print(f"  Allocated {tensor.numel() * tensor.element_size() / 1e6:.1f} MB on {device}")
    
    return tensor


def demonstrate_hybrid_placement(topology: Dict[str, Any]) -> None:
    """
    Demonstrate optimal tensor placement based on system topology.
    """
    print("\n=== Hybrid CPU-GPU Tensor Placement ===\n")
    
    # Check if hybrid placement is beneficial
    has_fast_interconnect = "NVLink-C2C" in topology["interconnect"]
    
    if has_fast_interconnect:
        print(f"Fast interconnect detected: {topology['interconnect']}")
        print("  Hybrid placement is highly beneficial\n")
    else:
        print(f"⚠ Interconnect: {topology['interconnect']}")
        print("  Hybrid placement has moderate benefit\n")
    
    # Scenario 1: Large model training
    print("Scenario 1: Large Model Training (7B parameters)")
    print("-" * 60)
    
    # Parameters on GPU (frequently accessed)
    params = allocate_tensor_with_numa_hint(
        (7_000_000_000 // 1000,),  # Simplified
        device="cuda:0",
        dtype=torch.float16,
        on_cpu_memory=False
    )
    print("  Model parameters on GPU")
    
    # Optimizer states - placement depends on interconnect
    if has_fast_interconnect:
        # With NVLink-C2C, put optimizer states on CPU
        optimizer_state_m = allocate_tensor_with_numa_hint(
            (7_000_000_000 // 1000,),
            device="cuda:0",
            dtype=torch.float32,
            on_cpu_memory=True  # On CPU with GPU access
        )
        print(f"  Optimizer momentum on CPU (via {topology['interconnect']})")
        
        optimizer_state_v = allocate_tensor_with_numa_hint(
            (7_000_000_000 // 1000,),
            device="cuda:0",
            dtype=torch.float32,
            on_cpu_memory=True
        )
        print(f"  Optimizer variance on CPU (via {topology['interconnect']})")
    else:
        # With slower interconnect, keep on GPU or use offloading strategically
        optimizer_state_m = allocate_tensor_with_numa_hint(
            (7_000_000_000 // 1000,),
            device="cuda:0",
            dtype=torch.float32,
            on_cpu_memory=False
        )
        print("  Optimizer momentum on GPU")
        
        optimizer_state_v = allocate_tensor_with_numa_hint(
            (7_000_000_000 // 1000,),
            device="cuda:0",
            dtype=torch.float32,
            on_cpu_memory=False
        )
        print("  Optimizer variance on GPU")
    
    total_gpu = params.numel() * params.element_size() / 1e9
    total_cpu = 0
    if has_fast_interconnect:
        total_cpu = (optimizer_state_m.numel() * optimizer_state_m.element_size() +
                     optimizer_state_v.numel() * optimizer_state_v.element_size()) / 1e9
    else:
        total_gpu += (optimizer_state_m.numel() * optimizer_state_m.element_size() +
                      optimizer_state_v.numel() * optimizer_state_v.element_size()) / 1e9
    
    print(f"\nMemory Distribution:")
    print(f"  GPU: {total_gpu:.2f} GB")
    if has_fast_interconnect:
        print(f"  CPU: {total_cpu:.2f} GB (pinned, GPU-accessible)")
        print(f"  GPU memory saved: {total_cpu:.2f} GB ({total_cpu / (total_gpu + total_cpu) * 100:.1f}%)")
    
    # Cleanup
    del params, optimizer_state_m, optimizer_state_v
    
    # Scenario 2: Inference with large KV cache
    print("\n\nScenario 2: Inference with Large KV Cache")
    print("-" * 60)
    
    print("  Model weights: GPU")
    
    if has_fast_interconnect:
        print(f"  KV cache: CPU memory (via {topology['interconnect']})")
        print("    → Store thousands of sequences")
        print("    → High-bandwidth access")
        print("    → Transparent to model code")
    else:
        print("  KV cache: GPU memory (primary)")
        print("  Overflow KV cache: CPU memory (with explicit management)")


def demonstrate_multi_gpu_placement(topology: Dict[str, Any]) -> None:
    """
    Demonstrate NUMA-aware placement for multi-GPU training.
    """
    if topology["num_gpus"] < 2:
        print("\nMulti-GPU placement requires 2+ GPUs")
        return
    
    print("\n=== NUMA-Aware Multi-GPU Placement ===\n")
    
    numa_nodes = topology["cpu_info"]["numa_nodes"]
    if numa_nodes is None:
        print("CPU NUMA topology unavailable:")
        print("  → Do not infer socket-local placement")
        print("  → Leave CPU binding unchanged until locality is known")
    elif numa_nodes > 1:
        print("NUMA-aware configuration recommended:")
        print()
        
        for gpu_id in range(min(topology["num_gpus"], 8)):
            numa_node = topology["numa_gpu_mapping"].get(gpu_id)
            print(f"Process {gpu_id} (GPU {gpu_id}):")
            if numa_node is None:
                print("  → GPU NUMA locality unavailable from platform APIs")
                print("  → Leave CPU binding unchanged until locality is known")
                print("  → Benefit: avoids binding to the wrong NUMA node")
            else:
                print(f"  → Bind to NUMA node {numa_node}")
                print(f"  → Allocate CPU tensors from NUMA node {numa_node}")
                print(f"  → DataLoader workers use NUMA node {numa_node} CPUs")
                print("  → Benefit: Reduced memory access latency")
            print()
    else:
        print("Single NUMA node system:")
        print("  → Standard multi-GPU setup")
        print("  → Use DDP or FSDP for data parallelism")
        print("  → No special NUMA binding needed")


def main():
    """Main demonstration."""
    print("=" * 80)
    print("CPU-GPU Topology-Aware Tensor Placement")
    print("=" * 80)
    print()
    
    # Detect topology
    topology = detect_system_topology()
    print_topology_info(topology)
    
    if topology["num_gpus"] == 0:
        print("\n⚠ No GPUs detected. Demonstration requires CUDA-capable GPU.")
        return
    
    # Demonstrate placement strategies
    demonstrate_hybrid_placement(topology)
    
    # Multi-GPU placement
    if topology["num_gpus"] >= 2:
        demonstrate_multi_gpu_placement(topology)
    
    # Best practices
    print("\n=== Best Practices for CPU-GPU Systems ===\n")
    
    if "NVLink-C2C" in topology["interconnect"]:
        print("For Grace-Hopper/Blackwell superchips:")
        print("  1. Place frequently-accessed data on GPU memory")
        print("  2. Place optimizer states and KV cache on CPU memory")
        print("  3. Use pinned memory for CPU allocations")
        print("  4. Bind processes to NUMA nodes matching their GPUs")
        print("  5. Use torch.cuda.Stream for overlapping CPU↔GPU transfers")
    else:
        print("For general CPU-GPU systems:")
        print("  1. Keep working set on GPU memory")
        print("  2. Use pinned memory for CPU-GPU transfers")
        print("  3. Minimize CPU↔GPU data movement")
        print("  4. Use CUDA streams for async transfers")
        print("  5. Consider model parallelism for large models")
    
    print()
    print("Expected Performance:")
    if topology["interconnect"]:
        print(f"  - Interconnect: {topology['interconnect']}")
    print(f"  - Total Memory: {topology['total_system_memory_gb']:.1f} GB")
    print(f"    CPU: {topology['cpu_info']['cpu_memory_gb']:.1f} GB")
    if topology["num_gpus"] > 0:
        print(f"    GPU: {topology['gpu_memory_per_device_gb'] * topology['num_gpus']:.1f} GB")
    print("=" * 80)


if __name__ == "__main__":
    main()
