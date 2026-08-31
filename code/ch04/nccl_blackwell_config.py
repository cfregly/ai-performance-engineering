#!/usr/bin/env python3
"""NCCL 2.28 Blackwell configuration.

NCCL discovers NVLink and NVLink-C2C from the hardware topology. This module
only exposes documented NCCL environment controls and leaves fabric discovery
to NCCL.

Key Features:
- NVLink 5.0 support (900 GB/s per GPU pair)
- Automatic NVLink/NVLink-C2C topology discovery
- Optimized algorithms for Blackwell topology

Requirements:
- PyTorch 2.10+
- NCCL 2.28+
- Multiple Blackwell GPUs
- CUDA 13.0+

Usage:
    from ch04.nccl_blackwell_config import configure_nccl_for_blackwell
    configure_nccl_for_blackwell()
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist


def _ib_devices_present() -> bool:
    if os.environ.get("NCCL_IB_HCA"):
        return True
    ib_path = Path("/sys/class/infiniband")
    if not ib_path.exists():
        return False
    try:
        return any(ib_path.iterdir())
    except OSError:
        return False


def configure_nccl_for_blackwell(
    algo: str = "Ring,Tree",
    verbose: bool = True,
) -> Dict[str, str]:
    """Configure NCCL 2.28 for optimal Blackwell performance.

    Args:
        algo: NCCL algorithms to use (Ring, Tree, or both)
        verbose: Print configuration details

    Returns:
        Dictionary of environment variables set
    """
    env_vars: Dict[str, str] = {}

    # 1. NCCL Protocol - Simple is best for Blackwell NVLink 5.0
    os.environ["NCCL_PROTO"] = "Simple"
    env_vars["NCCL_PROTO"] = "Simple"

    # 2. NCCL Algorithms - Ring + Tree for best performance
    os.environ["NCCL_ALGO"] = algo
    env_vars["NCCL_ALGO"] = algo

    # 3. Cross NIC - Enable for multi-node
    os.environ.setdefault("NCCL_CROSS_NIC", "1")
    env_vars["NCCL_CROSS_NIC"] = "1"

    # 4. P2P Level - Prefer NVLink when topology discovery reports it
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")  # NVLink level
    env_vars["NCCL_P2P_LEVEL"] = "NVL"

    # 5. IB (InfiniBand) optimizations for multi-node
    ib_disable_override = os.environ.get("NCCL_IB_DISABLE")
    if ib_disable_override is not None:
        env_vars["NCCL_IB_DISABLE"] = ib_disable_override
        if ib_disable_override == "0":
            os.environ.setdefault("NCCL_IB_HCA", "mlx5")
            env_vars["NCCL_IB_HCA"] = os.environ["NCCL_IB_HCA"]
    elif _ib_devices_present():
        os.environ.setdefault("NCCL_IB_DISABLE", "0")  # Enable IB when detected.
        os.environ.setdefault("NCCL_IB_HCA", "mlx5")  # Mellanox adapters
        env_vars["NCCL_IB_DISABLE"] = "0"
        env_vars["NCCL_IB_HCA"] = os.environ["NCCL_IB_HCA"]
    else:
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        env_vars["NCCL_IB_DISABLE"] = "1"

    # 6. Socket NUMA affinity
    os.environ.setdefault("NCCL_SOCKET_NTHREADS", "4")
    os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "8")
    env_vars["NCCL_SOCKET_NTHREADS"] = "4"
    env_vars["NCCL_NSOCKS_PERTHREAD"] = "8"

    # 7. Buffer sizes - Tuned for Blackwell
    os.environ.setdefault("NCCL_BUFFSIZE", str(32 * 1024 * 1024))  # 32 MB
    os.environ.setdefault("NCCL_LL_THRESHOLD", "0")  # Use low-latency
    env_vars["NCCL_BUFFSIZE"] = str(32 * 1024 * 1024)
    env_vars["NCCL_LL_THRESHOLD"] = "0"

    # 8. Graph support - Enable for torch.compile
    os.environ.setdefault("NCCL_GRAPH_REGISTER", "1")
    env_vars["NCCL_GRAPH_REGISTER"] = "1"

    # 9. Debug level (set to INFO for initial tuning, WARN for production)
    if verbose:
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,ENV")
        env_vars["NCCL_DEBUG"] = "INFO"
        env_vars["NCCL_DEBUG_SUBSYS"] = "INIT,GRAPH,ENV"

    if verbose:
        print("=" * 80)
        print("NCCL 2.28 Blackwell Configuration")
        print("=" * 80)
        for key, value in sorted(env_vars.items()):
            print(f"  {key}={value}")
        print("=" * 80)
        print("\nKey Features Enabled:")
        print("  Topology discovery: automatic (including NVLink/NVLink-C2C)")
        print(f"  Algorithms: {algo}")
        print("=" * 80)

    return env_vars


def configure_nccl_for_multigpu(
    *,
    num_gpus: int | None = None,
    enable_nvls: bool = True,
    num_channels: int | None = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Configure NCCL 2.28 optimized for multi-GPU Blackwell/GB topologies.

    Args:
        num_gpus: GPU count (defaults to all available GPUs)
        enable_nvls: Enable NVLink Sharp (NVLS) when available
        num_channels: Override NCCL channels (auto-derived if None)
        verbose: Print configuration details

    Returns:
        Dictionary of environment variables set
    """
    if num_gpus is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to configure NCCL for multi-GPU.")
        num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise ValueError(f"num_gpus must be >=2, got {num_gpus}")

    env_vars = configure_nccl_for_blackwell(
        algo="Ring,Tree",
        verbose=False,
    )

    if num_channels is None:
        if num_gpus >= 8:
            num_channels = 16
        elif num_gpus >= 4:
            num_channels = 12
        else:
            num_channels = 8

    os.environ["NCCL_NCHANNELS_PER_NET_PEER"] = str(num_channels)
    env_vars["NCCL_NCHANNELS_PER_NET_PEER"] = str(num_channels)

    if enable_nvls and num_gpus >= 4:
        os.environ["NCCL_NVLS_ENABLE"] = "1"
        env_vars["NCCL_NVLS_ENABLE"] = "1"
        os.environ["NCCL_ALGO"] = "Tree,Ring,NVLS"
        env_vars["NCCL_ALGO"] = "Tree,Ring,NVLS"
    else:
        os.environ["NCCL_ALGO"] = "Tree,Ring"
        env_vars["NCCL_ALGO"] = "Tree,Ring"

    if num_gpus >= 8:
        os.environ["NCCL_BUFFSIZE"] = str(64 * 1024 * 1024)
        os.environ["NCCL_MIN_NCHANNELS"] = str(max(8, num_channels // 2))
        os.environ["NCCL_MAX_NCHANNELS"] = str(num_channels)
    elif num_gpus >= 4:
        os.environ["NCCL_BUFFSIZE"] = str(32 * 1024 * 1024)
        os.environ["NCCL_MIN_NCHANNELS"] = str(max(4, num_channels // 2))
        os.environ["NCCL_MAX_NCHANNELS"] = str(num_channels)
    else:
        os.environ["NCCL_BUFFSIZE"] = str(16 * 1024 * 1024)
        os.environ["NCCL_MIN_NCHANNELS"] = str(max(2, num_channels // 2))
        os.environ["NCCL_MAX_NCHANNELS"] = str(num_channels)

    env_vars["NCCL_BUFFSIZE"] = os.environ["NCCL_BUFFSIZE"]
    env_vars["NCCL_MIN_NCHANNELS"] = os.environ["NCCL_MIN_NCHANNELS"]
    env_vars["NCCL_MAX_NCHANNELS"] = os.environ["NCCL_MAX_NCHANNELS"]

    if verbose:
        print("=" * 80)
        print(f"NCCL 2.28 Configuration for {num_gpus}x Blackwell/GB GPUs")
        print("=" * 80)
        for key in sorted(env_vars.keys()):
            if key.startswith("NCCL_"):
                print(f"  {key}={env_vars[key]}")
        print("=" * 80)

    return env_vars


def verify_nccl_configuration() -> Dict[str, Any]:
    """Verify NCCL configuration and GPU topology.

    Returns:
        Dictionary with configuration status
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    info: Dict[str, Any] = {
        "nccl_version": (torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else "unknown"),
        "num_gpus": torch.cuda.device_count(),
        "gpus": [],
    }

    # Check each GPU
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_info = {
            "id": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / 1e9,
            "is_blackwell": props.major in (10, 12),
        }
        info["gpus"].append(gpu_info)

    # Check P2P access
    if info["num_gpus"] >= 2:
        info["p2p_access"] = []
        for i in range(min(info["num_gpus"], 4)):  # Check first 4 GPUs
            for j in range(i + 1, min(info["num_gpus"], 4)):
                can_access = torch.cuda.can_device_access_peer(i, j)
                info["p2p_access"].append({"from": i, "to": j, "accessible": can_access})

    return info


def print_nccl_topology() -> None:
    """Print NCCL topology information."""
    info = verify_nccl_configuration()

    print("\n" + "=" * 80)
    print("NCCL Configuration & Topology")
    print("=" * 80)

    if "error" in info:
        print(f"Error: {info['error']}")
        return

    print(f"NCCL Version: {info['nccl_version']}")
    print(f"Number of GPUs: {info['num_gpus']}")
    print()
    print("GPU Details:")
    for gpu in info["gpus"]:
        blackwell_marker = " (Blackwell/GB series)" if gpu["is_blackwell"] else ""
        print(f"  GPU {gpu['id']}: {gpu['name']}{blackwell_marker}")
        print(f"    Compute Capability: {gpu['compute_capability']}")
        print(f"    Memory: {gpu['total_memory_gb']:.1f} GB")

    if "p2p_access" in info:
        print("\nP2P Access Matrix:")
        all_accessible = all(p["accessible"] for p in info["p2p_access"])
        for p in info["p2p_access"]:
            status = "✓" if p["accessible"] else "✗"
            print(f"  {status} GPU {p['from']} <-> GPU {p['to']}")
        if all_accessible:
            print("\n  ✓ All GPUs have P2P access (NVLink detected)")
        else:
            print("\n  ⚠ Warning: Not all GPUs have P2P access")

    print("=" * 80)


def configure_nccl_for_gb200_gb300(verbose: bool = True) -> Dict[str, str]:
    """Configure NCCL specifically for GB200/GB300 Grace-Blackwell Superchips.

    GB200/GB300 Architecture:
    - Grace CPU: 72 ARM Neoverse V2 cores
    - CPU Memory: 480GB-1TB LPDDR5X
    - NVLink-C2C: 900 GB/s coherent CPU-GPU bandwidth
    - Unified memory address space (CPU + GPU)

    Args:
        verbose: Print configuration details

    Returns:
        Dictionary of environment variables set
    """
    # Start with multi-GPU config as base
    env_vars = configure_nccl_for_multigpu(
        num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 4,
        enable_nvls=True,
        num_channels=8,
        verbose=False,
    )

    # GB200/GB300 specific enhancements. NCCL detects C2C from topology;
    # configure only documented CPU/network controls here.
    # 1. Grace-specific: Enhanced CPU-GPU affinity
    os.environ["NCCL_SOCKET_NTHREADS"] = "12"  # More threads for 72 ARM cores
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    env_vars["NCCL_SOCKET_NTHREADS"] = "12"
    env_vars["NCCL_NSOCKS_PERTHREAD"] = "4"

    # 2. Unified memory support
    os.environ["NCCL_CUMEM_ENABLE"] = "1"
    env_vars["NCCL_CUMEM_ENABLE"] = "1"

    # 3. Grace CPU network interface
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0,en0"
    env_vars["NCCL_SOCKET_IFNAME"] = "eth0,en0"

    # 4. InfiniBand settings for Grace
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    env_vars["NCCL_IB_GID_INDEX"] = "3"

    if verbose:
        print("=" * 80)
        print("NCCL Configuration for GB200/GB300 Grace-Blackwell Superchip")
        print("=" * 80)
        print("\nGrace CPU Specifications:")
        print("  Architecture: ARM Neoverse V2")
        print("  Cores: 72 (144 threads)")
        print("  Memory: 480GB-1TB LPDDR5X")
        print("  CPU-GPU: 900 GB/s (NVLink-C2C)")
        print("=" * 80)

    return env_vars


def detect_b200_multigpu_topology() -> Dict[str, Any]:
    """Detect multi-GPU Blackwell/GB GPU topology and NVLink configuration.

    Returns:
        Dictionary with topology information
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    num_gpus = torch.cuda.device_count()
    info: Dict[str, Any] = {
        "num_gpus": num_gpus,
        "is_b200_multigpu": False,
        "has_nvswitch": False,
        "has_grace_cpu": False,
        "nvlink_topology": "unknown",
        "gpus": [],
    }

    if num_gpus < 2:
        info["warning"] = f"Expected >=2 GPUs, found {num_gpus}"
        return info

    # Check if all GPUs are Blackwell B200
    all_blackwell = True
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        is_blackwell = props.major in (10, 12)
        all_blackwell = all_blackwell and is_blackwell
        info["gpus"].append(
            {
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1e9,
                "is_blackwell": is_blackwell,
            }
        )

    info["is_b200_multigpu"] = all_blackwell

    # Check P2P access pattern to infer topology
    if num_gpus >= 2:
        p2p_matrix = []
        for i in range(num_gpus):
            row = []
            for j in range(num_gpus):
                if i != j:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    row.append(can_access)
                else:
                    row.append(True)
            p2p_matrix.append(row)

        # Check if all-to-all (NVSwitch) or partial (direct NVLink)
        all_connected = all(all(row) for row in p2p_matrix)
        info["has_nvswitch"] = all_connected
        if all_connected:
            info["nvlink_topology"] = "nvswitch_all_to_all"
        else:
            info["nvlink_topology"] = "direct_nvlink_pairs"

    # Check for Grace CPU (GB200/GB300)
    try:
        import platform

        if platform.machine() == "aarch64":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                if "ARM" in cpuinfo or "Neoverse" in cpuinfo:
                    info["has_grace_cpu"] = True
    except FileNotFoundError:
        pass  # /proc/cpuinfo doesn't exist (not Linux)

    return info


def print_multigpu_topology() -> None:
    """Print topology information for multi-GPU Blackwell/GB configuration."""
    info = detect_b200_multigpu_topology()

    print("\n" + "=" * 80)
    print("Multi-GPU Blackwell/GB GPU Topology Detection")
    print("=" * 80)

    if "error" in info:
        print(f"Error: {info['error']}")
        return

    if "warning" in info:
        print(f"⚠ Warning: {info['warning']}")

    print(f"Number of GPUs: {info['num_gpus']}")
    status = "✓ Yes" if info["is_b200_multigpu"] else "✗ No"
    print(f"B200/GB multi-GPU Configuration: {status}")

    if info["is_b200_multigpu"]:
        print(f"\nTopology: {info['nvlink_topology']}")
        if info["has_nvswitch"]:
            print("  ✓ NVSwitch detected (all-to-all connectivity)")
        else:
            print("  ℹ Direct NVLink pairs")

        if info["has_grace_cpu"]:
            print("\n✓ Grace CPU detected (GB200/GB300 Superchip)")
        else:
            print("\nℹ Standard B200 configuration (PCIe or NVLink only)")

        print(f"\nTotal Resources:")
        print(f"  SMs: {148 * info['num_gpus']} (148 per GPU)")
        print(f"  Memory: {180 * info['num_gpus']} GB HBM3e")
        print(f"  Bandwidth: {7.8 * info['num_gpus']:.1f} TB/s aggregate")

    print("=" * 80)


def benchmark_nccl_allreduce(
    tensor_size_mb: int = 256,
    num_iterations: int = 100,
    warmup: int = 10,
) -> float:
    """Benchmark NCCL allreduce performance.

    Args:
        tensor_size_mb: Size of tensor in MB
        num_iterations: Number of iterations to benchmark
        warmup: Number of warmup iterations

    Returns:
        Bandwidth in GB/s
    """
    if not dist.is_initialized():
        print("Distributed not initialized. Call dist.init_process_group() first.")
        return 0.0

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())

    # Create tensor
    num_elements = tensor_size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
    tensor = torch.randn(num_elements, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    dist.barrier()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end.record()
    end.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_iterations

    # Calculate bandwidth
    data_size_gb = tensor_size_mb / 1024  # GB
    busbw = data_size_gb * 2 * (world_size - 1) / world_size / (elapsed_ms / 1000)

    if rank == 0:
        print(f"\nAllReduce Benchmark:")
        print(f"  Tensor size: {tensor_size_mb} MB")
        print(f"  Time per iteration: {elapsed_ms:.3f} ms")
        print(f"  Bus bandwidth: {busbw:.2f} GB/s")

    return busbw


def main() -> None:
    """Run NCCL configuration and benchmarks."""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    topology = detect_b200_multigpu_topology()
    is_grace_blackwell = topology.get("has_grace_cpu", False)

    if num_gpus >= 2:
        print(f"Detected {num_gpus} GPUs - using optimized multi-GPU Blackwell/GB configuration")
        print_multigpu_topology()
        print()
        if is_grace_blackwell:
            print("Grace CPU detected - using GB200/GB300 configuration")
            configure_nccl_for_gb200_gb300(verbose=True)
        else:
            configure_nccl_for_multigpu(num_gpus=num_gpus, verbose=True)
    else:
        print(f"Detected {num_gpus} GPUs - using standard Blackwell/GB configuration")
        configure_nccl_for_blackwell(verbose=True)
        print_nccl_topology()


if __name__ == "__main__":
    main()
