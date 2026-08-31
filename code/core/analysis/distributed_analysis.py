#!/usr/bin/env python3
"""
🌐 Distributed Training & Multi-Node Analysis

Real analysis for large distributed training clusters:
- Multi-node GPU topology discovery
- NCCL communication profiling
- Scaling efficiency analysis
- Load balancing detection
- Network bottleneck identification
- Optimal parallelism strategy recommendation

Supports:
- SLURM clusters
- Kubernetes (with GPU operator)
- Bare metal multi-node
- Cloud (AWS, GCP, Azure)

Usage:
    python -m analysis.distributed_analysis topology
    python -m analysis.distributed_analysis scaling --nodes 8
    python -m analysis.distributed_analysis comm-profile
    python -m analysis.distributed_analysis recommend --model llama-70b --nodes 4
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from types import SimpleNamespace
import typer
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# =============================================================================
# CLUSTER TOPOLOGY DISCOVERY
# =============================================================================

@dataclass
class GPUInfo:
    node: str
    gpu_id: int
    name: str
    memory_gb: float
    nvlink_connections: List[int]
    pcie_bandwidth_gbps: float


@dataclass 
class NodeInfo:
    hostname: str
    ip: str
    gpus: List[GPUInfo]
    cpu_cores: int
    memory_gb: float
    network_bandwidth_gbps: float
    ib_available: bool


@dataclass
class ClusterTopology:
    nodes: List[NodeInfo]
    total_gpus: int
    interconnect: str  # nvlink, nvswitch, pcie, ethernet, infiniband
    network_topology: str  # flat, fat-tree, dragonfly


class ClusterDiscovery:
    """Discover cluster topology and capabilities."""
    
    def __init__(self):
        self.hostname = socket.gethostname()
    
    def discover_local_gpus(self) -> List[Dict[str, Any]]:
        """Discover GPUs on local node."""
        gpus = []
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        # Calculate PCIe bandwidth
                        pcie_gen = int(parts[3]) if parts[3].isdigit() else 4
                        pcie_width = int(parts[4].replace('x', '')) if 'x' in parts[4] or parts[4].isdigit() else 16
                        pcie_bw = {3: 1.0, 4: 2.0, 5: 4.0}.get(pcie_gen, 2.0) * pcie_width  # GB/s
                        
                        gpus.append({
                            "id": int(parts[0]),
                            "name": parts[1],
                            "memory_gb": int(parts[2]) / 1024,
                            "pcie_bandwidth_gbps": pcie_bw,
                        })
        except Exception as e:
            print(f"Warning: Could not query GPUs: {e}")
        
        return gpus
    
    def discover_nvlink_topology(self) -> Dict[int, List[int]]:
        """Return peer GPU IDs from the NV# entries in ``nvidia-smi topo -m``.

        Local NVLink port IDs from ``nvlink -s`` do not identify peer GPUs.
        The matrix also does not distinguish direct links from NVSwitch paths.
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', 'topo', '-m'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise ValueError(result.stderr.strip() or f"nvidia-smi exited {result.returncode}")

            nvlink: Dict[int, List[int]] = {}
            gpu_columns = None
            for line in result.stdout.splitlines():
                cells = line.split()
                if not cells:
                    continue
                if gpu_columns is None:
                    columns = [
                        (index, int(match.group(1)))
                        for index, cell in enumerate(cells)
                        if (match := re.fullmatch(r"GPU(\d+)", cell, flags=re.IGNORECASE))
                    ]
                    is_data_row = any(
                        re.fullmatch(r"X|NV\d+|SYS|NODE|PHB|PXB|PIX", cell, flags=re.IGNORECASE)
                        for cell in cells[1:]
                    )
                    if columns and not is_data_row:
                        gpu_columns = columns
                    continue

                gpu_match = re.fullmatch(r"GPU(\d+)", cells[0], flags=re.IGNORECASE)
                if not gpu_match:
                    continue
                gpu_id = int(gpu_match.group(1))
                if len(cells) <= max(index for index, _ in gpu_columns) + 1:
                    raise ValueError(f"incomplete matrix row for GPU {gpu_id}")
                for index, _ in gpu_columns:
                    if not re.fullmatch(
                        r"X|NV[1-9]\d*|SYS|NODE|PHB|PXB|PIX", cells[index + 1], flags=re.IGNORECASE
                    ):
                        raise ValueError(f"unrecognized matrix entry for GPU {gpu_id}: {cells[index + 1]}")
                nvlink[gpu_id] = [
                    peer_id
                    for index, peer_id in gpu_columns
                    if peer_id != gpu_id
                    and re.fullmatch(r"NV[1-9]\d*", cells[index + 1], flags=re.IGNORECASE)
                ]

            if gpu_columns is None or set(nvlink) != {gpu_id for _, gpu_id in gpu_columns}:
                raise ValueError("missing GPU matrix header or rows")
            return nvlink
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            print(f"Warning: Could not query NVLink topology: {exc}", file=sys.stderr)
            return {}
    
    def discover_slurm_nodes(self) -> List[str]:
        """Discover nodes in SLURM allocation."""
        nodes = []
        
        # Check SLURM environment
        nodelist = os.environ.get('SLURM_JOB_NODELIST', '')
        if nodelist:
            try:
                result = subprocess.run(
                    ['scontrol', 'show', 'hostnames', nodelist],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    nodes = result.stdout.strip().split('\n')
            except Exception:
                pass
        
        return nodes if nodes else [self.hostname]
    
    def discover_kubernetes_nodes(self) -> List[str]:
        """Discover GPU nodes in Kubernetes cluster."""
        nodes = []
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'nodes', '-l', 'nvidia.com/gpu=true', '-o', 'jsonpath={.items[*].metadata.name}'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                nodes = result.stdout.strip().split()
        except Exception:
            pass
        
        return nodes if nodes else [self.hostname]
    
    def check_infiniband(self) -> Tuple[bool, float]:
        """Check if InfiniBand is available and get bandwidth."""
        try:
            result = subprocess.run(['ibstat'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'Active' in result.stdout:
                # Parse rate
                for line in result.stdout.split('\n'):
                    if 'Rate' in line:
                        rate = line.split(':')[1].strip()
                        bw = float(rate.split()[0]) if rate else 200.0
                        return True, bw
                return True, 200.0  # Default HDR IB
        except Exception:
            pass
        return False, 0.0
    
    def get_full_topology(self) -> Dict[str, Any]:
        """Get complete cluster topology."""
        # Discover nodes
        slurm_nodes = self.discover_slurm_nodes()
        k8s_nodes = self.discover_kubernetes_nodes()
        
        nodes = slurm_nodes if len(slurm_nodes) > 1 else k8s_nodes
        
        # Local GPU info
        local_gpus = self.discover_local_gpus()
        nvlink_topo = self.discover_nvlink_topology()
        ib_available, ib_bandwidth = self.check_infiniband()
        
        # Determine interconnect
        if any(nvlink_topo.values()) and len(local_gpus) > 1:
            interconnect = "nvlink"
        elif ib_available:
            interconnect = "infiniband"
        else:
            interconnect = "ethernet"
        
        return {
            "hostname": self.hostname,
            "nodes": nodes,
            "num_nodes": len(nodes),
            "local_gpus": local_gpus,
            "gpus_per_node": len(local_gpus),
            "total_gpus": len(nodes) * len(local_gpus),
            "nvlink_topology": nvlink_topo,
            "interconnect": interconnect,
            "infiniband": {
                "available": ib_available,
                "bandwidth_gbps": ib_bandwidth,
            },
            "scheduler": "slurm" if os.environ.get('SLURM_JOB_ID') else "kubernetes" if os.environ.get('KUBERNETES_SERVICE_HOST') else "standalone",
        }


# =============================================================================
# SCALING EFFICIENCY ANALYSIS
# =============================================================================

class ScalingAnalyzer:
    """Analyze distributed training scaling efficiency."""
    
    def __init__(self, topology: Dict[str, Any]):
        self.topology = topology
    
    def estimate_scaling_efficiency(self, model_params_b: float, 
                                    batch_size: int,
                                    sequence_length: int,
                                    num_gpus: int) -> Dict[str, Any]:
        """Estimate scaling efficiency for given configuration."""
        
        # Memory per GPU (rough estimate)
        bytes_per_param = 2  # FP16
        model_memory_gb = model_params_b * bytes_per_param
        
        # Activation memory (rough)
        hidden_dim = int((model_params_b * 1e9 / 100) ** 0.5)  # Rough estimate
        activation_memory_gb = batch_size * sequence_length * hidden_dim * 4 / 1e9
        
        # Communication volume (all-reduce)
        # For data parallel: 2 * model_size * (n-1)/n
        comm_volume_gb = 2 * model_memory_gb * (num_gpus - 1) / num_gpus
        
        # Estimate computation time (rough)
        flops_per_token = 6 * model_params_b * 1e9  # Forward + backward
        total_flops = flops_per_token * batch_size * sequence_length
        
        # Assume B200 at ~1000 TFLOPS effective
        gpu_tflops = 1000
        compute_time_ms = total_flops / (gpu_tflops * 1e12) * 1000
        
        # Communication time
        if self.topology.get('interconnect') == 'nvswitch':
            bandwidth_gbps = 900  # NVSwitch
        elif self.topology.get('interconnect') == 'nvlink':
            bandwidth_gbps = 600  # NVLink
        elif self.topology.get('infiniband', {}).get('available'):
            bandwidth_gbps = self.topology['infiniband']['bandwidth_gbps']
        else:
            bandwidth_gbps = 100  # Ethernet
        
        comm_time_ms = comm_volume_gb / bandwidth_gbps * 8 * 1000  # GB to Gb
        
        # Overlap factor (how much comm overlaps with compute)
        overlap_factor = 0.7 if num_gpus <= 8 else 0.5
        
        effective_comm_time = comm_time_ms * (1 - overlap_factor)
        total_time = compute_time_ms + effective_comm_time
        
        ideal_time = compute_time_ms
        scaling_efficiency = ideal_time / total_time if total_time > 0 else 1.0
        
        return {
            "num_gpus": num_gpus,
            "model_memory_gb": model_memory_gb,
            "activation_memory_gb": activation_memory_gb,
            "comm_volume_gb": comm_volume_gb,
            "compute_time_ms": compute_time_ms,
            "comm_time_ms": comm_time_ms,
            "effective_comm_time_ms": effective_comm_time,
            "total_time_ms": total_time,
            "scaling_efficiency": scaling_efficiency,
            "speedup": num_gpus * scaling_efficiency,
            "interconnect": self.topology.get('interconnect'),
            "bandwidth_gbps": bandwidth_gbps,
        }
    
    def recommend_parallelism(self, model_params_b: float,
                              gpu_memory_gb: float,
                              num_gpus: int,
                              sequence_length: int = 4096) -> Dict[str, Any]:
        """Recommend parallelism strategy."""
        
        # Model memory with FP16
        model_memory_gb_fp16 = model_params_b * 2
        
        # Optimizer states (Adam: 2x model for fp32 states)
        optimizer_memory_gb = model_params_b * 4 * 2
        
        # Gradients
        gradient_memory_gb = model_params_b * 2
        
        total_training_memory = model_memory_gb_fp16 + optimizer_memory_gb + gradient_memory_gb
        
        # Determine minimum tensor parallelism needed
        available_memory = gpu_memory_gb * 0.85  # Leave headroom
        
        min_tp = 1
        while total_training_memory / min_tp > available_memory and min_tp < num_gpus:
            min_tp *= 2
        
        # Determine if pipeline parallelism helps
        # PP is good for very deep models (>100 layers typically)
        estimated_layers = int(model_params_b * 1e9 / (12 * 4096 * 4096))  # Rough
        use_pp = estimated_layers > 80 and num_gpus >= 8
        
        pp_degree = 2 if use_pp and num_gpus >= 8 else 1
        tp_degree = min(min_tp, num_gpus // pp_degree)
        dp_degree = num_gpus // (tp_degree * pp_degree)
        
        # Check if FSDP would be better than DP
        use_fsdp = dp_degree > 1 and total_training_memory > available_memory * 0.5
        
        strategy = []
        if tp_degree > 1:
            strategy.append(f"Tensor Parallel: {tp_degree}")
        if pp_degree > 1:
            strategy.append(f"Pipeline Parallel: {pp_degree}")
        if dp_degree > 1:
            if use_fsdp:
                strategy.append(f"FSDP: {dp_degree}")
            else:
                strategy.append(f"Data Parallel: {dp_degree}")
        
        return {
            "model_params_b": model_params_b,
            "total_training_memory_gb": total_training_memory,
            "gpu_memory_gb": gpu_memory_gb,
            "num_gpus": num_gpus,
            "recommended": {
                "tensor_parallel": tp_degree,
                "pipeline_parallel": pp_degree,
                "data_parallel": dp_degree,
                "use_fsdp": use_fsdp,
            },
            "strategy_summary": " + ".join(strategy) if strategy else "Single GPU",
            "memory_per_gpu_gb": total_training_memory / (tp_degree * pp_degree),
            "estimated_efficiency": 0.85 if tp_degree <= 4 else 0.75 if tp_degree <= 8 else 0.65,
        }


# =============================================================================
# NCCL PROFILING
# =============================================================================

class NCCLProfiler:
    """Profile NCCL communication patterns."""
    
    @staticmethod
    def get_nccl_env_recommendations(topology: Dict[str, Any]) -> Dict[str, str]:
        """Get recommended NCCL environment variables."""
        
        env = {
            "NCCL_DEBUG": "WARN",
            "NCCL_IB_DISABLE": "0" if topology.get('infiniband', {}).get('available') else "1",
            "NCCL_P2P_DISABLE": "0",
        }
        
        # NVLink/NVSwitch optimizations
        if topology.get('interconnect') in ['nvlink', 'nvswitch']:
            env["NCCL_P2P_LEVEL"] = "NVL"
            env["NCCL_NVLS_ENABLE"] = "1"  # NVLink SHARP for Hopper/Blackwell
        
        # Multi-node with IB
        if topology.get('num_nodes', 1) > 1:
            if topology.get('infiniband', {}).get('available'):
                env["NCCL_IB_HCA"] = "mlx5"
                env["NCCL_IB_GID_INDEX"] = "3"
                env["NCCL_SOCKET_IFNAME"] = "eth0"
            env["NCCL_CROSS_NIC"] = "1"
        
        # Large clusters
        if topology.get('total_gpus', 1) > 64:
            env["NCCL_TREE_THRESHOLD"] = "0"  # Always use tree
            env["NCCL_ALGO"] = "Tree"
        
        return env
    
    @staticmethod
    def estimate_all_reduce_time(size_gb: float, num_gpus: int, 
                                 bandwidth_gbps: float, algo: str = "ring") -> float:
        """Estimate all-reduce time in milliseconds."""
        
        if algo == "ring":
            # Ring all-reduce: 2 * (n-1)/n * size / bandwidth
            return 2 * (num_gpus - 1) / num_gpus * size_gb / bandwidth_gbps * 8 * 1000
        elif algo == "tree":
            # Tree all-reduce: 2 * log2(n) * size / bandwidth
            import math
            return 2 * math.log2(num_gpus) * size_gb / bandwidth_gbps * 8 * 1000
        else:
            return size_gb / bandwidth_gbps * 8 * 1000


# =============================================================================
# CLI (Typer)
# =============================================================================

def cmd_topology(args):
    """Show cluster topology."""
    print("\n🌐 Cluster Topology Discovery")
    print("=" * 70)
    
    discovery = ClusterDiscovery()
    topo = discovery.get_full_topology()
    
    print(f"\n  Hostname: {topo['hostname']}")
    print(f"  Scheduler: {topo['scheduler']}")
    print(f"  Nodes: {topo['num_nodes']}")
    print(f"  GPUs per node: {topo['gpus_per_node']}")
    print(f"  Total GPUs: {topo['total_gpus']}")
    print(f"  Interconnect: {topo['interconnect']}")
    
    if topo['infiniband']['available']:
        print(f"  InfiniBand: {topo['infiniband']['bandwidth_gbps']} Gbps")
    
    print("\n  Local GPUs:")
    for gpu in topo['local_gpus']:
        print(f"    GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.0f} GB)")
    
    if topo['nvlink_topology']:
        print("\n  NVLink Topology:")
        for gpu_id, connections in topo['nvlink_topology'].items():
            print(f"    GPU {gpu_id} -> {connections}")
    
    print("\n  Recommended NCCL settings:")
    nccl_env = NCCLProfiler.get_nccl_env_recommendations(topo)
    for key, value in nccl_env.items():
        print(f"    export {key}={value}")
    
    print()
    return 0


def cmd_scaling(args):
    """Analyze scaling efficiency."""
    print("\n📈 Scaling Efficiency Analysis")
    print("=" * 70)
    
    discovery = ClusterDiscovery()
    topo = discovery.get_full_topology()
    analyzer = ScalingAnalyzer(topo)
    
    model_params = float(args.model_params or 70)
    
    print(f"\n  Model: {model_params}B parameters")
    print(f"  Interconnect: {topo['interconnect']}")
    print()
    
    print(f"  {'GPUs':<8} {'Efficiency':<12} {'Speedup':<12} {'Comm Time':<12} {'Compute'}")
    print("  " + "-" * 60)
    
    for num_gpus in [1, 2, 4, 8, 16, 32, 64, 128]:
        if num_gpus > topo['total_gpus'] * 4:  # Allow projecting beyond current cluster
            break
        
        result = analyzer.estimate_scaling_efficiency(
            model_params_b=model_params,
            batch_size=8 * num_gpus,
            sequence_length=4096,
            num_gpus=num_gpus,
        )
        
        eff = result['scaling_efficiency']
        eff_bar = "█" * int(eff * 10) + "░" * (10 - int(eff * 10))
        
        print(f"  {num_gpus:<8} {eff_bar} {eff*100:>5.1f}%  {result['speedup']:>6.1f}x     {result['comm_time_ms']:>6.1f}ms    {result['compute_time_ms']:>6.1f}ms")
    
    print()
    return 0


def cmd_recommend(args):
    """Recommend parallelism strategy."""
    print("\n🎯 Parallelism Strategy Recommendation")
    print("=" * 70)
    
    discovery = ClusterDiscovery()
    topo = discovery.get_full_topology()
    analyzer = ScalingAnalyzer(topo)
    
    model_params = float(args.model_params or 70)
    num_gpus = int(args.nodes or topo['total_gpus'])
    gpu_memory = topo['local_gpus'][0]['memory_gb'] if topo['local_gpus'] else 80
    
    result = analyzer.recommend_parallelism(
        model_params_b=model_params,
        gpu_memory_gb=gpu_memory,
        num_gpus=num_gpus,
    )
    
    print(f"\n  Model: {model_params}B parameters")
    print(f"  Training memory: {result['total_training_memory_gb']:.1f} GB")
    print(f"  Available GPUs: {num_gpus}")
    print(f"  GPU memory: {gpu_memory:.0f} GB each")
    
    print("\n  Recommended Strategy:")
    print("  " + "-" * 50)
    print(f"    {result['strategy_summary']}")
    print()
    print(f"    Tensor Parallel: {result['recommended']['tensor_parallel']}")
    print(f"    Pipeline Parallel: {result['recommended']['pipeline_parallel']}")
    print(f"    Data Parallel: {result['recommended']['data_parallel']}")
    print(f"    Use FSDP: {'Yes' if result['recommended']['use_fsdp'] else 'No'}")
    print()
    print(f"  Memory per GPU: {result['memory_per_gpu_gb']:.1f} GB")
    print(f"  Estimated efficiency: {result['estimated_efficiency']*100:.0f}%")
    
    # Generate launch command
    tp = result['recommended']['tensor_parallel']
    pp = result['recommended']['pipeline_parallel']
    dp = result['recommended']['data_parallel']
    
    print("\n  Example launch command:")
    print("  " + "-" * 50)
    print(f"""  torchrun --nproc_per_node={tp * pp * dp} \\
      --nnodes={num_gpus // (tp * pp * dp)} \\
      train.py \\
      --tensor-parallel-size {tp} \\
      --pipeline-parallel-size {pp} \\
      --data-parallel-size {dp}""")
    
    print()
    return 0


import typer

app = typer.Typer(help="Distributed training & parallelism planner")


@app.command("topology", help="Show cluster topology")
def typer_topology() -> None:
    cmd_topology(SimpleNamespace())


@app.command("scaling", help="Scaling efficiency")
def typer_scaling(
    model_params: float = typer.Option(70.0, "--model-params", help="Model params in billions"),
) -> None:
    cmd_scaling(SimpleNamespace(model_params=model_params))


@app.command("recommend", help="Recommend parallelism")
def typer_recommend(
    model_params: float = typer.Option(70.0, "--model-params", help="Model params in billions"),
    nodes: int = typer.Option(1, "--nodes", help="Number of GPUs"),
) -> None:
    cmd_recommend(SimpleNamespace(model_params=model_params, nodes=nodes))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
