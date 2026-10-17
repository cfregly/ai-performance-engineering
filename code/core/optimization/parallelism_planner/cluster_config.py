#!/usr/bin/env python3
"""
Multi-Node Cluster Configuration

Handles multi-node cluster topology and cross-node parallelism strategies.
Supports InfiniBand, RoCE, and Ethernet interconnects.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .topology_detector import TopologyInfo


class NetworkType(Enum):
    """Network interconnect type for multi-node communication."""
    INFINIBAND_HDR = "infiniband_hdr"      # 200 Gbps
    INFINIBAND_NDR = "infiniband_ndr"      # 400 Gbps
    INFINIBAND_XDR = "infiniband_xdr"      # 800 Gbps
    ROCE = "roce"                           # RDMA over Converged Ethernet
    ETHERNET_100G = "ethernet_100g"
    ETHERNET_400G = "ethernet_400g"
    ETHERNET_800G = "ethernet_800g"
    TCP = "tcp"                             # Standard TCP/IP


# Network bandwidth in GB/s (bidirectional)
NETWORK_BANDWIDTH = {
    NetworkType.INFINIBAND_HDR: 50,    # 200 Gbps = 25 GB/s x2 NICs
    NetworkType.INFINIBAND_NDR: 100,   # 400 Gbps = 50 GB/s x2 NICs
    NetworkType.INFINIBAND_XDR: 200,   # 800 Gbps = 100 GB/s x2 NICs
    NetworkType.ROCE: 50,              # Varies, assume 200 Gbps
    NetworkType.ETHERNET_100G: 25,     # 100 Gbps = 12.5 GB/s x2 NICs
    NetworkType.ETHERNET_400G: 100,    # 400 Gbps = 50 GB/s x2 NICs
    NetworkType.ETHERNET_800G: 200,    # 800 Gbps = 100 GB/s x2 NICs
    NetworkType.TCP: 10,               # Conservative estimate
}


@dataclass
class NodeSpec:
    """Specification for a single node in the cluster."""
    node_id: int
    hostname: str
    gpus_per_node: int
    gpu_memory_gb: float
    gpu_name: str
    nvlink_bandwidth_gbps: float
    has_nvswitch: bool
    numa_nodes: Optional[int] = None


@dataclass
class ClusterTopology:
    """Multi-node cluster topology information."""
    
    # Cluster composition
    num_nodes: int
    nodes: List[NodeSpec]
    total_gpus: int
    gpus_per_node: int
    
    # Network configuration
    network_type: NetworkType
    network_bandwidth_gbps: float
    nics_per_node: int
    rdma_capable: bool
    
    # GPU Direct features
    gpudirect_rdma: bool  # GPU can directly access network
    gpudirect_storage: bool  # GPU can directly access NVMe
    
    # NVLink across nodes (NVLink-C2C for Grace systems)
    cross_node_nvlink: bool
    cross_node_nvlink_bandwidth_gbps: float
    
    # Topology hints
    is_dgx_cluster: bool = False
    is_hgx_cluster: bool = False
    is_grace_hopper: bool = False
    is_grace_blackwell: bool = False
    
    @property
    def total_memory_gb(self) -> float:
        """Total GPU memory across cluster."""
        if self.nodes:
            return sum(n.gpu_memory_gb * n.gpus_per_node for n in self.nodes)
        return self.total_gpus * 80  # Default assumption
    
    @property
    def effective_network_bandwidth_gbps(self) -> float:
        """Effective inter-node bandwidth per GPU."""
        return self.network_bandwidth_gbps * self.nics_per_node / self.gpus_per_node
    
    def get_optimal_cross_node_parallelism(self) -> Dict[str, List[int]]:
        """Get recommended parallelism dimensions for cross-node communication."""
        recommendations = {
            "dp_across_nodes": [],
            "pp_across_nodes": [],
            "tp_within_node": [],
        }
        
        # TP should stay within node (NVLink)
        if self.gpus_per_node >= 8:
            recommendations["tp_within_node"] = [1, 2, 4, 8]
        elif self.gpus_per_node >= 4:
            recommendations["tp_within_node"] = [1, 2, 4]
        else:
            recommendations["tp_within_node"] = [1, 2]
        
        # PP can span nodes if we have good network
        if self.network_bandwidth_gbps >= 100 or self.cross_node_nvlink:
            recommendations["pp_across_nodes"] = list(range(1, min(self.num_nodes + 1, 9)))
        else:
            recommendations["pp_across_nodes"] = [1, 2]  # Limit PP for slower networks
        
        # DP across nodes (always works)
        recommendations["dp_across_nodes"] = list(range(1, self.num_nodes + 1))
        
        return recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "total_gpus": self.total_gpus,
            "gpus_per_node": self.gpus_per_node,
            "total_memory_gb": self.total_memory_gb,
            "network_type": self.network_type.value,
            "network_bandwidth_gbps": self.network_bandwidth_gbps,
            "nics_per_node": self.nics_per_node,
            "rdma_capable": self.rdma_capable,
            "gpudirect_rdma": self.gpudirect_rdma,
            "cross_node_nvlink": self.cross_node_nvlink,
            "effective_bandwidth_per_gpu_gbps": self.effective_network_bandwidth_gbps,
            "optimal_parallelism": self.get_optimal_cross_node_parallelism(),
        }


class ClusterDetector:
    """Detects multi-node cluster configuration."""
    
    def __init__(self):
        self._cached_cluster: Optional[ClusterTopology] = None
    
    def detect(
        self,
        single_node_topology: Optional[TopologyInfo] = None,
        force_refresh: bool = False,
    ) -> ClusterTopology:
        """Detect cluster topology.
        
        Args:
            single_node_topology: Optional pre-detected single node topology
            force_refresh: Force re-detection
            
        Returns:
            ClusterTopology for the cluster
        """
        if self._cached_cluster is not None and not force_refresh:
            return self._cached_cluster
        
        # Get node count from environment (SLURM, etc.)
        num_nodes = self._detect_num_nodes()
        gpus_per_node = self._detect_gpus_per_node(single_node_topology)
        
        # Detect network
        network_type, network_bw, nics, rdma = self._detect_network()
        
        # Detect GPU Direct capabilities
        gpudirect_rdma = self._detect_gpudirect_rdma()
        gpudirect_storage = self._detect_gpudirect_storage()
        
        # Detect cross-node NVLink (Grace systems)
        cross_nvlink, cross_nvlink_bw = self._detect_cross_node_nvlink(single_node_topology)
        
        # Build node specs
        nodes = self._build_node_specs(num_nodes, gpus_per_node, single_node_topology)
        
        # Detect cluster type
        is_dgx = self._is_dgx_cluster()
        is_hgx = self._is_hgx_cluster()
        is_grace_hopper = single_node_topology and single_node_topology.is_grace_cpu and \
                          any("hopper" in g.architecture for g in single_node_topology.gpus)
        # GB200 and GB300 GPUs carry the exact "blackwell" planner label (SM 10.0
        # and 10.3). SM 12.x GPUs are labeled "blackwell_sm12x" and are not
        # GB200/GB300 superchip GPUs.
        is_grace_blackwell = single_node_topology and single_node_topology.is_grace_cpu and \
                             any(g.architecture == "blackwell" for g in single_node_topology.gpus)
        
        cluster = ClusterTopology(
            num_nodes=num_nodes,
            nodes=nodes,
            total_gpus=num_nodes * gpus_per_node,
            gpus_per_node=gpus_per_node,
            network_type=network_type,
            network_bandwidth_gbps=network_bw,
            nics_per_node=nics,
            rdma_capable=rdma,
            gpudirect_rdma=gpudirect_rdma,
            gpudirect_storage=gpudirect_storage,
            cross_node_nvlink=cross_nvlink,
            cross_node_nvlink_bandwidth_gbps=cross_nvlink_bw,
            is_dgx_cluster=is_dgx,
            is_hgx_cluster=is_hgx,
            is_grace_hopper=is_grace_hopper,
            is_grace_blackwell=is_grace_blackwell,
        )
        
        self._cached_cluster = cluster
        return cluster
    
    def _detect_num_nodes(self) -> int:
        """Detect number of nodes in the cluster."""
        # Check SLURM
        if "SLURM_NNODES" in os.environ:
            return int(os.environ["SLURM_NNODES"])
        
        # Check PBS
        if "PBS_NUM_NODES" in os.environ:
            return int(os.environ["PBS_NUM_NODES"])
        
        # Check generic distributed env
        if "WORLD_SIZE" in os.environ and "LOCAL_WORLD_SIZE" in os.environ:
            world = int(os.environ["WORLD_SIZE"])
            local = int(os.environ["LOCAL_WORLD_SIZE"])
            return world // local
        
        # Single node
        return 1
    
    def _detect_gpus_per_node(self, topology: Optional[TopologyInfo]) -> int:
        """Detect GPUs per node."""
        if topology:
            return topology.num_gpus
        
        # Check environment
        if "LOCAL_WORLD_SIZE" in os.environ:
            return int(os.environ["LOCAL_WORLD_SIZE"])
        
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            return len([d for d in devices if d.strip()])
        
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return len([l for l in result.stdout.split("\n") if "GPU" in l])
        except Exception:
            pass
        
        return 8  # Default assumption
    
    def _detect_network(self) -> Tuple[NetworkType, float, int, bool]:
        """Detect network configuration."""
        network_type = NetworkType.TCP
        bandwidth = 10.0
        nics = 1
        rdma = False
        
        # Try to detect InfiniBand
        try:
            result = subprocess.run(
                ["ibstat"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "State: Active" in result.stdout:
                rdma = True
                # Parse rate
                rate_match = re.search(r'Rate:\s*(\d+)', result.stdout)
                if rate_match:
                    rate = int(rate_match.group(1))
                    if rate >= 800:
                        network_type = NetworkType.INFINIBAND_XDR
                        bandwidth = 200
                    elif rate >= 400:
                        network_type = NetworkType.INFINIBAND_NDR
                        bandwidth = 100
                    else:
                        network_type = NetworkType.INFINIBAND_HDR
                        bandwidth = 50
                
                # Count active ports
                nics = result.stdout.count("State: Active")
        except FileNotFoundError:
            pass
        
        # Check for high-speed Ethernet if no IB
        if network_type == NetworkType.TCP:
            try:
                result = subprocess.run(
                    ["ethtool", "-i", "eth0"],
                    capture_output=True, text=True, timeout=5
                )
                if "mlx5" in result.stdout.lower():
                    # Mellanox NIC, likely RoCE capable
                    network_type = NetworkType.ROCE
                    bandwidth = 50
                    rdma = True
            except Exception:
                pass
        
        return network_type, bandwidth, nics, rdma
    
    def _detect_gpudirect_rdma(self) -> bool:
        """Detect if GPUDirect RDMA is available."""
        # Check for nvidia-peermem module
        try:
            result = subprocess.run(
                ["lsmod"],
                capture_output=True, text=True, timeout=5
            )
            return "nvidia_peermem" in result.stdout or "nv_peer_mem" in result.stdout
        except Exception:
            return False
    
    def _detect_gpudirect_storage(self) -> bool:
        """Detect if GPUDirect Storage is available."""
        try:
            # Check for cufile library
            result = subprocess.run(
                ["ldconfig", "-p"],
                capture_output=True, text=True, timeout=5
            )
            return "libcufile" in result.stdout
        except Exception:
            return False
    
    def _detect_cross_node_nvlink(
        self, topology: Optional[TopologyInfo]
    ) -> Tuple[bool, float]:
        """Detect cross-node NVLink (Grace Superchip configurations)."""
        if topology and topology.has_nvlink_c2c:
            # Grace-Blackwell or Grace-Hopper with NVLink-C2C
            return True, 900.0  # NVLink-C2C bandwidth
        return False, 0.0
    
    def _build_node_specs(
        self,
        num_nodes: int,
        gpus_per_node: int,
        topology: Optional[TopologyInfo],
    ) -> List[NodeSpec]:
        """Build node specifications."""
        nodes = []
        
        for i in range(num_nodes):
            if topology and i == 0:
                # Use actual topology for first node
                node = NodeSpec(
                    node_id=i,
                    hostname=os.environ.get("HOSTNAME", f"node{i}"),
                    gpus_per_node=topology.num_gpus,
                    gpu_memory_gb=topology.gpus[0].memory_gb if topology.gpus else 80,
                    gpu_name=topology.gpus[0].name if topology.gpus else "Unknown",
                    nvlink_bandwidth_gbps=topology.max_nvlink_bandwidth_gbps,
                    has_nvswitch=topology.has_nvswitch,
                    numa_nodes=topology.numa_nodes if topology.numa_nodes > 0 else None,
                )
            else:
                # Assume identical nodes
                node = NodeSpec(
                    node_id=i,
                    hostname=f"node{i}",
                    gpus_per_node=gpus_per_node,
                    gpu_memory_gb=80,  # Default H100
                    gpu_name="NVIDIA GPU",
                    nvlink_bandwidth_gbps=600,
                    has_nvswitch=True,
                    numa_nodes=2,
                )
            nodes.append(node)
        
        return nodes
    
    def _is_dgx_cluster(self) -> bool:
        """Check if this is a DGX cluster."""
        try:
            result = subprocess.run(
                ["cat", "/etc/dgx-release"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _is_hgx_cluster(self) -> bool:
        """Check if this is an HGX system."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-q"],
                capture_output=True, text=True, timeout=10
            )
            return "HGX" in result.stdout
        except Exception:
            return False


# Preset cluster configurations
CLUSTER_PRESETS: Dict[str, ClusterTopology] = {}


def create_cluster_preset_dgx_h100_8x(num_nodes: int = 1) -> ClusterTopology:
    """Create DGX H100 cluster preset."""
    nodes = [
        NodeSpec(
            node_id=i,
            hostname=f"dgx-h100-{i}",
            gpus_per_node=8,
            gpu_memory_gb=80,
            gpu_name="NVIDIA H100 80GB HBM3",
            nvlink_bandwidth_gbps=600,
            has_nvswitch=True,
            numa_nodes=2,
        )
        for i in range(num_nodes)
    ]
    
    return ClusterTopology(
        num_nodes=num_nodes,
        nodes=nodes,
        total_gpus=num_nodes * 8,
        gpus_per_node=8,
        network_type=NetworkType.INFINIBAND_NDR,
        network_bandwidth_gbps=100,
        nics_per_node=8,
        rdma_capable=True,
        gpudirect_rdma=True,
        gpudirect_storage=True,
        cross_node_nvlink=False,
        cross_node_nvlink_bandwidth_gbps=0,
        is_dgx_cluster=True,
    )


def create_cluster_preset_dgx_gb200_nvl72() -> ClusterTopology:
    """Create DGX GB200 NVL72 cluster preset (72 GPUs across 36 Grace CPUs)."""
    nodes = [
        NodeSpec(
            node_id=i,
            hostname=f"gb200-{i}",
            gpus_per_node=2,  # 2 B200 per Grace CPU
            gpu_memory_gb=192,
            gpu_name="NVIDIA B200",
            nvlink_bandwidth_gbps=1800,  # 5th gen NVLink
            has_nvswitch=True,
            numa_nodes=1,
        )
        for i in range(36)
    ]
    
    return ClusterTopology(
        num_nodes=36,
        nodes=nodes,
        total_gpus=72,
        gpus_per_node=2,
        network_type=NetworkType.INFINIBAND_XDR,
        network_bandwidth_gbps=200,
        nics_per_node=2,
        rdma_capable=True,
        gpudirect_rdma=True,
        gpudirect_storage=True,
        cross_node_nvlink=True,  # NVLink-C2C between all
        cross_node_nvlink_bandwidth_gbps=900,
        is_grace_blackwell=True,
    )


def create_cluster_preset_b200_single_node(num_gpus: int = 8) -> ClusterTopology:
    """Create single-node B200 preset for a configurable GPU count."""
    if num_gpus < 2:
        raise ValueError("num_gpus must be >=2 for multi-GPU presets")

    has_nvswitch = num_gpus >= 8
    nodes = [
        NodeSpec(
            node_id=0,
            hostname="b200-node",
            gpus_per_node=num_gpus,
            gpu_memory_gb=192,
            gpu_name="NVIDIA B200",
            nvlink_bandwidth_gbps=900,
            has_nvswitch=has_nvswitch,
            numa_nodes=1,
        )
    ]

    return ClusterTopology(
        num_nodes=1,
        nodes=nodes,
        total_gpus=num_gpus,
        gpus_per_node=num_gpus,
        network_type=NetworkType.TCP,
        network_bandwidth_gbps=0,
        nics_per_node=0,
        rdma_capable=False,
        gpudirect_rdma=False,
        gpudirect_storage=False,
        cross_node_nvlink=False,
        cross_node_nvlink_bandwidth_gbps=0,
        is_grace_blackwell=True,
    )
