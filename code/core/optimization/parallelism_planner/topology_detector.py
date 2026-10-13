#!/usr/bin/env python3
"""
Enhanced Hardware Topology Detection

Detects the real hardware topology including:
- GPU interconnect mesh (NVLink, NVSwitch, PCIe)
- P2P accessibility matrix
- NUMA node mapping
- Interconnect bandwidths
- Multi-node configuration
"""

from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

from core.common.grace_platform import host_grace_soc_id, host_link_flags


def gpu_architecture_for_capability(major: int, minor: int) -> str:
    """Return the planner's GPU family label for one compute capability.

    Compute capability names the GPU instruction set, not the host platform.
    SM 10.0 and 10.3 are the datacenter Blackwell GPUs used in B200/GB200 and
    B300/GB300 systems. SM 12.x is the GeForce RTX 50, RTX PRO and
    GB10 / DGX Spark family, not a GB200/GB300 Superchip GPU. Grace pairing
    and NVLink-C2C are reported by CPU detection, never by this label.
    Unlisted capabilities keep their raw sm_XY label instead of borrowing a
    neighboring family.
    """
    if (major, minor) in ((10, 0), (10, 3)):
        return "blackwell"
    if major == 12:
        return "blackwell_sm12x"
    if major == 9:
        return "hopper"
    if major == 8:
        return "ampere"
    return f"sm_{major}{minor}"


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    compute_capability: str
    memory_gb: float
    num_sms: int
    architecture: str  # See gpu_architecture_for_capability.
    nvlink_capable: bool
    numa_node: Optional[int] = None
    pcie_bus_id: Optional[str] = None


@dataclass
class InterconnectInfo:
    """Information about GPU-to-GPU interconnect."""
    gpu_a: int
    gpu_b: int
    link_type: str  # NV18, NV12, NV4, PIX, PHB, SYS, etc.
    bandwidth_gbps: float
    is_nvlink: bool
    nvlink_count: int = 0


@dataclass  
class TopologyInfo:
    """Complete hardware topology information."""
    # GPU info
    num_gpus: int
    gpus: List[GPUInfo]
    total_memory_gb: float
    
    # Interconnect topology
    interconnects: List[InterconnectInfo]
    p2p_matrix: List[List[bool]]  # Can GPU i access GPU j directly?
    bandwidth_matrix: List[List[float]]  # Bandwidth in GB/s
    
    # NVLink/NVSwitch
    has_nvlink: bool
    has_nvswitch: bool  # Full mesh connectivity
    nvlink_version: Optional[str]  # 3.0, 4.0, 5.0
    max_nvlink_bandwidth_gbps: float
    
    # NUMA topology
    numa_nodes: int
    gpu_numa_mapping: Dict[int, int]  # GPU index -> NUMA node
    numa_distance_matrix: List[List[int]]  # NUMA distances
    
    # CPU info
    cpu_type: str  # platform.machine(), such as x86_64 or aarch64
    is_grace_cpu: bool  # NVIDIA SoC ID in sysfs
    has_nvlink_c2c: bool  # CUDA driver reports an NVLink-C2C host link for every GPU

    # Locality detection status
    gpu_numa_status: str = "unknown"  # unknown, partial, complete, synthetic

    # Provenance
    topology_source: str = "detected"  # detected, synthetic_preset
    topology_preset_name: Optional[str] = None
    
    # Network (for multi-node)
    network_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    infiniband_available: bool = False
    rdma_capable: bool = False
    
    # Cluster context
    num_nodes: int = 1
    gpus_per_node: int = 0
    
    def get_optimal_tp_sizes(self) -> List[int]:
        """Get optimal tensor parallel sizes based on topology."""
        sizes = [1]
        
        if self.has_nvswitch:
            # Full NVSwitch mesh - any power of 2 up to num_gpus works well
            size = 2
            while size <= self.num_gpus:
                sizes.append(size)
                size *= 2
        elif self.has_nvlink:
            # Partial NVLink - prefer smaller TP within NVLink domains
            # Find connected components
            visited = set()
            for i in range(self.num_gpus):
                if i not in visited:
                    component = self._find_nvlink_component(i)
                    visited.update(component)
                    if len(component) > 1 and len(component) not in sizes:
                        sizes.append(len(component))
            sizes = sorted(set(sizes))
        else:
            # PCIe only - TP=2 is usually max efficient
            if self.num_gpus >= 2:
                sizes.append(2)
        
        return sorted(sizes)
    
    def _find_nvlink_component(self, start: int) -> set:
        """Find all GPUs connected via NVLink to start GPU."""
        component = {start}
        queue = [start]
        while queue:
            gpu = queue.pop()
            for conn in self.interconnects:
                if conn.is_nvlink:
                    if conn.gpu_a == gpu and conn.gpu_b not in component:
                        component.add(conn.gpu_b)
                        queue.append(conn.gpu_b)
                    elif conn.gpu_b == gpu and conn.gpu_a not in component:
                        component.add(conn.gpu_a)
                        queue.append(conn.gpu_a)
        return component
    
    def get_recommended_pp_placement(self, stages: int) -> List[List[int]]:
        """Get recommended GPU placement for pipeline parallel stages.
        
        Returns list of GPU indices for each stage, optimizing for
        NVLink connectivity between adjacent stages.
        """
        if stages > self.num_gpus:
            raise ValueError(f"Cannot have {stages} stages with {self.num_gpus} GPUs")
        
        gpus_per_stage = self.num_gpus // stages
        placement = []
        
        if self.has_nvswitch:
            # With NVSwitch, any placement works - just divide evenly
            for i in range(stages):
                start = i * gpus_per_stage
                placement.append(list(range(start, start + gpus_per_stage)))
        else:
            # Try to keep adjacent stages on NVLink-connected GPUs
            # For now, simple contiguous allocation
            for i in range(stages):
                start = i * gpus_per_stage
                placement.append(list(range(start, start + gpus_per_stage)))
        
        return placement
    
    def get_cp_group_recommendations(self, cp_size: int) -> List[List[int]]:
        """Get recommended GPU groupings for context parallelism.
        
        Context parallelism benefits from high bandwidth ring topology.
        """
        if cp_size > self.num_gpus:
            raise ValueError(f"Cannot have CP size {cp_size} with {self.num_gpus} GPUs")
        
        num_groups = self.num_gpus // cp_size
        groups = []
        
        if self.has_nvswitch:
            # Any grouping works with NVSwitch
            for i in range(num_groups):
                groups.append(list(range(i * cp_size, (i + 1) * cp_size)))
        else:
            # Prefer NVLink-connected GPUs within each CP group
            groups.append(list(range(cp_size)))  # Simple for now
        
        return groups

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "num_gpus": self.num_gpus,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "compute_capability": g.compute_capability,
                    "memory_gb": g.memory_gb,
                    "num_sms": g.num_sms,
                    "architecture": g.architecture,
                    "nvlink_capable": g.nvlink_capable,
                    "numa_node": g.numa_node,
                }
                for g in self.gpus
            ],
            "total_memory_gb": self.total_memory_gb,
            "has_nvlink": self.has_nvlink,
            "has_nvswitch": self.has_nvswitch,
            "nvlink_version": self.nvlink_version,
            "max_nvlink_bandwidth_gbps": self.max_nvlink_bandwidth_gbps,
            "p2p_matrix": self.p2p_matrix,
            "bandwidth_matrix": self.bandwidth_matrix,
            "numa_nodes": self.numa_nodes,
            "gpu_numa_mapping": self.gpu_numa_mapping,
            "gpu_numa_status": self.gpu_numa_status,
            "topology_source": self.topology_source,
            "topology_preset_name": self.topology_preset_name,
            "cpu_type": self.cpu_type,
            "is_grace_cpu": self.is_grace_cpu,
            "has_nvlink_c2c": self.has_nvlink_c2c,
            "optimal_tp_sizes": self.get_optimal_tp_sizes(),
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "infiniband_available": self.infiniband_available,
            "rdma_capable": self.rdma_capable,
        }


@dataclass(frozen=True)
class SyntheticTopologyPreset:
    """Explicit synthetic topology preset that must be materialized before use."""

    preset_name: str
    topology: TopologyInfo
    description: str = ""

    def materialize(self) -> TopologyInfo:
        topo = deepcopy(self.topology)
        topo.gpu_numa_status = "synthetic"
        topo.topology_source = "synthetic_preset"
        topo.topology_preset_name = self.preset_name
        return topo


class TopologyDetector:
    """Detects hardware topology for parallelism planning."""
    
    # NVLink bandwidth per link (GB/s) by version
    NVLINK_BANDWIDTH = {
        "3.0": 50,   # A100
        "4.0": 50,   # H100
        "5.0": 100,  # B100/B200
    }
    
    def __init__(self):
        self._cached_topology: Optional[TopologyInfo] = None
    
    def detect(self, force_refresh: bool = False) -> TopologyInfo:
        """Detect hardware topology.
        
        Args:
            force_refresh: Force re-detection even if cached
            
        Returns:
            TopologyInfo with complete topology information
        """
        if self._cached_topology is not None and not force_refresh:
            return self._cached_topology
        
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - cannot detect GPU topology")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs detected")
        
        # Detect GPU information
        gpus = self._detect_gpus(num_gpus)
        
        # Detect interconnect topology
        interconnects, p2p_matrix, bandwidth_matrix = self._detect_interconnects(num_gpus)
        
        # Analyze NVLink topology
        has_nvlink, has_nvswitch, nvlink_version, max_bw = self._analyze_nvlink(
            interconnects, num_gpus
        )
        
        # Detect NUMA topology
        numa_nodes, gpu_numa_mapping, numa_distance, gpu_numa_status = self._detect_numa(num_gpus)
        
        # Detect CPU type
        cpu_type, is_grace, has_c2c = self._detect_cpu(num_gpus)
        
        # Detect network interfaces
        network_ifs, has_ib, has_rdma = self._detect_network()
        
        topology = TopologyInfo(
            num_gpus=num_gpus,
            gpus=gpus,
            total_memory_gb=sum(g.memory_gb for g in gpus),
            interconnects=interconnects,
            p2p_matrix=p2p_matrix,
            bandwidth_matrix=bandwidth_matrix,
            has_nvlink=has_nvlink,
            has_nvswitch=has_nvswitch,
            nvlink_version=nvlink_version,
            max_nvlink_bandwidth_gbps=max_bw,
            numa_nodes=numa_nodes,
            gpu_numa_mapping=gpu_numa_mapping,
            numa_distance_matrix=numa_distance,
            gpu_numa_status=gpu_numa_status,
            cpu_type=cpu_type,
            is_grace_cpu=is_grace,
            has_nvlink_c2c=has_c2c,
            network_interfaces=network_ifs,
            infiniband_available=has_ib,
            rdma_capable=has_rdma,
            num_nodes=1,  # Single node for now
            gpus_per_node=num_gpus,
        )
        
        self._cached_topology = topology
        return topology
    
    def _detect_gpus(self, num_gpus: int) -> List[GPUInfo]:
        """Detect information about each GPU."""
        gpus = []
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            arch = gpu_architecture_for_capability(props.major, props.minor)

            # Check NVLink capability (Volta+)
            nvlink_capable = props.major >= 7
            
            gpu = GPUInfo(
                index=i,
                name=props.name,
                compute_capability=f"{props.major}.{props.minor}",
                memory_gb=props.total_memory / (1024 ** 3),
                num_sms=props.multi_processor_count,
                architecture=arch,
                nvlink_capable=nvlink_capable,
            )
            gpus.append(gpu)
        
        return gpus
    
    def _detect_interconnects(
        self, num_gpus: int
    ) -> Tuple[List[InterconnectInfo], List[List[bool]], List[List[float]]]:
        """Detect GPU interconnect topology."""
        interconnects = []
        p2p_matrix = [[False] * num_gpus for _ in range(num_gpus)]
        bandwidth_matrix = [[0.0] * num_gpus for _ in range(num_gpus)]
        
        # Self-connections
        for i in range(num_gpus):
            p2p_matrix[i][i] = True
            bandwidth_matrix[i][i] = float('inf')  # Local access
        
        # Parse nvidia-smi topology
        topo_output = self._run_nvidia_smi_topo()
        link_info = self._parse_topology_matrix(topo_output, num_gpus)
        
        # Check P2P access via PyTorch
        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                try:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                except Exception:
                    can_access = False
                
                p2p_matrix[i][j] = can_access
                p2p_matrix[j][i] = can_access
                
                # Get link info from nvidia-smi
                link_type = link_info.get((i, j), "SYS")
                is_nvlink = link_type.startswith("NV")
                nvlink_count = 0
                bandwidth = 16.0  # Default PCIe Gen4 x16
                
                if is_nvlink:
                    # Parse NVX where X is number of links
                    match = re.match(r"NV(\d+)", link_type)
                    if match:
                        nvlink_count = int(match.group(1))
                        # Estimate bandwidth based on architecture
                        bandwidth = nvlink_count * 50  # Conservative estimate
                elif link_type == "PIX":
                    bandwidth = 32.0  # PCIe within same complex
                elif link_type == "PHB":
                    bandwidth = 32.0  # PCIe through host bridge
                elif link_type == "SYS":
                    bandwidth = 16.0  # System (QPI/UPI)
                
                bandwidth_matrix[i][j] = bandwidth
                bandwidth_matrix[j][i] = bandwidth
                
                interconnects.append(InterconnectInfo(
                    gpu_a=i,
                    gpu_b=j,
                    link_type=link_type,
                    bandwidth_gbps=bandwidth,
                    is_nvlink=is_nvlink,
                    nvlink_count=nvlink_count,
                ))
        
        return interconnects, p2p_matrix, bandwidth_matrix
    
    def _analyze_nvlink(
        self, interconnects: List[InterconnectInfo], num_gpus: int
    ) -> Tuple[bool, bool, Optional[str], float]:
        """Analyze NVLink topology."""
        nvlink_connections = [c for c in interconnects if c.is_nvlink]
        has_nvlink = len(nvlink_connections) > 0
        
        if not has_nvlink:
            return False, False, None, 0.0
        
        # Check for full mesh (NVSwitch)
        # With NVSwitch, every GPU pair has NVLink
        expected_connections = num_gpus * (num_gpus - 1) // 2
        has_nvswitch = len(nvlink_connections) == expected_connections
        
        # Determine NVLink version from link count and bandwidth
        max_links = max(c.nvlink_count for c in nvlink_connections)
        max_bandwidth = max(c.bandwidth_gbps for c in nvlink_connections)
        
        # Heuristic for version detection
        if max_links >= 18:
            nvlink_version = "5.0"  # B200 with NVSwitch
        elif max_links >= 12:
            nvlink_version = "4.0"  # H100 NVL
        elif max_links >= 6:
            nvlink_version = "4.0"  # H100 DGX
        else:
            nvlink_version = "3.0"  # A100
        
        return has_nvlink, has_nvswitch, nvlink_version, max_bandwidth
    
    def _detect_numa(
        self, num_gpus: int
    ) -> Tuple[int, Dict[int, int], List[List[int]], str]:
        """Detect NUMA topology."""
        gpu_numa_mapping = {}
        numa_nodes = 0

        for i in range(num_gpus):
            numa_node = self._get_gpu_numa_node(i)
            if numa_node is not None:
                gpu_numa_mapping[i] = numa_node
                numa_nodes = max(numa_nodes, numa_node + 1)

        if num_gpus > 0 and len(gpu_numa_mapping) == num_gpus:
            gpu_numa_status = "complete"
        elif gpu_numa_mapping:
            gpu_numa_status = "partial"
        else:
            gpu_numa_status = "unknown"
        
        # Read NUMA distance matrix
        numa_distance = self._read_numa_distances(numa_nodes)
        
        return numa_nodes, gpu_numa_mapping, numa_distance, gpu_numa_status
    
    def _get_gpu_numa_node(self, gpu_index: int) -> Optional[int]:
        """Get NUMA node for a GPU from sysfs."""
        try:
            # Find GPU PCI address
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None

        pci_id = result.stdout.strip().lower()
        if pci_id.startswith("00000000:"):
            pci_id = pci_id[4:]
        numa_path = Path(f"/sys/bus/pci/devices/{pci_id}/numa_node")
        if not numa_path.exists():
            return None
        try:
            value = int(numa_path.read_text().strip())
        except (OSError, ValueError):
            return None
        return value if value >= 0 else None
        return None
    
    def _read_numa_distances(self, numa_nodes: int) -> List[List[int]]:
        """Read NUMA distance matrix."""
        if numa_nodes <= 0:
            return []
        distances = [[10] * numa_nodes for _ in range(numa_nodes)]  # Default: local
        
        try:
            for i in range(numa_nodes):
                distance_path = Path(f"/sys/devices/system/node/node{i}/distance")
                if distance_path.exists():
                    values = distance_path.read_text().strip().split()
                    for j, val in enumerate(values[:numa_nodes]):
                        distances[i][j] = int(val)
        except (OSError, ValueError):
            pass
        
        return distances
    
    def _detect_cpu(self, num_gpus: int) -> Tuple[str, bool, bool]:
        """Return the CPU type, NVIDIA Grace identity and NVLink-C2C host-link status.

        Grace identity comes from the NVIDIA SoC ID in sysfs. NVLink-C2C is
        reported only when the CUDA driver puts every GPU on an NVLink-C2C host
        link, because a Grace server can also hold PCIe GPUs. See
        core.common.grace_platform.
        """
        import platform

        cpu_type = platform.machine()
        is_grace = host_grace_soc_id() is not None
        has_c2c = num_gpus > 0 and all(
            host_link_flags(torch.cuda.get_device_properties(index), index)[0]
            for index in range(num_gpus)
        )
        return cpu_type, is_grace, has_c2c
    
    def _detect_network(self) -> Tuple[List[Dict[str, Any]], bool, bool]:
        """Detect network interfaces for multi-node communication."""
        interfaces = []
        has_ib = False
        has_rdma = False
        
        try:
            # Check for InfiniBand
            result = subprocess.run(
                ["ibstat"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "State: Active" in result.stdout:
                has_ib = True
                has_rdma = True
                # Parse IB ports
                for line in result.stdout.split('\n'):
                    if 'Rate:' in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            interfaces.append({
                                "type": "InfiniBand",
                                "rate_gbps": int(match.group(1)),
                            })
        except FileNotFoundError:
            pass
        
        try:
            # Check for high-speed Ethernet
            result = subprocess.run(
                ["ip", "-o", "link", "show"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'state UP' in line:
                        # Get interface name
                        match = re.search(r'\d+:\s+(\w+):', line)
                        if match:
                            iface = match.group(1)
                            if iface not in ['lo', 'docker0']:
                                interfaces.append({
                                    "type": "Ethernet",
                                    "name": iface,
                                })
        except FileNotFoundError:
            pass
        
        return interfaces, has_ib, has_rdma
    
    def _run_nvidia_smi_topo(self) -> str:
        """Run nvidia-smi topo -m and return output."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception:
            return ""
    
    def _parse_topology_matrix(
        self, topo_output: str, num_gpus: int
    ) -> Dict[Tuple[int, int], str]:
        """Parse nvidia-smi topology matrix output."""
        link_info = {}
        
        if not topo_output:
            return link_info
        
        lines = topo_output.strip().split('\n')
        
        # Find the matrix rows (lines starting with "GPU")
        for line in lines:
            if not line.strip().startswith('GPU'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # Extract GPU index
            gpu_match = re.match(r'GPU(\d+)', parts[0])
            if not gpu_match:
                continue
            
            gpu_i = int(gpu_match.group(1))
            
            # Parse connections to other GPUs
            for j, part in enumerate(parts[1:]):
                if j >= num_gpus:
                    break
                if part != 'X':  # X is self
                    link_info[(min(gpu_i, j), max(gpu_i, j))] = part
        
        return link_info
    
    def format_topology_report(self, topology: Optional[TopologyInfo] = None) -> str:
        """Generate a human-readable topology report."""
        topo = topology or self.detect()
        
        lines = [
            "=" * 70,
            "HARDWARE TOPOLOGY REPORT",
            "=" * 70,
            "",
            f"GPUs: {topo.num_gpus} x {topo.gpus[0].name if topo.gpus else 'Unknown'}",
            f"Total Memory: {topo.total_memory_gb:.1f} GB",
            f"Architecture: {topo.gpus[0].architecture if topo.gpus else 'Unknown'}",
            "",
            "INTERCONNECT TOPOLOGY:",
            f"  NVLink: {'Yes' if topo.has_nvlink else 'No'}",
        ]
        
        if topo.has_nvlink:
            lines.extend([
                f"  NVSwitch (Full Mesh): {'Yes' if topo.has_nvswitch else 'No'}",
                f"  NVLink Version: {topo.nvlink_version or 'Unknown'}",
                f"  Max Bandwidth: {topo.max_nvlink_bandwidth_gbps:.0f} GB/s",
            ])
        
        lines.extend([
            "",
            "CPU TOPOLOGY:",
            f"  Type: {topo.cpu_type}",
            f"  Grace CPU: {'Yes' if topo.is_grace_cpu else 'No'}",
            f"  NVLink-C2C: {'Yes' if topo.has_nvlink_c2c else 'No'}",
            f"  Topology Source: {topo.topology_source}",
            f"  NUMA Nodes: {topo.numa_nodes if topo.gpu_numa_status != 'unknown' else 'unknown'}",
            f"  GPU NUMA Status: {topo.gpu_numa_status}",
        ])

        if topo.topology_preset_name:
            lines.append(f"  Preset: {topo.topology_preset_name}")
        
        if topo.gpu_numa_mapping:
            lines.append(f"  GPU-NUMA Mapping: {topo.gpu_numa_mapping}")
        
        lines.extend([
            "",
            "OPTIMAL PARALLELISM:",
            f"  Recommended TP sizes: {topo.get_optimal_tp_sizes()}",
        ])
        
        if topo.infiniband_available:
            lines.extend([
                "",
                "NETWORK:",
                f"  InfiniBand: Available",
                f"  RDMA: {'Yes' if topo.rdma_capable else 'No'}",
            ])
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def detect_topology() -> TopologyInfo:
    """Convenience function to detect topology."""
    detector = TopologyDetector()
    return detector.detect()


if __name__ == "__main__":
    detector = TopologyDetector()
    try:
        topology = detector.detect()
        print(detector.format_topology_report(topology))
        print("\nJSON output:")
        print(json.dumps(topology.to_dict(), indent=2))
    except RuntimeError as e:
        print(f"Error: {e}")
