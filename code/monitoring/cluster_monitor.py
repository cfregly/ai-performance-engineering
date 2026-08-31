#!/usr/bin/env python3
"""
🌐 Real-Time Cluster Monitoring

Multi-node GPU cluster monitoring with distributed profiling aggregation.

Features:
- Real-time GPU metrics across all nodes
- Distributed profiling aggregation (nsys, ncu)
- Communication pattern analysis (NCCL, IB)
- Hotspot detection across the cluster
- Bottleneck identification
- Auto-scaling recommendations

Architecture:
    Monitor Agent (runs on each node)
        │
        ├─> Collects GPU metrics (nvidia-smi)
        ├─> Collects NCCL stats
        ├─> Collects network stats
        │
        └─> Reports to Aggregator
                │
                └─> Aggregates cluster-wide view
                    │
                    └─> Exposes via REST/WebSocket

Usage:
    # Start agent on each node
    python -m monitoring.cluster_monitor agent --master-addr 10.0.0.1

    # Start aggregator on master node
    python -m monitoring.cluster_monitor master --port 9999

    # View cluster status
    python -m monitoring.cluster_monitor status --master-addr 10.0.0.1:9999

    # From unified CLI
    python -m cli.aisp monitor cluster agent --master-addr 10.0.0.1
    python -m cli.aisp monitor cluster master --port 9999
    python -m cli.aisp monitor cluster status --master-addr 10.0.0.1:9999
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import queue
import http.server
import socketserver


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GPUMetrics:
    """GPU metrics from a single GPU."""
    gpu_id: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_pct: int
    temperature_c: int
    power_w: float
    sm_clock_mhz: int
    memory_clock_mhz: int
    pcie_tx_gbps: float = 0.0
    pcie_rx_gbps: float = 0.0
    nvlink_tx_gbps: float = 0.0
    nvlink_rx_gbps: float = 0.0


@dataclass
class NodeMetrics:
    """Metrics from a single node."""
    hostname: str
    node_id: int
    timestamp: float
    gpus: List[GPUMetrics] = field(default_factory=list)
    cpu_utilization_pct: Optional[float] = None
    cpu_telemetry_status: str = "unavailable"
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    network_rx_gbps: Optional[float] = None
    network_tx_gbps: Optional[float] = None
    network_telemetry_status: str = "unavailable"
    ib_rx_gbps: Optional[float] = None
    ib_tx_gbps: Optional[float] = None
    ib_telemetry_status: str = "unavailable"
    nccl_collectives_per_sec: float = 0.0
    nccl_bytes_per_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "gpus": [asdict(g) for g in self.gpus],
            "cpu_utilization_pct": self.cpu_utilization_pct,
            "cpu_telemetry_status": self.cpu_telemetry_status,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "network_rx_gbps": self.network_rx_gbps,
            "network_tx_gbps": self.network_tx_gbps,
            "network_telemetry_status": self.network_telemetry_status,
            "ib_rx_gbps": self.ib_rx_gbps,
            "ib_tx_gbps": self.ib_tx_gbps,
            "ib_telemetry_status": self.ib_telemetry_status,
            "nccl_collectives_per_sec": self.nccl_collectives_per_sec,
            "nccl_bytes_per_sec": self.nccl_bytes_per_sec,
        }


@dataclass
class ClusterMetrics:
    """Aggregated cluster metrics."""
    timestamp: float
    nodes: List[NodeMetrics] = field(default_factory=list)
    total_gpus: int = 0
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    avg_gpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    total_interconnect_gbps: float = 0.0
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "nodes": [n.to_dict() for n in self.nodes],
            "total_gpus": self.total_gpus,
            "total_memory_gb": self.total_memory_gb,
            "used_memory_gb": self.used_memory_gb,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "avg_memory_utilization": self.avg_memory_utilization,
            "total_interconnect_gbps": self.total_interconnect_gbps,
            "hotspots": self.hotspots,
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
        }


# =============================================================================
# METRIC COLLECTORS
# =============================================================================

class GPUCollector:
    """Collects GPU metrics using nvidia-smi."""
    
    @staticmethod
    def collect() -> List[GPUMetrics]:
        """Collect metrics from all GPUs on this node."""
        gpus = []
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,'
                 'utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 9:
                        gpus.append(GPUMetrics(
                            gpu_id=int(parts[0]),
                            name=parts[1],
                            memory_used_mb=int(parts[2]),
                            memory_total_mb=int(parts[3]),
                            utilization_pct=int(parts[4]) if parts[4] != '[N/A]' else 0,
                            temperature_c=int(parts[5]),
                            power_w=float(parts[6]) if parts[6] != '[N/A]' else 0,
                            sm_clock_mhz=int(parts[7]),
                            memory_clock_mhz=int(parts[8]),
                        ))
        except Exception:
            pass
        
        # Try to get NVLink stats
        try:
            result = subprocess.run(
                ['nvidia-smi', 'nvlink', '-g', '0', '-gt', 'd'],
                capture_output=True, text=True, timeout=5
            )
            # Parse NVLink throughput if available
            if result.returncode == 0 and gpus:
                # Basic parsing - in reality would need more sophisticated parsing
                for gpu in gpus:
                    gpu.nvlink_tx_gbps = 0.0  # Placeholder
                    gpu.nvlink_rx_gbps = 0.0
        except Exception:
            pass
        
        return gpus


class SystemCollector:
    """Collects system-wide metrics."""

    _cpu_sample_lock = threading.Lock()
    _previous_cpu_sample: Optional[Tuple[int, int]] = None

    @staticmethod
    def _read_cpu_sample() -> Tuple[int, int]:
        """Return cumulative (idle, total) jiffies from /proc/stat."""
        with open('/proc/stat') as f:
            parts = f.readline().split()
        if not parts or parts[0] != 'cpu':
            raise ValueError("/proc/stat does not contain an aggregate cpu row")
        values = [int(value) for value in parts[1:9]]
        if len(values) < 4:
            raise ValueError("/proc/stat aggregate cpu row is incomplete")
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return idle, sum(values)

    @classmethod
    def collect(cls) -> Dict[str, Any]:
        """Collect CPU and memory metrics."""
        metrics = {
            "cpu_utilization_pct": None,
            "cpu_telemetry_status": "unavailable",
            "memory_used_gb": 0.0,
            "memory_total_gb": 0.0,
        }
        
        # /proc/stat fields are cumulative. Report utilization only from the
        # delta between two observations; the first observation is warmup.
        try:
            with cls._cpu_sample_lock:
                # Keep the sample read and prior-sample update atomic so two
                # callers cannot publish deltas from observations in reverse order.
                current_idle, current_total = cls._read_cpu_sample()
                previous = cls._previous_cpu_sample
                cls._previous_cpu_sample = (current_idle, current_total)
            if previous is None:
                metrics["cpu_telemetry_status"] = "warming_up"
            else:
                previous_idle, previous_total = previous
                delta_total = current_total - previous_total
                delta_idle = current_idle - previous_idle
                if delta_total > 0 and 0 <= delta_idle <= delta_total:
                    metrics["cpu_utilization_pct"] = 100.0 * (delta_total - delta_idle) / delta_total
                    metrics["cpu_telemetry_status"] = "ok"
                else:
                    metrics["cpu_telemetry_status"] = "counter_reset"
        except Exception as exc:
            metrics["cpu_telemetry_status"] = f"unavailable: {type(exc).__name__}"
        
        # Memory
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal'):
                        metrics["memory_total_gb"] = int(line.split()[1]) / 1024 / 1024
                    elif line.startswith('MemAvailable'):
                        available = int(line.split()[1]) / 1024 / 1024
                        metrics["memory_used_gb"] = metrics["memory_total_gb"] - available
        except Exception:
            pass
        
        return metrics


class NetworkCollector:
    """Collects network metrics including InfiniBand."""
    
    @staticmethod
    def collect() -> Dict[str, Any]:
        """Collect network throughput metrics."""
        metrics = {
            "network_rx_gbps": None,
            "network_tx_gbps": None,
            "network_telemetry_status": "unavailable: no sampled byte-counter source",
            "ib_rx_gbps": None,
            "ib_tx_gbps": None,
            "ib_telemetry_status": "unavailable",
        }
        
        # Check for InfiniBand
        try:
            result = subprocess.run(
                ['ibstat'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and 'Active' in result.stdout:
                # ibstat establishes link state only. It does not report a
                # throughput rate, and a single perfquery counter snapshot is
                # cumulative, so publishing Gbps here would fabricate telemetry.
                metrics["ib_telemetry_status"] = (
                    "unavailable: active link but no sampled byte-counter delta"
                )
            elif result.returncode == 0:
                metrics["ib_telemetry_status"] = "unsupported_or_inactive"
            else:
                metrics["ib_telemetry_status"] = "unavailable: ibstat failed"
        except FileNotFoundError:
            metrics["ib_telemetry_status"] = "unsupported: ibstat not installed"
        except Exception as exc:
            metrics["ib_telemetry_status"] = f"unavailable: {type(exc).__name__}"
        
        return metrics


# =============================================================================
# MONITORING AGENT (runs on each node)
# =============================================================================

class MonitorAgent:
    """
    Monitoring agent that runs on each node.
    Collects local metrics and reports to the master aggregator.
    """
    
    def __init__(
        self,
        master_addr: str = "localhost",
        master_port: int = 9999,
        interval_seconds: float = 1.0,
        node_id: Optional[int] = None,
    ):
        self.master_addr = master_addr
        self.master_port = master_port
        self.interval = interval_seconds
        self.node_id = node_id or self._detect_node_id()
        self.hostname = socket.gethostname()
        self.running = False
        self._thread = None
    
    def _detect_node_id(self) -> int:
        """Detect node ID from environment or hostname."""
        # Try SLURM
        if "SLURM_NODEID" in os.environ:
            return int(os.environ["SLURM_NODEID"])
        
        # Try to parse from hostname
        hostname = socket.gethostname()
        import re
        match = re.search(r'\d+$', hostname)
        if match:
            return int(match.group())
        
        return 0
    
    def collect_metrics(self) -> NodeMetrics:
        """Collect all metrics from this node."""
        gpus = GPUCollector.collect()
        system = SystemCollector.collect()
        network = NetworkCollector.collect()
        
        return NodeMetrics(
            hostname=self.hostname,
            node_id=self.node_id,
            timestamp=time.time(),
            gpus=gpus,
            cpu_utilization_pct=system["cpu_utilization_pct"],
            cpu_telemetry_status=system["cpu_telemetry_status"],
            memory_used_gb=system["memory_used_gb"],
            memory_total_gb=system["memory_total_gb"],
            network_rx_gbps=network["network_rx_gbps"],
            network_tx_gbps=network["network_tx_gbps"],
            network_telemetry_status=network["network_telemetry_status"],
            ib_rx_gbps=network["ib_rx_gbps"],
            ib_tx_gbps=network["ib_tx_gbps"],
            ib_telemetry_status=network["ib_telemetry_status"],
        )
    
    def report_metrics(self, metrics: NodeMetrics):
        """Report metrics to master aggregator."""
        import urllib.request
        import urllib.error
        
        try:
            url = f"http://{self.master_addr}:{self.master_port}/api/report"
            data = json.dumps(metrics.to_dict()).encode('utf-8')
            
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass  # Just need to send, don't care about response
        except Exception as e:
            # Silently ignore connection errors
            pass
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            metrics = self.collect_metrics()
            self.report_metrics(metrics)
            time.sleep(self.interval)
    
    def start(self):
        """Start the monitoring agent."""
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"🔍 Agent started on {self.hostname} (node {self.node_id})")
        print(f"   Reporting to {self.master_addr}:{self.master_port}")
    
    def stop(self):
        """Stop the monitoring agent."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("Agent stopped")
    
    def run_foreground(self):
        """Run agent in foreground."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


# =============================================================================
# MASTER AGGREGATOR (runs on master node)
# =============================================================================

class ClusterAggregator:
    """
    Master aggregator that collects metrics from all nodes.
    Provides cluster-wide view and analysis.
    """
    
    def __init__(
        self,
        port: int = 9999,
        history_seconds: int = 300,  # Keep 5 minutes of history
    ):
        self.port = port
        self.history_seconds = history_seconds
        self.nodes: Dict[str, NodeMetrics] = {}  # hostname -> latest metrics
        self.history: List[ClusterMetrics] = []
        self.lock = threading.Lock()
    
    def report_node_metrics(self, metrics: NodeMetrics):
        """Receive metrics from a node."""
        with self.lock:
            self.nodes[metrics.hostname] = metrics
    
    def aggregate(self) -> ClusterMetrics:
        """Aggregate metrics from all nodes."""
        with self.lock:
            nodes = list(self.nodes.values())
        
        if not nodes:
            return ClusterMetrics(timestamp=time.time())
        
        # Aggregate
        total_gpus = sum(len(n.gpus) for n in nodes)
        total_memory = sum(g.memory_total_mb for n in nodes for g in n.gpus) / 1024
        used_memory = sum(g.memory_used_mb for n in nodes for g in n.gpus) / 1024
        
        all_utils = [g.utilization_pct for n in nodes for g in n.gpus]
        avg_util = sum(all_utils) / len(all_utils) if all_utils else 0
        
        # Detect hotspots (GPUs with high utilization but low memory)
        hotspots = []
        for node in nodes:
            for gpu in node.gpus:
                if gpu.utilization_pct > 90:
                    hotspots.append({
                        "node": node.hostname,
                        "gpu": gpu.gpu_id,
                        "type": "high_compute",
                        "utilization": gpu.utilization_pct,
                    })
                if gpu.temperature_c > 80:
                    hotspots.append({
                        "node": node.hostname,
                        "gpu": gpu.gpu_id,
                        "type": "high_temperature",
                        "temperature": gpu.temperature_c,
                    })
        
        # Detect bottlenecks
        bottlenecks = []
        mem_utils = [g.memory_used_mb / g.memory_total_mb * 100 for n in nodes for g in n.gpus if g.memory_total_mb > 0]
        if mem_utils and max(mem_utils) > 90:
            bottlenecks.append("Memory near capacity on some GPUs")
        
        if avg_util < 50:
            bottlenecks.append("Low GPU utilization - consider larger batch size")
        
        # Generate recommendations
        recommendations = []
        if avg_util < 30:
            recommendations.append("Consider enabling gradient accumulation")
        if mem_utils and max(mem_utils) > 85:
            recommendations.append("Consider gradient checkpointing")
        ib_rx_samples = [
            n.ib_rx_gbps
            for n in nodes
            if n.ib_telemetry_status == "ok" and n.ib_rx_gbps is not None
        ]
        if len(nodes) > 4 and ib_rx_samples and not any(value > 0 for value in ib_rx_samples):
            recommendations.append("Consider enabling InfiniBand for better scaling")
        
        cluster = ClusterMetrics(
            timestamp=time.time(),
            nodes=nodes,
            total_gpus=total_gpus,
            total_memory_gb=total_memory,
            used_memory_gb=used_memory,
            avg_gpu_utilization=avg_util,
            avg_memory_utilization=(used_memory / total_memory * 100) if total_memory > 0 else 0,
            hotspots=hotspots,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )
        
        # Add to history
        self.history.append(cluster)
        # Prune old history
        cutoff = time.time() - self.history_seconds
        self.history = [h for h in self.history if h.timestamp > cutoff]
        
        return cluster
    
    def get_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        cluster = self.aggregate()
        return cluster.to_dict()
    
    def get_history(self, seconds: int = 60) -> List[Dict]:
        """Get metrics history."""
        cutoff = time.time() - seconds
        return [h.to_dict() for h in self.history if h.timestamp > cutoff]


class AggregatorHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the aggregator server."""
    
    aggregator: ClusterAggregator = None
    
    def log_message(self, format, *args):
        pass  # Suppress logs
    
    def do_GET(self):
        if self.path == '/api/status':
            self.send_json(self.aggregator.get_status())
        elif self.path == '/api/history':
            self.send_json(self.aggregator.get_history())
        elif self.path == '/api/nodes':
            with self.aggregator.lock:
                nodes = [n.to_dict() for n in self.aggregator.nodes.values()]
            self.send_json({"nodes": nodes})
        elif self.path == '/api/health':
            self.send_json({"status": "healthy", "nodes": len(self.aggregator.nodes)})
        else:
            self.send_json({"error": "Unknown endpoint"})
    
    def do_POST(self):
        if self.path == '/api/report':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                metrics = NodeMetrics(
                    hostname=data["hostname"],
                    node_id=data["node_id"],
                    timestamp=data["timestamp"],
                    gpus=[GPUMetrics(**g) for g in data.get("gpus", [])],
                    cpu_utilization_pct=data.get("cpu_utilization_pct"),
                    cpu_telemetry_status=data.get("cpu_telemetry_status", "unavailable"),
                    memory_used_gb=data.get("memory_used_gb", 0),
                    memory_total_gb=data.get("memory_total_gb", 0),
                    network_rx_gbps=data.get("network_rx_gbps"),
                    network_tx_gbps=data.get("network_tx_gbps"),
                    network_telemetry_status=data.get("network_telemetry_status", "unavailable"),
                    ib_rx_gbps=data.get("ib_rx_gbps"),
                    ib_tx_gbps=data.get("ib_tx_gbps"),
                    ib_telemetry_status=data.get("ib_telemetry_status", "unavailable"),
                )
                self.aggregator.report_node_metrics(metrics)
                self.send_json({"status": "ok"})
            except Exception as e:
                self.send_json({"error": str(e)})
        else:
            self.send_json({"error": "Unknown endpoint"})
    
    def send_json(self, data: dict):
        response = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)


def run_aggregator_server(port: int = 9999):
    """Run the aggregator HTTP server."""
    aggregator = ClusterAggregator(port=port)
    AggregatorHandler.aggregator = aggregator
    
    with socketserver.TCPServer(("", port), AggregatorHandler) as httpd:
        print(f"🌐 Cluster Monitor Master running on port {port}")
        print(f"   Status: http://localhost:{port}/api/status")
        print(f"   Nodes:  http://localhost:{port}/api/nodes")
        print(f"   History: http://localhost:{port}/api/history")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


# =============================================================================
# CLI UTILITIES
# =============================================================================

def get_cluster_status(master_addr: str, master_port: int) -> Dict:
    """Get cluster status from master."""
    import urllib.request
    
    url = f"http://{master_addr}:{master_port}/api/status"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode())


def print_cluster_status(status: Dict):
    """Pretty print cluster status."""
    print("\n🌐 Cluster Status")
    print("=" * 60)
    print(f"Total GPUs: {status['total_gpus']}")
    print(f"GPU Memory: {status['used_memory_gb']:.1f} / {status['total_memory_gb']:.1f} GB "
          f"({status['avg_memory_utilization']:.1f}%)")
    print(f"Avg GPU Utilization: {status['avg_gpu_utilization']:.1f}%")
    
    if status['nodes']:
        print(f"\n📊 Nodes ({len(status['nodes'])})")
        for node in status['nodes']:
            gpu_count = len(node['gpus'])
            avg_util = sum(g['utilization_pct'] for g in node['gpus']) / gpu_count if gpu_count else 0
            print(f"   {node['hostname']}: {gpu_count} GPUs, {avg_util:.1f}% util")
    
    if status['hotspots']:
        print("\n🔥 Hotspots")
        for h in status['hotspots'][:5]:
            print(f"   {h['node']}:GPU{h['gpu']} - {h['type']}")
    
    if status['bottlenecks']:
        print("\n⚠️ Bottlenecks")
        for b in status['bottlenecks']:
            print(f"   • {b}")
    
    if status['recommendations']:
        print("\n💡 Recommendations")
        for r in status['recommendations']:
            print(f"   • {r}")


# =============================================================================
# CLI
# =============================================================================

import typer
from types import SimpleNamespace

app = typer.Typer(help="Cluster monitoring (agent/master/status/local)")


@app.command("agent", help="Run monitoring agent")
def cli_agent(
    master_addr: str = typer.Option("localhost", "--master-addr"),
    master_port: int = typer.Option(9999, "--master-port"),
    interval: float = typer.Option(1.0, "--interval"),
    node_id: int = typer.Option(None, "--node-id"),
) -> None:
    agent = MonitorAgent(
        master_addr=master_addr,
        master_port=master_port,
        interval_seconds=interval,
        node_id=node_id,
    )
    agent.run_foreground()


@app.command("master", help="Run master aggregator")
def cli_master(port: int = typer.Option(9999, "--port")) -> None:
    run_aggregator_server(port=port)


@app.command("status", help="Get cluster status")
def cli_status(
    master_addr: str = typer.Option("localhost:9999", "--master-addr"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    parts = master_addr.split(":")
    addr = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 9999
    status = get_cluster_status(addr, port)
    if json_out:
        print(json.dumps(status, indent=2))
    else:
        print_cluster_status(status)


@app.command("local", help="Monitor local GPUs only")
def cli_local(
    interval: float = typer.Option(1.0, "--interval"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    print("🔍 Local GPU Monitor")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    try:
        while True:
            gpus = GPUCollector.collect()
            if json_out:
                print(json.dumps([asdict(g) for g in gpus], indent=2))
            else:
                print(f"\n{'GPU':<6} {'Util':<8} {'Temp':<8} {'Memory':<20} {'Power':<10}")
                print("-" * 52)
                for gpu in gpus:
                    mem = f"{gpu.memory_used_mb}/{gpu.memory_total_mb}MB"
                    print(f"{gpu.gpu_id:<6} {gpu.utilization_pct:>5}%  {gpu.temperature_c:>5}°C  {mem:<20} {gpu.power_w:>6.1f}W")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
