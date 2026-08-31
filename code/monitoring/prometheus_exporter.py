#!/usr/bin/env python3
"""Prometheus exporter for AI performance metrics.

Exposes GPU, training, and inference metrics in Prometheus format for
monitoring Blackwell performance in production.

Requires PyTorch with CUDA support to collect GPU metrics.
"""

import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional
import threading

import typer

try:
    import torch  # type: ignore
except Exception as exc:  # pragma: no cover
    torch = None
    _torch_import_error = exc

from core.utils.logger import get_logger

logger = get_logger(__name__)


def format_prometheus_metrics(metrics: Dict[str, float]) -> str:
    """Serialize a snapshot with one HELP/TYPE pair per metric family."""
    lines = []
    emitted = set()
    for name, value in sorted(metrics.items()):
        family = name.split('{', 1)[0]
        if family not in emitted:
            if family.startswith('gpu_'):
                description = 'GPU metric'
            elif family.startswith('training_'):
                description = 'Training metric'
            elif family.startswith('inference_'):
                description = 'Inference metric'
            else:
                description = None
            if description is not None:
                lines.append(f'# HELP {family} {description}')
                lines.append(f'# TYPE {family} gauge')
            emitted.add(family)
        lines.append(f'{name} {value}')
    return '\n'.join(lines) + '\n'


class MetricsCollector:
    """Collect GPU and training/inference metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
        
        if not torch:
            raise RuntimeError(
                f"PyTorch is required for prometheus_exporter; install torch with CUDA support "
                f"and re-run (import error: {_torch_import_error})"
            )
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device_count = torch.cuda.device_count()
            logger.info(f"Monitoring {self.device_count} GPUs")
        else:
            raise RuntimeError("CUDA not available; GPU metrics exporter requires CUDA-enabled PyTorch")
    
    def collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU utilization and memory metrics."""
        metrics = {}
        
        if not self.cuda_available or not torch:
            return metrics
        
        for gpu_id in range(self.device_count):
            try:
                # cudaMemGetInfo reports device-wide free/total memory. PyTorch's
                # allocator counters below are process-local and are labeled as
                # such; they must not be presented as whole-GPU utilization.
                free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
                used_bytes = total_bytes - free_bytes
                mem_total = total_bytes / (1024**3)
                mem_used = used_bytes / (1024**3)
                process_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                process_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                
                metrics[f'gpu_process_memory_allocated_gb{{gpu="{gpu_id}"}}'] = process_allocated
                metrics[f'gpu_process_memory_reserved_gb{{gpu="{gpu_id}"}}'] = process_reserved
                metrics[f'gpu_memory_used_gb{{gpu="{gpu_id}"}}'] = mem_used
                metrics[f'gpu_memory_total_gb{{gpu="{gpu_id}"}}'] = mem_total
                if total_bytes > 0:
                    metrics[f'gpu_memory_utilization{{gpu="{gpu_id}"}}'] = used_bytes / total_bytes
                
                # Device properties
                props = torch.cuda.get_device_properties(gpu_id)
                metrics[f'gpu_sm_count{{gpu="{gpu_id}"}}'] = props.multi_processor_count
                
            except Exception as e:
                logger.warning(f"Error collecting metrics for GPU {gpu_id}: {e}")
        
        return metrics
    
    def collect_training_metrics(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        tokens_per_sec: float,
        batch_size: int
    ):
        """Record training metrics.
        
        Args:
            step: Training step
            loss: Current loss value
            learning_rate: Current learning rate
            tokens_per_sec: Throughput in tokens/sec
            batch_size: Current batch size
        """
        with self.lock:
            self.metrics['training_step'] = step
            self.metrics['training_loss'] = loss
            self.metrics['training_learning_rate'] = learning_rate
            self.metrics['training_tokens_per_sec'] = tokens_per_sec
            self.metrics['training_batch_size'] = batch_size
    
    def collect_inference_metrics(
        self,
        requests_per_sec: float,
        ttft_ms: float,
        tpot_ms: float,
        batch_size: int,
        queue_depth: int
    ):
        """Record inference metrics.
        
        Args:
            requests_per_sec: Requests per second
            ttft_ms: Time to first token (ms)
            tpot_ms: Time per output token (ms)
            batch_size: Average batch size
            queue_depth: Current queue depth
        """
        with self.lock:
            self.metrics['inference_requests_per_sec'] = requests_per_sec
            self.metrics['inference_ttft_ms'] = ttft_ms
            self.metrics['inference_tpot_ms'] = tpot_ms
            self.metrics['inference_batch_size'] = batch_size
            self.metrics['inference_queue_depth'] = queue_depth
    
    def format_prometheus(self) -> str:
        """Format metrics in Prometheus exposition format."""
        # Snapshot application metrics under the writer lock, then collect GPU
        # telemetry without holding that lock across CUDA calls/formatting.
        with self.lock:
            application_metrics = dict(self.metrics)
        return format_prometheus_metrics({**application_metrics, **self.collect_gpu_metrics()})


class PrometheusHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus scraping."""
    
    collector: Optional[MetricsCollector] = None
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/metrics':
            # Return Prometheus metrics
            metrics = self.collector.format_prometheus()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.end_headers()
            self.wfile.write(metrics.encode('utf-8'))
        elif self.path == '/health':
            # Health check
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK\n')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class PrometheusExporter:
    """Prometheus exporter for AI metrics."""
    
    def __init__(self, port: int = 9090):
        """Initialize exporter.
        
        Args:
            port: Port to expose metrics on
        """
        self.port = port
        self.collector = MetricsCollector()
        self.server = None
        self.server_thread = None
        
        # Set collector on handler class
        PrometheusHandler.collector = self.collector
    
    def start(self):
        """Start Prometheus exporter server."""
        self.server = HTTPServer(('0.0.0.0', self.port), PrometheusHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        
        logger.info(f"Prometheus exporter started on port {self.port}")
        logger.info(f"Metrics endpoint: http://localhost:{self.port}/metrics")
        logger.info(f"Health endpoint: http://localhost:{self.port}/health")
    
    def stop(self):
        """Stop exporter server."""
        if self.server:
            self.server.shutdown()
            logger.info("Prometheus exporter stopped")
    
    def update_training_metrics(self, **kwargs):
        """Update training metrics."""
        self.collector.collect_training_metrics(**kwargs)
    
    def update_inference_metrics(self, **kwargs):
        """Update inference metrics."""
        self.collector.collect_inference_metrics(**kwargs)


app = typer.Typer(help="Prometheus Exporter for AI metrics")


def _demo_loop(exporter: PrometheusExporter) -> None:
    """Simulate training/inference metrics when --demo is enabled."""
    print(f"\n{'='*60}")
    print("Prometheus Exporter - Demo Mode")
    print(f"{'='*60}")
    print(f"Metrics URL: http://localhost:{exporter.port}/metrics")
    print(f"Health URL: http://localhost:{exporter.port}/health")
    print("\nSimulating training metrics...")
    print("Press Ctrl+C to stop\n")

    try:
        step = 0
        while True:
            exporter.update_training_metrics(
                step=step,
                loss=3.5 - (step * 0.001),  # Decreasing loss
                learning_rate=1e-4,
                tokens_per_sec=10000 + (step * 10),
                batch_size=32,
            )

            exporter.update_inference_metrics(
                requests_per_sec=100.0,
                ttft_ms=50.0,
                tpot_ms=10.0,
                batch_size=8,
                queue_depth=5,
            )

            step += 1
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping exporter...")
        exporter.stop()


@app.command("serve")
def serve(
    port: int = typer.Option(9100, "--port", "-p", help="Port to run the exporter on"),
    demo: bool = typer.Option(False, "--demo", help="Run with simulated metrics"),
) -> None:
    """Start the Prometheus exporter."""
    exporter = PrometheusExporter(port=port)
    exporter.start()

    if demo:
        _demo_loop(exporter)
        return

    print(f"\n{'='*60}")
    print("Prometheus Exporter Started")
    print(f"{'='*60}")
    print(f"Metrics URL: http://localhost:{port}/metrics")
    print(f"Health URL: http://localhost:{port}/health")
    print("\nExporter running. Press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping exporter...")
        exporter.stop()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
