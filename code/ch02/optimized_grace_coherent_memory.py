#!/usr/bin/env python3
"""Optimized: zero-copy update of a pinned host buffer over NVLink-C2C.

The baseline stages each iteration through pageable H2D and D2H copies. Here the
GPU reads and writes the same host buffer in place, so the data still starts and
ends in host memory but no staging copies run.
"""

import ctypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.grace_platform import host_link_flags
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.utils.logger import get_logger
from core.utils.pinned_host_view import pinned_host_cuda_view

logger = get_logger(__name__)

_CUDA_SKIP_REASON = "SKIPPED: grace_coherent_memory requires CUDA"
_GRACE_SKIP_REASON = (
    "SKIPPED: grace_coherent_memory requires Grace-Blackwell coherent memory support "
    "(the CUDA driver must report an NVLink-C2C host link to a Grace CPU, as on GB200 and GB300)."
)


def _normalize_pci_bus_id(bus_id: str) -> str:
    value = str(bus_id).strip().lower()
    if value.startswith("00000000:"):
        return value[4:]
    return value


def _query_gpu_pci_bus_id(gpu_id: int) -> Optional[str]:
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    bus_id = result.stdout.strip()
    return _normalize_pci_bus_id(bus_id) if bus_id else None


def _gpu_numa_node_from_sysfs(gpu_id: int) -> Optional[int]:
    bus_id = _query_gpu_pci_bus_id(gpu_id)
    if not bus_id:
        return None
    numa_path = Path(f"/sys/bus/pci/devices/{bus_id}/numa_node")
    try:
        raw = numa_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _cpus_for_numa_node(numa_node: int) -> List[int]:
    cpulist_path = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
    if not cpulist_path.exists():
        return []
    ranges = cpulist_path.read_text(encoding="utf-8").strip().split(",")
    cpus: List[int] = []
    for cpu_range in ranges:
        if not cpu_range:
            continue
        if "-" in cpu_range:
            start, end = cpu_range.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(cpu_range))
    return cpus


class OptimizedGraceCoherentMemory:
    """GPU updates a pinned host buffer in place, with no staging copies."""

    strategy = "zero_copy"

    def __init__(self, size_mb: int = 256, iterations: int = 100):
        if not torch.cuda.is_available():
            raise RuntimeError(_CUDA_SKIP_REASON)
        self.size_mb = size_mb
        self.iterations = iterations
        self.device = torch.device("cuda")

        # Check if we're on Grace-Blackwell
        self.is_grace_blackwell = self._detect_grace_blackwell()
        if not self.is_grace_blackwell:
            raise RuntimeError(_GRACE_SKIP_REASON)

    def _detect_grace_blackwell(self) -> bool:
        """Return whether the CUDA driver puts this GPU on an NVLink-C2C host link to Grace.

        Compute capability cannot decide this. GB200 and GB300 GPUs report SM
        10.0 and 10.3, the same as B200 and B300 GPUs on PCIe in x86 servers.
        The zero-copy step reads pinned host memory over the link, so it does
        not need the driver's pageable-memory (ATS) flag. Driver query errors
        propagate.
        """
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        nvlink_c2c, _ = host_link_flags(props, device_index)
        return nvlink_c2c

    def _bind_numa_node(self):
        """Bind to NUMA node closest to GPU (Grace-Blackwell specific)."""
        if not self.is_grace_blackwell:
            return

        gpu_id = torch.cuda.current_device()
        numa_node = _gpu_numa_node_from_sysfs(gpu_id)
        if numa_node is None:
            logger.info(
                "GPU NUMA node unavailable for GPU %s; leaving process affinity unchanged", gpu_id
            )
            return

        # Try to pin this process' CPU affinity to the NUMA node's CPUs
        try:
            cpus = _cpus_for_numa_node(numa_node)
            if cpus:
                os.sched_setaffinity(0, cpus)
                logger.info(f"Bound CPU affinity to NUMA node {numa_node}: {cpus}")
        except Exception as e:  # pragma: no cover - best effort
            logger.debug(f"NUMA CPU affinity binding failed: {e}")

        # Best-effort memory preference using libnuma if available
        try:
            libnuma = ctypes.CDLL("libnuma.so.1")
            if libnuma.numa_available() != -1:
                libnuma.numa_run_on_node(ctypes.c_int(numa_node))
                libnuma.numa_set_preferred(ctypes.c_int(numa_node))
                logger.info(f"Set NUMA memory preference to node {numa_node}")
        except Exception as e:  # pragma: no cover - optional path
            logger.debug(f"NUMA memory binding skipped: {e}")

    def setup(self):
        """Allocate the pinned host buffer and its zero-copy GPU view."""
        num_elements = (self.size_mb * 1024 * 1024) // 4  # float32

        # Bind to optimal NUMA node
        self._bind_numa_node()

        device = torch.device("cuda", torch.cuda.current_device())
        # Same RNG draw as the baseline's pageable host buffer, then pinned so the
        # GPU can address it directly.
        self.cpu_data = torch.randn(num_elements, dtype=torch.float32).pin_memory()
        self.gpu_data = pinned_host_cuda_view(self.cpu_data, device)
        self._one = torch.ones((), dtype=torch.float32, device=device)
        logger.info(f"Using zero-copy pinned host buffer ({self.size_mb}MB)")

    def run_step(self) -> float:
        """Compute ``2x + 1`` on the host buffer with one GPU pass and no copies."""
        torch.cuda.synchronize()
        start = time.perf_counter()

        # One fused pass reads each element across the CPU-GPU link once and
        # writes it back once. Separate mul_ and add_ calls would cross it twice.
        # 2x is exact in fp32, so this matches the baseline's mul_ then add_ bit for bit.
        torch.add(self._one, self.gpu_data, alpha=2.0, out=self.gpu_data)

        torch.cuda.synchronize()
        end = time.perf_counter()
        return end - start

    def cleanup(self):
        """Clean up resources."""
        del self.gpu_data
        del self.cpu_data
        del self._one
        torch.cuda.empty_cache()


class OptimizedGraceCoherentMemoryBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness-friendly wrapper around the optimized coherent memory path."""

    allowed_benchmark_fn_antipatterns = ("sync", "host_transfer")

    def __init__(self, size_mb: int = 256, iterations: int = 100):
        super().__init__()
        self._impl = OptimizedGraceCoherentMemory(
            size_mb=size_mb,
            iterations=iterations,
        )
        # Each step reads the whole host buffer across the CPU-GPU link and writes
        # it back, the same payload as the baseline's H2D plus D2H copies.
        bytes_per_iter = size_mb * 1024 * 1024 * 2
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )
        self.elapsed_s: Optional[float] = None
        self.bandwidth_gb_s: Optional[float] = None
        self.size_mb = size_mb
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        # Inputs use the RNG already seeded by the harness.
        self._impl.setup()
        self._verify_output_buffer = torch.empty_like(self._impl.cpu_data[:1000])
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )

    def benchmark_fn(self) -> None:
        elapsed = self._impl.run_step()
        self.elapsed_s = elapsed
        self.bandwidth_gb_s = self._workload.bytes_per_iteration / elapsed / 1e9
        self.output = None

    def capture_verification_payload(self) -> None:
        # Compare the host buffer, which is where the baseline also leaves its result.
        if self.output is None:
            if self._verify_output_buffer is None:
                raise RuntimeError("setup() must be called before verification")
            output_slice = self._impl.cpu_data[: self._verify_output_buffer.numel()].detach()
            self._verify_output_buffer.copy_(output_slice)
            self.output = self._verify_output_buffer
        self._set_verification_payload(
            inputs={
                "cpu_data": self._impl.cpu_data,
                "gpu_data": self._impl.gpu_data,
            },
            output=self.output,
            batch_size=self._impl.cpu_data.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32
                if torch.cuda.is_available()
                else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self._impl.cleanup()
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=max(1, self._impl.iterations),
            warmup=5,
            enable_memory_tracking=False,
            # benchmark_fn mutates the transfer buffers, so adaptive iteration
            # expansion would change the observable output used for verification.
            adaptive_iterations=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return the bytes each step moves across the CPU-GPU link."""
        # core.benchmark.metrics has no NVLink-C2C peak, and its NVLink peak is the
        # GPU-to-GPU link, so this reports achieved bandwidth without an efficiency.
        bytes_moved = self._workload.bytes_per_iteration
        metrics = {"transfer.bytes": bytes_moved}
        if self.elapsed_s is not None:
            metrics["transfer.achieved_gbps"] = bytes_moved / self.elapsed_s / 1e9
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.elapsed_s is None:
            return "Benchmark did not run"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedGraceCoherentMemoryBenchmark()


def run_benchmark(
    size_mb: int = 256, iterations: int = 100, profile: str = "none", **kwargs
) -> Dict[str, Any]:
    """Run optimized Grace coherent memory benchmark."""

    benchmark = OptimizedGraceCoherentMemoryBenchmark(size_mb=size_mb, iterations=iterations)
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(
            iterations=max(1, iterations),
            warmup=5,
            profile_mode=profile,
        ),
    )
    result = harness.benchmark(benchmark, name="optimized_grace_coherent_memory")

    return {
        "mean_time_ms": result.timing.mean_ms if result.timing else 0.0,
        "is_grace_blackwell": getattr(benchmark, "_impl", None).is_grace_blackwell
        if hasattr(benchmark, "_impl")
        else False,
        "strategy": getattr(benchmark, "_impl", None).strategy
        if hasattr(benchmark, "_impl")
        else "",
        "size_mb": size_mb,
        "iterations": iterations,
    }
