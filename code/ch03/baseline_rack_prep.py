"""Rack-prep baseline: topology-unaware pageable staging."""

from __future__ import annotations

from typing import List, Optional
import os

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

from ch03.grace_blackwell_topology import NICInfo, discover_nics, format_cpulist


class BaselineRackPrepBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Stages a NIC-style batch without locality hints or overlap."""

    def __init__(self):
        super().__init__()
        self.seq_len = 4096
        self.hidden_size = 4096
        self.host_batch: Optional[torch.Tensor] = None
        self.device_batch: Optional[torch.Tensor] = None
        self.norm: Optional[nn.Module] = None
        self.nic_snapshot: List[NICInfo] = []
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        bytes_per_iter = self.seq_len * self.hidden_size * 4  # float32 bytes
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.nic_snapshot = discover_nics()
        self.host_batch = torch.randn(self.seq_len, self.hidden_size, dtype=torch.float32)
        self.device_batch = torch.empty_like(self.host_batch, device=self.device)
        self.norm = nn.LayerNorm(self.hidden_size, device=self.device)
        self._verify_output_buffer = torch.empty_like(self.device_batch)
        self._payload_parameter_count = sum(p.numel() for p in self.norm.parameters())
        
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.host_batch is not None and self.device_batch is not None and self.norm is not None
        with torch.inference_mode(), self._nvtx_range("baseline_rack_prep"):
            self.device_batch.copy_(self.host_batch, non_blocking=False)
            self.output = self.norm(self.device_batch)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"host_batch": self.host_batch, "device_batch": self.device_batch},
            output=self._verify_output_buffer,
            batch_size=self.host_batch.shape[0],
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1.0, 10.0),
        )

    def teardown(self) -> None:
        self.host_batch = None
        self.device_batch = None
        self.norm = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=12, warmup=10)

    def validate_result(self) -> Optional[str]:
        if self.host_batch is None or self.norm is None:
            return "Host batch or model not initialized"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        if not self.nic_snapshot:
            return None
        summaries = [f"{n.name}:numa={n.numa_node},cpus={format_cpulist(n.local_cpus)}" for n in self.nic_snapshot]
        return {"nic_layouts": len(self.nic_snapshot), "nic_summary": ";".join(summaries)}


def get_benchmark() -> BaseBenchmark:
    return BaselineRackPrepBenchmark()
