"""Pinned-prefetch MLP baseline: synchronous host batches each iteration.

This benchmark demonstrates inefficient data loading - using non-pinned memory
and blocking H2D copies. The optimized version uses pinned memory and prefetching.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import os
import torch
import torch.nn as nn
from core.benchmark.smoke import is_smoke_mode

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.optimization.allocator_tuning import log_allocator_guidance
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselinePinnedPrefetchMLPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Uses blocking host-to-device copies without pinned-memory prefetch."""

    def __init__(self):
        super().__init__()
        low_mem = is_smoke_mode()
        # Match optimized workload for fair comparison
        self.input_dim = 2048 if low_mem else 4096
        self.hidden_dim = 2048 if low_mem else 4096
        self.output_dim = 1024 if low_mem else 2048
        self.batch_size = 512 if low_mem else 1024  # Large batch = significant H2D
        self.num_batches = 4 if low_mem else 8
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.host_batches: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.batch_idx = 0
        self._batch_count = 0
        self.output: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        # Training benchmarks don't support jitter check - outputs change due to weight updates
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(self.batch_size * self.input_dim * 4),  # float32
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        log_allocator_guidance("ch03/baseline_pinned_prefetch_mlp", optimized=False)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim),
        ).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())

        for _ in range(self.num_batches):
            self.host_batches.append(torch.randn(self.batch_size, self.input_dim, dtype=torch.float32))
            self.targets.append(torch.randn(self.batch_size, self.output_dim, dtype=torch.float32))
        self._batch_count = self.num_batches
        
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.optimizer is not None

        idx = self.batch_idx % self._batch_count
        host_x = self.host_batches[idx]
        host_y = self.targets[idx]
        self.batch_idx += 1

        with self._nvtx_range("baseline_pinned_prefetch_mlp"):
            x = self.to_device(host_x)  # blocking copy (tensor not pinned)
            y = self.to_device(host_y)
            out = self.model(x)
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Store output for verification
        self.output = out.detach_()
        self._payload_x = x
        self._payload_y = y

    def capture_verification_payload(self) -> None:
        x = self._payload_x
        y = self._payload_y
        self._set_verification_payload(
            inputs={"data": x, "target": y},
            output=self.output,
            batch_size=self.batch_size,
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
        self.model = None
        self.optimizer = None
        self.host_batches = []
        self.targets = []
        self._batch_count = 0
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        low_mem = is_smoke_mode()
        # Minimum warmup=5 even in smoke mode to exclude JIT overhead
        return BenchmarkConfig(iterations=5 if low_mem else 20, warmup=5 if low_mem else 10)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 0),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselinePinnedPrefetchMLPBenchmark()
