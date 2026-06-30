"""Pinned-prefetch MLP optimization: pinned-memory prefetch + overlap.

This benchmark demonstrates efficient data loading - using pinned memory
and prefetching with double-buffering. Uses the same model architecture
as baseline for fair verification comparison.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.optimization.allocator_tuning import log_allocator_guidance
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class Prefetcher:
    """Double-buffered prefetcher from pinned host memory to device."""

    def __init__(self, device: torch.device, host_batches: List[torch.Tensor], targets: List[torch.Tensor]):
        self.device = device
        self.host_batches = host_batches
        self.targets = targets
        self.batch_count = len(host_batches)
        self.copy_stream = torch.cuda.Stream()
        self.buffers = [
            torch.empty_like(host_batches[0], device=device, dtype=host_batches[0].dtype),
            torch.empty_like(host_batches[0], device=device, dtype=host_batches[0].dtype),
        ]
        self.target_bufs = [
            torch.empty_like(targets[0], device=device, dtype=targets[0].dtype),
            torch.empty_like(targets[0], device=device, dtype=targets[0].dtype),
        ]
        self.cur_slot = 0
        self.next_slot = 1
        self.batch_idx = 0
        self._inflight = False
        self._prefetch()

    def _prefetch(self, wait_stream: Optional[torch.cuda.Stream] = None) -> None:
        host_idx = self.batch_idx % self.batch_count
        self.batch_idx += 1
        with torch.cuda.stream(self.copy_stream):
            if wait_stream is not None:
                self.copy_stream.wait_stream(wait_stream)
            self.buffers[self.next_slot].copy_(self.host_batches[host_idx], non_blocking=True)
            self.target_bufs[self.next_slot].copy_(self.targets[host_idx], non_blocking=True)
        self._inflight = True

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)
        if self._inflight:
            self.cur_slot, self.next_slot = self.next_slot, self.cur_slot
            self._prefetch(wait_stream=current_stream)
        return self.buffers[self.cur_slot], self.target_bufs[self.cur_slot]


class OptimizedPinnedPrefetchMLPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Pinned-memory prefetch with the same MLP workload as the baseline."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.host_batches: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.prefetcher: Optional[Prefetcher] = None
        self.output: Optional[torch.Tensor] = None
        self._payload_inputs: Optional[torch.Tensor] = None
        self._payload_targets: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        # Training benchmarks don't support jitter check - outputs change due to weight updates
        # Larger transfers to make H2D optimization measurable on high-bandwidth GPUs
        # The prefetcher benefit is proportional to (H2D time / compute time)
        from core.benchmark.smoke import is_smoke_mode
        low_mem = is_smoke_mode()
        self.input_dim = 2048 if low_mem else 4096
        self.hidden_dim = 2048 if low_mem else 4096  
        self.output_dim = 1024 if low_mem else 2048
        self.batch_size = 512 if low_mem else 1024  # Large batch = significant H2D
        self.num_batches = 4 if low_mem else 8
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(self.batch_size * self.input_dim * 4),  # float32
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        log_allocator_guidance("ch03/optimized_pinned_prefetch_mlp", optimized=True)
        # Use same model architecture as baseline for fair comparison
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),  # Same activation as baseline
            nn.Linear(self.hidden_dim, self.output_dim),
        ).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())

        # Use pinned memory for efficient async H2D transfer (the optimization)
        for _ in range(self.num_batches):
            self.host_batches.append(torch.randn(self.batch_size, self.input_dim, dtype=torch.float32, pin_memory=True))
            self.targets.append(torch.randn(self.batch_size, self.output_dim, dtype=torch.float32, pin_memory=True))
        self.prefetcher = Prefetcher(self.device, self.host_batches, self.targets)
        
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        assert (
            self.model is not None
            and self.optimizer is not None
            and self.prefetcher is not None
        )

        inputs, targets = self.prefetcher.next()
        with self._nvtx_range("optimized_pinned_prefetch_mlp"):
            out = self.model(inputs)
            loss = torch.nn.functional.mse_loss(out, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        # Store output for verification
        self.output = out.detach_()
        self._payload_inputs = inputs
        self._payload_targets = targets

    def capture_verification_payload(self) -> None:
        if self._payload_inputs is None or self._payload_targets is None:
            raise RuntimeError("benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"data": self._payload_inputs, "target": self._payload_targets},
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
        self.prefetcher = None
        self.output = None
        self._payload_inputs = None
        self._payload_targets = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        from core.benchmark.smoke import is_smoke_mode
        low_mem = is_smoke_mode()
        return BenchmarkConfig(iterations=5 if low_mem else 20, warmup=5 if low_mem else 10)

    def get_custom_streams(self) -> list["torch.cuda.Stream"]:
        if self.prefetcher is None:
            return []
        return [self.prefetcher.copy_stream]

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 0),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.prefetcher is None:
            return "Prefetcher not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPinnedPrefetchMLPBenchmark()
