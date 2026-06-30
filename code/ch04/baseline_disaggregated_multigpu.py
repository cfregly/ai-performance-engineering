"""baseline_disaggregated_multigpu.py - Baseline monolithic inference in multi-GPU context."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.distributed as dist

from core.utils.compile_utils import compile_model
from core.benchmark.gpu_requirements import skip_if_insufficient_gpus
from core.common.device_utils import resolve_local_rank

from typing import Optional

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class BaselineDisaggregatedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Monolithic inference (prefill and decode share resources across GPUs).
    
    Disaggregated inference: This baseline does not separate prefill and decode phases.
    Both phases compete for same GPU resources, causing interference and poor utilization.
    """
    multi_gpu_required = True
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.prefill_input = None
        self.decode_input = None
        self.output = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.batch_size = 2
        self.prefill_len = 512
        self.hidden_dim = 256
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        tokens = self.batch_size * (self.prefill_len + 1)  # include decode token
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model and inputs."""
        skip_if_insufficient_gpus()

        # Only initialize distributed when launched under torchrun.
        if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            local_rank = resolve_local_rank()
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    device_id=local_rank,
                )
            self.is_distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Baseline: Monolithic inference - prefill and decode share same resources
        # Disaggregated inference separates prefill (parallel) and decode (autoregressive)
        # This baseline does not separate prefill and decode phases
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )
        self.model = self.model.to(self.device).eval()
        
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters()) * 2
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Simulate prefill (long context) and decode (single token) inputs
        self.prefill_input = torch.randn(self.batch_size, self.prefill_len, self.hidden_dim, device=self.device)
        self.decode_input = torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Monolithic inference."""
        with nvtx_range("baseline_disaggregated_multigpu", enable=self._enable_nvtx):
            with torch.inference_mode():
                # Baseline: Monolithic inference
                # Prefill and decode phases share same resources across GPUs
                # This causes interference - prefill blocks decode and vice versa
                # Disaggregated inference separates these phases for better efficiency
                
                # Process prefill (long context) - competes with decode for resources
                prefill_output = self.model(self.prefill_input)
                
                # Synchronize across GPUs
                if self.is_distributed:
                    dist.all_reduce(prefill_output, op=dist.ReduceOp.SUM)
                    prefill_output = prefill_output / self.world_size
                
                # Process decode (autoregressive) - competes with prefill for resources
                decode_output = self.model(self.decode_input)
                
                # Synchronize across GPUs
                if self.is_distributed:
                    dist.all_reduce(decode_output, op=dist.ReduceOp.SUM)
                    decode_output = decode_output / self.world_size
                self.output = decode_output
                
            # Baseline: No separation - both phases interfere with each other
                # This leads to poor GPU utilization and latency spikes

    def capture_verification_payload(self) -> None:
        if self.prefill_input is None or self.decode_input is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"prefill": self.prefill_input, "decode": self.decode_input},
            output=self.output.to(dtype=torch.float32),
            batch_size=int(self.batch_size),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.prefill_input = None
        self.decode_input = None
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            multi_gpu_required=True,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.prefill_input is None or self.decode_input is None:
            return "Inputs not initialized"
        return None
    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (1e-5, 1e-5)


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineDisaggregatedBenchmark()
