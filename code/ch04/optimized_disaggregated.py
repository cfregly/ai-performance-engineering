"""optimized_disaggregated.py - Optimized disaggregated inference (single GPU)."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.distributed as dist
import copy

from typing import Optional

from ch04.reduction_common import ReusableReductionMlp
from core.common.device_utils import resolve_local_rank
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin

class OptimizedDisaggregatedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Disaggregated inference using separate model copies per phase."""
    
    def __init__(self):
        super().__init__()
        self.prefill_model = None
        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self.output = None
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.batch_size = 2
        self.prefill_len = 512
        self.hidden_dim = 256
        self._payload_parameter_count = 0
        tokens = self.batch_size * (self.prefill_len + 1)  # include decode token
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize separate models for prefill and decode."""
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
        
        # Optimization: Disaggregated inference
        # Separate models/resources for prefill and decode phases
        # Prefill: Parallel processing, compute-intensive, can use multiple GPUs
        # Decode: Autoregressive, latency-sensitive, dedicated GPU resources
        
        # Prefill model (optimized for parallel processing)
        base_model = ReusableReductionMlp(self.hidden_dim, self.hidden_dim * 2).to(self.device).eval()
        self.prefill_model = base_model
        # Decode model uses identical weights; disaggregation changes placement/scheduling, not math.
        self.decode_model = copy.deepcopy(base_model)
        model_param_count = sum(p.numel() for p in base_model.parameters())
        self._payload_parameter_count = model_param_count * 2
        
        if self.is_distributed:
            # In disaggregated setup, prefill and decode can use different GPU groups.
            self.prefill_model = nn.parallel.DistributedDataParallel(self.prefill_model)
            self.decode_model = nn.parallel.DistributedDataParallel(self.decode_model)
        
        # Simulate prefill (long context) and decode (single token) inputs
        self.prefill_input = torch.randn(self.batch_size, self.prefill_len, self.hidden_dim, device=self.device)
        self.decode_input = torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Disaggregated inference."""
        assert self.prefill_model is not None and self.decode_model is not None
        assert self.prefill_input is not None and self.decode_input is not None
        with self._nvtx_range("optimized_disaggregated"):
            with torch.inference_mode():
                # Process prefill on dedicated prefill GPUs (parallel, compute-intensive)
                prefill_output = self.prefill_model(self.prefill_input)
                
                # Synchronize prefill across GPUs
                if self.is_distributed:
                    dist.all_reduce(prefill_output, op=dist.ReduceOp.SUM)
                    prefill_output.div_(self.world_size)
                
                # Process decode on dedicated decode GPUs (autoregressive, latency-sensitive)
                decode_output = self.decode_model(self.decode_input)
                
                # Synchronize decode across GPUs
                if self.is_distributed:
                    dist.all_reduce(decode_output, op=dist.ReduceOp.SUM)
                    decode_output.div_(self.world_size)
                self.output = decode_output

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
        self.prefill_model = None
        self.decode_model = None
        self.prefill_input = None
        self.decode_input = None
        self.output = None
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
        iterations=10,
            warmup=5,
        )
    
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
        if self.prefill_model is None or self.decode_model is None:
            return "Models not initialized"
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
    return OptimizedDisaggregatedBenchmark()
