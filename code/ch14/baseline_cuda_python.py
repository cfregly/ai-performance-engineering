"""baseline_cuda_python.py - Baseline PyTorch operations without native CUDA Python.

This demonstrates standard PyTorch eager mode operations that could benefit
from native CUDA Python kernels. The baseline uses PyTorch's built-in
operators which go through the dispatcher and may not be optimal for
custom fusion patterns.

Native CUDA Python (cuda.core, cuda.bindings) provides:
- Direct kernel authoring in Python without Triton
- Lower-level control than Triton with higher-level than raw CUDA C++
- JIT compilation with nvrtc
- First-class CUDA support announced in PyTorch 2.10+
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineCudaPythonBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Standard PyTorch operations (no native CUDA Python).
    
    Demonstrates common operations that are candidates for CUDA Python:
    
    1. **Fused LayerNorm + Activation**: Separate ops with intermediate storage
    2. **Custom reduction patterns**: Using PyTorch primitives
    3. **Elementwise with masking**: Standard masked operations
    
    These patterns have overhead from:
    - Multiple kernel launches
    - Intermediate tensor allocations  
    - Dispatcher overhead per operation
    
    Native CUDA Python can fuse these into single kernels.
    """
    
    def __init__(self):
        super().__init__()
        self.batch = 64
        self.seq_len = 2048
        self.hidden = 4096
        self.input = None
        self.weight = None
        self.bias = None
        self.output = None
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._flops = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verification_payload = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors for operations."""
        torch.manual_seed(42)
        
        # Input tensor (batch, seq, hidden)
        self.input = torch.randn(
            self.batch, self.seq_len, self.hidden, 
            device=self.device, dtype=torch.float16
        )
        
        # LayerNorm parameters
        self.weight = torch.ones(self.hidden, device=self.device, dtype=torch.float16)
        self.bias = torch.zeros(self.hidden, device=self.device, dtype=torch.float16)
        
        # Mask for sparse operations
        self.mask = torch.rand(self.batch, self.seq_len, device=self.device) > 0.3
        
        # Output buffer
        self.output = None
        
        # Calculate FLOPs for LayerNorm + GELU
        # LayerNorm: 5n (mean, var, normalize, scale, shift)
        # GELU: ~10n (approximation with tanh)
        n_elements = self.batch * self.seq_len * self.hidden
        self._flops = 15 * n_elements
        
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard PyTorch eager operations.
        
        Pattern: LayerNorm → GELU → Masked selection
        
        Each operation is a separate kernel launch with dispatcher overhead.
        Intermediate tensors are allocated for each step.
        """
        with torch.inference_mode(), self._nvtx_range("baseline_cuda_python"):
            # Step 1: Layer normalization (separate kernel)
            normalized = F.layer_norm(
                self.input, 
                normalized_shape=(self.hidden,),
                weight=self.weight,
                bias=self.bias,
            )
            
            # Step 2: GELU activation (separate kernel, intermediate tensor)
            activated = F.gelu(normalized)
            
            # Step 3: Masked fill (separate kernel)
            # Zero out positions where mask is False
            activated.masked_fill_(~self.mask.unsqueeze(-1), 0.0)
            
            # Step 4: Residual connection (separate kernel)
            self.output = activated + self.input
        

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "input": self.input.detach(),
                "mask": self.mask.detach(),
                "weight": self.weight.detach(),
                "bias": self.bias.detach(),
            },
            output=self.output.detach(),
            batch_size=int(self.batch),
            parameter_count=0,
            output_tolerance=(1e-3, 1e-3),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.weight = None
        self.bias = None
        self.mask = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        total_elements = self.batch * self.seq_len * self.hidden
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=float(total_elements * 2),
            total_bytes=float(total_elements * 2 * 3),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None:
            return "Input not initialized"
        if self.output is None:
            return "Output not computed"
        if torch.isnan(self.output).any():
            return "Output contains NaN"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineCudaPythonBenchmark()
