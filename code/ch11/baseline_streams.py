"""baseline_streams.py - Sequential data transfer and compute (baseline).

Chapter 11: CUDA Streams and Concurrency

This baseline demonstrates sequential execution where each data chunk must:
1. Transfer from host to device (H2D)
2. Compute on the GPU
3. Synchronize before processing the next chunk

The sequential pattern creates bubbles where the GPU compute unit is idle
during memory transfers and vice versa.
"""

from __future__ import annotations

from typing import Optional, List

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import canonicalize_nvtx_name


class BaselineStreamsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Sequential execution - no overlap between H2D transfers and compute."""
    
    def __init__(self):
        super().__init__()
        self.host_data: Optional[List[torch.Tensor]] = None
        self.device_data: Optional[List[torch.Tensor]] = None
        self.results: Optional[List[torch.Tensor]] = None
        self._scratch0: Optional[torch.Tensor] = None
        self._scratch1: Optional[torch.Tensor] = None
        self._scratch_pair: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._chunk_triplets: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._verify_output: Optional[torch.Tensor] = None
        self._verify_indices: tuple[int, int, int] = ()
        self._verify_slice_len = 0
        self.N = 5_000_000  # Elements per chunk - balanced for H2D/compute overlap
        self.num_chunks = 20  # More chunks to amortize pipeline startup
        # Stream benchmark - fixed dimensions for overlap measurement
        processed = float(self.N * self.num_chunks)
        # Register at init time so pre-run compliance checks pass; setup() may update
        # this if the workload is adjusted before allocation.
        self.register_workload_metadata(
            tokens_per_iteration=processed,
            requests_per_iteration=float(self.num_chunks),
        )
    
    def setup(self) -> None:
        """Setup: Initialize pinned host memory and device buffers."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create pinned host memory for async transfers
        self.host_data = [
            torch.randn(self.N, dtype=torch.float32).pin_memory() 
            for _ in range(self.num_chunks)
        ]
        
        # Pre-allocate device buffers
        self.device_data = [
            torch.empty(self.N, dtype=torch.float32, device=self.device)
            for _ in range(self.num_chunks)
        ]
        self.results = [
            torch.empty(self.N, dtype=torch.float32, device=self.device)
            for _ in range(self.num_chunks)
        ]
        self._scratch0 = torch.empty(self.N, dtype=torch.float32, device=self.device)
        self._scratch1 = torch.empty(self.N, dtype=torch.float32, device=self.device)
        self._scratch_pair = (self._scratch0, self._scratch1)
        self._chunk_triplets = list(zip(self.host_data, self.device_data, self.results, strict=True))
        self._verify_slice_len = min(256, self.N)
        self._verify_indices = (0, self.num_chunks // 2, self.num_chunks - 1)
        self._verify_output = torch.empty(
            self._verify_slice_len * len(self._verify_indices),
            dtype=torch.float32,
            device=self.device,
        )
        
        self._synchronize()
        processed = float(self.N * self.num_chunks)
        self.register_workload_metadata(
            tokens_per_iteration=processed,
            requests_per_iteration=float(self.num_chunks),
        )
    
    def _compute(self, data: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Compute-intensive operation on GPU data.
        
        Multiple trig operations to ensure compute time is meaningful
        relative to H2D transfer time for proper overlap demonstration.
        """
        if self._scratch_pair is None:
            raise RuntimeError("setup() must initialize compute scratch buffers")
        scratch0, scratch1 = self._scratch_pair
        result = data
        for _ in range(3):  # Multiple passes to increase compute time
            torch.sin(result, out=scratch0)
            torch.cos(result, out=scratch1)
            scratch0.mul_(scratch1)
            scratch0.add_(result, alpha=0.1)
            torch.tanh(scratch0, out=out)
            torch.sigmoid(scratch0, out=scratch1)
            out.add_(scratch1, alpha=0.5)
            result = out
        return out
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential H2D transfer then compute for each chunk.
        
        Pattern: H2D -> Sync -> Compute -> Sync -> H2D -> Sync -> Compute -> ...
        
        This creates idle time because:
        - GPU compute units are idle during H2D transfers
        - Memory controller is idle during compute
        """
        if not self._chunk_triplets:
            raise RuntimeError("setup() must initialize chunk views")
        with torch.inference_mode(), self._nvtx_range("baseline_streams_sequential"):
            for host_chunk, device_chunk, result_chunk in self._chunk_triplets:
                # Transfer data from host to device (blocking)
                device_chunk.copy_(host_chunk)
                
                # Compute on device
                self._compute(device_chunk, result_chunk)
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        self.results = None
        self._scratch0 = None
        self._scratch1 = None
        self._scratch_pair = None
        self._chunk_triplets = []
        self._verify_output = None
        self._verify_indices = ()
        self._verify_slice_len = 0
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[canonicalize_nvtx_name("baseline_streams_sequential")],
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        from core.benchmark.metrics import compute_stream_metrics
        return compute_stream_metrics(
            sequential_time_ms=getattr(self, '_last_elapsed_ms', None),
            overlapped_time_ms=None,
            num_streams=1,  # Sequential uses default stream only
            num_operations=self.num_chunks * 2,  # transfer + compute per chunk
        )

    def capture_verification_payload(self) -> None:
        if self.host_data is None or self.results is None or self._verify_output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        slice_len = self._verify_slice_len
        sample = self.host_data[0][:slice_len]
        with torch.no_grad():
            for slot, result_idx in enumerate(self._verify_indices):
                start = slot * slice_len
                self._verify_output[start : start + slice_len].copy_(self.results[result_idx][:slice_len])
        self._set_verification_payload(
            inputs={"host_data": sample},
            output=self._verify_output,
            batch_size=int(self.num_chunks),
            parameter_count=int(self.N * self.num_chunks),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
        )


def get_benchmark() -> BaselineStreamsBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineStreamsBenchmark()
