"""optimized_streams.py - Pipelined H2D transfers overlapping with compute (optimized).

Chapter 11: CUDA Streams and Concurrency

This optimized version demonstrates stream-based pipelining where:
1. H2D transfers happen on one stream
2. Compute happens on another stream
3. The streams overlap - while computing on chunk N, transfer chunk N+1

This eliminates the idle time from sequential execution by keeping both
the memory controller and compute units busy simultaneously.

Key optimization technique: Double-buffered pipelining with CUDA streams
"""

from __future__ import annotations

from typing import Optional, List

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import canonicalize_nvtx_name


class OptimizedStreamsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Pipelined execution - overlap H2D transfers with compute using streams.
    
    Stream Pipeline Architecture:
    
    Time ->
    Stream H2D:    [Transfer 0]  [Transfer 1]  [Transfer 2]  ...
    Stream Compute:              [Compute 0]   [Compute 1]   [Compute 2] ...
    
    Each compute waits for its transfer, but transfers and computes overlap.
    """
    
    def __init__(self):
        super().__init__()
        self.host_data: Optional[List[torch.Tensor]] = None
        self.device_data: Optional[List[torch.Tensor]] = None
        self.results: Optional[List[torch.Tensor]] = None
        self.stream_h2d: Optional[torch.cuda.Stream] = None
        self.stream_compute: Optional[torch.cuda.Stream] = None
        self._scratch0: Optional[torch.Tensor] = None
        self._scratch1: Optional[torch.Tensor] = None
        self._scratch_pair: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._chunk_triplets: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._pipeline_steps: List[
            tuple[
                torch.Tensor,
                torch.Tensor,
                Optional[tuple[torch.Tensor, torch.Tensor]],
            ]
        ] = []
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
        """Setup: Initialize streams, pinned memory, and device buffers."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create streams for pipelining
        self.stream_h2d = torch.cuda.Stream()
        self.stream_compute = torch.cuda.Stream()
        
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
        self._pipeline_steps = []
        for chunk_idx, (_host_chunk, device_chunk, result_chunk) in enumerate(self._chunk_triplets):
            next_transfer: Optional[tuple[torch.Tensor, torch.Tensor]] = None
            if chunk_idx + 1 < len(self._chunk_triplets):
                next_host, next_device, _next_result = self._chunk_triplets[chunk_idx + 1]
                next_transfer = (next_host, next_device)
            self._pipeline_steps.append((device_chunk, result_chunk, next_transfer))
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
        """Benchmark: Pipelined H2D transfer overlapping with compute.
        
        Key insight: While GPU is computing on chunk i, we transfer chunk i+1.
        This keeps both memory controller and compute units busy.
        
        Pipeline stages:
        1. Start first H2D transfer
        2. For each chunk:
           - Wait for its transfer to complete (on compute stream)
           - Start next transfer (if any)
           - Compute on current chunk
        3. Synchronize all streams
        """
        if not self._chunk_triplets or not self._pipeline_steps:
            raise RuntimeError("setup() must initialize chunk views")
        chunks = self._chunk_triplets
        pipeline_steps = self._pipeline_steps
        with torch.inference_mode(), self._nvtx_range("streams_pipelined"):
            # Stage 0: Kick off first transfer
            first_host, first_device, _ = chunks[0]
            with torch.cuda.stream(self.stream_h2d):
                first_device.copy_(first_host, non_blocking=True)
            
            for device_chunk, result_chunk, next_transfer in pipeline_steps:
                # Ensure this chunk's transfer is complete before computing
                self.stream_compute.wait_stream(self.stream_h2d)
                
                # Start next transfer while we compute (double buffering)
                if next_transfer is not None:
                    next_host, next_device = next_transfer
                    with torch.cuda.stream(self.stream_h2d):
                        next_device.copy_(next_host, non_blocking=True)
                
                # Compute on current chunk
                with torch.cuda.stream(self.stream_compute):
                    self._compute(device_chunk, result_chunk)
            
            current = torch.cuda.current_stream(device=self.device)
            current.wait_stream(self.stream_compute)
            current.wait_stream(self.stream_h2d)
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        self.results = None
        self.stream_h2d = None
        self.stream_compute = None
        self._scratch0 = None
        self._scratch1 = None
        self._scratch_pair = None
        self._chunk_triplets = []
        self._pipeline_steps = []
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
            nsys_nvtx_include=[canonicalize_nvtx_name("streams_pipelined")],
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        from core.benchmark.metrics import compute_stream_metrics
        return compute_stream_metrics(
            sequential_time_ms=None,
            overlapped_time_ms=getattr(self, '_last_elapsed_ms', None),
            num_streams=2,  # H2D stream + compute stream
            num_operations=self.num_chunks * 2,  # transfer + compute per chunk
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.results is None:
            return "Results not initialized"
        for i, r in enumerate(self.results):
            if not torch.isfinite(r).all():
                return f"Result {i} contains non-finite values"
        return None

    declare_all_streams = False

    def get_custom_streams(self):
        if self.stream_h2d is None or self.stream_compute is None:
            return None
        return [self.stream_h2d, self.stream_compute]

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


def get_benchmark() -> OptimizedStreamsBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedStreamsBenchmark()
