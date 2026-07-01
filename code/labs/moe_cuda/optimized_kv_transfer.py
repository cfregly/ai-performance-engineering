"""labs.moe_cuda/optimized_kv_transfer.py - Overlapped KV transfers."""

from __future__ import annotations

from typing import List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class OptimizedKVTransferBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Prefill compute + KV transfer with CUDA-stream pipelining."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024  # Must match baseline for valid comparison
        self.chunk_size = 256
        self.num_chunks = 32
        self.pipeline_depth = 2
        self.dtype = torch.float16
        self.input_chunks: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None
        self.workspace: Optional[torch.Tensor] = None
        self.kv_dest: Optional[torch.Tensor] = None
        self.compute_stream = torch.cuda.Stream()
        self.copy_stream = torch.cuda.Stream()
        # Use per-chunk events to avoid unsafe reuse that can mis-order waits.
        self.compute_done_events: List[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=False, blocking=False)
            for _ in range(self.num_chunks)
        ]
        self._compute_chunk_specs: List[tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]] = []
        self._copy_chunk_specs: List[tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]] = []
        self._chunk_spec_counts: tuple[int, int] = (0, 0)
        self._expected_chunk_spec_counts: tuple[int, int] = (0, 0)
        tokens = self.num_chunks * self.chunk_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )
        self._payload_meta: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_view: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._enable_nvtx = False

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda KV transfer requires CUDA")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.input_chunks = torch.randn(
            self.num_chunks,
            self.chunk_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.weight = torch.randn(
            self.hidden_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.workspace = torch.empty_like(self.input_chunks)
        self.kv_dest = torch.empty_like(self.input_chunks)
        input_views = self.input_chunks.unbind(0)
        workspace_views = self.workspace.unbind(0)
        dest_views = self.kv_dest.unbind(0)
        self._compute_chunk_specs = list(zip(input_views, workspace_views, self.compute_done_events, strict=True))
        self._copy_chunk_specs = list(zip(workspace_views, dest_views, self.compute_done_events, strict=True))
        self._chunk_spec_counts = (
            len(self._compute_chunk_specs),
            len(self._copy_chunk_specs),
        )
        self._expected_chunk_spec_counts = (self.num_chunks, self.num_chunks)
        self._output_view = self.kv_dest[0, :1, : min(8, self.hidden_size)]
        self._verify_output_buffer = torch.empty_like(self._output_view, dtype=torch.float32)
        self._payload_meta = torch.tensor([self.hidden_size], dtype=torch.int64, device="cpu")
        
        # Warmup the streams so the steady-state path doesn't include one-time setup overhead.
        with torch.cuda.stream(self.compute_stream):
            first_input, first_workspace, _ = self._compute_chunk_specs[0]
            torch.matmul(first_input, self.weight, out=first_workspace)
        with torch.cuda.stream(self.copy_stream):
            first_workspace, first_dest, _ = self._copy_chunk_specs[0]
            first_dest.copy_(first_workspace)
        self._synchronize()

    def _launch_compute(
        self,
        chunk: torch.Tensor,
        workspace_chunk: torch.Tensor,
        event: torch.cuda.Event,
    ) -> None:
        assert self.weight is not None
        torch.matmul(chunk, self.weight, out=workspace_chunk)
        event.record(self.compute_stream)

    def _launch_copy(
        self,
        workspace_chunk: torch.Tensor,
        dest_chunk: torch.Tensor,
        event: torch.cuda.Event,
    ) -> None:
        self.copy_stream.wait_event(event)
        dest_chunk.copy_(workspace_chunk)

    def benchmark_fn(self) -> None:
        if (
            self.input_chunks is None
            or self.weight is None
            or self.workspace is None
            or self.kv_dest is None
        ):
            raise RuntimeError("Buffers not initialized")
        if self._chunk_spec_counts != self._expected_chunk_spec_counts:
            raise RuntimeError("Chunk views not initialized")

        with nvtx_range("moe_cuda_kv_overlap", enable=self._enable_nvtx):
            with torch.inference_mode():
                # Reduce Python overhead by issuing all compute on one stream context
                # and all dependent copies on a second stream context.
                with torch.cuda.stream(self.compute_stream):
                    for chunk, workspace_chunk, compute_event in self._compute_chunk_specs:
                        self._launch_compute(chunk, workspace_chunk, compute_event)
                with torch.cuda.stream(self.copy_stream):
                    for workspace_chunk, dest_chunk, compute_event in self._copy_chunk_specs:
                        self._launch_copy(workspace_chunk, dest_chunk, compute_event)
                torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
        if self.kv_dest is None:
            raise RuntimeError("KV destination missing")
        self.output = self._output_view
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        meta = self._payload_meta
        if meta is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"meta": meta},
            output=self._verify_output_buffer,
            batch_size=1,
            parameter_count=0,
            precision_flags={},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.input_chunks = None
        self.weight = None
        self.workspace = None
        self.kv_dest = None
        self._compute_chunk_specs = []
        self._copy_chunk_specs = []
        self._chunk_spec_counts = (0, 0)
        self._expected_chunk_spec_counts = (0, 0)
        self.output = None
        self._output_view = None
        self._verify_output_buffer = None
        self._payload_meta = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=10)  # Match baseline steady-state warmup

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def validate_result(self) -> Optional[str]:
        if any(t is None for t in (self.input_chunks, self.weight, self.workspace, self.kv_dest)):
            return "Buffers not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    return OptimizedKVTransferBenchmark()
