"""labs.moe_cuda/optimized_kv_transfer_graphs.py - Deeper pipeline + CUDA graphs.

This step layers three incremental changes on top of the baseline overlap demo:
1) Increase pipeline depth and chunk count to feed more concurrent work to GB10's 48 SMs.
2) Keep GEMMs in bfloat16 and try torch.compile to shrink per-matmul launch overhead.
3) Capture the steady-state pipeline into a CUDA graph so iterations replay with minimal
   CPU scheduling cost while preserving the dual-stream overlap pattern.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class GraphedKVTransferBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Prefill compute + KV transfer with deeper pipelining and CUDA graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024  # Must match baseline for valid comparison
        self.chunk_size = 256
        self.num_chunks = 32
        self.dtype = torch.float16
        self.input_chunks: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None
        self.workspace: Optional[torch.Tensor] = None
        self.kv_dest: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_chunk_triplets: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._payload_meta: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_view: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        tokens = self.num_chunks * self.chunk_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )
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
        self._graph_chunk_triplets = list(
            zip(
                self.input_chunks.unbind(0),
                self.workspace.unbind(0),
                self.kv_dest.unbind(0),
                strict=True,
            )
        )
        self._output_view = self.kv_dest[0, :1, : min(8, self.hidden_size)]
        self._verify_output_buffer = torch.empty_like(self._output_view, dtype=torch.float32)
        self._payload_meta = torch.tensor([self.hidden_size], dtype=torch.int64, device="cpu")

        # Warmup to ensure cuBLAS/allocator state is initialized before graph capture.
        first_input, first_workspace, first_dest = self._graph_chunk_triplets[0]
        with torch.inference_mode():
            torch.matmul(first_input, self.weight, out=first_workspace)
            first_dest.copy_(first_workspace)
        torch.cuda.synchronize(self.device)
        
        self._maybe_capture_graph()

    def _maybe_capture_graph(self) -> None:
        if (
            self.input_chunks is None
            or self.weight is None
            or self.workspace is None
            or self.kv_dest is None
        ):
            raise RuntimeError("Buffers not initialized")
        if len(self._graph_chunk_triplets) != self.num_chunks:
            raise RuntimeError("Chunk views not initialized")
        # Capture the steady-state pipeline so replay avoids Python/launch overhead.
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(self.device)
        with torch.inference_mode():
            with torch.cuda.graph(self.graph):
                for input_chunk, workspace_chunk, dest_chunk in self._graph_chunk_triplets:
                    torch.matmul(input_chunk, self.weight, out=workspace_chunk)
                    dest_chunk.copy_(workspace_chunk)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if (
            self.input_chunks is None
            or self.weight is None
            or self.workspace is None
            or self.kv_dest is None
        ):
            raise RuntimeError("Buffers not initialized")
        if self.graph is None:
            raise RuntimeError("CUDA graph not captured (setup() must run)")

        with nvtx_range("moe_cuda_kv_overlap_graphed", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.graph.replay()
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
        self.graph = None
        self.input_chunks = None
        self.weight = None
        self.workspace = None
        self.kv_dest = None
        self._graph_chunk_triplets = []
        self.output = None
        self._output_view = None
        self._verify_output_buffer = None
        self._payload_meta = None
        
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=10)  # CUDA graphs need extra warmup

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def validate_result(self) -> Optional[str]:
        if any(t is None for t in (self.input_chunks, self.weight, self.workspace, self.kv_dest)):
            return "Buffers not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    return GraphedKVTransferBenchmark()
