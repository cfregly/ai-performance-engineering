"""labs.moe_cuda/optimized_kv_transfer_direct_graphs.py - Direct KV writes with graph replay.

This variant models the serving path where compute can write directly into the
KV destination layout and the steady-state iteration can be replayed from a
CUDA Graph. It removes both the intermediate workspace copy and Python launch
scheduling from the hot path.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark
from core.profiling.nvtx_helper import nvtx_range
from labs.moe_cuda.optimized_kv_transfer_direct import DirectKVTransferBenchmark


class DirectGraphedKVTransferBenchmark(DirectKVTransferBenchmark):
    """Direct KV destination writes captured as a steady-state CUDA graph."""

    def __init__(self) -> None:
        super().__init__()
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def setup(self) -> None:
        super().setup()
        self._maybe_capture_graph()

    def _maybe_capture_graph(self) -> None:
        if self.input_chunks is None or self.weight is None or self.kv_dest is None:
            raise RuntimeError("Buffers not initialized")

        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(self.device)
        with torch.inference_mode():
            with torch.cuda.graph(self.graph):
                torch.matmul(self.input_chunks, self.weight, out=self.kv_dest)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.graph is None:
            raise RuntimeError("CUDA graph not captured (setup() must run)")

        with nvtx_range("moe_cuda_kv_direct_destination_graphed", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.graph.replay()
        self.output = self._output_view
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def teardown(self) -> None:
        self.graph = None
        super().teardown()


def get_benchmark() -> BaseBenchmark:
    return DirectGraphedKVTransferBenchmark()
