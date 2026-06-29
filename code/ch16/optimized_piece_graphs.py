"""Optimized piece-graph benchmark: cached regional CUDA graphs for steady-state replay.

Splits the model into two pieces (head/tail) and captures a CUDA graph per piece.
The baseline captures the full model monolithically.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from ch16.piece_graphs_model import RegionalPieceGraph

resolve_device = partial(
    require_cuda_device,
    "CUDA-capable GPU required for piece-graph benchmarks.",
)


RegionGraph = Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]


class OptimizedPieceGraphsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Caches two smaller CUDA graphs per sequence bucket (piece graph strategy)."""

    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model: Optional[RegionalPieceGraph] = None
        self.seq_len = 256
        self.hidden_dim_val = 768
        self.n_layers_val = 12
        self.num_heads = 8
        self.graph_cache: Dict[int, Tuple[RegionGraph, RegionGraph]] = {}
        self.parameter_count: int = 0
        self._verify_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._payload_verify_tokens: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._enable_nvtx = False

        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.seq_len),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.seq_len),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.model = RegionalPieceGraph(
            hidden_dim=self.hidden_dim_val,
            n_layers=self.n_layers_val,
            num_heads=self.num_heads,
        ).to(
            self.device, dtype=torch.float16
        ).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self._verify_input = torch.randn(
            1,
            self.seq_len,
            self.hidden_dim_val,
            device=self.device,
            dtype=torch.float16,
        )
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        self._capture_piece_graphs()

    def _capture_piece_graphs(self) -> None:
        assert self.model is not None
        if self._verify_input is None:
            raise RuntimeError("Verification input must be initialized before CUDA graph capture")
        self.graph_cache.clear()
        torch.cuda.synchronize(self.device)

        seq_len = self.seq_len
        head_input = torch.empty(
            1, seq_len, self.model.hidden_dim, device=self.device, dtype=torch.float16
        )
        head_output = torch.empty_like(head_input)
        tail_input = torch.empty_like(head_input)
        tail_output = torch.empty_like(head_input)

        head_input.copy_(self._verify_input)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            head_output.copy_(self.model.region_head(self.model.input_proj(head_input)))
        torch.cuda.synchronize(self.device)

        head_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(head_graph):
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                head_output.copy_(self.model.region_head(self.model.input_proj(head_input)))

        tail_input.copy_(head_output)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            tail_output.copy_(self.model.output_proj(self.model.region_tail(tail_input)))
        torch.cuda.synchronize(self.device)

        tail_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(tail_graph):
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                tail_output.copy_(self.model.output_proj(self.model.region_tail(tail_input)))

        self.graph_cache[seq_len] = (
            (head_graph, head_input, head_output),
            (tail_graph, tail_input, tail_output),
        )
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")

        seq_len = self.seq_len
        (head_graph, head_input, head_output), (tail_graph, tail_input, tail_output) = self.graph_cache[seq_len]

        head_input.copy_(self._verify_input)
        with nvtx_range("piece_graph_head", enable=self._enable_nvtx):
            head_graph.replay()
        tail_input.copy_(head_output)
        with nvtx_range("piece_graph_tail", enable=self._enable_nvtx):
            tail_graph.replay()
        self.output = tail_output
        self._payload_verify_tokens = self._verify_input

    def capture_verification_payload(self) -> None:
        verify_tokens = self._payload_verify_tokens
        if verify_tokens is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": verify_tokens},
            output=self._verify_output_buffer,
            batch_size=verify_tokens.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self._verify_input = None
        self._payload_verify_tokens = None
        self._verify_output_buffer = None
        self.graph_cache.clear()
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            measurement_timeout_seconds=240,
            setup_timeout_seconds=120,
            use_subprocess=True,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

def get_benchmark() -> BaseBenchmark:
    return OptimizedPieceGraphsBenchmark()
