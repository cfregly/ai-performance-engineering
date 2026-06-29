"""baseline_piece_graphs.py - Monolithic CUDA graph capture baseline for piece-graphs.

Captures the full model in a single CUDA graph for steady-state replay.
The optimized variant captures smaller "piece graphs" (head/tail regions) and
replays them separately.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
from ch16.piece_graphs_model import RegionalPieceGraph  # noqa: E402


GraphCacheEntry = Tuple["torch.cuda.CUDAGraph", torch.Tensor, torch.Tensor]


class BaselinePieceGraphsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Monolithic graph capture baseline for piece-graphs."""

    def __init__(self):
        super().__init__()
        self.model: Optional[RegionalPieceGraph] = None
        self.seq_len = 256
        self.hidden_dim = 768
        self.n_layers = 12
        self.num_heads = 8

        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0

        self.graph_cache: dict[int, GraphCacheEntry] = {}
        self._seq_len_used: Optional[int] = None
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
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            num_heads=self.num_heads,
        ).to(self.device, dtype=torch.float16).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        self._verify_input = torch.randn(
            1,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)

        self.graph_cache.clear()
        static_input = torch.empty_like(self._verify_input)
        static_output = torch.empty_like(self._verify_input)
        static_input.copy_(self._verify_input)

        torch.cuda.synchronize(self.device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            static_output.copy_(self.model(static_input))
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                static_output.copy_(self.model(static_input))

        torch.cuda.synchronize(self.device)
        self.graph_cache[self.seq_len] = (graph, static_input, static_output)

    def benchmark_fn(self) -> None:
        with nvtx_range("baseline_piece_graphs", enable=self._enable_nvtx):
            if self.model is None or self._verify_input is None:
                raise RuntimeError("Model/inputs not initialized")
            entry = self.graph_cache.get(self.seq_len)
            if entry is None:
                raise RuntimeError("Missing CUDA graph for configured sequence length")
            graph, static_input, static_output = entry
            static_input.copy_(self._verify_input)
            graph.replay()
            self.output = static_output
            self._seq_len_used = self.seq_len

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
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
        self._verify_output_buffer = None
        self.graph_cache.clear()
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            measurement_timeout_seconds=240,
            setup_timeout_seconds=120,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

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

    def validate_result(self) -> Optional[str]:
        if self.model is None or self._verify_input is None:
            return "Model/inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselinePieceGraphsBenchmark()
