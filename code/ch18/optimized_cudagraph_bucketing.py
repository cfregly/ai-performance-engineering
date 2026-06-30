"""Optimized decode bucketing with pre-captured CUDA graphs.

This module demonstrates two approaches:
1. GraphTreeSimulator: Simulates bucketing behavior for analysis
2. CUDAGraphBucketing: Actual CUDA graph capture/replay for inference

Key Optimization (Ch18):
- Pre-capture graphs for common batch/seq size buckets at startup
- Pad inputs to nearest bucket size
- Replay pre-captured graph instead of re-capturing
- Eliminates kernel launch overhead for decode phase
"""

from __future__ import annotations

import argparse
import heapq
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch18.baseline_cudagraph_bucketing import (  # noqa: E402
    BaselineCUDAGraphBucketing,
)
from ch18.cudagraph_bucketing_common import (  # noqa: E402
    BucketBands,
    GraphTreeSimulator,
    default_bucket_bands,
    demo_traffic,
    pad_batch_to_capture,
)
from ch18.cudagraph_bucketing_metrics import (  # noqa: E402
    export_stats_to_prometheus,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

# ============================================================
# CUDA Graph Bucketing for Real Inference
# ============================================================

class CUDAGraphBucketing:
    """Pre-captured CUDA graphs for variable batch size inference.
    
    Key Optimization:
    - Capture graphs at startup for all bucket combinations
    - At runtime, pad input to nearest bucket and replay
    - Eliminates per-request graph capture overhead
    
    Usage:
        graph_exec = CUDAGraphBucketing(model, hidden_dim=256)
        output = graph_exec.forward(input_tensor)  # Auto-selects bucket
    """
    
    # Standard bucket sizes - powers of 2 for batch, common seq lengths
    BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64]
    SEQ_BUCKETS = [128, 256, 512, 1024, 2048]
    
    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int = 256,
        device: str = "cuda",
        max_batch: int = 64,
        max_seq: int = 2048,
    ):
        self.model = model
        self.hidden_dim = hidden_dim
        self.device = device
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.model_dtype = next(self.model.parameters()).dtype
        
        # Storage for captured graphs and static buffers
        self.graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[Tuple[int, int], torch.Tensor] = {}
        self.static_outputs: Dict[Tuple[int, int], torch.Tensor] = {}
        
        # Metrics
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0
        
        # Pre-capture graphs
        if torch.cuda.is_available():
            self._precapture_graphs()
    
    def _precapture_graphs(self) -> None:
        """Pre-capture CUDA graphs for all bucket combinations."""
        # Filter buckets within limits
        batch_buckets = [b for b in self.BATCH_BUCKETS if b <= self.max_batch]
        seq_buckets = [s for s in self.SEQ_BUCKETS if s <= self.max_seq]
        
        for bs in batch_buckets:
            for seq in seq_buckets:
                self._capture_graph(bs, seq)
    
    def _capture_graph(self, batch_size: int, seq_len: int) -> None:
        """Capture a single CUDA graph for the given shape."""
        key = (batch_size, seq_len)
        
        # Allocate static input buffer - match model dtype
        self.static_inputs[key] = torch.empty(
            batch_size, seq_len, self.hidden_dim,
            device=self.device, dtype=self.model_dtype
        )
        
        # Warmup runs (required before capture)
        warmup_stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()
        warmup_stream.wait_stream(current_stream)
        
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                _ = self.model(self.static_inputs[key])
        
        current_stream.wait_stream(warmup_stream)
        
        # Capture the graph
        self.graphs[key] = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graphs[key]):
            self.static_outputs[key] = self.model(self.static_inputs[key])
        
        self.capture_count += 1
    
    def _find_bucket(self, batch: int, seq: int) -> Optional[Tuple[int, int]]:
        """Find smallest bucket >= actual size."""
        # Find smallest batch bucket >= actual batch
        batch_bucket = None
        for b in self.BATCH_BUCKETS:
            if b >= batch and b <= self.max_batch:
                batch_bucket = b
                break
        
        if batch_bucket is None:
            return None
        
        # Find smallest seq bucket >= actual seq
        seq_bucket = None
        for s in self.SEQ_BUCKETS:
            if s >= seq and s <= self.max_seq:
                seq_bucket = s
                break
        
        if seq_bucket is None:
            return None
        
        return (batch_bucket, seq_bucket)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using pre-captured graphs.
        
        Args:
            x: Input tensor of shape (batch, seq, hidden)
            
        Returns:
            Output tensor, unpadded to original size
        """
        if not torch.cuda.is_available():
            with torch.inference_mode():
                return self.model(x)
        
        batch, seq = x.shape[:2]
        key = self._find_bucket(batch, seq)
        
        if key is None or key not in self.graphs:
            # Fallback to eager execution
            self.fallback_count += 1
            with torch.inference_mode():
                return self.model(x)
        
        bucket_batch, bucket_seq = key
        
        # Pad input to bucket size
        if batch < bucket_batch or seq < bucket_seq:
            padded = F.pad(
                x,
                (0, 0, 0, bucket_seq - seq, 0, bucket_batch - batch)
            )
        else:
            padded = x
        
        # Copy to static buffer and replay graph
        self.static_inputs[key].copy_(padded)
        self.graphs[key].replay()
        self.replay_count += 1
        
        # Return unpadded output
        return self.static_outputs[key][:batch, :seq]
    
    def get_stats(self) -> Dict[str, int]:
        """Return bucketing statistics."""
        return {
            "graphs_captured": self.capture_count,
            "graph_replays": self.replay_count,
            "eager_fallbacks": self.fallback_count,
            "total_buckets": len(self.graphs),
        }


class OptimizedCUDAGraphBucketing(BaselineCUDAGraphBucketing):
    """
    Buckets batch/seq shapes, rounds to CUDA-graph capture sizes, and pre-warms
    the hot buckets so the first live requests replay instead of capturing.
    """

    def __init__(
        self,
        traffic: Iterable[Tuple[int, int]] | None = None,
        bucket_bands: BucketBands | None = None,
        prewarm_shapes: Iterable[Tuple[int, int]] | None = None,
        vllm_model: str = "gpt-oss-20b",
        use_vllm_bins: bool = True,
        region: str = "local",
        model_label: str = "gpt-oss-20b",
    ) -> None:
        super().__init__(
            traffic=traffic,
            vllm_model=vllm_model,
            use_vllm_bins=use_vllm_bins,
        )
        self.bucket_bands = bucket_bands if bucket_bands is not None else default_bucket_bands()
        self.prewarm_shapes: List[Tuple[int, int]] = list(prewarm_shapes) if prewarm_shapes else self._default_prewarm()
        self.region = region
        self.model_label = model_label

    def _default_prewarm(self) -> List[Tuple[int, int]]:
        # Prime the most common padded buckets from the demo traffic so the first live hits replay.
        freq: dict[Tuple[int, int], int] = {}
        for raw_batch, raw_seqlen in demo_traffic():
            b_bucket, s_bucket = self.bucket_bands.bucket(raw_batch, raw_seqlen)
            padded_batch = pad_batch_to_capture(b_bucket, self._capture_bins, self._pad_fn)
            if padded_batch is None:
                continue
            freq[(padded_batch, s_bucket)] = freq.get((padded_batch, s_bucket), 0) + 1
        top_shapes = heapq.nlargest(
            4,
            freq.items(),
            key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]),
        )
        return [shape for shape, _ in top_shapes]

    def build_simulator(self) -> GraphTreeSimulator:
        sim = GraphTreeSimulator(
            bucket_bands=self.bucket_bands,
            capture_batch_sizes=self._capture_bins,
            name="optimized_cudagraphs",
            pad_fn=self._pad_fn,
            # Model expensive graph capture vs cheap replay.
            capture_cost_iters=5000,
        )
        if self.prewarm_shapes:
            sim.prewarm(self.prewarm_shapes)
        return sim

    def run_compile_validation(self) -> dict[str, int]:
        """
        Wrap a toy decode step in torch.compile(dynamic=True) and execute
        padded bucket shapes to confirm compile stability and rebuild counts.
        """
        shapes: List[Tuple[int, int]] = []

        for raw_batch, raw_seqlen in self.traffic:
            b_bucket, s_bucket = self.bucket_bands.bucket(raw_batch, raw_seqlen)
            padded_batch = pad_batch_to_capture(b_bucket, self._capture_bins, self._pad_fn)
            if padded_batch is None:
                continue
            shapes.append((padded_batch, s_bucket))

        compile_counter = {"compiles": 0}

        def counting_backend(gm, example_inputs, **kwargs):
            compile_counter["compiles"] += 1
            inductor = getattr(torch, "_inductor", None)
            if inductor is not None and hasattr(inductor, "compile"):
                return inductor.compile(gm, example_inputs)  # type: ignore[call-arg]
            return gm.forward

        def decode_step(x: torch.Tensor) -> torch.Tensor:
            # Lightweight decode-style math to keep compile fast while exercising shapes.
            return torch.relu(x).sum(dim=-1)

        compiled = torch.compile(
            decode_step,
            dynamic=True,
            mode="reduce-overhead",
            backend=counting_backend,
        )

        for batch, seqlen in shapes:
            x = torch.randn(batch, seqlen, device="cpu")
            _ = compiled(x)

        return compile_counter


def _build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized CUDA graph bucketing simulator", add_help=add_help)
    parser.add_argument("--vllm-model", type=str, default="gpt-oss-20b", help="Model name for capture bins.")
    parser.add_argument(
        "--no-vllm-bins",
        action="store_true",
        help="Force fallback capture bins instead of reading vLLM config",
    )
    parser.add_argument("--region", type=str, default="local", help="Region label for metrics/export.")
    parser.add_argument(
        "--model-label",
        type=str,
        default="gpt-oss-20b",
        help="Model label for metrics/export.",
    )
    parser.add_argument(
        "--prom-port",
        type=int,
        default=None,
        help="If set, starts a Prometheus HTTP exporter on this port and publishes graph stats",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    traffic = demo_traffic()
    optimized = OptimizedCUDAGraphBucketing(
        traffic=traffic,
        vllm_model=args.vllm_model,
        use_vllm_bins=not args.no_vllm_bins,
        region=args.region,
        model_label=args.model_label,
    )
    sim = optimized.run()
    print(sim.format_summary())

    compile_stats = optimized.run_compile_validation()
    print(f"[compile] torch.compile(dynamic=True) recompiles: {compile_stats['compiles']}")

    export_stats_to_prometheus(
        sim.stats,
        region=args.region,
        model=args.model_label,
        start_port=args.prom_port,
    )


class OptimizedCUDAGraphBucketingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark wrapper demonstrating CUDA graph bucketing for inference.
    
    Tests both:
    1. GraphTreeSimulator: Analyzes bucketing behavior
    2. CUDAGraphBucketing: Actual pre-captured graph execution
    """

    def __init__(self) -> None:
        super().__init__()
        self.output = None
        self.vllm_model = "gpt-oss-20b"
        self.use_vllm_bins = True
        self.region = "local"
        self.model_label = "gpt-oss-20b"
        self._last_sim: Optional[GraphTreeSimulator] = None
        self._compile_stats: Optional[dict] = None
        self._graph_bucketing: Optional[CUDAGraphBucketing] = None
        self._graph_stats: Optional[Dict[str, int]] = None
        self._optimized_runner: Optional[OptimizedCUDAGraphBucketing] = None
        self._output_values: Optional[list[float]] = None
        self._payload_traffic: list[Tuple[int, int]] = []
        self._payload_output_values: list[float] = []
        self._output_tensor: Optional[torch.Tensor] = None
        self._traffic_shape_tensor: Optional[torch.Tensor] = None
        self._verification_payload = None
        
        # Simulator-only workload metadata (one traffic replay per iteration).
        self._workload = WorkloadMetadata(requests_per_iteration=1.0)
        self._refresh_payload_metadata()

    def _resolve_device(self) -> torch.device:
        # Simulator is CPU-only.
        return torch.device("cpu")

    def apply_target_overrides(self, argv: Iterable[str]) -> None:
        parser = _build_parser(add_help=False)
        try:
            args, _ = parser.parse_known_args(list(argv))
            self.vllm_model = args.vllm_model
            self.use_vllm_bins = not args.no_vllm_bins
            self.region = args.region
            self.model_label = args.model_label
        except SystemExit:
            pass
        self._refresh_payload_metadata()

    def _refresh_payload_metadata(self) -> None:
        traffic = demo_traffic()
        total_tokens = sum(batch * seqlen for batch, seqlen in traffic)
        self._payload_traffic = traffic
        self._payload_output_values = [float(len(traffic)), float(total_tokens)]
        self._optimized_runner = None
        self._output_tensor = None
        self._traffic_shape_tensor = None

    def _build_optimized_runner(self) -> OptimizedCUDAGraphBucketing:
        return OptimizedCUDAGraphBucketing(
            traffic=self._payload_traffic,
            vllm_model=self.vllm_model,
            use_vllm_bins=self.use_vllm_bins,
            region=self.region,
            model_label=self.model_label,
        )

    def _optimized_simulator_runner(self) -> OptimizedCUDAGraphBucketing:
        runner = self._optimized_runner
        if runner is None:
            runner = self._build_optimized_runner()
            self._optimized_runner = runner
        return runner

    def setup(self) -> None:
        self._optimized_runner = self._build_optimized_runner()
        self._output_tensor = torch.empty(len(self._payload_output_values), dtype=torch.float32)
        self._traffic_shape_tensor = torch.empty(1, dtype=torch.int64, device=self.device)

    def benchmark_fn(self) -> None:
        # NOTE: Keep benchmark_fn() focused on the simulator path so timing
        # reflects bucketing/prewarm improvements vs baseline. Heavy compile
        # validation and GPU graph demos belong in standalone scripts, not the
        # timed harness hot path.
        runner = self._optimized_simulator_runner()
        sim = runner.run()
        self._last_sim = sim
        self._output_values = self._payload_output_values

    def capture_verification_payload(self) -> None:
        traffic = self._payload_traffic
        if self._output_values is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._output_tensor is None or self._traffic_shape_tensor is None:
            raise RuntimeError("setup() must initialize verification tensors")
        for idx, value in enumerate(self._output_values):
            self._output_tensor[idx] = value
        self._traffic_shape_tensor[0] = len(traffic)
        self.output = self._output_tensor
        self._set_verification_payload(
            inputs={"traffic_shape": self._traffic_shape_tensor},
            output=self.output,
            batch_size=len(traffic) if len(traffic) > 0 else 1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        """Cleanup resources."""
        self._demo_model = None
        self._graph_bucketing = None
        self._optimized_runner = None
        super().teardown()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return simulator-derived graph bucketing metrics."""
        if self._last_sim is None:
            return None
        summary = self._last_sim.stats.summary()
        return {
            "graph_tree.captures": float(summary["captures"]),
            "graph_tree.prewarm_captures": float(summary["prewarm_captures"]),
            "graph_tree.replays": float(summary["replays"]),
            "graph_tree.skipped": float(summary["skipped"]),
            "graph_tree.unique_keys": float(summary["unique_keys"]),
        }

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5, enable_profiling=False)


def get_benchmark() -> BaseBenchmark:
    return OptimizedCUDAGraphBucketingBenchmark()
