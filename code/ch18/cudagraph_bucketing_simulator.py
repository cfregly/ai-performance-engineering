"""CUDA graph bucketing analysis tool (Chapter 18).

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

from ch18.cudagraph_bucketing_common import (  # noqa: E402
    DEFAULT_CAPTURE_BATCH_SIZES,
    BucketBands,
    GraphTreeSimulator,
    capture_bins_from_vllm_config,
    default_bucket_bands,
    demo_traffic,
    load_vllm_config,
    pad_batch_to_capture,
    pad_fn_from_vllm_config,
)
from ch18.cudagraph_bucketing_metrics import (  # noqa: E402
    export_stats_to_prometheus,
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
        
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _ = self.model(self.static_inputs[key])
        
        current_stream.wait_stream(warmup_stream)
        
        # Capture the graph
        self.graphs[key] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graphs[key]):
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
            return self.model(x)
        
        batch, seq = x.shape[:2]
        key = self._find_bucket(batch, seq)
        
        if key is None or key not in self.graphs:
            # Fallback to eager execution
            self.fallback_count += 1
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


class BaselineCUDAGraphBucketing:
    """
    Simulates decode traffic without shape bucketing or pre-warming.

    Every distinct (batch, seqlen) pair becomes a fresh CUDA graph node,
    which is why captures grow quickly when request shapes wander.
    """

    def __init__(
        self,
        traffic: Iterable[Tuple[int, int]] | None = None,
        vllm_model: str = "gpt-oss-20b",
        use_vllm_bins: bool = True,
    ) -> None:
        self.traffic = list(traffic) if traffic is not None else demo_traffic()
        self.vllm_model = vllm_model
        self.use_vllm_bins = use_vllm_bins
        self._vllm_config = load_vllm_config(vllm_model) if use_vllm_bins else None
        self._capture_bins = (
            capture_bins_from_vllm_config(self._vllm_config)
            if self._vllm_config
            else DEFAULT_CAPTURE_BATCH_SIZES
        )
        self._pad_fn = pad_fn_from_vllm_config(self._vllm_config) if self._vllm_config else None

    def build_simulator(self) -> GraphTreeSimulator:
        bands = BucketBands(batch_buckets=[], seqlen_buckets=[])
        return GraphTreeSimulator(
            bucket_bands=bands,
            capture_batch_sizes=self._capture_bins,
            name="baseline_cudagraphs",
            pad_fn=self._pad_fn,
            # Model expensive graph capture vs cheap replay.
            capture_cost_iters=5000,
        )

    def run(self) -> GraphTreeSimulator:
        sim = self.build_simulator()
        sim.run(self.traffic)
        return sim


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
    parser = argparse.ArgumentParser(description="CUDA graph bucketing simulator (tool)", add_help=add_help)
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=("baseline", "optimized", "compare"),
        help="Scenario to run: baseline (unbucketed), optimized (bucketed + prewarm), or compare.",
    )
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
        "--compile-validation",
        action="store_true",
        help="Run torch.compile(dynamic=True) shape validation (optimized mode only).",
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
    baseline_sim: Optional[GraphTreeSimulator] = None
    optimized_sim: Optional[GraphTreeSimulator] = None

    if args.mode in ("baseline", "compare"):
        baseline = BaselineCUDAGraphBucketing(
            traffic=traffic,
            vllm_model=args.vllm_model,
            use_vllm_bins=not args.no_vllm_bins,
        )
        baseline_sim = baseline.run()
        print(baseline_sim.format_summary())

    if args.mode in ("optimized", "compare"):
        optimized = OptimizedCUDAGraphBucketing(
            traffic=traffic,
            vllm_model=args.vllm_model,
            use_vllm_bins=not args.no_vllm_bins,
            region=args.region,
            model_label=args.model_label,
        )
        optimized_sim = optimized.run()
        print(optimized_sim.format_summary())

        if args.compile_validation:
            compile_stats = optimized.run_compile_validation()
            print(f"[compile] torch.compile(dynamic=True) recompiles: {compile_stats['compiles']}")

        export_stats_to_prometheus(
            optimized_sim.stats,
            region=args.region,
            model=args.model_label,
            start_port=args.prom_port,
        )

    if args.mode == "compare" and baseline_sim is not None and optimized_sim is not None:
        base = baseline_sim.stats.summary()
        opt = optimized_sim.stats.summary()
        print(
            "[compare] captures: {b} -> {o}, unique_keys: {bk} -> {ok}".format(
                b=base["captures"],
                o=opt["captures"],
                bk=base["unique_keys"],
                ok=opt["unique_keys"],
            )
        )


if __name__ == "__main__":
    main()
