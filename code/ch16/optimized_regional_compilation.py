"""optimized_regional_compilation.py - Optimized: per-layer compile regions plus CUDA graph replay.

Demonstrates the solution: compile hot regions of the large transformer independently
instead of compiling the entire model monolithically. We capture per-sequence CUDA
graphs (regional buckets) so each bucket replays instantly without re-hitting Python
overhead or torch.compile re-specialization churn. The end result is a deterministic,
BF16 fast-path that still mirrors the baseline workload.
"""

from __future__ import annotations

from functools import partial
import torch
import torch.nn as nn

from typing import Dict, Optional, Tuple

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
from core.utils.compile_utils import maybe_nested_compile_region
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import nvtx_range, get_nvtx_enabled
from ch16.baseline_regional_compilation import (
    DummyTransformer,
    select_regional_compilation_choice,
)
resolve_device = partial(require_cuda_device, "CUDA required for ch16")


@maybe_nested_compile_region
def _run_compiled_layer(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return layer(x)


class RegionalCompilationTransformer(DummyTransformer):
    """Optimized model that routes each block through a compile-friendly region."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            layer_out = _run_compiled_layer(layer, x)
            layer_out.add_(x)
            x = layer_out
        return x


GraphCacheEntry = Tuple["torch.cuda.CUDAGraph", torch.Tensor, torch.Tensor]


class OptimizedRegionalCompilationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Regional compilation via CUDA graph capture for a fixed bucket."""

    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model: Optional[DummyTransformer] = None
        self.choice = select_regional_compilation_choice()
        self.sequence_schedule = [self.choice["seq_len"]]
        self._sequence_schedule_len = len(self.sequence_schedule)
        self.max_seq_len = self.choice["seq_len"]
        self.d_model = self.choice["d_model"]
        self._iteration = 0
        self.compiled_layers = 0
        self.output: Optional[torch.Tensor] = None
        self.graph_cache: Dict[int, GraphCacheEntry] = {}
        self.capture_ms: float = 0.0
        self._verify_input: Optional[torch.Tensor] = None
        self._payload_verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._input_views: Dict[int, torch.Tensor] = {}
        self.parameter_count: int = 0
        self._enable_nvtx = False
        tokens = self.max_seq_len * self.d_model
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        # Use the larger preset so region capture speedups have room to show.
        candidate = self.choice
        self.d_model = candidate["d_model"]
        self.max_seq_len = candidate["seq_len"]
        self.sequence_schedule = [self.max_seq_len]
        self._sequence_schedule_len = len(self.sequence_schedule)
        model = RegionalCompilationTransformer(
            n_layers=candidate["n_layers"],
            d_model=self.d_model,
            d_ff=candidate["d_ff"],
        ).to(self.device, dtype=torch.bfloat16).eval()
        self.model = model
        self.parameter_count = sum(p.numel() for p in model.parameters())

        # IMPORTANT: Keep inputs identical to the baseline and avoid any extra
        # verification-only forward passes in benchmark_fn(). Verification uses
        # the output from the timed run.
        self._verify_input = torch.randn(
            1,
            self.max_seq_len,
            self.d_model,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._payload_verify_input = self._verify_input
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        self._input_views = {self.max_seq_len: self._verify_input}
        self._prepare_cuda_graphs()
        tokens = self.max_seq_len * self.d_model
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def _input_view_for(self, seq_len: int) -> torch.Tensor:
        if self._verify_input is None:
            raise RuntimeError("Verification input must be initialized before slicing")
        view = self._input_views.get(seq_len)
        if view is not None:
            return view
        view = self._verify_input if seq_len == self.max_seq_len else self._verify_input[:, :seq_len]
        self._input_views[seq_len] = view
        return view

    def _prepare_cuda_graphs(self) -> None:
        """Capture CUDA graphs per sequence length to eliminate Python overhead."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before CUDA graph capture")
        if self._verify_input is None:
            raise RuntimeError("Verification input must be initialized before CUDA graph capture")
        self.graph_cache.clear()
        torch.cuda.synchronize()
        unique_lengths = sorted(set(self.sequence_schedule))
        capture_start = torch.cuda.Event(enable_timing=True)
        capture_stop = torch.cuda.Event(enable_timing=True)
        capture_stream = torch.cuda.current_stream(self.device)
        capture_start.record(capture_stream)

        for seq_len in unique_lengths:
            static_input = self._input_view_for(seq_len)
            # Warm-up and capture under inference mode to avoid autograd state.
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                static_output = self.model(static_input)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    static_output = self.model(static_input)
            self.graph_cache[seq_len] = (graph, static_input, static_output)
        self.compiled_layers = len(self.graph_cache)
        capture_stop.record(capture_stream)
        torch.cuda.synchronize()
        self.capture_ms = float(capture_start.elapsed_time(capture_stop))

    def benchmark_fn(self) -> None:
        if self.model is None:
            raise RuntimeError("Optimized model not initialized")
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")

        seq_len = self.sequence_schedule[self._iteration % self._sequence_schedule_len]
        self._iteration += 1
        ran_graph = self._run_with_cuda_graph(seq_len, self._enable_nvtx)
        if not ran_graph:
            raise RuntimeError("CUDA graph replay missing for expected sequence length bucket")

        self._payload_verify_input = self._input_view_for(seq_len)

    def capture_verification_payload(self) -> None:
        verify_input = self._payload_verify_input
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if verify_input is None:
            raise RuntimeError("Verification input not initialized")
        verify_output = self._verify_output_buffer[
            : self.output.shape[0],
            : self.output.shape[1],
            : self.output.shape[2],
        ]
        verify_output.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": verify_input},
            output=verify_output,
            batch_size=1,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def run(self, compare_eager: bool = False) -> torch.Tensor:
        """Run a single forward pass for demo/validation."""
        if self.model is None or self._verify_input is None:
            raise RuntimeError("Optimized model not initialized")

        seq_len = self.sequence_schedule[0]
        input_view = self._input_view_for(seq_len)
        ran_graph = self._run_with_cuda_graph(seq_len, enable_nvtx=False)
        if not ran_graph:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                _ = self.model(input_view)
        if compare_eager:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                _ = self.model(input_view)
        self._synchronize()
        if self.output is None:
            raise RuntimeError("run() did not produce output")
        return self.output

    def _run_with_cuda_graph(self, seq_len: int, enable_nvtx: bool) -> bool:
        """Replay a cached CUDA graph for the requested sequence length."""
        entry = self.graph_cache.get(seq_len)
        if entry is None:
            return False
        graph, _static_input, static_output = entry
        with nvtx_range("regional_compilation[cuda_graph]", enable=enable_nvtx):
            graph.replay()
        self.output = static_output
        return True

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=10,
            setup_timeout_seconds=240,
            measurement_timeout_seconds=240,
            use_subprocess=True,
            adaptive_iterations=False,
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        metrics = compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )
        metrics.update(
            {
                "regional_compilation.capture_ms": float(self.capture_ms),
                "regional_compilation.graph_bucket_count": float(len(self.graph_cache)),
                "regional_compilation.steady_state_only": 1.0,
            }
        )
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.model is None or self._verify_input is None:
            return "Model/input not initialized"
        return None

    def teardown(self) -> None:
        self.model = None
        self._verify_input = None
        self._payload_verify_input = None
        self._verify_output_buffer = None
        self._input_views = {}
        self.graph_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedRegionalCompilationBenchmark()


def main():
    """Run the optimized benchmark."""
    benchmark = OptimizedRegionalCompilationBenchmark()

    print("\n" + "=" * 80)
    print("Regional Compilation Benchmark Demo")
    print("=" * 80)

    benchmark.setup()
    output = benchmark.run(compare_eager=True)
    print(f"\n[OK] Optimized completed: output shape {output.shape}")
    print(f"   Captured CUDA graph buckets: {benchmark.compiled_layers}")
    print("   Path: per-layer compile regions + fixed-shape CUDA graph replay")
    benchmark.teardown()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("Baseline: eager transformer blocks in the timed region.")
    print("Optimized: the same blocks routed through nested compile regions and replayed via CUDA graphs.")
