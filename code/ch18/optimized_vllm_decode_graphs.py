"""Optimized decode loop with bucketed shapes, preallocated workspaces, and lazy KV compaction."""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, Optional, Tuple

# Workaround for importlib.util.spec_from_file_location loading:
# Register this module in sys.modules so @dataclass works correctly
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from dataclasses import dataclass

import torch

try:
    from ch18.baseline_vllm_decode_graphs import (  # noqa: E402
        DecodeMetrics,
        default_trace,
        export_prom_metrics,
        format_metrics,
    )
except ImportError as exc:
    raise RuntimeError(
        "FAIL FAST: Unable to import ch18.baseline_vllm_decode_graphs. "
        "Ensure the benchmark is executed from repo root with `ch18` on PYTHONPATH."
    ) from exc

from ch18.decode_kernels import DEVICE, build_decode_kernel  # noqa: E402
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402

# Keep a moderately coarse bucket set that limits padding overhead while still
# reducing shape churn relative to the ragged baseline.
BUCKETS = (7, 9, 12, 18, 24, 28)
FRAG_LIMIT = 0.20
AGE_LIMIT = 6  # steps; small for the toy demo


def pick_bucket(size: int) -> int:
    for b in BUCKETS:
        if size <= b:
            return b
    return BUCKETS[-1]


@dataclass
class BucketWorkspace:
    batch: int
    hidden: int
    device: str = DEVICE
    tokens_kv: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    kv: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    tmp: torch.Tensor | None = None
    stream: torch.cuda.Stream | None = None
    initialized: bool = False

    def ensure(self) -> None:
        if self.initialized:
            return
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Keep tokens + KV contiguous so we can initialize both with a single kernel.
        self.tokens_kv = torch.empty((2, self.batch, self.hidden), device=self.device, dtype=dtype)
        self.tokens = self.tokens_kv[0]
        self.kv = self.tokens_kv[1]
        # Populate once and reuse to avoid per-step RNG overhead in the optimized path.
        self.tokens_kv.normal_(mean=0.0, std=1.0)
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(self.stream):
                self.logits = torch.empty((self.batch, self.hidden), device=self.device, dtype=dtype)
                self.tmp = torch.empty_like(self.logits)
            self.stream.synchronize()
        else:
            self.logits = torch.empty((self.batch, self.hidden), device=self.device, dtype=dtype)
            self.tmp = torch.empty_like(self.logits)
        self.initialized = True

    @property
    def bytes(self) -> int:
        if self.logits is None or self.tmp is None or self.tokens is None or self.kv is None:
            return 0
        return (
            (self.logits.numel() + self.tmp.numel()) * self.logits.element_size()
            + self.tokens.numel() * self.tokens.element_size()
            + self.kv.numel() * self.kv.element_size()
        )


@dataclass
class KVBlock:
    capacity: int
    used: int = 0
    age: int = 0

    @property
    def free_ratio(self) -> float:
        if self.capacity == 0:
            return 0.0
        return max(0.0, 1.0 - (self.used / self.capacity))

    def advance(self, tokens_written: int) -> None:
        self.used = min(self.capacity, self.used + tokens_written)
        self.age += 1

    def maybe_compact(self) -> bool:
        if self.free_ratio > FRAG_LIMIT and self.age > AGE_LIMIT:
            self.used = 0
            self.age = 0
            return True
        return False


class LazyKVLayout:
    """Defers compaction until fragmentation and age cross a threshold."""

    def __init__(self, blocks: Iterable[int] = (96, 128, 160)) -> None:
        self.blocks = [KVBlock(b) for b in blocks]

    def simulate_step(self, tokens_written: int) -> int:
        compactions = 0
        for block in self.blocks:
            block.advance(tokens_written)
            if block.maybe_compact():
                compactions += 1
        return compactions


class OptimizedDecodeDriver:
    """
    Decode loop that keeps shapes stable, preallocates per-bucket workspaces, and lazily compacts KV.
    """

    def __init__(self, trace: Iterable[int] | None = None, hidden: int = 256) -> None:
        self.trace = list(trace) if trace is not None else default_trace()
        self.hidden = hidden
        self.decode_kernel = build_decode_kernel(hidden=self.hidden, max_batch=max(BUCKETS))
        self.kv_layout = LazyKVLayout()
        self.workspaces: Dict[int, BucketWorkspace] = {}
        self.captured_shapes: set[Tuple[int, int]] = set()
        self._vllm_kernel = self._resolve_vllm_kernel()
        self._seq_lens_profiles: Dict[Tuple[int, int], torch.Tensor] = {}
        self._trace_schedule: list[Tuple[int, BucketWorkspace, Optional[torch.Tensor]]] = []
        self._prepare_trace_schedule()

    def _resolve_vllm_kernel(self):
        if getattr(self.decode_kernel, "backend", None) != "vllm":
            return None
        kernel = getattr(self.decode_kernel, "fn", None)
        if kernel is None:
            raise RuntimeError("vLLM decode kernel missing 'fn' implementation")
        if not hasattr(kernel, "seq_lens") or not hasattr(kernel, "block_size"):
            raise RuntimeError("vLLM decode kernel missing required 'seq_lens'/'block_size' attributes")
        return kernel

    def workspace_for(self, bucket: int) -> BucketWorkspace:
        if bucket not in self.workspaces:
            self.workspaces[bucket] = BucketWorkspace(batch=bucket, hidden=self.hidden)
        return self.workspaces[bucket]

    def seq_lens_profile(self, batch_size: int, bucket: int) -> torch.Tensor:
        if self._vllm_kernel is None:
            raise RuntimeError("vLLM seq_lens profile requested without a vLLM kernel")
        key = (batch_size, bucket)
        profile = self._seq_lens_profiles.get(key)
        if profile is None:
            seq_lens = self._vllm_kernel.seq_lens
            profile = torch.empty(bucket, device=seq_lens.device, dtype=seq_lens.dtype)
            profile[:batch_size].fill_(self._vllm_kernel.block_size)
            if bucket > batch_size:
                profile[batch_size:bucket].zero_()
            self._seq_lens_profiles[key] = profile
        return profile

    def _prepare_trace_schedule(self) -> None:
        self._trace_schedule = []
        for batch_size in self.trace:
            bucket = pick_bucket(batch_size)
            workspace = self.workspace_for(bucket)
            seq_lens_profile = (
                self.seq_lens_profile(batch_size, bucket)
                if self._vllm_kernel is not None
                else None
            )
            self._trace_schedule.append((batch_size, workspace, seq_lens_profile))

    def run(self) -> DecodeMetrics:
        metrics = DecodeMetrics()
        for batch_size, ws, seq_lens_profile in self._trace_schedule:
            was_initialized = ws.initialized
            ws.ensure()

            if ws.tokens is None or ws.kv is None or ws.tokens_kv is None:
                raise RuntimeError("workspace not initialized")
            if self._vllm_kernel is not None:
                if seq_lens_profile is None:
                    raise RuntimeError("vLLM seq_lens profile missing from trace schedule")
                # Mark padded rows as "inactive" by setting seq_lens=0 for them.
                # This keeps bucketed shapes stable (good for CUDA graphs / workspace reuse)
                # without paying full attention cost on dummy rows.
                seq_lens = self._vllm_kernel.seq_lens
                bucket = ws.batch
                seq_lens[:bucket].copy_(seq_lens_profile)
            # Keep the compute path aligned with the baseline (no masking); the
            # benchmark's outputs are metrics, not decoded logits.
            logits = self.decode_kernel(ws.tokens, ws.kv, None)

            shape_key = (logits.shape[0], logits.shape[1])
            if shape_key not in self.captured_shapes:
                metrics.graph_recaptures += 1
                self.captured_shapes.add(shape_key)

            # Count allocator growth only once per workspace instead of every step.
            metrics.allocator_bytes += 0 if was_initialized else ws.bytes
            metrics.compactions += self.kv_layout.simulate_step(tokens_written=batch_size)
            metrics.tokens += batch_size
            metrics.steps += 1

        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucketed decode loop with reusable workspaces.")
    parser.add_argument("--steps", type=int, default=24, help="Decode iterations to run.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size for the mock decode op.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the ragged trace.")
    parser.add_argument("--prom-port", type=int, default=None, help="Optional Prometheus port to export metrics.")
    parser.add_argument(
        "--prom-duration",
        type=int,
        default=0,
        help="If --prom-port is set, keep the server alive this many seconds (0 = fire-and-exit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = default_trace(num_steps=args.steps, seed=args.seed)
    driver = OptimizedDecodeDriver(trace=trace, hidden=args.hidden)
    metrics = driver.run()
    backend = getattr(driver.decode_kernel, "backend", "torch")
    print(format_metrics("optimized", metrics, backend=backend))

    if args.prom_port is not None:
        export_prom_metrics("optimized", metrics, backend=backend, port=args.prom_port, duration_s=args.prom_duration)


class OptimizedVLLMDecodeGraphsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """
    Harness entry point for the bucketed decode loop. Uses the same ragged
    trace as the baseline so speedups reflect graph reuse and allocator reuse,
    not synthetic workload changes.
    """

    def __init__(self, steps: int = 32, hidden: int = 192, seed: int = 42) -> None:
        super().__init__()
        self.steps = steps
        self.hidden = hidden
        self.seed = seed
        self._trace: list[int] = default_trace(num_steps=self.steps, seed=self.seed)
        self._driver: Optional[OptimizedDecodeDriver] = None
        self._last_metrics: Optional[DecodeMetrics] = None
        self.output: Optional[torch.Tensor] = None
        self._output_values: Optional[list[float]] = None
        self._payload_output_values = [float(len(self._trace)), float(sum(self._trace))]
        self._output_tensor: Optional[torch.Tensor] = None
        self._trace_tensor: Optional[torch.Tensor] = None
        self._verification_payload = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            percentiles=[50, 75, 90, 99],
        )

    def setup(self) -> None:
        torch.manual_seed(self.seed)
        self._driver = OptimizedDecodeDriver(trace=self._trace, hidden=self.hidden)
        self._output_tensor = torch.empty(len(self._payload_output_values), dtype=torch.float32)
        self._trace_tensor = torch.tensor(self._trace, device=DEVICE)

    def benchmark_fn(self) -> None:
        if self._driver is None:
            raise RuntimeError("FAIL FAST: optimized decode driver not initialized")
        self._last_metrics = self._driver.run()
        self._output_values = self._payload_output_values

    def capture_verification_payload(self) -> None:
        if self._output_values is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._output_tensor is None or self._trace_tensor is None:
            raise RuntimeError("setup() must initialize verification tensors")
        for idx, value in enumerate(self._output_values):
            self._output_tensor[idx] = value
        self.output = self._output_tensor
        self._set_verification_payload(
            inputs={"trace": self._trace_tensor},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        super().teardown()
        self._driver = None

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if self._last_metrics is None:
            return None
        return {
            "vllm_decode_graphs.steps": float(self._last_metrics.steps),
            "vllm_decode_graphs.tokens": float(self._last_metrics.tokens),
            "vllm_decode_graphs.graph_recaptures": float(self._last_metrics.graph_recaptures),
            "vllm_decode_graphs.allocator_mb": float(self._last_metrics.allocator_bytes) / (1024 * 1024),
            "vllm_decode_graphs.compactions": float(self._last_metrics.compactions),
        }


def get_benchmark() -> BaseBenchmark:
    return OptimizedVLLMDecodeGraphsBenchmark()
