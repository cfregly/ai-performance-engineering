"""Baseline decode loop that triggers CUDA graph churn, allocator growth, and eager KV compaction."""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Workaround for importlib.util.spec_from_file_location loading:
# Register this module in sys.modules so @dataclass works correctly
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from dataclasses import dataclass

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402

# Defer decode_kernels import so the harness can load this module while still
# surfacing dependency failures as explicit FAIL FAST errors.
DECODE_KERNEL_AVAILABLE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
build_decode_kernel = None
try:
    from ch18.decode_kernels import DEVICE as _DEV  # noqa: E402
    from ch18.decode_kernels import build_decode_kernel as _build  # noqa: E402
    DEVICE = _DEV
    build_decode_kernel = _build
    DECODE_KERNEL_AVAILABLE = True
except ImportError:
    pass


def default_trace(num_steps: int = 24, seed: int = 0) -> List[int]:
    """Generate a ragged decode schedule to mimic continuous batching."""
    rng = random.Random(seed)
    candidates = (3, 5, 7, 9, 12, 15, 18, 24, 28)
    return [rng.choice(candidates) for _ in range(num_steps)]


@dataclass
class KVBlock:
    capacity: int
    used: int = 0

    @property
    def free(self) -> int:
        return max(0, self.capacity - self.used)

    def fill(self, tokens: int) -> None:
        self.used = min(self.capacity, self.used + tokens)

    def compact(self) -> None:
        self.used = 0


class NaiveKVLayout:
    """Compacts every step, which models the "micro-motion" overhead."""

    def __init__(
        self,
        blocks: Sequence[int] = (64, 96, 128, 192),
        *,
        hidden: int = 256,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.blocks: List[KVBlock] = [KVBlock(b) for b in blocks]
        # Scratch buffers model eager KV compaction/copy per block.
        self._scratch: List[torch.Tensor] = [
            torch.zeros((b, hidden), device=device, dtype=dtype) for b in blocks
        ]

    def simulate_step(self, tokens_written: int) -> int:
        """Write tokens into blocks, then compact all blocks eagerly."""
        compactions = 0
        for block, scratch in zip(self.blocks, self._scratch):
            block.fill(tokens_written)
            block.compact()
            # Simulate a zero-fill of the block during compaction to model realistic copy traffic.
            scratch.fill_(0.0)
            scratch.add_(float(tokens_written))
            compactions += 1
        return compactions


@dataclass
class DecodeMetrics:
    steps: int = 0
    tokens: int = 0
    graph_recaptures: int = 0
    allocator_bytes: int = 0
    compactions: int = 0


class BaselineDecodeDriver:
    """
    Decode loop without buckets or preallocation.

    Every ragged batch forces a new graph capture, fresh allocator activity, and
    aggressive KV compaction.
    """

    def __init__(self, trace: Iterable[int] | None = None, hidden: int = 256) -> None:
        self.trace = list(trace) if trace is not None else default_trace()
        self.hidden = hidden
        self.decode_kernel = build_decode_kernel(hidden=self.hidden, max_batch=max(self.trace or [1]))
        dtype = torch.float16 if getattr(self.decode_kernel, "backend", "") == "vllm" else torch.float32
        self.kv_layout = NaiveKVLayout(hidden=self.hidden, device=DEVICE, dtype=dtype)
        self.captured_shapes: set[Tuple[int, int]] = set()
        self._token_bank = [
            torch.randn(batch_size, self.hidden, device=DEVICE, dtype=dtype)
            for batch_size in self.trace
        ]
        self._kv_bank = [
            torch.randn(batch_size, self.hidden, device=DEVICE, dtype=dtype)
            for batch_size in self.trace
        ]

    def run(self) -> DecodeMetrics:
        metrics = DecodeMetrics()
        for batch_size, tokens, kv in zip(self.trace, self._token_bank, self._kv_bank):
            logits = self.decode_kernel(tokens, kv, None)

            shape_key = (logits.shape[0], logits.shape[1])
            if shape_key not in self.captured_shapes:
                metrics.graph_recaptures += 1
                self.captured_shapes.add(shape_key)

            metrics.allocator_bytes += logits.numel() * logits.element_size()
            metrics.compactions += self.kv_layout.simulate_step(tokens_written=batch_size)
            metrics.tokens += batch_size
            metrics.steps += 1

        return metrics


def format_metrics(label: str, metrics: DecodeMetrics, backend: str = "torch") -> str:
    mb = metrics.allocator_bytes / (1024 * 1024)
    return (
        f"[{label}] backend={backend}, steps={metrics.steps}, tokens={metrics.tokens}, "
        f"graph_captures={metrics.graph_recaptures}, allocator_mb={mb:.1f}, "
        f"compactions={metrics.compactions}"
    )


def export_prom_metrics(
    label: str,
    metrics: DecodeMetrics,
    backend: str,
    port: int,
    duration_s: int,
) -> None:
    """Expose graph/allocator counters alongside vLLM metrics naming style."""
    try:
        from prometheus_client import Counter, Gauge, start_http_server
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[warn] prometheus_client unavailable, skipping export: {exc}")
        return

    graph_recaptures = Counter(
        "vllm:decode_graph_recaptures_total",
        "Decode graph recaptures (bucket drift) observed by the demo driver.",
        ["variant", "backend"],
    )
    allocator_bytes = Gauge(
        "vllm:decode_allocator_bytes",
        "Bytes attributed to decode workspaces in the demo driver.",
        ["variant", "backend"],
    )
    kv_compactions = Counter(
        "vllm:decode_kv_compactions_total",
        "Number of KV compactions triggered by the demo driver.",
        ["variant", "backend"],
    )

    start_http_server(port)
    graph_recaptures.labels(variant=label, backend=backend).inc(metrics.graph_recaptures)
    allocator_bytes.labels(variant=label, backend=backend).set(metrics.allocator_bytes)
    kv_compactions.labels(variant=label, backend=backend).inc(metrics.compactions)

    if duration_s > 0:
        print(f"[metrics] exporting on :{port} for {duration_s}s")
        threading.Event().wait(timeout=duration_s)
        print(f"[metrics] Prometheus window elapsed on :{port}")
        return

    print(f"[metrics] exported once on :{port} (process will exit)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline ragged decode loop (no buckets/prealloc).")
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
    if not DECODE_KERNEL_AVAILABLE:
        raise RuntimeError(
            "FAIL FAST: vllm_decode_graphs dependencies unavailable for CLI run. "
            "Install the pinned serving stack and verify `ch18.decode_kernels` imports."
        )
    args = parse_args()
    trace = default_trace(num_steps=args.steps, seed=args.seed)
    driver = BaselineDecodeDriver(trace=trace, hidden=args.hidden)
    metrics = driver.run()
    backend = getattr(driver.decode_kernel, "backend", "torch")
    print(format_metrics("baseline", metrics, backend=backend))

    if args.prom_port is not None:
        export_prom_metrics("baseline", metrics, backend=backend, port=args.prom_port, duration_s=args.prom_duration)


class VLLMDecodeGraphsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """
    Runs the ragged decode driver under the harness so we capture graph
    recaptures, allocator growth, and eager KV compaction costs.
    """

    def __init__(self, steps: int = 32, hidden: int = 192, seed: int = 42) -> None:
        if not DECODE_KERNEL_AVAILABLE:
            raise RuntimeError(
                "FAIL FAST: vllm_decode_graphs dependencies unavailable. "
                "Install the pinned serving stack and verify `ch18.decode_kernels` imports."
            )
        super().__init__()
        self.steps = steps
        self.hidden = hidden
        self.seed = seed
        self._trace: List[int] = default_trace(num_steps=self.steps, seed=self.seed)
        self._driver: Optional[BaselineDecodeDriver] = None
        self._last_metrics: Optional[DecodeMetrics] = None
        self.output: Optional[torch.Tensor] = None
        self._output_values: Optional[list[float]] = None
        self._payload_output_values = [float(len(self._trace)), float(sum(self._trace))]
        self._output_tensor: Optional[torch.Tensor] = None
        self._trace_tensor: Optional[torch.Tensor] = None
        self._verification_payload = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def get_config(self) -> BenchmarkConfig:
        # Keep iterations modest—the kernel is compiled once and re-used.
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            percentiles=[50, 75, 90, 99],
        )

    def setup(self) -> None:
        torch.manual_seed(self.seed)
        self._driver = BaselineDecodeDriver(trace=self._trace, hidden=self.hidden)
        self._output_tensor = torch.empty(len(self._payload_output_values), dtype=torch.float32)
        self._trace_tensor = torch.tensor(self._trace, device=DEVICE)

    def benchmark_fn(self) -> None:
        if self._driver is None:
            raise RuntimeError("FAIL FAST: decode driver not initialized")
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
    return VLLMDecodeGraphsBenchmark()
