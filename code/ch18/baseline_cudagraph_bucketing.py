"""Baseline decode bucketing demo: unbucketed shapes cause many CUDA graph captures."""

from __future__ import annotations

import argparse
from typing import Iterable, Optional, Tuple

import torch

from ch18.cudagraph_bucketing_common import (  # noqa: E402
    DEFAULT_CAPTURE_BATCH_SIZES,
    BucketBands,
    GraphTreeSimulator,
    capture_bins_from_vllm_config,
    demo_traffic,
    load_vllm_config,
    pad_fn_from_vllm_config,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402


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


def _build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline CUDA graph bucketing simulator", add_help=add_help)
    parser.add_argument("--vllm-model", type=str, default="gpt-oss-20b", help="Model name for capture bins.")
    parser.add_argument(
        "--no-vllm-bins",
        action="store_true",
        help="Force fallback capture bins instead of reading vLLM config",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    baseline = BaselineCUDAGraphBucketing(
        vllm_model=args.vllm_model,
        use_vllm_bins=not args.no_vllm_bins,
    )
    sim = baseline.run()
    print(sim.format_summary())


class BaselineCUDAGraphBucketingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark wrapper so the simulator can run via aisp bench."""

    def __init__(self) -> None:
        super().__init__()
        self.vllm_model = "gpt-oss-20b"
        self.use_vllm_bins = True
        self._last = None
        self.output: Optional[torch.Tensor] = None
        self._baseline_runner: Optional[BaselineCUDAGraphBucketing] = None
        self._output_values: Optional[list[float]] = None
        self._payload_traffic: list[Tuple[int, int]] = []
        self._payload_output_values: list[float] = []
        self._output_tensor: Optional[torch.Tensor] = None
        self._traffic_shape_tensor: Optional[torch.Tensor] = None
        self._verification_payload = None
        self.register_workload_metadata(requests_per_iteration=1.0)
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
        except SystemExit:
            # Ignore parse errors in override path.
            pass
        self._refresh_payload_metadata()

    def _refresh_payload_metadata(self) -> None:
        traffic = demo_traffic()
        total_tokens = sum(batch * seqlen for batch, seqlen in traffic)
        self._payload_traffic = traffic
        self._payload_output_values = [float(len(traffic)), float(total_tokens)]
        self._baseline_runner = None
        self._output_tensor = None
        self._traffic_shape_tensor = None

    def _build_baseline_runner(self) -> BaselineCUDAGraphBucketing:
        return BaselineCUDAGraphBucketing(
            traffic=self._payload_traffic,
            vllm_model=self.vllm_model,
            use_vllm_bins=self.use_vllm_bins,
        )

    def _baseline_simulator_runner(self) -> BaselineCUDAGraphBucketing:
        runner = self._baseline_runner
        if runner is None:
            runner = self._build_baseline_runner()
            self._baseline_runner = runner
        return runner

    def setup(self) -> None:
        self._baseline_runner = self._build_baseline_runner()
        self._output_tensor = torch.empty(len(self._payload_output_values), dtype=torch.float32)
        self._traffic_shape_tensor = torch.empty(1, dtype=torch.int64)

    def benchmark_fn(self) -> None:
        runner = self._baseline_simulator_runner()
        sim = runner.run()
        self._last = sim
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
            inputs={
                "traffic_shape": self._traffic_shape_tensor,
            },
            output=self.output,
            batch_size=len(traffic) if len(traffic) > 0 else 1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return simulator-derived graph bucketing metrics."""
        if self._last is None:
            return None
        summary = self._last.stats.summary()
        return {
            "graph_tree.captures": float(summary["captures"]),
            "graph_tree.prewarm_captures": float(summary["prewarm_captures"]),
            "graph_tree.replays": float(summary["replays"]),
            "graph_tree.skipped": float(summary["skipped"]),
            "graph_tree.unique_keys": float(summary["unique_keys"]),
        }

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5, enable_profiling=False)


def get_benchmark() -> BaseBenchmark:
    return BaselineCUDAGraphBucketingBenchmark()
