"""Baseline inference placement policy (intentionally cross-node heavy)."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ch15.placement_sim import (  # noqa: E402
    PlacementConfig,
    PlacementSimulator,
    percentile,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402


class _PlacementBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Shared scaffolding for placement simulations."""

    def __init__(self, cfg: PlacementConfig, prefix: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.prefix = prefix
        self.sessions = 64
        self.simulator = PlacementSimulator()
        self._summary: Dict[str, float] = {}
        self.output = None  # Simulation metrics as tensor
        self._output_values: list[float] = []
        self.register_workload_metadata(requests_per_iteration=float(self.sessions))
        self._verify_cfg = torch.tensor(
            [cfg.prefill_tp_size, cfg.decode_tp_size, cfg.decode_microbatch],
            dtype=torch.int64,
        )

    def setup(self) -> None:
        self._previous_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.cfg.dtype)  # type: ignore[arg-type]

    def benchmark_fn(self) -> None:
        run = self.simulator.simulate(self.cfg, sessions=self.sessions, seed=17)
        ttft_p50 = percentile(run.ttft_ms, 50)
        ttft_p95 = percentile(run.ttft_ms, 95)
        decode_p50 = percentile(run.decode_ms, 50)
        decode_p95 = percentile(run.decode_ms, 95)
        total_ms = run.ttft_total_ms + run.decode_total_ms
        tput_tokens_s = run.tokens_processed / max(total_ms / 1000.0, 1e-6)
        self._total_tokens = int(run.tokens_processed)
        self._total_requests = int(run.sessions)

        self._summary = {
            f"{self.prefix}.ttft_p50_ms": ttft_p50,
            f"{self.prefix}.ttft_p95_ms": ttft_p95,
            f"{self.prefix}.decode_p50_ms": decode_p50,
            f"{self.prefix}.decode_p95_ms": decode_p95,
            f"{self.prefix}.tokens_per_s_est": tput_tokens_s,
            f"{self.prefix}.cross_node_kv_moves": float(run.cross_node_kv_moves),
            f"{self.prefix}.cross_node_collectives": float(run.cross_node_collectives),
            f"{self.prefix}.prefill_collective_ms": run.prefill_collective_ms,
            f"{self.prefix}.decode_collective_ms": run.decode_collective_ms,
            f"{self.prefix}.remote_expert_ms": run.remote_expert_ms,
        }
        # Capture simulation metrics as tensor for verification
        self._output_values = [
            ttft_p50, ttft_p95, decode_p50, decode_p95, tput_tokens_s,
            float(run.cross_node_kv_moves), float(run.cross_node_collectives),
        ]
        self.output = None

    def capture_verification_payload(self) -> None:
        if not self._output_values:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self.output = torch.tensor(self._output_values, dtype=torch.float32)
        self._set_verification_payload(
            inputs={"config": self._verify_cfg},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        previous = getattr(self, "_previous_default_dtype", None)
        if isinstance(previous, torch.dtype):
            torch.set_default_dtype(previous)
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=60,
            timeout_multiplier=2.0,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics for inference_placement."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=int(getattr(self, '_total_tokens', self.cfg.batch_size)),
            total_requests=int(getattr(self, '_total_requests', self.sessions)),
            batch_size=int(getattr(self, 'batch_size', self.cfg.batch_size)),
            max_batch_size=int(getattr(self, 'max_batch_size', self.cfg.batch_size)),
        )


class BaselineInferencePlacementBenchmark(_PlacementBenchmark):
    """Naive placement with cross-node TP/EP and non-sticky decode."""

    def __init__(self) -> None:
        cfg = PlacementConfig(
            prefill_tp_size=8,
            prefill_span_nodes=True,
            decode_tp_size=2,
            decode_span_nodes=True,
            decode_microbatch=16,
            remote_expert_fraction=0.35,
            router_sticky_decode=False,
            kv_transfer_policy="allow_cross_node",
            notes="Cross-node TP/EP, larger decode microbatches, no KV locality.",
        )
        super().__init__(cfg, prefix="placement_baseline")


def get_benchmark() -> BaseBenchmark:
    return BaselineInferencePlacementBenchmark()
