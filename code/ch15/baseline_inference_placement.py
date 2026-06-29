"""Baseline inference placement policy (intentionally cross-node heavy)."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ch15.placement_sim import (  # noqa: E402
    PlacementConfig,
    PlacementSimulator,
    percentiles,
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
        self._summary_keys = (
            f"{prefix}.ttft_p50_ms",
            f"{prefix}.ttft_p95_ms",
            f"{prefix}.decode_p50_ms",
            f"{prefix}.decode_p95_ms",
            f"{prefix}.tokens_per_s_est",
            f"{prefix}.cross_node_kv_moves",
            f"{prefix}.cross_node_collectives",
            f"{prefix}.prefill_collective_ms",
            f"{prefix}.decode_collective_ms",
            f"{prefix}.remote_expert_ms",
        )
        self._summary: Dict[str, float] = {key: 0.0 for key in self._summary_keys}
        self.output = None  # Simulation metrics as tensor
        self._output_values: list[float] = [0.0] * 7
        self._output_tensor: Optional[torch.Tensor] = None
        self._output_values_ready = False
        self.register_workload_metadata(requests_per_iteration=float(self.sessions))
        self._verify_cfg = torch.tensor(
            [cfg.prefill_tp_size, cfg.decode_tp_size, cfg.decode_microbatch],
            dtype=torch.int64,
        )

    def setup(self) -> None:
        self._previous_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.cfg.dtype)  # type: ignore[arg-type]
        self._output_tensor = torch.empty(len(self._output_values), dtype=torch.float32)
        self._output_values_ready = False

    def benchmark_fn(self) -> None:
        run = self.simulator.simulate(self.cfg, sessions=self.sessions, seed=17)
        ttft_p50, ttft_p95 = percentiles(run.ttft_ms, (50, 95))
        decode_p50, decode_p95 = percentiles(run.decode_ms, (50, 95))
        total_ms = run.ttft_total_ms + run.decode_total_ms
        tput_tokens_s = run.tokens_processed / max(total_ms / 1000.0, 1e-6)
        self._total_tokens = int(run.tokens_processed)
        self._total_requests = int(run.sessions)

        summary = self._summary
        keys = self._summary_keys
        summary[keys[0]] = ttft_p50
        summary[keys[1]] = ttft_p95
        summary[keys[2]] = decode_p50
        summary[keys[3]] = decode_p95
        summary[keys[4]] = tput_tokens_s
        summary[keys[5]] = float(run.cross_node_kv_moves)
        summary[keys[6]] = float(run.cross_node_collectives)
        summary[keys[7]] = run.prefill_collective_ms
        summary[keys[8]] = run.decode_collective_ms
        summary[keys[9]] = run.remote_expert_ms
        # Capture simulation metrics as tensor for verification
        self._output_values[0] = ttft_p50
        self._output_values[1] = ttft_p95
        self._output_values[2] = decode_p50
        self._output_values[3] = decode_p95
        self._output_values[4] = tput_tokens_s
        self._output_values[5] = float(run.cross_node_kv_moves)
        self._output_values[6] = float(run.cross_node_collectives)
        self._output_values_ready = True
        self.output = None

    def capture_verification_payload(self) -> None:
        if not self._output_values_ready:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._output_tensor is None:
            raise RuntimeError("setup() must initialize verification tensors")
        for idx, value in enumerate(self._output_values):
            self._output_tensor[idx] = value
        self.output = self._output_tensor
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
        self._output_tensor = None
        self._output_values_ready = False
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
