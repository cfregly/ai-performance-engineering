"""Baseline runtime scheduling benchmark (sequential prep + per-token streaming).

Models host overhead in vLLM-like serving without async scheduling or stream interval.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, Dict

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from ch16.runtime_scheduler_common import RuntimeSchedulerWorkload, SchedulerScenario


class BaselineRuntimeSchedulerBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: CPU prep + GPU compute + per-token streaming, no overlap."""

    allowed_benchmark_fn_antipatterns = ("sync",)

    def __init__(self) -> None:
        super().__init__()
        self.output: Optional[torch.Tensor] = None
        self.workload: Optional[RuntimeSchedulerWorkload] = None
        self.scenarios: Tuple[SchedulerScenario, ...] = ()
        self._custom_metrics: Dict[str, float] = {}
        self._enable_nvtx = False
        self._verification_dummy: Optional[torch.Tensor] = None
        self._verification_output_buffer: Optional[torch.Tensor] = None

        # Pareto-like scenarios: latency-focused and throughput-focused.
        self.scenarios = (
            SchedulerScenario(
                name="latency",
                concurrency=1,
                decode_steps=16,
                tokens_per_step=16,
                stream_interval=1,
                matmul_dim=1024,
            ),
            SchedulerScenario(
                name="throughput",
                concurrency=32,
                decode_steps=16,
                tokens_per_step=16,
                stream_interval=1,
                matmul_dim=1536,
            ),
        )
        tokens = sum(s.concurrency * s.decode_steps * s.tokens_per_step for s in self.scenarios)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(sum(s.concurrency for s in self.scenarios)),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration or 0.0,
            tokens_per_iteration=self._workload.tokens_per_iteration or 0.0,
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # Reduce CPU variance for deterministic prep cost.
        torch.set_num_threads(1)
        self.workload = RuntimeSchedulerWorkload(self.device, self.scenarios)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self._verification_dummy = torch.zeros(1, device=self.device)
        max_dim = max(scenario.matmul_dim for scenario in self.scenarios)
        self._verification_output_buffer = torch.empty(
            max_dim,
            max_dim,
            device=self.device,
            dtype=torch.float32,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _run_scenario(self, scenario: SchedulerScenario) -> float:
        if self.workload is None:
            raise RuntimeError("Workload not initialized")
        total_tokens = scenario.concurrency * scenario.decode_steps * scenario.tokens_per_step
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(scenario.decode_steps):
                _ = self.workload.cpu_prepare()
                self.output = self.workload.gpu_compute(scenario)
                torch.cuda.synchronize(self.device)
                # Per-token streaming: send every token individually.
                for _ in range(scenario.concurrency * scenario.tokens_per_step):
                    self.workload.stream_send(1)
        end = time.perf_counter()
        elapsed = max(end - start, 1e-9)
        self._custom_metrics[f"{scenario.name}.tps_per_gpu"] = total_tokens / elapsed
        self._custom_metrics[f"{scenario.name}.tps_per_user"] = (
            (scenario.decode_steps * scenario.tokens_per_step) / elapsed
        )
        self._custom_metrics[f"{scenario.name}.elapsed_ms"] = elapsed * 1000.0
        return elapsed

    def benchmark_fn(self) -> None:
        with nvtx_range("runtime_scheduler_baseline", enable=self._enable_nvtx):
            for scenario in self.scenarios:
                self._run_scenario(scenario)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if (
            self.output is None
            or self._verification_dummy is None
            or self._verification_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() did not produce output")
        verify_output = self._verification_output_buffer[
            : self.output.shape[0],
            : self.output.shape[1],
        ]
        verify_output.copy_(self.output)
        self._set_verification_payload(
            inputs={"dummy": self._verification_dummy},
            output=verify_output,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return self._custom_metrics

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)


def get_benchmark() -> BaseBenchmark:
    return BaselineRuntimeSchedulerBenchmark()
