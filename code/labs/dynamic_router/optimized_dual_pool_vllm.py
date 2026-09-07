"""Optimized vLLM dual-pool benchmark: dedicated prefill and decode pools."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router import vllm_runner
from labs.dynamic_router.topology import detect_topology
from labs.dynamic_router.verification import (
    metric_row_buffer,
    numeric_metric_values,
    require_verification_output,
    scalar_int_buffer,
)
from labs.dynamic_router.vllm_runner import run_dual_pool_vllm_with_topology


class OptimizedDualPoolVllmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs vLLM with disaggregated prefill and decode pools to cut TTFT tails."""

    multi_gpu_required = True
    _is_deterministic = True
    input_jitter_bounds = {"prompt_token_ids": (0, 2)}

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.output: Optional[torch.Tensor] = None
        self._metric_values: Optional[list[float]] = None
        self._metric_output_buffer: Optional[torch.Tensor] = None
        self._mode_input: Optional[torch.Tensor] = None
        self._prompt_token_ids: Optional[torch.Tensor] = None
        self._prompt_lengths = vllm_runner.dual_pool_prompt_lengths(
            vllm_runner._CLI_ARGS
        )
        self._topology = None
        self._summary_ready = False
        request_count = len(self._prompt_lengths)
        self.register_workload_metadata(
            requests_per_iteration=float(request_count),
            tokens_per_iteration=float(
                sum(self._prompt_lengths)
                + request_count * max(1, vllm_runner._CLI_ARGS.max_tokens)
            ),
        )

    def setup(self) -> None:
        self._mode_input = scalar_int_buffer(self, "_mode_input", 1)
        self._prompt_token_ids = vllm_runner.build_prompt_token_ids(
            self._prompt_lengths
        )
        self._topology = detect_topology(max_gpus=torch.cuda.device_count())

    def benchmark_fn(self) -> None:
        if self._mode_input is None or int(self._mode_input[0]) != 1:
            raise RuntimeError("setup() must initialize dual-pool routing mode")
        if self._prompt_token_ids is None:
            raise RuntimeError("setup() must initialize live prompt-token input")
        self._summary = run_dual_pool_vllm_with_topology(
            "dual",
            topology_snapshot=self._topology,
            cli_args=vllm_runner._CLI_ARGS,
            prompt_token_ids=self._prompt_token_ids,
        )
        self._summary_ready = True

    def capture_verification_payload(self) -> None:
        if not self._summary_ready:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._prompt_token_ids is None:
            raise RuntimeError("setup() must initialize live prompt-token input")
        require_verification_output(self._summary)
        metric_values = numeric_metric_values(self._summary, self._metric_values)
        self._metric_values = metric_values
        self.output = metric_row_buffer(self, metric_values)
        self._set_verification_payload(
            inputs={
                "prompt_token_ids": self._prompt_token_ids,
                "mode": scalar_int_buffer(self, "_mode_input", 1),
            },  # dual
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.output = None
        self._metric_values = None
        self._metric_output_buffer = None
        self._mode_input = None
        self._prompt_token_ids = None
        self._topology = None
        self._summary_ready = False
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5, multi_gpu_required=True)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDualPoolVllmBenchmark()
