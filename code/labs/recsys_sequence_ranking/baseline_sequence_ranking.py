"""Baseline session-ranking path with conservative sparse lookups and scoring."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.recsys_sequence_ranking.recsys_sequence_ranking_common import (
    RankingInputs,
    RankingModelState,
    RankingWorkspace,
    apply_cli_overrides,
    baseline_forward,
    build_inputs,
    build_model_state,
    build_workspace,
    default_workload,
    ranking_metrics,
    prepare_workspace_for_inputs,
    requests_per_iteration,
    tokens_per_iteration,
)


class BaselineSequenceRankingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline sequence-ranking path with explicit per-step sparse work."""

    preferred_ncu_replay_mode = "application"

    def __init__(self) -> None:
        super().__init__()
        self.workload = default_workload()
        self.inputs: Optional[RankingInputs] = None
        self.state: Optional[RankingModelState] = None
        self.workspace: Optional[RankingWorkspace] = None
        self.output: Optional[torch.Tensor] = None
        self._custom_metrics: dict[str, float] = {}
        self._verification_sequence_mask: Optional[torch.Tensor] = None
        self._refresh_workload_metadata()

    def _refresh_workload_metadata(self) -> None:
        self._workload = WorkloadMetadata(
            requests_per_iteration=requests_per_iteration(self.workload),
            tokens_per_iteration=tokens_per_iteration(self.workload),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.recsys_sequence_ranking requires CUDA for fair comparison")
        self.inputs = build_inputs(self.workload, self.device)
        self.state = build_model_state(self.workload, self.device)
        self.workspace = build_workspace(self.workload, self.device)
        prepare_workspace_for_inputs(self.inputs, self.workspace)
        self._verification_sequence_mask = self.inputs.sequence_mask.to(torch.int32)
        self.output = None
        self._custom_metrics = ranking_metrics(
            self.workload,
            self.inputs,
            score_backend="torch",
            compile_enabled=False,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.state is None or self.workspace is None:
            raise RuntimeError("Benchmark state is not initialized")
        with torch.inference_mode():
            self.output = baseline_forward(self.inputs, self.state, self.workspace)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if (
            self.inputs is None
            or self.output is None
            or self.state is None
            or self._verification_sequence_mask is None
        ):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={
                "sequence_ids": self.inputs.sequence_ids,
                "sequence_mask": self._verification_sequence_mask,
                "sequence_lengths": self.inputs.sequence_lengths,
                "context_ids": self.inputs.context_ids,
                "candidate_ids": self.inputs.candidate_ids,
            },
            output=self.output,
            batch_size=self.workload.batch_size,
            parameter_count=self.state.parameter_count,
            precision_flags={"fp16": False, "bf16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        self.inputs = None
        self.state = None
        self.workspace = None
        self.output = None
        self._verification_sequence_mask = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            measurement_timeout_seconds=90,
            setup_timeout_seconds=90,
            profiling_warmup=0,
            profiling_iterations=1,
            ncu_replay_mode="application",
            nsys_nvtx_include=["compute_kernel:profile"],
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._custom_metrics)

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.shape != (self.workload.batch_size, self.workload.num_candidates):
            return "Unexpected output shape"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None

    def apply_target_overrides(self, argv: list[str]) -> None:
        self.workload = apply_cli_overrides(self.workload, argv)
        self._refresh_workload_metadata()


def get_benchmark() -> BaseBenchmark:
    return BaselineSequenceRankingBenchmark()
