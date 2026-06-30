"""Shared benchmarking helpers for the MoE parallelism lab."""

from __future__ import annotations

from typing import List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

# Try to import plan module - may not be available in all builds
_PLAN_AVAILABLE = False
_PLAN_ERROR: Optional[str] = None

try:
    from .plan import (  # noqa: E402
        ClusterSpec,
        DEFAULT_CLUSTER,
        DEFAULT_MODEL,
        ModelSpec,
        get_default_cluster_spec,
        get_default_model_spec,
        ParallelismPlan,
        PlanEvaluator,
        PlanReport,
        format_report,
    )
    _PLAN_AVAILABLE = True
except ModuleNotFoundError as exc:
    _PLAN_ERROR = f"moe_parallelism plan module unavailable: {exc}"
    # Define stubs so the module can import
    DEFAULT_CLUSTER = None
    DEFAULT_MODEL = None
    ClusterSpec = None  # type: ignore
    ModelSpec = None  # type: ignore
    def get_default_cluster_spec(): return None
    def get_default_model_spec(): return None
    ParallelismPlan = None  # type: ignore
    PlanEvaluator = None  # type: ignore
    PlanReport = None  # type: ignore
    def format_report(*args, **kwargs): return ""


def is_plan_available() -> bool:
    """Check if the plan module is available."""
    return _PLAN_AVAILABLE


def get_plan_error() -> Optional[str]:
    """Get the error message if plan is not available."""
    return _PLAN_ERROR


class _SkipPlanBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Skip benchmark when plan module is not available."""
    
    def __init__(self) -> None:
        super().__init__()
        self.metrics = torch.randn((1, 1), dtype=torch.float32)
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)
        self.verification_not_applicable_reason = _PLAN_ERROR or "moe_parallelism plan module unavailable"

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5)
    
    def benchmark_fn(self) -> None:
        raise RuntimeError(f"SKIPPED: {_PLAN_ERROR or 'moe_parallelism plan module unavailable'}")


def get_skip_benchmark() -> BaseBenchmark:
    """Return a skip benchmark for use when plan is unavailable."""
    return _SkipPlanBenchmark()


class PlanBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """CPU-only benchmark that emits a plan analysis report."""

    def __init__(
        self,
        plan: ParallelismPlan,
        cluster: Optional[ClusterSpec] = None,
        model: Optional[ModelSpec] = None,
    ) -> None:
        self._override_device = torch.device("cpu")
        super().__init__()
        resolved_cluster = cluster
        resolved_model = model
        if resolved_cluster is None or resolved_model is None:
            default_cluster = get_default_cluster_spec()
            default_model = get_default_model_spec()
            resolved_cluster = resolved_cluster or default_cluster
            resolved_model = resolved_model or default_model
        self.plan = plan
        self.cluster = resolved_cluster
        self.model = resolved_model
        self.evaluator = PlanEvaluator(resolved_cluster, resolved_model)
        self.report: Optional[PlanReport] = None
        self._summary: Optional[str] = None
        self._config: Optional[BenchmarkConfig] = None
        self.metrics: Optional[torch.Tensor] = None
        self._summary_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def _resolve_device(self) -> torch.device:  # type: ignore[override]
        return self._override_device

    def setup(self) -> None:
        torch.manual_seed(42)
        self.report = None
        self._summary = None
        if self.metrics is None or tuple(self.metrics.shape) != (1, 3):
            self.metrics = torch.zeros((1, 3), dtype=torch.float32)
        if self._summary_buffer is None or tuple(self._summary_buffer.shape) != (1, 3):
            self._summary_buffer = torch.empty((1, 3), dtype=torch.float32)

    def benchmark_fn(self) -> None:
        report = self.evaluator.analyze(self.plan)
        self.report = report
        self._summary = format_report(report)
        self._finalize_output(
            [
                float(report.estimated_step_ms),
                float(report.throughput_tokens_per_s),
                float(report.memory_per_device_gb),
            ]
        )
        if self.output is None or self.metrics is None:
            raise RuntimeError("Failed to produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.metrics is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"metrics": self.metrics},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.report = None
        self._summary = None
        self.metrics = None
        self._summary_buffer = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        if self._config is None:
            config = BenchmarkConfig(iterations=1, warmup=5)
            self._config = config
        return self._config

    def validate_result(self) -> Optional[str]:
        if self.report is None:
            return "Plan analysis missing"
        return None

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        if self.report is None:
            return None
        throughput = max(self.report.throughput_tokens_per_s, 0.0)
        return WorkloadMetadata(
            requests_per_iteration=throughput,
            tokens_per_iteration=throughput,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return MoE parallelism plan metrics."""
        if self.report is None:
            return None
        return {
            "moe.estimated_step_ms": self.report.estimated_step_ms,
            "moe.throughput_tokens_per_s": self.report.throughput_tokens_per_s,
            "moe.memory_per_device_gb": self.report.memory_per_device_gb,
        }

    def print_summary(self) -> None:
        if self._summary:
            print(self._summary)

    def _finalize_output(self, metric_values: List[float]) -> None:
        expected_shape = (1, len(metric_values))
        if (
            self.metrics is None
            or self._summary_buffer is None
            or tuple(self.metrics.shape) != expected_shape
            or tuple(self._summary_buffer.shape) != expected_shape
        ):
            raise RuntimeError("setup() must preallocate metric buffers for PlanBenchmark")
        for index, value in enumerate(metric_values):
            self._summary_buffer[0, index] = float(value)
        self.output = self._summary_buffer


def run_benchmark(benchmark: PlanBenchmark) -> None:
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    # Re-run the analysis in-process so we can show the report even if the harness
    # executed the benchmark in a worker thread/subprocess.
    report = benchmark.evaluator.analyze(benchmark.plan)
    print(format_report(report))
    timing_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"Synthetic step time (from analysis): {report.estimated_step_ms:.1f} ms")
    print(f"Projected throughput: {report.throughput_tokens_per_s:,.0f} tokens/s")
    print(f"Harness runtime to generate report: {timing_ms:.4f} ms")
