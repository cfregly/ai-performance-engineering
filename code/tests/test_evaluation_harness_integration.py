"""Real CPU lifecycle coverage for opt-in evaluation provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from core.benchmark.evaluation_provenance import (
    DatasetDeclaration,
    EvaluationContract,
    EvaluatorDeclaration,
    FeatureLineage,
)
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    ExecutionMode,
)


class EvaluationLifecycleBenchmark(BaseBenchmark):
    """Importable CPU benchmark used by both thread and isolated-process tests."""

    allow_cpu = True
    allowed_benchmark_fn_antipatterns = ("io",)

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.protected_path = ""
        self.mutation_phase = "none"
        self.failure_phase = "none"
        self.threshold = 0.5
        self.mutation_done = False
        self.benchmark_calls = 0
        self.input_tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def _mutate_if_requested(self, phase: str) -> None:
        if self.mutation_done or not self.mutation_phase.startswith(phase):
            return
        if self.mutation_phase.endswith("source"):
            Path(self.protected_path).write_text("changed during evaluation\n", encoding="utf-8")
        elif self.mutation_phase.endswith("threshold"):
            self.threshold = 0.75
        else:  # pragma: no cover - test fixture configuration guard
            raise AssertionError(f"unknown mutation: {self.mutation_phase}")
        self.mutation_done = True

    def setup(self) -> None:
        self.input_tensor = torch.tensor([1.0], device=self.device)
        self._mutate_if_requested("setup")
        if self.failure_phase == "setup":
            raise RuntimeError("intentional setup failure")

    def benchmark_fn(self) -> torch.Tensor:
        if self.input_tensor is None:
            raise RuntimeError("setup() must initialize input_tensor")
        self.benchmark_calls += 1
        if self.benchmark_calls > 5:
            self._mutate_if_requested("timing")
            if self.failure_phase == "timing":
                raise RuntimeError("intentional timing failure")
        self.output = self.input_tensor + 1.0
        return self.output

    def teardown(self) -> None:
        self._mutate_if_requested("teardown")
        if self.failure_phase == "teardown":
            raise RuntimeError("intentional teardown failure")

    def validate_result(self) -> Optional[str]:
        return None

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self.input_tensor is None:
            raise RuntimeError("setup() must initialize input_tensor")
        return {"input": self.input_tensor}

    def get_verify_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_input_signature(self) -> dict[str, object]:
        return {"shape": (1,), "dtype": "torch.float32"}

    def get_evaluation_contract(self) -> Optional[EvaluationContract]:
        if not self.protected_path:
            return None
        return EvaluationContract(
            dataset=DatasetDeclaration(
                name="cpu-fixture",
                version="1",
                content_sha256="0" * 64,
            ),
            sample_ids_by_split={
                "train": ("train-1",),
                "test": ("test-1",),
                "holdout": ("holdout-1",),
            },
            evaluator=EvaluatorDeclaration(
                name="accuracy",
                version="1",
                thresholds={"minimum": self.threshold},
            ),
            feature_lineage=(
                FeatureLineage(feature="value", source_fields=("value",)),
            ),
            label_fields=("label",),
            protected_sources=(Path(self.protected_path),),
        )


def _harness(*, subprocess_mode: bool = False) -> BenchmarkHarness:
    execution_mode = ExecutionMode.SUBPROCESS if subprocess_mode else ExecutionMode.THREAD
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=2,
        warmup=5,
        use_subprocess=subprocess_mode,
        execution_mode=execution_mode,
        enable_profiling=False,
        enable_memory_tracking=False,
        enable_cleanup=False,
        lock_gpu_clocks=False,
        enforce_environment_validation=False,
        detect_setup_precomputation=False,
        adaptive_iterations=False,
        clear_compile_cache=False,
        measurement_timeout_seconds=15,
    )
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)


def _benchmark(tmp_path: Path, mutation_phase: str = "none") -> EvaluationLifecycleBenchmark:
    protected = tmp_path / "protected_source.py"
    protected.write_text("stable source\n", encoding="utf-8")
    benchmark = EvaluationLifecycleBenchmark()
    benchmark.protected_path = str(protected)
    benchmark.mutation_phase = mutation_phase
    return benchmark


def test_thread_lifecycle_serializes_pass_receipt_into_result_and_manifest(tmp_path: Path) -> None:
    run = _harness().benchmark_with_manifest(_benchmark(tmp_path), run_id="evaluation-pass")

    assert run.result.evaluation is not None
    assert run.result.evaluation.status == "PASS"
    assert run.result.evaluation.finalized_at is not None
    assert run.manifest is not None
    assert run.manifest.evaluation is not None
    assert run.manifest.evaluation.model_dump(mode="json") == run.result.evaluation.model_dump(mode="json")
    payload = run.model_dump(mode="json")
    assert payload["result"]["evaluation"]["status"] == "PASS"
    assert payload["manifest"]["evaluation"]["status"] == "PASS"


def test_default_benchmark_path_keeps_evaluation_provenance_absent() -> None:
    benchmark = EvaluationLifecycleBenchmark()
    run = _harness().benchmark_with_manifest(benchmark, run_id="evaluation-absent")

    assert run.result.evaluation is None
    assert run.manifest is not None
    assert run.manifest.evaluation is None


def test_setup_source_mutation_rejects_timing_result(tmp_path: Path) -> None:
    result = _harness().benchmark(_benchmark(tmp_path, "setup_source"))

    assert result.evaluation is not None
    assert result.evaluation.status == "FAIL"
    assert result.timing.iterations == 0
    assert "protected_source_mutated" in {failure.code for failure in result.evaluation.failures}


def test_timed_threshold_mutation_rejects_timing_result(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path, "timing_threshold")
    result = _harness().benchmark(benchmark)

    assert benchmark.benchmark_calls > 5
    assert result.evaluation is not None
    assert result.evaluation.status == "FAIL"
    assert result.timing.iterations == 0
    failure_codes = {failure.code for failure in result.evaluation.failures}
    assert {"evaluator_threshold_drift", "evaluation_contract_drift"} <= failure_codes


def test_teardown_source_mutation_rejects_timing_result(tmp_path: Path) -> None:
    result = _harness().benchmark(_benchmark(tmp_path, "teardown_source"))

    assert result.evaluation is not None
    assert result.evaluation.status == "FAIL"
    assert result.timing.iterations == 0
    assert "protected_source_mutated" in {failure.code for failure in result.evaluation.failures}


def test_teardown_failure_still_finalizes_source_mutation_receipt(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path, "teardown_source")
    benchmark.failure_phase = "teardown"
    result = _harness().benchmark(benchmark)

    assert result.evaluation is not None
    assert result.evaluation.finalized_at is not None
    assert result.evaluation.status == "FAIL"
    assert "protected_source_mutated" in {failure.code for failure in result.evaluation.failures}
    assert any("Teardown failed: intentional teardown failure" in error for error in result.errors)


def test_subprocess_worker_detects_drift_and_transports_receipt_to_manifest(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path, "timing_threshold")
    run = _harness(subprocess_mode=True).benchmark_with_manifest(
        benchmark,
        run_id="evaluation-subprocess-fail",
    )

    assert benchmark.threshold == 0.5
    assert run.result.evaluation is not None
    assert run.result.evaluation.status == "FAIL"
    assert run.result.timing.iterations == 0
    failure_codes = {failure.code for failure in run.result.evaluation.failures}
    assert {"evaluator_threshold_drift", "evaluation_contract_drift"} <= failure_codes
    assert run.manifest is not None
    assert run.manifest.evaluation is not None
    assert run.manifest.evaluation.model_dump(mode="json") == run.result.evaluation.model_dump(mode="json")
