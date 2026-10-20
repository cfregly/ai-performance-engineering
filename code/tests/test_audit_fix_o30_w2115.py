"""W2-115: ch20 workload metadata must count timed input rows as tokens.

These CPU tests run the real benchmark setup, the real verification helpers
and the stored expectation files. They are metadata and control-plane checks,
not GPU evidence.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from ch20.baseline_autotuning import BaselineAutotuningBenchmark
from ch20.baseline_bf16_mlp import BaselineBF16MLPBenchmark
from ch20.baseline_moe import BaselineMoeBenchmark
from ch20.baseline_training_single import BaselineTrainingSingleBenchmark
from ch20.optimized_autotuning import OptimizedAutotuningBenchmark
from ch20.optimized_bf16_mlp import OptimizedBF16MLPBenchmark
from ch20.optimized_moe import OptimizedMoeBenchmark
from ch20.optimized_training_single import OptimizedTrainingSingleBenchmark
from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verification import compare_workload_metrics
from core.benchmark.verify_runner import VerifyRunner

CH20_ROOT = Path(__file__).resolve().parents[1] / "ch20"

BASELINE_TIMED_INPUTS = [
    pytest.param(BaselineBF16MLPBenchmark, "x", None, id="bf16_mlp"),
    pytest.param(BaselineMoeBenchmark, "inputs", None, id="moe"),
    pytest.param(BaselineTrainingSingleBenchmark, "inputs", "_train_step_range", id="training_single"),
    pytest.param(BaselineAutotuningBenchmark, "inputs", None, id="autotuning"),
]

PAIRS = [
    pytest.param(BaselineBF16MLPBenchmark, OptimizedBF16MLPBenchmark, id="bf16_mlp"),
    pytest.param(BaselineMoeBenchmark, OptimizedMoeBenchmark, id="moe"),
    pytest.param(BaselineTrainingSingleBenchmark, OptimizedTrainingSingleBenchmark, id="training_single"),
    pytest.param(BaselineAutotuningBenchmark, OptimizedAutotuningBenchmark, id="autotuning"),
]

# ch20:moe has no stored expectation entry. test_wave2_low_111_123_regressions.py checks that.
STORED_EXPECTATIONS = [
    pytest.param(filename, example, baseline_factory, optimized_factory, id=f"{filename}:{example}")
    for filename in ("expectations_b200.json", "expectations_4x_gb200.json")
    for example, baseline_factory, optimized_factory in (
        ("bf16_mlp", BaselineBF16MLPBenchmark, OptimizedBF16MLPBenchmark),
        ("training_single", BaselineTrainingSingleBenchmark, OptimizedTrainingSingleBenchmark),
        ("autotuning", BaselineAutotuningBenchmark, OptimizedAutotuningBenchmark),
    )
]


@pytest.mark.parametrize(("factory", "input_attr", "step_range_attr"), BASELINE_TIMED_INPUTS)
def test_w2_115_tokens_per_iteration_matches_timed_input_rows(
    factory: Callable[[], object],
    input_attr: str,
    step_range_attr: str | None,
) -> None:
    benchmark = factory()
    workload = benchmark.get_workload_metadata()
    benchmark.device = torch.device("cpu")
    benchmark.setup()
    try:
        inputs = getattr(benchmark, input_attr)
        assert isinstance(inputs, torch.Tensor)
        # benchmark_fn runs one pass over the input per entry in this range.
        passes = 1 if step_range_attr is None else len(getattr(benchmark, step_range_attr))
    finally:
        benchmark.teardown()

    assert inputs.dim() == 2
    rows, hidden = inputs.shape
    assert hidden > 1
    assert passes >= 1
    assert workload.tokens_per_iteration == float(rows * passes)
    assert workload.tokens_per_iteration != float(rows * hidden * passes)


@pytest.mark.parametrize(("baseline_factory", "optimized_factory"), PAIRS)
def test_w2_115_pair_workload_metrics_pass_verify_gate(
    tmp_path: Path,
    baseline_factory: Callable[[], object],
    optimized_factory: Callable[[], object],
) -> None:
    runner = VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path / "quarantine"),
    )
    baseline_metrics = runner._extract_workload_metrics(baseline_factory())
    optimized_metrics = runner._extract_workload_metrics(optimized_factory())

    assert baseline_metrics["tokens_per_iteration"] == optimized_metrics["tokens_per_iteration"]
    matched, deltas = compare_workload_metrics(baseline_metrics, optimized_metrics)
    assert matched, deltas


@pytest.mark.parametrize(("filename", "example", "baseline_factory", "optimized_factory"), STORED_EXPECTATIONS)
def test_w2_115_stored_tokens_per_s_uses_current_token_unit(
    filename: str,
    example: str,
    baseline_factory: Callable[[], object],
    optimized_factory: Callable[[], object],
) -> None:
    # The harness divides tokens and requests per iteration by the same mean time,
    # so a stored entry in the current unit keeps their ratio.
    metrics = json.loads((CH20_ROOT / filename).read_text(encoding="utf-8"))["examples"][example]["metrics"]
    for prefix, factory in (
        ("baseline_throughput", baseline_factory),
        ("best_optimized_throughput", optimized_factory),
    ):
        workload = factory().get_workload_metadata()
        expected_ratio = workload.tokens_per_iteration / workload.requests_per_iteration
        stored_ratio = metrics[f"{prefix}.tokens_per_s"] / metrics[f"{prefix}.requests_per_s"]
        assert stored_ratio == pytest.approx(expected_ratio, rel=1e-9), prefix
