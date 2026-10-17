"""Follow-up regressions for audit fixes on benchmark pairs and their siblings.

Covers the NanoChat side-stream check (W1-036), the GB300 software pipelining
status note (W2-091), the ch13 precisionmixed token unit (follow-up to W2-115)
and the ch04 naive NVLink DDP loop (W2-043). These tests run on the CPU. They
check metadata, update math and documentation against the code. They are not
GPU performance evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

import ch04.ddp_nvlink_naive as ddp_nvlink_naive
from ch13.baseline_precisionmixed import BaselinePrecisionMixedBenchmark
from ch13.optimized_precisionmixed import OptimizedPrecisionMixedBenchmark
from core.benchmark.models import TimingStats
from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verification import compare_workload_metrics
from core.benchmark.verify_runner import VerifyRunner
from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode

CODE_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# W1-036 follow-up: the NanoChat side-stream parity check holds compile off
# ---------------------------------------------------------------------------


class _StopAfterEngine(Exception):
    """Ends the lab check right after it builds its real Engine."""


def test_nanochat_side_stream_check_builds_an_uncompiled_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import labs.nanochat_fullstack.test_new_optimizations as lab_checks

    built = []

    class RecordingEngine(lab_checks.Engine):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)
            raise _StopAfterEngine

    monkeypatch.delenv("NANOCHAT_DISABLE_COMPILE", raising=False)
    # Build the check's model on the CPU. The Engine constructor and its compile
    # gate run for real. On Blackwell, a compiled Engine reports the decode mode
    # side_stream_torch_compile and fails the check's side_stream assertion.
    monkeypatch.setattr(lab_checks, "_require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(lab_checks, "Engine", RecordingEngine)

    for config in (
        lab_checks._config(enable_persistent_decode=True),
        lab_checks._config(
            use_flash_attn_varlen=True, enable_persistent_decode=True
        ),
    ):
        with pytest.raises(_StopAfterEngine):
            lab_checks._compare_decode(config)

    assert len(built) == 2
    for engine in built:
        assert engine.allow_compile is False
        assert engine.compile_status == "disabled"
        assert engine.compile_status_reason == "Engine(allow_compile=False)"


# ---------------------------------------------------------------------------
# W2-091 follow-up: the GB300 pipelining doc marks its TMA numbers as history
# ---------------------------------------------------------------------------


def test_gb300_pipelining_doc_status_note_matches_the_shipped_lab() -> None:
    doc = (CODE_ROOT / "docs" / "gb300-software-pipelining.md").read_text(encoding="utf-8")
    lines = doc.splitlines()
    assert lines[0].startswith("# GB300 software_pipelining lab")
    verdict_index = next(i for i, line in enumerate(lines) if line.startswith("**Verdict"))
    note_lines = [line for line in lines[1:verdict_index] if line.startswith(">")]
    assert note_lines, "the status note must sit between the title and the verdict"
    note = " ".join(line.lstrip("> ").strip() for line in note_lines)

    assert note.startswith("**Status (2026-10-12, W2-091):**")
    assert "They are historical and do not describe the current lab." in note
    assert ";" not in note

    kernels = (
        CODE_ROOT / "labs" / "software_pipelining" / "software_pipelining_kernels.cu"
    ).read_text(encoding="utf-8")
    assert "optimized_tile_pipeline_tma_kernel" not in kernels
    assert "`optimized_tile_pipeline_tma_kernel` and its" in note
    block_threads = int(re.search(r"constexpr int kBlockThreads = (\d+);", kernels).group(1))
    elems_per_thread = int(re.search(r"constexpr int kElemsPerThread = (\d+);", kernels).group(1))
    assert f"tile size ({block_threads * elems_per_thread} floats)" in note
    assert f"block size ({block_threads} threads)" in note
    assert "constexpr int kPipelineStages = 2;" in kernels
    assert "two-stage `cuda::pipeline`" in note

    notes_source = (
        CODE_ROOT / "labs" / "software_pipelining" / "optimized_tile_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "TMA" not in notes_source
    assert "TMA notes string" in note


# ---------------------------------------------------------------------------
# W2-115 follow-up: ch13 precisionmixed counts timed input rows as tokens
# ---------------------------------------------------------------------------


def test_ch13_precisionmixed_counts_timed_input_rows_as_tokens() -> None:
    benchmark = BaselinePrecisionMixedBenchmark()
    workload = benchmark.get_workload_metadata()
    registered = benchmark._workload_metadata
    benchmark.device = torch.device("cpu")
    benchmark.setup()
    try:
        inputs = benchmark.inputs
        assert isinstance(inputs, torch.Tensor)
        rows, hidden = inputs.shape
        # benchmark_fn runs one forward and backward pass over every row per micro step.
        passes = len(benchmark._micro_step_range)
    finally:
        benchmark.teardown()

    assert hidden > 1
    assert passes > 1
    assert workload.tokens_per_iteration == float(rows * passes)
    assert workload.tokens_per_iteration != float(rows * hidden * passes)
    assert workload.requests_per_iteration == float(passes)
    assert registered.tokens_per_iteration == workload.tokens_per_iteration
    assert registered.requests_per_iteration == workload.requests_per_iteration


def test_ch13_precisionmixed_pair_workload_metrics_pass_verify_gate(tmp_path: Path) -> None:
    runner = VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path / "quarantine"),
    )
    baseline_metrics = runner._extract_workload_metrics(BaselinePrecisionMixedBenchmark())
    optimized_metrics = runner._extract_workload_metrics(OptimizedPrecisionMixedBenchmark())

    assert baseline_metrics == optimized_metrics == {
        "requests_per_iteration": 4.0,
        "tokens_per_iteration": 2048.0,
    }
    matched, deltas = compare_workload_metrics(baseline_metrics, optimized_metrics)
    assert matched, deltas


@pytest.mark.parametrize(
    "filename", ("expectations_b200.json", "expectations_2x_b200.json", "expectations_4x_gb200.json")
)
def test_ch13_stored_precisionmixed_tokens_per_s_uses_the_row_unit(filename: str) -> None:
    metrics = json.loads((CODE_ROOT / "ch13" / filename).read_text(encoding="utf-8"))["examples"][
        "precisionmixed"
    ]["metrics"]
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(device=torch.device("cpu"))
    )
    for prefix, factory in (
        ("baseline_throughput", BaselinePrecisionMixedBenchmark),
        ("best_optimized_throughput", OptimizedPrecisionMixedBenchmark),
    ):
        latency_ms = metrics[f"{prefix}.latency_ms"]
        timing = TimingStats(
            mean_ms=latency_ms,
            median_ms=latency_ms,
            std_ms=0.0,
            min_ms=latency_ms,
            max_ms=latency_ms,
            iterations=1,
            warmup_iterations=0,
        )
        # The harness's own throughput path, fed the stored mean time.
        stats = harness._compute_throughput_stats(
            timing, harness._resolve_workload_metadata(factory())
        )
        # The timing fields were not touched, so the stored request rate still matches.
        assert stats.requests_per_s == pytest.approx(metrics[f"{prefix}.requests_per_s"], rel=1e-12)
        assert stats.tokens_per_s == pytest.approx(metrics[f"{prefix}.tokens_per_s"], rel=1e-12)


# ---------------------------------------------------------------------------
# W2-043 follow-up: the naive NVLink DDP loop uses the overlap loop's update math
# ---------------------------------------------------------------------------


def test_ddp_nvlink_naive_docstring_matches_the_multi_gpu_skip_contract() -> None:
    doc = " ".join((ddp_nvlink_naive.__doc__ or "").split())
    assert "Falls back to single-GPU" not in doc
    assert "Requires at least two GPUs. setup() raises a SKIPPED: diagnostic otherwise." in doc
    assert "the same update math as ddp_nvlink_overlap.py" in doc

    source = Path(ddp_nvlink_naive.__file__).read_text(encoding="utf-8")
    assert "if self._model_count == 1:" not in source

    bench = ddp_nvlink_naive.get_benchmark()
    if torch.cuda.device_count() < 2:
        with pytest.raises(RuntimeError, match=r"^SKIPPED: "):
            bench.setup()
        assert bench.models == []


def _replica_weight_grads(
    weights: list[torch.Tensor], biases: list[torch.Tensor], micro_inputs: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Weight gradients of mean(square(x W^T + b)) for each replica, in float64."""
    grads = []
    for idx, x in enumerate(micro_inputs):
        weight = weights[idx].clone().requires_grad_(True)
        y = x.detach().double().cpu() @ weight.T + biases[idx]
        y.square().mean().backward()
        grads.append(weight.grad)
    return grads


def test_ddp_nvlink_naive_applies_one_replica_averaged_update_per_iteration() -> None:
    bench = ddp_nvlink_naive.get_benchmark()
    bench.device = torch.device("cpu")
    # setup() checks for two GPUs and then calls this helper with one cuda device
    # per replica. Two CPU replicas run the same update path on any host.
    bench._build_replicas(["cpu", "cpu"])
    try:
        # Scaling the inputs makes the second-order gap between one update and one
        # step per microbatch about 1000x larger than FP32 rounding.
        for micro_inputs in bench._inputs:
            for x in micro_inputs:
                x.mul_(16.0)
        weights = [model.weight.detach().double().clone() for model in bench.models]
        biases = [model.bias.detach().double().clone() for model in bench.models]
        replicas = len(bench.models)

        summed = torch.zeros_like(weights[0])
        for micro_inputs in bench._inputs:
            for grad in _replica_weight_grads(weights, biases, micro_inputs):
                summed += grad
        expected_delta = -1e-3 * summed / replicas

        # The old loop stepped after every microbatch. Prove this test can tell them apart.
        stepped = [weight.clone() for weight in weights]
        for micro_inputs in bench._inputs:
            average = sum(_replica_weight_grads(stepped, biases, micro_inputs)) / replicas
            stepped = [weight - 1e-3 * average for weight in stepped]
        assert float((stepped[0] - weights[0] - expected_delta).abs().max()) > 1e-6

        versions = [model.weight._version for model in bench.models]
        bench.benchmark_fn()
        for idx, model in enumerate(bench.models):
            delta = model.weight.detach().double() - weights[idx]
            torch.testing.assert_close(delta, expected_delta, rtol=0.0, atol=1e-7)
        # Exactly one in-place weight update per replica per iteration.
        assert [model.weight._version - before for model, before in zip(bench.models, versions)] == [
            1
        ] * replicas
        bench.capture_verification_payload()
    finally:
        bench.teardown()
    assert bench._model_update_groups == []
