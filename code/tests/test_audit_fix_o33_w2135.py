"""Regressions for W2-135: the RecSys README must describe both Triton kernels.

The optimized sequence-ranking path runs a fused Triton kernel for sequence
pooling plus context-table aggregation, and a second Triton kernel for
candidate scoring. `--score-backend torch` replaces only the scoring kernel.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import shlex
from pathlib import Path

import pytest
import torch

import labs.recsys_sequence_ranking.recsys_sequence_ranking_common as ranking_common
from labs.recsys_sequence_ranking.baseline_sequence_ranking import (
    BaselineSequenceRankingBenchmark,
)
from labs.recsys_sequence_ranking.optimized_sequence_ranking import (
    OptimizedSequenceRankingBenchmark,
)
from labs.recsys_sequence_ranking.recsys_sequence_ranking_common import (
    TRITON_AVAILABLE,
    SequenceRankingWorkload,
    apply_cli_overrides,
    build_inputs,
    default_workload,
    ranking_metrics,
)

README_PATH = (
    Path(__file__).resolve().parents[1] / "labs" / "recsys_sequence_ranking" / "README.md"
)
TARGET = "labs/recsys_sequence_ranking:sequence_ranking"
SMALL_SHAPE_ARGS = [
    "--batch-size", "4",
    "--seq-len", "6",
    "--num-tables", "3",
    "--embedding-dim", "16",
    "--hidden-dim", "24",
    "--num-candidates", "8",
    "--item-vocab-size", "128",
    "--context-vocab-size", "64",
    "--min-history-len", "2",
]


def _readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _section(title: str) -> str:
    match = re.search(
        rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)",
        _readme(),
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"README is missing the '## {title}' section"
    return match.group(1)


def _bullet_for_flag(section: str, flag: str) -> str:
    bullets = [line for line in section.splitlines() if line.startswith("- ") and flag in line]
    assert len(bullets) == 1, f"expected one bullet documenting {flag!r}, found {bullets}"
    return bullets[0]


def _changed_fields(argv: list[str]) -> set[str]:
    base = default_workload()
    merged = apply_cli_overrides(base, argv)
    return {
        field.name
        for field in dataclasses.fields(SequenceRankingWorkload)
        if getattr(base, field.name) != getattr(merged, field.name)
    }


def test_summary_and_optimized_path_name_the_fused_triton_pooling_kernel() -> None:
    assert re.search(r"pools .* in one fused Triton kernel", _section("Summary"))

    optimized_bullets = [
        line for line in _section("Optimized Path").splitlines() if line.startswith("- ")
    ]
    assert any(
        "fused Triton kernel" in line and "pooling" in line and "context-table" in line
        for line in optimized_bullets
    ), optimized_bullets

    layout_rows = [
        line
        for line in _section("Directory Layout").splitlines()
        if line.startswith("| `optimized_sequence_ranking.py`")
    ]
    assert len(layout_rows) == 1
    assert "Triton pooling" in layout_rows[0]
    assert "Triton candidate scoring" in layout_rows[0]


def test_score_backend_torch_bullet_does_not_claim_to_isolate_torch_compile() -> None:
    bullet = _bullet_for_flag(_section("Running the Benchmarks"), "--score-backend torch")

    assert "torch.compile" not in bullet
    assert "replaces only the Triton candidate-scoring kernel" in bullet
    assert "Sequence pooling still runs in its fused Triton kernel" in bullet
    assert "no flag turns that kernel off" in bullet


def test_disable_compile_bullet_says_both_triton_kernels_still_run() -> None:
    bullet = _bullet_for_flag(_section("Running the Benchmarks"), "--disable-compile")

    assert "runs the user tower eagerly" in bullet
    assert "Both Triton kernels still run" in bullet


def test_notes_point_to_metrics_the_benchmark_reports() -> None:
    notes = _section("Notes")
    assert "fused Triton kernels for sequence pooling and candidate scoring" in notes
    assert "`ranking.triton_available`" in notes
    assert "`ranking.score_backend_triton`" in notes

    workload = SequenceRankingWorkload(**{**default_workload().__dict__, "batch_size": 2})
    inputs = build_inputs(workload, torch.device("cpu"))
    metrics = ranking_metrics(workload, inputs, score_backend="torch", compile_enabled=False)
    assert metrics["ranking.score_backend_triton"] == 0.0
    assert "ranking.triton_available" in inspect.getsource(OptimizedSequenceRankingBenchmark.setup)


def test_readme_has_no_semicolons_in_prose() -> None:
    prose_lines = []
    in_code_block = False
    for line in _readme().splitlines():
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            prose_lines.append(line)
    assert [line for line in prose_lines if ";" in line] == []


def test_documented_target_args_reach_the_real_workload_parser() -> None:
    documented = re.findall(rf'--target-extra-arg {re.escape(TARGET)}="([^"]+)"', _readme())
    concrete = [args for args in documented if args != "--flag value"]
    assert {"--disable-compile", "--score-backend torch"} <= set(concrete)

    for args in concrete:
        assert _changed_fields(shlex.split(args)), (
            f"README documents {args!r}, but apply_cli_overrides leaves the workload unchanged"
        )


def test_isolation_flags_change_only_their_own_setting() -> None:
    assert _changed_fields(["--disable-compile"]) == {"use_compile"}
    assert _changed_fields(["--score-backend", "torch"]) == {"score_backend"}
    # The README says no flag turns off the pooling kernel. A new pooling
    # switch on the workload must come with a README update.
    assert [
        field.name
        for field in dataclasses.fields(SequenceRankingWorkload)
        if "pool" in field.name
    ] == []


@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="SKIPPED: CUDA and Triton are required to run the fused pooling and scoring kernels",
)
@pytest.mark.parametrize(
    ("score_backend", "expect_triton_scoring"),
    [("triton", True), ("torch", False)],
)
def test_optimized_benchmark_runs_triton_pooling_for_every_score_backend(
    monkeypatch: pytest.MonkeyPatch,
    score_backend: str,
    expect_triton_scoring: bool,
) -> None:
    argv = [*SMALL_SHAPE_ARGS, "--disable-compile", "--score-backend", score_backend]
    baseline = BaselineSequenceRankingBenchmark()
    optimized = OptimizedSequenceRankingBenchmark()
    baseline.apply_target_overrides(argv)
    optimized.apply_target_overrides(argv)
    baseline.setup()
    optimized.setup()
    try:
        calls = {"pooling": 0, "triton_scoring": 0, "torch_scoring": 0}

        def _counting(name: str, fn):
            def _wrapped(*args, **kwargs):
                calls[name] += 1
                return fn(*args, **kwargs)

            return _wrapped

        # The wrappers count calls and then run the real kernels. They are
        # installed after setup() so warmup launches are not counted.
        monkeypatch.setattr(
            ranking_common,
            "_sequence_context_user_input_triton_prepared",
            _counting("pooling", ranking_common._sequence_context_user_input_triton_prepared),
        )
        monkeypatch.setattr(
            ranking_common,
            "_candidate_scores_triton_prepared",
            _counting("triton_scoring", ranking_common._candidate_scores_triton_prepared),
        )
        monkeypatch.setattr(
            ranking_common,
            "candidate_scores_torch",
            _counting("torch_scoring", ranking_common.candidate_scores_torch),
        )

        baseline.benchmark_fn()
        optimized.benchmark_fn()
        torch.cuda.synchronize()

        assert optimized.score_backend == score_backend
        assert optimized.compile_enabled is False
        metrics = optimized.get_custom_metrics()
        assert metrics["ranking.triton_available"] == 1.0
        assert metrics["ranking.score_backend_triton"] == (1.0 if expect_triton_scoring else 0.0)
        assert calls == {
            "pooling": 1,
            "triton_scoring": 1 if expect_triton_scoring else 0,
            "torch_scoring": 0 if expect_triton_scoring else 1,
        }
        torch.testing.assert_close(optimized.output, baseline.output, rtol=1e-5, atol=1e-5)
    finally:
        baseline.teardown()
        optimized.teardown()


def test_optimized_module_and_class_docstrings_name_both_triton_kernels() -> None:
    import labs.recsys_sequence_ranking.optimized_sequence_ranking as optimized_module

    assert optimized_module.__doc__ == (
        "Optimized session-ranking path with fused Triton pooling, a compiled user tower, "
        "and Triton scoring."
    )
    assert OptimizedSequenceRankingBenchmark.__doc__ == (
        "Optimized sequence-ranking path using fused Triton pooling, a compiled tower, "
        "and Triton scoring."
    )
    for doc in (optimized_module.__doc__, OptimizedSequenceRankingBenchmark.__doc__):
        assert "vectorized" not in doc
    # The docstrings describe the default path: compile on and the auto score backend.
    workload = default_workload()
    assert workload.use_compile is True
    assert workload.score_backend == "auto"
    assert "_sequence_context_user_input_triton_prepared" in inspect.getsource(
        ranking_common.optimized_forward_preallocated
    )
