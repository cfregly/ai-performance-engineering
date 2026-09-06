"""Continuation-mode failure isolation for the real benchmark command loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.benchmark import bench_commands
from core.benchmark.e2e_sweep import _E2EAbort, _benchmark_stage_details_from_output
from core.harness import run_benchmarks as run_benchmarks_module


def _write_pair(chapter_dir: Path, example: str) -> None:
    chapter_dir.mkdir(parents=True, exist_ok=True)
    source = "def get_benchmark():\n    raise RuntimeError('GPU execution is not used by this test')\n"
    (chapter_dir / f"baseline_{example}.py").write_text(source, encoding="utf-8")
    (chapter_dir / f"optimized_{example}.py").write_text(source, encoding="utf-8")


def _prepare_cuda_hidden_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(bench_commands, "BENCHMARK_AVAILABLE", True)
    monkeypatch.setattr(bench_commands, "TEST_FUNCTIONS_AVAILABLE", True)
    monkeypatch.setattr(bench_commands, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(bench_commands, "get_gpu_state", lambda **_kwargs: {})
    monkeypatch.setattr(run_benchmarks_module.torch.cuda, "is_available", lambda: False)


def test_continuation_records_one_target_preflight_and_attempts_the_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench_root = tmp_path / "bench"
    chapter_dir = bench_root / "ch01"
    _write_pair(chapter_dir, "blocked")
    _write_pair(chapter_dir, "later")
    _prepare_cuda_hidden_control(monkeypatch)

    preflight_calls: list[set[str]] = []

    def selective_preflight(_chapter_dirs, chapter_filters, **_kwargs):
        examples = {
            example
            for selected in chapter_filters.values()
            for example in selected
        }
        preflight_calls.append(examples)
        if examples == {"blocked"}:
            return ["ch01: synthetic target preflight failure"]
        return []

    attempted_examples: list[list[str] | None] = []
    real_test_chapter = bench_commands.test_chapter

    def tracking_test_chapter(**kwargs):
        attempted_examples.append(kwargs["only_examples"])
        return real_test_chapter(**kwargs)

    monkeypatch.setattr(
        bench_commands,
        "_preflight_target_coverage_and_assets",
        selective_preflight,
    )
    monkeypatch.setattr(bench_commands, "test_chapter", tracking_test_chapter)

    result = bench_commands._execute_benchmarks(
        targets=["ch01:blocked", "ch01:later"],
        bench_root=bench_root,
        output_format="json",
        profile_type="none",
        suite_timeout=0,
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="target_preflight_isolation",
        exit_on_failure=False,
    )

    assert preflight_calls == [{"blocked"}, {"later"}]
    assert attempted_examples == [["later"]]
    assert result["total_failed"] == 1
    assert result["preflight_failed"] is True
    assert [entry["status"] for entry in result["results"]] == [
        "failed_preflight",
        "skipped",
    ]
    failed_target = result["results"][0]["benchmarks"][0]
    assert failed_target["example"] == "blocked"
    assert failed_target["status"] == "failed_error"
    assert "synthetic target preflight failure" in failed_target["error"]

    persisted = json.loads(Path(result["output_json"]).read_text(encoding="utf-8"))
    assert persisted["results"] == result["results"]
    stage_details = _benchmark_stage_details_from_output(result["output_json"])
    assert stage_details is not None
    assert stage_details["target_outcomes"] == [
        {"target": "ch01:blocked", "status": "failed_error"}
    ]


@pytest.mark.parametrize("split_across_chapters", [False, True])
def test_continuation_records_ordinary_exception_and_attempts_later_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split_across_chapters: bool,
) -> None:
    bench_root = tmp_path / "bench"
    first_chapter = bench_root / "ch01"
    later_chapter = bench_root / ("ch02" if split_across_chapters else "ch01")
    _write_pair(first_chapter, "first")
    _write_pair(later_chapter, "later")
    _prepare_cuda_hidden_control(monkeypatch)

    attempted: list[tuple[str, list[str] | None]] = []
    real_test_chapter = bench_commands.test_chapter

    def fault_then_real_test_chapter(**kwargs):
        call = (kwargs["chapter_dir"].name, kwargs["only_examples"])
        attempted.append(call)
        if len(attempted) == 1:
            raise RuntimeError("synthetic ordinary chapter failure")
        return real_test_chapter(**kwargs)

    monkeypatch.setattr(bench_commands, "test_chapter", fault_then_real_test_chapter)
    targets = (
        ["ch01:first", "ch02:later"]
        if split_across_chapters
        else ["ch01:first", "ch01:later"]
    )

    result = bench_commands._execute_benchmarks(
        targets=targets,
        bench_root=bench_root,
        output_format="json",
        profile_type="none",
        suite_timeout=0,
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id=f"exception_isolation_{int(split_across_chapters)}",
        exit_on_failure=False,
    )

    assert len(attempted) == 2
    assert attempted[0] == ("ch01", ["first"])
    assert attempted[1] == (
        "ch02" if split_across_chapters else "ch01",
        ["later"],
    )
    assert result["total_failed"] == 1
    assert [entry["status"] for entry in result["results"]] == [
        "failed_error",
        "skipped",
    ]
    failed_target = result["results"][0]["benchmarks"][0]
    assert failed_target["example"] == "first"
    assert failed_target["status"] == "failed_error"
    assert failed_target["error"] == "RuntimeError: synthetic ordinary chapter failure"

    persisted = json.loads(Path(result["output_json"]).read_text(encoding="utf-8"))
    assert persisted["results"] == result["results"]


@pytest.mark.parametrize(
    "abort",
    [KeyboardInterrupt(), SystemExit(17), _E2EAbort("synthetic e2e abort")],
)
def test_failure_boundary_does_not_capture_run_level_abort(
    monkeypatch: pytest.MonkeyPatch,
    abort: BaseException,
) -> None:
    def raise_abort(**_kwargs):
        raise abort

    monkeypatch.setattr(bench_commands, "test_chapter", raise_abort)

    with pytest.raises(type(abort)):
        bench_commands._run_test_chapter_with_failure_boundary(isolate_failures=True)
