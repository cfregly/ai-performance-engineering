"""Follow-up regressions for the harness, CLI, MCP and docs audit fixes (WP-H).

Covers OOF-011..017, OOF-025, OOF-068/069, OOF-101..106, OOF-108 and OOF-116,
plus the apply_target_overrides hooks the stricter harness requires. Each test
runs the real repository code. The only patched seams are the engine's handler
(to point it at a temporary bench root), LLM availability, the chapter runner
inside the CLI batch loop, the harness compliance gate used as a stop point
before benchmark execution, the CUDA device lookup of the persistent decode lab
(this host has no GPU) and a stand-in for the missing triton import.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

import core.engine as engine_module
import core.llm as llm_module
import mcp.mcp_server as mcp_server
from core import profile_artifacts
from core.analysis.analyze_expectations import classify_entry
from core.benchmark import bench_commands
from core.benchmark.defaults import get_defaults
from core.benchmark.expectations import (
    MAX_MEAN_SKEW_RATIO,
    UPDATE_EXPECTATIONS_HELP_TEXT,
    ExpectationEntry,
    ExpectationsStore,
    RunProvenance,
)
from core.benchmark.validate_expectations import validate_expectations, validate_file
from core.harness import run_benchmarks
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    LaunchVia,
)
from core.perf_core import PerfCore
from core.profiling.profiler_wrapper import (
    render_ncu_python_profile_wrapper,
    render_nsys_python_profile_wrapper,
    render_torch_python_profile_wrapper,
)
from core.utils.chapter_compare_template import get_last_load_error, load_benchmark
from scripts.full_virtualized_rerun import _build_entry, _write_eligible_expectation

CODE_ROOT = Path(__file__).resolve().parents[1]
DECODE_LAB = CODE_ROOT / "labs" / "decode_optimization"


# ---------------------------------------------------------------------------
# OOF-011, OOF-012, OOF-013: one --update-expectations wording everywhere
# ---------------------------------------------------------------------------


def _bench_option_help(callback_name: str, parameter: str) -> str:
    commands = [c for c in bench_commands.app.registered_commands if c.callback.__name__ == callback_name]
    assert len(commands) == 1, callback_name
    return inspect.signature(commands[0].callback).parameters[parameter].default.help


def test_update_expectations_wording_matches_what_the_store_refuses() -> None:
    assert f"by more than {MAX_MEAN_SKEW_RATIO:g}x is still refused" in UPDATE_EXPECTATIONS_HELP_TEXT
    assert "even when they regress or provenance differs" in UPDATE_EXPECTATIONS_HELP_TEXT

    for callback in ("run", "run_tier1", "run_e2e"):
        help_text = _bench_option_help(callback, "update_expectations")
        assert help_text == f"{UPDATE_EXPECTATIONS_HELP_TEXT} Useful for refreshing baselines on new hardware."
    assert _bench_option_help("explore", "update_expectations") == f"{UPDATE_EXPECTATIONS_HELP_TEXT} (recommended)"

    descriptions = {
        name: mcp_server.TOOLS[name].input_schema["properties"]["update_expectations"]["description"]
        for name in ("run_benchmarks", "benchmark_explore", "benchmark_e2e_sweep")
    }
    assert descriptions["run_benchmarks"].startswith(UPDATE_EXPECTATIONS_HELP_TEXT)
    assert descriptions["benchmark_e2e_sweep"].startswith(UPDATE_EXPECTATIONS_HELP_TEXT)
    assert descriptions["benchmark_explore"] == f"{UPDATE_EXPECTATIONS_HELP_TEXT} (recommended)"

    assert "help=UPDATE_EXPECTATIONS_HELP_TEXT" in inspect.getsource(run_benchmarks.main)
    for surface in (
        inspect.getsource(bench_commands),
        inspect.getsource(mcp_server),
        (CODE_ROOT / "docs" / "benchmark_harness_guide.md").read_text(encoding="utf-8"),
    ):
        assert "orce-write" not in surface


def test_harness_guide_describes_the_write_flags_and_the_refused_evidence() -> None:
    guide = (CODE_ROOT / "docs" / "benchmark_harness_guide.md").read_text(encoding="utf-8")
    assert "on improvements only" not in guide
    assert (
        "No flag stores timing whose mean exceeds its p75, or its median when no p75 is recorded, "
        f"by more than {MAX_MEAN_SKEW_RATIO:g}x."
    ) in guide
    assert "reported as `failed_expectation_invalid`" in guide
    assert (
        "run's git commit, hardware key, profile, execution environment, validity profile and "
        "`--target-extra-arg` values match the stored entry"
    ) in guide
    # Those are exactly the fields that make a write a provenance mismatch.
    stored = RunProvenance("a", "b200", "minimal", "t", 1, 1, "bare_metal", "strict")
    changed = RunProvenance("b", "4x_gb200", "deep_dive", "t", 1, 1, "virtualized", "portable", None, ["--x"])
    assert stored.mismatch_fields(changed) == [
        "git_commit",
        "hardware_key",
        "profile_name",
        "execution_environment",
        "validity_profile",
        "target_override_argv",
    ]


def _write_pair(chapter_dir: Path, example: str) -> None:
    chapter_dir.mkdir(parents=True, exist_ok=True)
    source = "def get_benchmark():\n    raise RuntimeError('GPU execution is not used by this test')\n"
    (chapter_dir / f"baseline_{example}.py").write_text(source, encoding="utf-8")
    (chapter_dir / f"optimized_{example}.py").write_text(source, encoding="utf-8")


def test_cli_run_total_counts_failed_expectation_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench_root = tmp_path / "bench"
    _write_pair(bench_root / "ch01", "refused")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(bench_commands, "BENCHMARK_AVAILABLE", True)
    monkeypatch.setattr(bench_commands, "TEST_FUNCTIONS_AVAILABLE", True)
    monkeypatch.setattr(bench_commands, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(bench_commands, "get_gpu_state", lambda **_kwargs: {})
    monkeypatch.setattr(run_benchmarks.torch.cuda, "is_available", lambda: False)

    def refused_chapter(**kwargs: Any) -> dict[str, Any]:
        # A summary that names the status but omits the 'failed' total, so only
        # the per-status keys can count it.
        return {
            "chapter": "ch01",
            "status": "completed",
            "benchmarks": [{"example": "refused", "status": "failed_expectation_invalid"}],
            "manifests": [],
            "summary": {"total_benchmarks": 1, "successful": 0, "failed_expectation_invalid": 1},
        }

    monkeypatch.setattr(bench_commands, "test_chapter", refused_chapter)
    result = bench_commands._execute_benchmarks(
        targets=["ch01:refused"],
        bench_root=bench_root,
        output_format="json",
        profile_type="none",
        suite_timeout=0,
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="expectation_invalid_count",
        exit_on_failure=False,
    )
    assert result["total_failed"] == 1

    isolated = bench_commands._isolated_benchmark_failure_result(
        chapter_id="ch01", only_examples=["refused"], status="failed_error", error="boom"
    )
    assert isolated["summary"]["failed_expectation_invalid"] == 0


# ---------------------------------------------------------------------------
# OOF-014: the validator says what --fix can and cannot repair
# ---------------------------------------------------------------------------


def _expectation_payload(**metrics: Any) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "hardware_key": "b200",
        "examples": {
            "example": {
                "example": "example",
                "type": "python",
                "metadata": {"optimization_goal": "speed", "best_optimization_speedup": 2.0},
                "metrics": {"baseline_time_ms": 10.0, "best_optimized_time_ms": 5.0, **metrics},
                "provenance": {
                    "git_commit": "abc",
                    "hardware_key": "b200",
                    "profile_name": "minimal",
                    "timestamp": "2026-10-13T00:00:00",
                    "iterations": 20,
                    "warmup_iterations": 5,
                },
            }
        },
    }


def test_validator_fix_reports_timing_issues_that_need_a_fresh_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import core.benchmark.validate_expectations as validator

    assert "Timing outliers" in (validator.__doc__ or "")
    path = tmp_path / "expectations_b200.json"
    # A wrong stored speedup (fixable) and a baseline mean 1.6x its p75 (not fixable).
    path.write_text(
        json.dumps(_expectation_payload(best_speedup=9.0, baseline_p75_ms=6.25, best_optimized_p75_ms=5.0)),
        encoding="utf-8",
    )
    before = {issue.issue_type for issue in validate_file(path).issues}
    assert "timing_outlier_skew" in before and len(before) >= 2

    exit_code = validate_expectations(paths=[str(path)], fix=True, backup=False)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert [issue.issue_type for issue in validate_file(path).issues] == ["timing_outlier_skew"]
    assert re.search(r"Recomputed derived fields\. 1 of \d+ issue\(s\) remain", output), output
    assert "Fixed" not in output
    assert "--fix recomputes derived ratios and fills missing provenance only" in output
    assert "timing_outlier_skew and invalid_timing_percentile issues need a fresh run" in output


# ---------------------------------------------------------------------------
# OOF-015: a refused benchmark is graded as a failure, not by its speedup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["failed_expectation_invalid", "failed_verification", "failed_profiler"])
def test_classify_entry_fails_every_failed_status_regardless_of_speedup(tmp_path: Path, status: str) -> None:
    chapter_dir = tmp_path / "ch01"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "baseline_example.py").write_text("print('baseline')\n", encoding="utf-8")
    (chapter_dir / "optimized_example.py").write_text("print('optimized')\n", encoding="utf-8")
    benchmark = {
        "example": "skewed",
        "status": status,
        "best_speedup": 3.0,
        "optimization_goal": "speed",
        "baseline_file": "baseline_example.py",
        "optimizations": [{"file": "optimized_example.py", "status": "succeeded", "speedup": 3.0}],
        "error": "Timing rejected",
    }

    entry = classify_entry(
        chapter_name="ch01", benchmark=benchmark, artifact_dir="runs/test", threshold=1.05, repo_root=tmp_path
    )

    assert entry is not None
    assert entry.severity == "fail"


# ---------------------------------------------------------------------------
# OOF-016: portable rerun writes get the median check
# ---------------------------------------------------------------------------


def _rerun_entry(bench: dict[str, Any], best_opt: dict[str, Any]) -> ExpectationEntry:
    return _build_entry(
        bench=bench,
        best_opt=best_opt,
        hardware_key="b200",
        profile_name="none",
        git_commit="def456",
        execution_environment="virtualized",
        dmi_product_name=None,
        validity_profile="portable",
        iterations=20,
        warmup_iterations=5,
    )


def test_portable_rerun_refuses_a_mean_skewed_past_its_median(tmp_path: Path) -> None:
    (tmp_path / "ch01").mkdir()
    stored = _rerun_entry(
        {"example": "performance", "type": "python", "baseline_time_ms": 10.0, "baseline_median_ms": 10.0},
        {"time_ms": 5.0, "median_ms": 5.0},
    )
    store = ExpectationsStore(tmp_path / "ch01", "b200", allow_mixed_provenance=True)
    assert store.update_entry("performance", stored).status != "rejected"
    store.save()

    bench = {
        "example": "performance",
        "type": "python",
        "baseline_time_ms": 10.0,
        "baseline_median_ms": 7.0,
        "baseline_p90_ms": 30.0,
    }
    best_opt = {"time_ms": 5.0, "median_ms": 4.9, "p90_ms": 6.0}
    entry = _rerun_entry(bench, best_opt)
    assert (entry.baseline_median_ms, entry.best_optimized_median_ms) == (7.0, 4.9)
    assert (entry.baseline_p90_ms, entry.best_optimized_p90_ms) == (30.0, 6.0)

    action, reasons, details = _write_eligible_expectation(
        repo_root=tmp_path,
        target="ch01:performance",
        bench=bench,
        entry=entry,
        hardware_key="b200",
        allow_portable_expectations_update=True,
    )

    assert (action, reasons) == ("queued", ["expectation_write_rejected"])
    issue_types = {issue["issue_type"] for issue in details["update_result"]["validation_issues"]}
    assert issue_types == {"timing_outlier_skew"}


# ---------------------------------------------------------------------------
# OOF-017, OOF-101, OOF-102: stored decode rows
# ---------------------------------------------------------------------------


def test_decode_expectations_hold_no_invented_percentiles_or_fp8_labeled_fp4_rows() -> None:
    # The old harness stored p75 = 0 when the configured percentiles omitted 75.
    for path in sorted(CODE_ROOT.glob("**/expectations_*.json")):
        if "artifacts" in path.relative_to(CODE_ROOT).parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, entry in (payload.get("examples") or {}).items():
            for name, value in (entry.get("metrics") or {}).items():
                if name.endswith(("_p75_ms", "_median_ms")) and value is not None:
                    assert value > 0, (path.relative_to(CODE_ROOT), key, name, value)

    for path in sorted(DECODE_LAB.glob("expectations_*.json")):
        report = validate_file(path)
        assert not report.issues, (path.name, [issue.to_dict() for issue in report.issues])

    # These rows ran FP8 DelayedScaling under the FP4 label (decode_common.py at
    # each commit wrapped NVFP4BlockScaling in DelayedScaling). A fresh row from
    # a strict B200 run after the fix is allowed.
    fp8_labeled_fp4_runs = {
        "expectations_b200.json": "04ce7a016ff48e9486d3ecc3a6c963aed019d193",
        "expectations_2x_b200.json": "4fe709b0cf0876851d645e87c589d44dc5a59a91",
    }
    for filename, commit in fp8_labeled_fp4_runs.items():
        examples = json.loads((DECODE_LAB / filename).read_text(encoding="utf-8"))["examples"]
        stored = examples.get("decode_fp4")
        if stored is not None:
            assert stored["provenance"]["git_commit"] != commit, filename


# ---------------------------------------------------------------------------
# OOF-025: --reproducible keeps seed 42 and deterministic algorithms
# ---------------------------------------------------------------------------


class _DefaultsConfigBenchmark(BaseBenchmark):
    """A benchmark whose get_config returns every BenchmarkConfig default."""

    allow_cpu = True

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=3, warmup=5)


def _runner_merge(reproducible: bool) -> BenchmarkConfig:
    base = BenchmarkConfig(
        iterations=20,
        warmup=5,
        enable_memory_tracking=True,
        seed=42 if reproducible else None,
        deterministic=reproducible,
    )
    locked = run_benchmarks._compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
        reproducible=reproducible,
    )
    return run_benchmarks._merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DefaultsConfigBenchmark(),
        defaults_obj=get_defaults(),
        locked_fields=locked,
    )


def test_reproducible_locks_seed_and_deterministic_against_benchmark_defaults() -> None:
    assert BenchmarkConfig().deterministic is False
    merged = _runner_merge(reproducible=True)
    assert (merged.seed, merged.deterministic) == (42, True)
    # Iterations stay unlocked, so the merge still applies the benchmark's own count.
    assert merged.iterations == 3

    plain = _runner_merge(reproducible=False)
    assert (plain.seed, plain.deterministic) == (get_defaults().seed, False)
    assert "reproducible=reproducible," in inspect.getsource(run_benchmarks._test_chapter_impl)


def test_no_benchmark_config_sets_its_own_seed() -> None:
    # BenchmarkHarness.benchmark lets a benchmark's config replace a harness value
    # that equals the default. The --reproducible seed (42) is also the default
    # seed, so a benchmark that set its own seed would still replace it there.
    offenders = []
    for path in sorted([*CODE_ROOT.glob("ch*/**/*.py"), *CODE_ROOT.glob("labs/**/*.py")]):
        if "third_party" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "BenchmarkConfig":
                for keyword in node.keywords:
                    if keyword.arg == "seed" and ast.unparse(keyword.value) not in ("None", "42"):
                        offenders.append(f"{path.relative_to(CODE_ROOT)}:{node.lineno}")
    assert offenders == []


# ---------------------------------------------------------------------------
# OOF-068, OOF-108: --target-extra-arg must reach the benchmark or fail it
# ---------------------------------------------------------------------------


class _StoppedBeforeExecutionError(Exception):
    """Raised at the harness compliance gate, after the override checks."""


class _CpuBenchmark(BaseBenchmark):
    allow_cpu = True

    def __init__(self, launch_via: LaunchVia = LaunchVia.PYTHON) -> None:
        super().__init__()
        self._launch_via = launch_via
        self.ran = False

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(launch_via=self._launch_via)

    def setup(self) -> None:
        self.ran = True

    def benchmark_fn(self) -> None:
        self.ran = True


class _RejectingHookBenchmark(_CpuBenchmark):
    def apply_target_overrides(self, argv: list[str]) -> None:
        raise ValueError(f"unsupported override {argv}")


def _override_config(label: str, argv: list[str]) -> BenchmarkConfig:
    return BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=5,
        use_subprocess=False,
        enable_profiling=False,
        enable_memory_tracking=False,
        enforce_environment_validation=False,
        target_label=label,
        target_extra_args={label: argv},
    )


def _benchmark_until_execution(monkeypatch: pytest.MonkeyPatch, benchmark: BaseBenchmark, config: BenchmarkConfig) -> None:
    import core.benchmark.quarantine as quarantine

    def stop(_benchmark: object) -> None:
        raise _StoppedBeforeExecutionError

    monkeypatch.setattr(quarantine, "detect_skip_flags", stop)
    BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config).benchmark(benchmark)


def test_target_override_without_a_hook_fails_the_target(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = _CpuBenchmark()
    with pytest.raises(ValueError, match=r"--target-extra-arg was given for ch01:example, but _CpuBenchmark has no apply_target_overrides hook"):
        _benchmark_until_execution(monkeypatch, benchmark, _override_config("ch01:example", ["--size", "8"]))
    assert benchmark.ran is False

    # Without overrides the same benchmark reaches execution.
    with pytest.raises(_StoppedBeforeExecutionError):
        _benchmark_until_execution(monkeypatch, _CpuBenchmark(), _override_config("ch01:example", []))


def test_target_override_hook_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = _RejectingHookBenchmark()
    with pytest.raises(ValueError, match=r"unsupported override \['--backend', 'cudnn'\]"):
        _benchmark_until_execution(monkeypatch, benchmark, _override_config("ch01:example", ["--backend", "cudnn"]))
    assert benchmark.ran is False
    assert getattr(benchmark, "_applied_target_override_signature", None) is None


def test_torchrun_launch_takes_overrides_as_script_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    # The benchmark's own config selects torchrun, which only the merge reveals.
    with pytest.raises(_StoppedBeforeExecutionError):
        _benchmark_until_execution(
            monkeypatch,
            _CpuBenchmark(launch_via=LaunchVia.TORCHRUN),
            _override_config("labs/train_distributed:ddp_multigpu", ["--overlap-optimizer"]),
        )

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig())
    config = _override_config("ch01:example", ["--size", "8"])
    harness._apply_target_overrides(_CpuBenchmark(), config, launcher_final=False)
    with pytest.raises(ValueError, match="has no apply_target_overrides hook"):
        harness._apply_target_overrides(_CpuBenchmark(), config)


def test_real_lab_targets_fail_on_ignored_or_rejected_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    # The ch06 warp-divergence baseline has no hook, so its run would ignore the values.
    divergence = load_benchmark(CODE_ROOT / "ch06" / "baseline_warp_divergence_ilp.py")
    assert divergence is not None, get_last_load_error()
    assert not callable(getattr(divergence, "apply_target_overrides", None))
    with pytest.raises(ValueError, match=f"{type(divergence).__name__} has no apply_target_overrides hook"):
        _benchmark_until_execution(
            monkeypatch, divergence, _override_config("ch06:warp_divergence_ilp", ["--elements", "1024"])
        )

    # The cuDNN SDPA hooks reject a bare --backend, and that error now fails the target.
    sdpa = load_benchmark(CODE_ROOT / "labs" / "cudnn_sdpa_bench" / "baseline_flash_sdp.py")
    assert sdpa is not None, get_last_load_error()
    default_backend = sdpa.backend
    with pytest.raises(ValueError, match="--baseline-backend and --optimized-backend"):
        _benchmark_until_execution(
            monkeypatch, sdpa, _override_config("labs/cudnn_sdpa_bench:flash_sdp", ["--backend", "cudnn"])
        )
    assert sdpa.backend == default_backend


# ---------------------------------------------------------------------------
# OOF-069: an expectation entry records the override argv it ran with
# ---------------------------------------------------------------------------


def _provenance(argv: list[str] | None) -> RunProvenance:
    return RunProvenance(
        git_commit="abc",
        hardware_key="b200",
        profile_name="minimal",
        timestamp="2026-10-13T00:00:00",
        iterations=20,
        warmup_iterations=5,
        execution_environment="bare_metal",
        validity_profile="strict",
        target_override_argv=argv,
    )


def test_provenance_records_and_compares_the_override_argv() -> None:
    default_run = _provenance(None)
    override_run = _provenance(["--baseline-backend", "cudnn", "--optimized-backend", "flash"])

    assert "target_override_argv" not in default_run.to_dict()
    assert RunProvenance.from_dict(override_run.to_dict()) == override_run
    assert RunProvenance.from_dict(default_run.to_dict()) == default_run
    assert override_run.mismatch_fields(default_run) == ["target_override_argv"]
    with pytest.raises(ValueError, match="must be a list of strings"):
        RunProvenance.from_dict({**override_run.to_dict(), "target_override_argv": "--backend cudnn"})

    config = BenchmarkConfig(
        target_label="labs_cudnn_sdpa_bench:flash_sdp",
        target_extra_args={"labs/cudnn_sdpa_bench:flash_sdp": ["--baseline-backend", "cudnn"]},
    )
    assert run_benchmarks._resolve_target_override_argv(config) == ["--baseline-backend", "cudnn"]
    assert run_benchmarks._resolve_target_override_argv(BenchmarkConfig(target_label="ch01:example")) is None
    assert "target_override_argv=_resolve_target_override_argv(baseline_config)" in inspect.getsource(
        run_benchmarks._test_chapter_impl
    )


def test_override_run_cannot_replace_a_default_run_bar_without_a_mixed_provenance_write(tmp_path: Path) -> None:
    def entry(argv: list[str] | None) -> ExpectationEntry:
        return ExpectationEntry(
            example="flash_sdp",
            type="python",
            optimization_goal="speed",
            baseline_time_ms=10.0,
            best_optimized_time_ms=5.0,
            provenance=_provenance(argv),
        )

    store = ExpectationsStore(tmp_path, "b200")
    assert store.update_entry("flash_sdp", entry(None)).status == "updated"
    store.save()

    override_argv = ["--baseline-backend", "cudnn", "--optimized-backend", "flash"]
    rejected = ExpectationsStore(tmp_path, "b200", accept_regressions=True).update_entry(
        "flash_sdp", entry(override_argv)
    )
    assert rejected.status == "rejected"
    assert [issue.issue_type for issue in rejected.validation_issues] == ["provenance_mismatch"]
    assert "target_override_argv" in rejected.message

    mixed = ExpectationsStore(tmp_path, "b200", accept_regressions=True, allow_mixed_provenance=True)
    assert mixed.update_entry("flash_sdp", entry(override_argv)).status != "rejected"
    mixed.save()
    stored = json.loads((tmp_path / "expectations_b200.json").read_text(encoding="utf-8"))
    assert stored["examples"]["flash_sdp"]["provenance"]["target_override_argv"] == override_argv


# ---------------------------------------------------------------------------
# OOF-104: the dashboard loaders use the same trace rule as analyze_bottlenecks
# ---------------------------------------------------------------------------


_MTIME_NS = 1_790_000_000 * 1_000_000_000


def _write_profile_file(path: Path, text: str, *, seconds: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(_MTIME_NS + seconds * 1_000_000_000,) * 2)
    return path


def _capture_with_newer_non_trace_json(root: Path) -> Path:
    capture = root / "artifacts" / "runs" / "followup_run" / "profiles" / "tools" / "torch" / "capture"
    events = [
        {"ph": "X", "name": "gemm_kernel", "cat": "kernel", "ts": 1000, "dur": 600},
        {"ph": "X", "name": "Memcpy HtoD", "cat": "gpu_memcpy", "ts": 1700, "dur": 300},
        {"ph": "X", "name": "aten::mm", "cat": "cpu_op", "ts": 900, "dur": 50},
    ]
    trace = _write_profile_file(capture / "trace.json", json.dumps({"traceEvents": events}), seconds=0)
    _write_profile_file(capture / "metadata.json", json.dumps({"mode": "full"}), seconds=1)
    _write_profile_file(capture / "key_averages_full.json", json.dumps([{"name": "aten::mm"}]), seconds=2)
    _write_profile_file(capture / "torch_profile_summary.json", json.dumps({"top_ops": []}), seconds=3)
    _write_profile_file(
        root / "artifacts" / "runs" / "other_run" / "profiles" / "hta" / "export.json",
        '{"id":0,"table":"StringIds","value":"[Unknown]"}\n{"id":1,"table":"StringIds","value":"main"}\n',
        seconds=4,
    )
    _write_profile_file(
        root / "artifacts" / "runs" / "mcp-fixtures" / "profiles" / "bench" / "pair_manifest.json",
        json.dumps({"baseline": "a.json", "optimized": "b.json"}),
        seconds=5,
    )
    return trace


def test_dashboard_loaders_pick_the_newest_chrome_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _capture_with_newer_non_trace_json(tmp_path)

    flame = profile_artifacts.load_flame_graph_data(tmp_path)
    assert flame["trace_path"] == str(trace)
    assert flame["value"] == 950
    assert "error" not in flame and "warnings" not in flame
    kernels = profile_artifacts.load_kernel_breakdown(flame)
    assert kernels["kernels"][0]["name"] == "gemm_kernel"

    timeline = profile_artifacts.load_cpu_gpu_timeline(tmp_path)
    assert timeline["trace_path"] == str(trace)
    assert [event["name"] for event in timeline["gpu"]] == ["gemm_kernel"]
    assert timeline["summary"]["total_time_ms"] == pytest.approx(1.1)

    # The engine's bottleneck analysis and the dashboard read the same trace.
    core = PerfCore(bench_root=tmp_path)
    monkeypatch.setattr(engine_module, "_get_handler", lambda: core)
    profile = engine_module.get_engine().analyze.bottlenecks(mode="profile")["profile"]
    assert profile["trace_path"] == str(trace)
    assert core.get_flame_graph_data()["trace_path"] == str(trace)


def test_dashboard_loaders_report_profile_dirs_without_a_trace(tmp_path: Path) -> None:
    capture = tmp_path / "artifacts" / "runs" / "followup_run" / "profiles" / "capture"
    _write_profile_file(capture / "torch_profile_summary.json", json.dumps({"top_ops": []}), seconds=1)
    _write_profile_file(capture / "pair_manifest.json", json.dumps({"baseline": "a.json"}), seconds=2)

    for payload in (profile_artifacts.load_flame_graph_data(tmp_path), profile_artifacts.load_cpu_gpu_timeline(tmp_path)):
        assert "trace_path" not in payload
        assert "None of the 2 JSON file(s) there holds a traceEvents list" in payload["message"]

    empty = profile_artifacts.load_flame_graph_data(tmp_path / "missing")
    assert "None of the 0 JSON file(s)" in empty["message"]

    trace = _write_profile_file(capture / "trace.json", json.dumps({"traceEvents": []}), seconds=0)
    flame = profile_artifacts.load_flame_graph_data(tmp_path)
    assert flame["trace_path"] == str(trace)
    assert flame["warnings"] == [f"Chrome trace {trace} has no events"]

    _write_profile_file(capture / "trace.json", "{broken", seconds=0)
    broken = profile_artifacts.load_cpu_gpu_timeline(tmp_path)
    assert broken["trace_path"] == str(trace)
    assert broken["error"].startswith(f"Could not check {trace} for Chrome trace events")


# ---------------------------------------------------------------------------
# OOF-103, OOF-105, OOF-106, OOF-116: CLI, MCP and docs parity
# ---------------------------------------------------------------------------


def test_cli_analyze_bottlenecks_passes_focus_to_the_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from cli.aisp import app

    trace = _capture_with_newer_non_trace_json(tmp_path)
    monkeypatch.setattr(engine_module, "_get_handler", lambda: PerfCore(bench_root=tmp_path))
    monkeypatch.setattr(llm_module, "is_available", lambda: False)
    runner = CliRunner()

    result = runner.invoke(app, ["analyze", "bottlenecks", "--mode", "both", "--focus", "memory"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["focus"] == "memory"
    assert payload["profile"]["trace_path"] == str(trace)
    assert payload["llm"]["error_type"] == "llm_unavailable"

    rejected = runner.invoke(app, ["analyze", "bottlenecks", "--mode", "profile", "--focus", "compute"])
    assert rejected.exit_code == 2
    assert "applies only to the LLM explanation" in rejected.output

    tool = mcp_server.TOOLS["analyze_bottlenecks"]
    assert tool.input_schema["properties"]["analysis_type"]["enum"] == ["bottleneck", "memory", "compute"]


def test_api_reference_matches_the_cli_for_bottlenecks_and_profile_stubs() -> None:
    reference = (CODE_ROOT / "docs" / "api-reference.md").read_text(encoding="utf-8")
    assert (
        "| `bottlenecks(mode, focus)` | Rule-based bottlenecks in the newest captured Chrome trace, "
        "with an optional LLM explanation | `aisp analyze bottlenecks` | `analyze_bottlenecks` |"
    ) in reference
    assert 'engine.analyze.bottlenecks(mode="llm", focus="memory")' in reference
    assert "| Not implemented (`aisp profile kernels` exits 1) | `profile_kernels` |" in reference
    assert "| Not implemented (`aisp profile memory` exits 1) | `profile_memory` |" in reference
    assert inspect.signature(engine_module.AnalyzeDomain.bottlenecks).parameters["focus"].default == "bottleneck"


# ---------------------------------------------------------------------------
# Hooks for targets that used to read --target-extra-arg values from sys.argv
# ---------------------------------------------------------------------------


def _apply_overrides(benchmark: BaseBenchmark, label: str, argv: list[str]) -> None:
    config = BenchmarkConfig(target_label=label, target_extra_args={label: argv})
    BenchmarkHarness(config=config)._apply_target_overrides(benchmark, config)


RIGHT_SIZED_LABEL = "labs/persistent_decode:right_sized_decode"


def test_right_sized_decode_baseline_runs_the_overridden_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    import labs.persistent_decode.baseline_persistent_decode as baseline_module
    from labs.persistent_decode import baseline_right_sized_decode as right_sized
    from labs.persistent_decode.persistent_decode_common import DecodeOptions, get_decode_options

    # This host has no CUDA, so the naive decode loop runs on the CPU device.
    monkeypatch.setattr(baseline_module, "resolve_device", lambda: torch.device("cpu"))
    # The module no longer reads its flags from the process arguments.
    monkeypatch.setattr(sys, "argv", ["aisp", "--tier", "small"])
    bench = right_sized.get_benchmark()
    assert (bench.batch, bench.seq_len, bench.head_dim) == (8, 32, 64)

    _apply_overrides(bench, RIGHT_SIZED_LABEL, ["--tier", "large", "--quantization", "fp16"])

    assert bench.decode_options == DecodeOptions(tier="large", quantization="fp16")
    assert (bench.batch, bench.batch_size, bench.seq_len, bench.head_dim) == (12, 12, 64, 64)
    assert bench.options.quantization == "fp16"
    assert bench._workload_metadata.tokens_per_iteration == 12 * 64
    # The override stays with this benchmark. Other persistent decode targets keep the defaults.
    assert get_decode_options() == DecodeOptions()

    bench.setup()
    try:
        assert get_decode_options() == DecodeOptions()
        assert tuple(bench.inputs.q.shape) == (12, 64, 64)
        assert bench.inputs.q.dtype == torch.float16
        bench.benchmark_fn()
        # Checks every sequence and token against an independent reference.
        bench.capture_verification_payload()
    finally:
        bench.teardown()

    for argv in (["--tier", "huge"], ["--tie", "large"], ["--block-k", "48"], ["--backend", "graphs"]):
        with pytest.raises(ValueError, match="right_sized_decode baseline override"):
            _apply_overrides(right_sized.get_benchmark(), RIGHT_SIZED_LABEL, argv)
    with pytest.raises(ValueError, match="num_programs must be positive"):
        _apply_overrides(right_sized.get_benchmark(), RIGHT_SIZED_LABEL, ["--num-programs", "0"])


@pytest.fixture
def right_sized_triton_module(monkeypatch: pytest.MonkeyPatch):
    """Import optimized_right_sized_decode, with a stand-in for triton where it is missing.

    The stand-in only satisfies the import. No Triton kernel runs: the test
    builds the benchmark and applies overrides, which only compute launch
    parameters.
    """
    if importlib.util.find_spec("triton") is None:
        stub = types.ModuleType("triton")
        stub.jit = lambda fn: fn
        language = types.ModuleType("triton.language")
        stub.language = language
        monkeypatch.setitem(sys.modules, "triton", stub)
        monkeypatch.setitem(sys.modules, "triton.language", language)
    names = (
        "labs.persistent_decode.optimized_persistent_decode_triton",
        "labs.persistent_decode.optimized_right_sized_decode",
    )
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("labs.persistent_decode.optimized_right_sized_decode")
    try:
        yield module
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_right_sized_decode_optimized_hook_rebuilds_launch_parameters(
    right_sized_triton_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from labs.persistent_decode.persistent_decode_common import DecodeOptions, get_decode_options

    triton_module = sys.modules["labs.persistent_decode.optimized_persistent_decode_triton"]
    monkeypatch.setattr(triton_module, "resolve_device", lambda: torch.device("cpu"))
    bench = right_sized_triton_module.get_benchmark()
    assert (bench.block_k, bench.num_programs, bench._launch_grid) == (64, 8, (8,))

    _apply_overrides(bench, RIGHT_SIZED_LABEL, ["--tier", "small", "--num-programs", "1"])

    assert bench.decode_options == DecodeOptions(tier="small", num_programs=1)
    assert (bench.batch, bench.seq_len, bench._num_items) == (4, 32, 4)
    assert (bench.block_k, bench.num_programs, bench._block_k_const, bench._launch_grid) == (32, 1, 32, (1,))
    assert bench._workload_metadata.tokens_per_iteration == 4 * 32
    assert get_decode_options() == DecodeOptions()

    # The CUDA graphs backend is its own target, so these flags no longer exist here.
    for argv in (["--backend", "graphs"], ["--graph-mode", "full"]):
        with pytest.raises(ValueError, match="Unsupported right_sized_decode optimized override"):
            _apply_overrides(right_sized_triton_module.get_benchmark(), RIGHT_SIZED_LABEL, argv)


def test_trtllm_hooks_parse_explicit_arguments_and_rebuild_the_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from labs.trtllm_phi_3_5_moe import trtllm_common
    from labs.trtllm_phi_3_5_moe.baseline_trtllm_phi_3_5_moe import BaselineTrtLlmPhi35MoeBenchmark
    from labs.trtllm_phi_3_5_moe.optimized_trtllm_phi_3_5_moe import OptimizedTrtLlmPhi35MoeBenchmark

    label = "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe"
    defaults = trtllm_common.parse_trtllm_args([])
    # The constructors no longer read the process arguments.
    monkeypatch.setattr(sys, "argv", ["aisp", "--batch-size", "7"])
    argv = [
        "--model-path", str(tmp_path / "model"),
        "--engine-path", str(tmp_path / "engine"),
        "--prompt-len", "64",
        "--max-new-tokens", "16",
        "--batch-size", "2",
    ]
    for cls in (BaselineTrtLlmPhi35MoeBenchmark, OptimizedTrtLlmPhi35MoeBenchmark):
        bench = cls()
        assert (bench.prompt_len, bench.max_new_tokens, bench.batch_size) == (
            defaults.prompt_len,
            defaults.max_new_tokens,
            defaults.batch_size,
        )
        _apply_overrides(bench, label, argv)
        assert bench.model_path == tmp_path / "model"
        assert (bench.prompt_len, bench.max_new_tokens, bench.batch_size, bench.vocab_slice) == (
            64,
            16,
            2,
            defaults.vocab_slice,
        )
        for workload in (bench._workload, bench._workload_metadata):
            assert (workload.requests_per_iteration, workload.tokens_per_iteration) == (2.0, 80.0)
        if cls is OptimizedTrtLlmPhi35MoeBenchmark:
            assert bench.engine_path == tmp_path / "engine"

        for bad, message in (
            (["--prompt-len", "0"], "--prompt-len must be positive"),
            (["--model", "x"], "Unsupported trtllm_phi_3_5_moe override"),
            (["--batch-size", "two"], "Invalid trtllm_phi_3_5_moe override"),
        ):
            with pytest.raises(ValueError, match=message):
                _apply_overrides(cls(), label, bad)


def test_rack_prep_variants_share_one_strict_override_parser() -> None:
    from ch03.baseline_rack_prep import BaselineRackPrepBenchmark, discover_requested_nics
    from ch03.optimized_rack_prep import OptimizedRackPrepBenchmark

    label = "ch03:rack_prep"
    for cls in (BaselineRackPrepBenchmark, OptimizedRackPrepBenchmark):
        bench = cls()
        assert (bench.reserve_cores, bench.preferred_nics, bench.apply_affinity) == (2, [], False)
        _apply_overrides(bench, label, ["--reserve", "4", "--nic", "eth0", "--apply"])
        assert (bench.reserve_cores, bench.preferred_nics, bench.apply_affinity) == (4, ["eth0"], True)
        for bad, message in (
            (["--reserv", "4"], "Unsupported rack_prep override"),
            (["--bogus"], "Unsupported rack_prep override"),
            (["--reserve", "x"], "Invalid rack_prep override"),
            (["--reserve", "-1"], "--reserve must be zero or more CPUs"),
            (["--nic", " "], "--nic needs a NIC name"),
        ):
            with pytest.raises(ValueError, match=message):
                _apply_overrides(cls(), label, bad)

    with pytest.raises(RuntimeError, match="requested with --nic were not found"):
        discover_requested_nics(["aisp-missing-nic0"])

    # A requested --apply that cannot be honored fails setup before any staging.
    optimized = OptimizedRackPrepBenchmark()
    _apply_overrides(optimized, label, ["--apply"])
    with pytest.raises(RuntimeError, match="--apply (needs a NIC with local CPUs|writes IRQ/RPS/XPS affinity and needs root)"):
        optimized.setup()
    assert optimized.host_buffers == []


_CPU_BENCHMARK_MODULE = """
import os

import torch

from core.harness.benchmark_harness import BaseBenchmark


class CpuOverrideBenchmark(BaseBenchmark):
    allow_cpu = True

    def setup(self):
        self.value = torch.ones(4)

    def benchmark_fn(self):
        self.value = self.value * 2
{hook}

def get_benchmark():
    return CpuOverrideBenchmark()
"""

_RECORDING_HOOK = """
    def apply_target_overrides(self, argv):
        with open(os.environ["AISP_HOOK_MARKER"], "w", encoding="utf-8") as handle:
            handle.write(" ".join(argv))
"""


def _run_profile_wrapper(tmp_path: Path, render: Any, *, hook: str) -> subprocess.CompletedProcess[str]:
    module_path = tmp_path / "cpu_override_benchmark.py"
    module_path.write_text(_CPU_BENCHMARK_MODULE.format(hook=hook), encoding="utf-8")
    common = {
        "benchmark_path": module_path,
        "target_label": "ch01:cpu_override",
        "target_override_argv": ["--scale", "3"],
        "validity_profile": "portable",
        "lock_gpu_clocks_flag": False,
        "gpu_sm_clock_mhz": None,
        "gpu_mem_clock_mhz": None,
    }
    if render is render_nsys_python_profile_wrapper:
        wrapper = render(nvtx_includes=None, **common)
    elif render is render_ncu_python_profile_wrapper:
        wrapper = render(
            configured_nvtx_includes=None,
            profile_type="minimal",
            ncu_metric_set="minimal",
            pm_sampling_interval=None,
            ncu_replay_mode=None,
            profile_nvtx_label="profile",
            **common,
        )
    else:
        wrapper = render(torch_output=tmp_path / "trace.json", **common)
    wrapper_path = tmp_path / "profile_wrapper.py"
    wrapper_path.write_text(wrapper, encoding="utf-8")
    environment = {
        **os.environ,
        "PYTHONPATH": str(CODE_ROOT),
        "CUDA_VISIBLE_DEVICES": "",
        "AISP_HOOK_MARKER": str(tmp_path / "hook_marker.txt"),
    }
    return subprocess.run(
        [sys.executable, str(wrapper_path)],
        cwd=CODE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize(
    "render",
    [render_nsys_python_profile_wrapper, render_ncu_python_profile_wrapper, render_torch_python_profile_wrapper],
    ids=["nsys", "ncu", "torch"],
)
def test_profile_wrappers_fail_when_overrides_have_no_hook(tmp_path: Path, render: Any) -> None:
    completed = _run_profile_wrapper(tmp_path, render, hook="")

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert (
        "--target-extra-arg was given for ch01:cpu_override, but CpuOverrideBenchmark has no "
        "apply_target_overrides hook"
    ) in completed.stderr


def test_profile_wrapper_runs_a_benchmark_whose_hook_accepts_the_overrides(tmp_path: Path) -> None:
    completed = _run_profile_wrapper(tmp_path, render_nsys_python_profile_wrapper, hook=_RECORDING_HOOK)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "hook_marker.txt").read_text(encoding="utf-8") == "--scale 3"
