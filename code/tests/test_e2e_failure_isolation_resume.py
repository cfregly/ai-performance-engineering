"""Focused controls for full-sweep target isolation, routing, and resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.benchmark import bench_commands, e2e_sweep
from core.harness import run_benchmarks as run_benchmarks_module

_CODE_ROOT = Path(__file__).resolve().parents[1]


def _summary(*outcomes: tuple[str, str]) -> dict[str, object]:
    return {
        "target_outcomes": [
            {"target": target, "status": status} for target, status in outcomes
        ]
    }


def test_resume_retains_later_successes_by_frozen_target_identity() -> None:
    frozen = ["ch01:a", "ch01:b", "ch02:c"]
    first_attempt = {
        "verified_targets": frozen,
        "benchmark_summary": _summary(
            ("ch01:a", "failed_error"),
            ("ch01:b", "succeeded"),
            ("ch02:c", "succeeded"),
        ),
    }

    successes = e2e_sweep._successful_targets_from_attempts(
        [first_attempt], frozen_targets=frozen
    )

    assert successes == ["ch01:b", "ch02:c"]
    assert e2e_sweep._remaining_targets_after_successful_targets(
        frozen, successful_targets=successes
    ) == ["ch01:a"]
    assert e2e_sweep._completed_units_from_successful_targets(
        frozen, successful_targets=successes
    ) == ["ch02"]

    retry = {
        "verified_targets": ["ch01:a"],
        "benchmark_summary": _summary(("ch01:a", "succeeded")),
    }
    assert e2e_sweep._successful_targets_from_attempts(
        [first_attempt, retry], frozen_targets=frozen
    ) == frozen


def test_resume_uses_latest_verified_outcome_and_rejects_invalid_identity() -> None:
    frozen = ["ch01:a", "ch01:b", "ch02:c"]
    first_attempt = {
        "verified_targets": frozen,
        "benchmark_summary": _summary(
            ("ch01:a", "succeeded"),
            ("ch01:b", "succeeded"),
            ("ch02:c", "succeeded"),
        ),
    }
    retry = {
        "verified_targets": ["ch01:b"],
        "benchmark_summary": _summary(("ch01:b", "failed_verification")),
    }
    reordered = {
        "verified_targets": ["ch02:c", "ch01:b"],
        "benchmark_summary": _summary(
            ("ch02:c", "succeeded"),
            ("ch01:b", "succeeded"),
        ),
    }

    successes = e2e_sweep._successful_targets_from_attempts(
        [first_attempt, retry, reordered], frozen_targets=frozen
    )

    assert successes == ["ch01:a", "ch02:c"]
    assert e2e_sweep._remaining_targets_after_successful_targets(
        frozen, successful_targets=successes
    ) == ["ch01:b"]


def test_newer_verified_attempt_without_summary_invalidates_older_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = ["ch01:a", "ch01:b"]
    attempts = [
        {
            "run_id": "initial",
            "verified_targets": frozen,
            "benchmark_summary": _summary(
                ("ch01:a", "succeeded"),
                ("ch01:b", "succeeded"),
            ),
        },
        {
            "run_id": "retry",
            "verified_targets": ["ch01:a"],
        },
    ]
    monkeypatch.setattr(
        e2e_sweep,
        "_validate_benchmark_run_manifest",
        lambda **_kwargs: None,
    )

    verified, issues = e2e_sweep._verified_full_sweep_attempts(
        attempts,
        repo_root=tmp_path,
        artifacts_dir=None,
        expected_git_commit="a" * 40,
    )

    assert verified == attempts
    assert issues == ["retry: benchmark output lacks a valid summary"]
    assert e2e_sweep._successful_targets_from_attempts(
        verified,
        frozen_targets=frozen,
    ) == ["ch01:b"]


def test_static_routing_reads_attached_literal_metadata_without_import(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch99"
    chapter_dir.mkdir()
    (chapter_dir / "shared.py").write_text(
        "raise AssertionError('routing must not import workload modules')\n"
        "class SharedBenchmark:\n"
        "    pass\n",
        encoding="utf-8",
    )
    wrapper = (
        "from ch99.shared import SharedBenchmark\n"
        "from core.benchmark.wrapper_utils import attach_benchmark_metadata\n"
        "def get_benchmark():\n"
        "    bench = SharedBenchmark(multi_gpu=True)\n"
        "    return attach_benchmark_metadata(bench, __file__)\n"
    )
    (chapter_dir / "baseline_attached.py").write_text(wrapper, encoding="utf-8")
    (chapter_dir / "optimized_attached.py").write_text(wrapper, encoding="utf-8")

    routing = bench_commands._collect_benchmark_routing(chapter_dir)

    assert routing["attached"].minimum_gpu_count == 2
    assert routing["attached"].requires_torchrun is False


def test_observed_multigpu_misroutes_use_declared_source_metadata() -> None:
    required = {
        "ch04:gradient_compression_fp16_comm_only_multigpu",
        "ch04:gradient_compression_fp16_multigpu",
        "ch04:gradient_compression_int8_comm_only_multigpu",
        "ch04:gradient_compression_int8_multigpu",
        "ch15:prefill_decode_disagg_multigpu",
    }
    explicitly_single = {"ch04:disaggregated", "ch04:reinit_comm"}

    inventory = {
        entry["target"]: entry for entry in e2e_sweep._iter_discovered_targets(_CODE_ROOT)
    }

    assert required <= inventory.keys()
    assert all(inventory[target]["minimum_gpu_count"] >= 2 for target in required)
    assert all(inventory[target]["multi_gpu"] is True for target in required)
    assert all(inventory[target]["requires_torchrun"] is False for target in required)
    assert all(inventory[target]["minimum_gpu_count"] == 1 for target in explicitly_single)
    assert all(inventory[target]["multi_gpu"] is False for target in explicitly_single)


def _write_real_cpu_pair(chapter_dir: Path) -> None:
    source = '''from __future__ import annotations
import torch
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

class CpuControlBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True
    signature_equivalence_group = "e2e_cpu_control"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.input = torch.arange(16, dtype=torch.float32)
        self.output = None
        self.register_workload_metadata(samples_per_iteration=16.0)

    def setup(self):
        self.output = None

    def benchmark_fn(self):
        self.output = (self.input + 1.0).square()

    def capture_verification_payload(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn did not execute")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output,
            batch_size=16,
            output_tolerance=(0.0, 0.0),
        )

    def validate_result(self):
        expected = (self.input + 1.0).square()
        if self.output is None or not torch.equal(self.output, expected):
            return "real CPU output mismatch"
        return None

    def get_config(self):
        return BenchmarkConfig(
            device=self.device,
            iterations=1,
            warmup=5,
            use_subprocess=True,
            adaptive_iterations=False,
            min_run_time_ms=0.0,
            min_total_duration_ms=0.0,
            enable_memory_tracking=False,
        )

    def get_optimization_goal(self):
        return "comparison"

def get_benchmark():
    return CpuControlBenchmark()
'''
    (chapter_dir / "baseline_real_cpu.py").write_text(source, encoding="utf-8")
    (chapter_dir / "optimized_real_cpu.py").write_text(source, encoding="utf-8")


def test_preflight_failure_does_not_block_real_cpu_subprocess_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench_root = tmp_path / "bench"
    chapter_dir = bench_root / "ch99"
    chapter_dir.mkdir(parents=True)
    blocked_source = (
        "def get_benchmark():\n"
        "    raise AssertionError('preflight-blocked target factory was called')\n"
    )
    (chapter_dir / "baseline_blocked.py").write_text(blocked_source, encoding="utf-8")
    (chapter_dir / "optimized_blocked.py").write_text(blocked_source, encoding="utf-8")
    _write_real_cpu_pair(chapter_dir)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(bench_commands, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(bench_commands, "get_gpu_state", lambda **_kwargs: {})
    monkeypatch.setattr(run_benchmarks_module, "dump_environment_and_capabilities", lambda: None)
    # test_chapter has a process-level CUDA gate before it can discover an
    # allow_cpu benchmark. Keep that control-plane gate open while the real
    # isolated child still sees CUDA_VISIBLE_DEVICES="" and executes on CPU.
    actual_torch = run_benchmarks_module.torch

    class _CudaGateProxy:
        def __getattr__(self, name):
            return getattr(actual_torch.cuda, name)

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    class _TorchGateProxy:
        cuda = _CudaGateProxy()

        def __getattr__(self, name):
            return getattr(actual_torch, name)

    monkeypatch.setattr(run_benchmarks_module, "torch", _TorchGateProxy())
    monkeypatch.setattr(run_benchmarks_module, "detect_expectation_key", lambda: "cpu")
    monkeypatch.setattr(run_benchmarks_module, "get_gpu_state", lambda **_kwargs: {})
    monkeypatch.setattr(run_benchmarks_module, "reset_cuda_state", lambda **_kwargs: None)
    monkeypatch.setattr(run_benchmarks_module, "reset_gpu_state", lambda: None)
    monkeypatch.setattr(run_benchmarks_module, "clean_build_directories", lambda _path: None)

    def selective_preflight(_chapter_dirs, chapter_filters, **_kwargs):
        examples = {
            example for selected in chapter_filters.values() for example in selected
        }
        return ["blocked by real control preflight"] if examples == {"blocked"} else []

    monkeypatch.setattr(
        bench_commands,
        "_preflight_target_coverage_and_assets",
        selective_preflight,
    )
    real_test_chapter = bench_commands.test_chapter

    def cpu_control_test_chapter(**kwargs):
        kwargs["enforce_environment_validation"] = False
        return real_test_chapter(**kwargs)

    monkeypatch.setattr(bench_commands, "test_chapter", cpu_control_test_chapter)

    result = bench_commands._execute_benchmarks(
        targets=["ch99:blocked", "ch99:real_cpu"],
        bench_root=bench_root,
        output_format="json",
        profile_type="none",
        suite_timeout=0,
        validity_profile="portable",
        iterations=1,
        warmup=5,
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="e2e_failure_isolation_cpu_control",
        exit_on_failure=False,
    )

    assert result["total_failed"] == 1
    assert result["total_successful"] == 1
    assert [entry["status"] for entry in result["results"]] == [
        "failed_preflight",
        "completed",
    ]
    successful = result["results"][1]["benchmarks"][0]
    assert successful["example"] == "real_cpu"
    assert successful["status"] == "succeeded"
    assert successful["baseline_time_ms"] > 0
    assert successful["optimizations"][0]["time_ms"] > 0
    assert successful["verification"]["passed"] is True

    persisted = json.loads(Path(result["output_json"]).read_text(encoding="utf-8"))
    assert [entry["status"] for entry in persisted["results"]] == [
        "failed_preflight",
        "completed",
    ]


def test_all_explicit_preflight_failures_still_get_terminal_target_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench_root = tmp_path / "bench"
    chapter_dir = bench_root / "ch99"
    chapter_dir.mkdir(parents=True)
    discoverable = "def get_benchmark():\n    raise AssertionError('must not execute')\n"
    for example in ("first", "second"):
        (chapter_dir / f"baseline_{example}.py").write_text(
            discoverable, encoding="utf-8"
        )
        (chapter_dir / f"optimized_{example}.py").write_text(
            discoverable, encoding="utf-8"
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(bench_commands, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(bench_commands, "get_gpu_state", lambda **_kwargs: {})

    def reject_each(_chapter_dirs, chapter_filters, **_kwargs):
        example = next(iter(next(iter(chapter_filters.values()))))
        return [f"{example} rejected by control preflight"]

    monkeypatch.setattr(
        bench_commands,
        "_preflight_target_coverage_and_assets",
        reject_each,
    )
    monkeypatch.setattr(
        bench_commands,
        "test_chapter",
        lambda **_kwargs: pytest.fail("preflight-rejected target was executed"),
    )

    result = bench_commands._execute_benchmarks(
        targets=["ch99:first", "ch99:second"],
        bench_root=bench_root,
        output_format="json",
        profile_type="none",
        suite_timeout=0,
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="all_preflight_failures_are_terminal",
        exit_on_failure=False,
    )

    assert result["total_failed"] == 2
    assert result["preflight_failed"] is True
    assert [entry["status"] for entry in result["results"]] == [
        "failed_preflight",
        "failed_preflight",
    ]
    assert [
        entry["benchmarks"][0]["example"] for entry in result["results"]
    ] == ["first", "second"]
