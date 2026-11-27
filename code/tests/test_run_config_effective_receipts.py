"""Real CPU checks for run-level locks and effective configuration receipts."""

from __future__ import annotations

import argparse
import importlib
import inspect
import textwrap
from pathlib import Path

import pytest
import torch

from core.benchmark.models import BenchmarkResult, TimingStats
from core.harness import benchmark_harness as harness_module
from core.harness import run_benchmarks as run_benchmarks_module
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    ExecutionMode,
    LaunchVia,
    _benchmark_config_receipt,
    _is_chapter_or_labs_benchmark,
)
from core.harness.run_benchmarks import (
    _expectation_provenance_from_result,
    _timing_configuration_receipt,
)
from core.utils.chapter_compare_template import load_benchmark


class _ConfigReceiptBenchmark(BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.mode = "forward"
        self.output: torch.Tensor | None = None

    def apply_target_overrides(self, argv: list[str]) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--mode", choices=("forward", "fwd_bwd"), required=True)
        args = parser.parse_args(argv)
        self.mode = args.mode

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            device=self.device,
            iterations=20 if self.mode == "forward" else 10,
            warmup=5,
            adaptive_iterations=False,
            force_synchronize=False,
            single_gpu=False,
            profile_type="minimal",
            deterministic=False,
            seed=7,
            enable_memory_tracking=False,
        )

    def setup(self) -> None:
        self.input = torch.tensor([1.0], dtype=torch.float32)
        self.output = None

    def benchmark_fn(self) -> None:
        self.output = self.input + 0.0

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        return {"mode": torch.tensor([int(self.mode == "fwd_bwd")], dtype=torch.int64)}

    def get_verify_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")
        return self.output

    def get_output_tolerance(self) -> tuple[float, float]:
        return 0.0, 0.0

    def get_input_signature(self) -> dict[str, str]:
        return {"mode": self.mode}

    def validate_result(self) -> str | None:
        if self.output is None or not torch.equal(self.output, torch.tensor([1.0])):
            return "CPU control output mismatch"
        return None

    def get_custom_metrics(self) -> dict[str, float]:
        config = self._config
        return {
            "probe.mode_fwd_bwd": float(self.mode == "fwd_bwd"),
            "probe.iterations": float(config.iterations),
            "probe.warmup": float(config.warmup),
            "probe.force_synchronize": float(config.force_synchronize),
            "probe.single_gpu": float(config.single_gpu),
            "probe.profile_none": float(config.profile_type == "none"),
            "probe.deterministic": float(config.deterministic),
            "probe.seed": float(config.seed or 0),
        }


def _cpu_config(*, use_subprocess: bool, locked_fields: list[str]) -> BenchmarkConfig:
    return BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=3,
        warmup=5,
        adaptive_iterations=False,
        use_subprocess=use_subprocess,
        execution_mode=ExecutionMode.SUBPROCESS if use_subprocess else ExecutionMode.THREAD,
        force_synchronize=True,
        single_gpu=True,
        profile_type="none",
        enable_profiling=False,
        deterministic=True,
        seed=42,
        locked_fields=locked_fields,
        target_label="labs/top_k_kernel:top_k",
        target_extra_args={
            "labs/top_k_kernel:top_k": ["--mode", "fwd_bwd"],
        },
        enable_memory_tracking=False,
        enforce_environment_validation=False,
        detect_setup_precomputation=False,
        clear_compile_cache=False,
        reset_memory_pool=False,
        audit_stream_sync=False,
        cross_validate_timing=False,
        monitor_gpu_state=False,
        track_memory_allocations=False,
    )


@pytest.mark.parametrize("use_subprocess", [False, True], ids=["thread", "subprocess"])
def test_explicit_run_settings_survive_worker_merge_and_receipts(use_subprocess: bool) -> None:
    locked_fields = [
        "iterations",
        "warmup",
        "force_synchronize",
        "single_gpu",
        "profile_type",
        "deterministic",
        "seed",
    ]
    benchmark = _ConfigReceiptBenchmark()
    config = _cpu_config(use_subprocess=use_subprocess, locked_fields=locked_fields)

    run = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config).benchmark_with_manifest(
        benchmark,
        run_id=f"config-receipt-{'subprocess' if use_subprocess else 'thread'}",
    )

    assert run.result.errors == []
    assert run.result.custom_metrics == {
        "probe.mode_fwd_bwd": 1.0,
        "probe.iterations": 3.0,
        "probe.warmup": 5.0,
        "probe.force_synchronize": 1.0,
        "probe.single_gpu": 1.0,
        "probe.profile_none": 1.0,
        "probe.deterministic": 1.0,
        "probe.seed": 42.0,
    }
    applied = benchmark._applied_config_receipt
    assert applied["receipt_source"] == "worker"
    assert set(locked_fields) <= set(applied["locked_fields"])
    assert applied["iterations"] == 3
    assert applied["warmup"] == 5
    assert applied["force_synchronize"] is True
    assert applied["single_gpu"] is True
    assert applied["profile_type"] == "none"
    assert applied["deterministic"] is True
    assert applied["seed"] == 42
    assert applied["execution_mode"] == "thread"
    assert applied["use_subprocess"] is False

    assert run.manifest is not None and run.manifest.config is not None
    configured = run.manifest.config["configured_config"]
    assert configured["receipt_source"] == "configured"
    assert configured["execution_mode"] == ("subprocess" if use_subprocess else "thread")
    assert configured["use_subprocess"] is use_subprocess
    assert run.manifest.config["applied_config"] == applied

    timing_receipt = _timing_configuration_receipt(
        iterations=config.iterations,
        warmup=config.warmup,
        adaptive_iterations=config.adaptive_iterations,
        timing=run.result.timing,
        configured_config=_benchmark_config_receipt(config),
        applied_config=applied,
    )
    assert timing_receipt["configured_config"]["execution_mode"] == (
        "subprocess" if use_subprocess else "thread"
    )
    assert timing_receipt["config_receipt_source"] == "worker"
    assert timing_receipt["applied_config"]["execution_mode"] == "thread"
    assert timing_receipt["launcher_config"] is None
    assert timing_receipt["applied_iterations"] == 3
    assert timing_receipt["applied_warmup_iterations"] == 5


class _WorkerSetupFailure(_ConfigReceiptBenchmark):
    def setup(self) -> None:
        raise RuntimeError("isolated worker setup failed")


def test_failed_isolated_worker_has_no_applied_config_receipt() -> None:
    benchmark = _WorkerSetupFailure()
    config = _cpu_config(use_subprocess=True, locked_fields=[])
    run = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config).benchmark_with_manifest(
        benchmark, run_id="missing-worker-receipt",
    )
    assert any("isolated worker setup failed" in failure for failure in run.result.errors)
    assert getattr(benchmark, "_applied_config_receipt", None) is None
    assert "applied_config" not in run.manifest.config
    assert run.manifest.config["configured_config"]["execution_mode"] == "subprocess"


def test_subprocess_fallback_attests_the_thread_that_ran(tmp_path: Path) -> None:
    class PolicyProbe(_ConfigReceiptBenchmark):
        def get_custom_metrics(self) -> dict[str, float]:
            return {
                **super().get_custom_metrics(),
                "probe.actual_deterministic": float(torch.are_deterministic_algorithms_enabled()),
            }

    benchmark = PolicyProbe()
    benchmark._module_file_override = str(tmp_path / "missing_benchmark_module.py")
    config = _cpu_config(use_subprocess=True, locked_fields=[])
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        run = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config).benchmark_with_manifest(
            benchmark, run_id="subprocess-fallback-thread",
        )
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)

    assert not run.result.errors
    assert run.result.custom_metrics["probe.actual_deterministic"] == 1.0
    configured = run.manifest.config["configured_config"]
    applied = run.manifest.config["applied_config"]
    assert configured["execution_mode"] == "subprocess"
    assert applied["receipt_source"] == "worker"
    assert applied["execution_mode"] == "thread"
    assert applied["use_subprocess"] is False


def test_unlocked_mode_dependent_defaults_apply_after_target_override() -> None:
    benchmark = _ConfigReceiptBenchmark()
    config = _cpu_config(use_subprocess=False, locked_fields=["profile_type"])
    config.iterations = BenchmarkConfig().iterations
    config.force_synchronize = False
    config.single_gpu = False
    config.deterministic = False
    config.seed = None

    result = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config).benchmark(benchmark)

    assert result.errors == []
    assert result.custom_metrics["probe.mode_fwd_bwd"] == 1.0
    assert result.custom_metrics["probe.iterations"] == 10.0
    assert benchmark._applied_config_receipt["receipt_source"] == "worker"
    assert benchmark._applied_config_receipt["iterations"] == 10


@pytest.mark.parametrize(
    ("source", "is_chapter_or_lab"),
    [
        ("tests/benchmark.py", False),
        ("ch12/benchmark.py", True),
        ("ch120/benchmark.py", True),
        ("labs/top_k_kernel/benchmark.py", True),
    ],
)
def test_chapter_policy_matches_benchmark_directories(
    monkeypatch: pytest.MonkeyPatch, source: str, is_chapter_or_lab: bool,
) -> None:
    code_root = Path("/dev/labs/checkout/code")
    monkeypatch.setattr(harness_module, "__file__", str(code_root / "core/harness/benchmark_harness.py"))
    monkeypatch.setattr(inspect, "getfile", lambda _type: str(code_root / source))
    assert _is_chapter_or_labs_benchmark(_ConfigReceiptBenchmark()) is is_chapter_or_lab


@pytest.mark.parametrize(
    "target_module",
    [
        "ch12.baseline_nvfp4_mlp",
        "labs.async_input_pipeline.optimized_async_input_pipeline",
    ],
)
def test_direct_factories_preserve_the_target_directory(target_module: str) -> None:
    module = importlib.import_module(target_module)
    benchmark = module.get_benchmark()
    assert benchmark.__class__.__module__.startswith("core.")
    assert Path(benchmark._module_file_override).samefile(module.__file__)
    assert _is_chapter_or_labs_benchmark(benchmark)


def test_shared_class_keeps_chapter_policy_in_the_isolated_worker(tmp_path: Path) -> None:
    module = tmp_path / "ch99" / "baseline_wrapper.py"
    module.parent.mkdir()
    module.write_text(
        "from tests.test_run_config_effective_receipts import _ConfigReceiptBenchmark\n"
        "\ndef get_benchmark():\n    return _ConfigReceiptBenchmark()\n",
    )
    benchmark = load_benchmark(module)
    assert benchmark is not None and _is_chapter_or_labs_benchmark(benchmark)
    config = _cpu_config(use_subprocess=True, locked_fields=["profile_type"])
    config.iterations = BenchmarkConfig().iterations
    config.min_run_time_ms = 0.0
    config.min_total_duration_ms = 0.0

    result = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config).benchmark(benchmark)

    assert not result.errors
    assert result.custom_metrics["probe.mode_fwd_bwd"] == 1.0
    assert result.custom_metrics["probe.iterations"] == config.iterations


def test_timing_and_expectation_receipts_use_worker_applied_counts() -> None:
    receipt = _timing_configuration_receipt(
        iterations=20,
        warmup=5,
        adaptive_iterations=False,
        timing=None,
        configured_config={"iterations": 20, "warmup": 5, "adaptive_iterations": False},
        applied_config={
            "receipt_source": "worker",
            "iterations": 10,
            "warmup": 7,
            "adaptive_iterations": False,
        },
    )

    provenance = _expectation_provenance_from_result(
        {"baseline_timing_configuration": receipt},
        git_commit="a" * 40,
        hardware_key="cpu-config-receipt",
        profile_name="none",
        timestamp="2026-11-27T00:00:00+00:00",
        execution_environment="unknown",
        validity_profile="strict",
        dmi_product_name=None,
    )

    assert receipt["configured_iterations"] == 20
    assert receipt["configured_warmup_iterations"] == 5
    assert receipt["config_receipt_source"] == "worker"
    assert receipt["applied_iterations"] == 10
    assert receipt["applied_warmup_iterations"] == 7
    assert provenance.iterations == 10
    assert provenance.warmup_iterations == 7


def test_torchrun_config_receipt_is_labeled_launcher_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _ConfigReceiptBenchmark()
    config = _cpu_config(use_subprocess=False, locked_fields=[])
    config.launch_via = LaunchVia.TORCHRUN
    config.target_label = None
    config.target_extra_args = {}
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    def _torchrun_result(
        _benchmark: BaseBenchmark,
        resolved_config: BenchmarkConfig,
    ) -> BenchmarkResult:
        assert resolved_config.launch_via == LaunchVia.TORCHRUN
        return BenchmarkResult(
            timing=TimingStats(
                mean_ms=1.0,
                median_ms=1.0,
                std_ms=0.0,
                min_ms=1.0,
                max_ms=1.0,
                iterations=4,
                warmup_iterations=2,
                raw_times_ms=[1.0] * 4,
            ),
            device="cpu",
        )

    monkeypatch.setattr(harness, "_benchmark_with_torchrun", _torchrun_result)

    run = harness.benchmark_with_manifest(benchmark, run_id="launcher-receipt")

    launcher_receipt = benchmark._applied_config_receipt
    assert launcher_receipt["receipt_source"] == "launcher"
    assert launcher_receipt["launch_via"] == "torchrun"
    assert run.manifest is not None and run.manifest.config is not None
    assert run.manifest.config["launcher_config"] == launcher_receipt
    assert "applied_config" not in run.manifest.config

    timing_receipt = _timing_configuration_receipt(
        iterations=config.iterations,
        warmup=config.warmup,
        adaptive_iterations=config.adaptive_iterations,
        timing=run.result.timing,
        configured_config=_benchmark_config_receipt(config, source="configured"),
        applied_config=launcher_receipt,
    )
    assert timing_receipt["config_receipt_source"] == "launcher"
    assert timing_receipt["launcher_config"] == launcher_receipt
    assert timing_receipt["applied_config"] is None
    assert timing_receipt["applied_iterations"] is None
    provenance = _expectation_provenance_from_result(
        {"baseline_timing_configuration": timing_receipt},
        git_commit="a" * 40,
        hardware_key="cpu-launcher-receipt",
        profile_name="none",
        timestamp="2026-11-27T00:00:00+00:00",
        execution_environment="unknown",
        validity_profile="strict",
        dmi_product_name=None,
    )
    assert provenance.iterations == 4
    assert provenance.warmup_iterations == 2


def _write_chapter_pair(chapter_dir: Path, *, use_subprocess: bool) -> None:
    source = textwrap.dedent(
        f"""\
        from __future__ import annotations

        import argparse
        import torch

        from core.benchmark.verification_mixin import VerificationPayloadMixin
        from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


        class CpuConfigReceiptBenchmark(VerificationPayloadMixin, BaseBenchmark):
            allow_cpu = True
            signature_equivalence_group = "chapter_config_receipt"
            signature_equivalence_ignore_fields = ("precision_flags",)

            def __init__(self):
                super().__init__()
                self.device = torch.device("cpu")
                self.mode = "forward"
                self.input = torch.arange(4, dtype=torch.float32)
                self.output = None
                self.register_workload_metadata(samples_per_iteration=4.0)

            def apply_target_overrides(self, argv):
                parser = argparse.ArgumentParser(add_help=False)
                parser.add_argument("--mode", choices=("forward", "fwd_bwd"), required=True)
                self.mode = parser.parse_args(argv).mode

            def setup(self):
                self.output = None

            def benchmark_fn(self):
                self.output = self.input + 1.0

            def capture_verification_payload(self):
                if self.output is None:
                    raise RuntimeError("benchmark_fn did not execute")
                self._set_verification_payload(
                    inputs={{"input": self.input}},
                    output=self.output,
                    batch_size=4,
                    output_tolerance=(0.0, 0.0),
                )

            def validate_result(self):
                expected = self.input + 1.0
                if self.output is None or not torch.equal(self.output, expected):
                    return "CPU config receipt output mismatch"
                return None

            def get_config(self):
                return BenchmarkConfig(
                    device=self.device,
                    iterations=99 if self.mode == "forward" else 10,
                    warmup=9,
                    use_subprocess={use_subprocess!r},
                    adaptive_iterations=False,
                    min_run_time_ms=0.0,
                    min_total_duration_ms=0.0,
                    force_synchronize=False,
                    single_gpu=False,
                    profile_type="minimal",
                    deterministic=False,
                    seed=7,
                    enable_memory_tracking=False,
                )

            def get_custom_metrics(self):
                config = self._config
                return {{
                    "probe.mode_fwd_bwd": float(self.mode == "fwd_bwd"),
                    "probe.iterations": float(config.iterations),
                    "probe.warmup": float(config.warmup),
                    "probe.force_synchronize": float(config.force_synchronize),
                    "probe.single_gpu": float(config.single_gpu),
                    "probe.profile_none": float(config.profile_type == "none"),
                    "probe.deterministic": float(config.deterministic),
                    "probe.seed": float(config.seed or 0),
                }}

            def get_optimization_goal(self):
                return "comparison"


        def get_benchmark():
            return CpuConfigReceiptBenchmark()
        """
    )
    (chapter_dir / "baseline_config_receipt.py").write_text(source, encoding="utf-8")
    (chapter_dir / "optimized_config_receipt.py").write_text(source, encoding="utf-8")


def _open_cpu_control_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        run_benchmarks_module,
        "dump_environment_and_capabilities",
        lambda: None,
    )
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
    monkeypatch.setattr(
        run_benchmarks_module,
        "reset_cuda_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(run_benchmarks_module, "reset_gpu_state", lambda: None)
    monkeypatch.setattr(
        run_benchmarks_module,
        "clean_build_directories",
        lambda _path: None,
    )


@pytest.mark.parametrize("use_subprocess", [False, True], ids=["thread", "subprocess"])
def test_chapter_runner_preserves_explicit_settings_and_effective_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_subprocess: bool,
) -> None:
    chapter_dir = tmp_path / "ch99"
    chapter_dir.mkdir()
    _write_chapter_pair(chapter_dir, use_subprocess=use_subprocess)
    _open_cpu_control_gate(monkeypatch)

    result = run_benchmarks_module._test_chapter_impl(
        chapter_dir,
        enable_profiling=False,
        profile_type="none",
        reproducible=True,
        iterations=3,
        warmup=5,
        single_gpu=True,
        force_synchronize=True,
        validity_profile="portable",
        enforce_environment_validation=False,
        only_examples=["config_receipt"],
        target_extra_args={
            "ch99:config_receipt": ["--mode", "fwd_bwd"],
        },
        verify_input=True,
        verify_output=True,
        only_python=True,
        fail_on_no_benchmarks=True,
    )

    assert result["status"] == "completed"
    benchmark_result = result["benchmarks"][0]
    assert benchmark_result["status"] == "succeeded"
    expected_metrics = {
        "probe.mode_fwd_bwd": 1.0,
        "probe.iterations": 3.0,
        "probe.warmup": 5.0,
        "probe.force_synchronize": 1.0,
        "probe.single_gpu": 1.0,
        "probe.profile_none": 1.0,
        "probe.deterministic": 1.0,
        "probe.seed": 42.0,
    }
    assert benchmark_result["baseline_custom_metrics"] == expected_metrics
    assert benchmark_result["optimizations"][0]["custom_metrics"] == expected_metrics

    receipt = benchmark_result["baseline_timing_configuration"]
    configured = receipt["configured_config"]
    applied = receipt["applied_config"]
    assert receipt["config_receipt_source"] == "worker"
    assert configured["receipt_source"] == "configured"
    assert applied["receipt_source"] == "worker"
    expected_locks = {
        "iterations",
        "warmup",
        "force_synchronize",
        "single_gpu",
        "profile_type",
        "deterministic",
        "seed",
    }
    assert expected_locks <= set(configured["locked_fields"])
    assert expected_locks <= set(applied["locked_fields"])
    assert configured["execution_mode"] == ("subprocess" if use_subprocess else "thread")
    assert configured["use_subprocess"] is use_subprocess
    assert applied["execution_mode"] == "thread"
    assert applied["use_subprocess"] is False
    assert applied["iterations"] == receipt["applied_iterations"] == 3
    assert applied["warmup"] == receipt["applied_warmup_iterations"] == 5
    assert applied["force_synchronize"] is True
    assert applied["single_gpu"] is True
    assert applied["profile_type"] == "none"
    assert applied["deterministic"] is True
    assert applied["seed"] == 42

    baseline_manifest = next(
        entry["manifest"] for entry in result["manifests"] if entry["variant"] == "baseline"
    )
    assert baseline_manifest["config"]["configured_config"] == configured
    assert baseline_manifest["config"]["applied_config"] == applied
