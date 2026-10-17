from __future__ import annotations

import argparse
import importlib.util
import sys
import textwrap
from typing import Optional

import pytest
import torch

from core.benchmark.target_overrides import parse_target_extra_args
from core.harness import run_benchmarks as run_benchmarks_module
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    _lookup_target_extra_args,
)


@pytest.mark.parametrize(
    "entries",
    [
        ["missing-separator"],
        ["=--flag value"],
        ["ch01:gemm="],
        ['ch01:gemm="unterminated'],
        ["ch01:gemm=--one", "ch01:gemm=--two"],
        ["labs/demo:case=--one", "labs_demo:case=--two"],
    ],
)
def test_target_extra_args_reject_malformed_or_duplicate_entries(entries: list[str]) -> None:
    with pytest.raises(ValueError, match="target-extra-arg"):
        parse_target_extra_args(entries)


def test_target_extra_args_parse_quoted_arguments() -> None:
    assert parse_target_extra_args(['labs/demo:case=--label "two words"']) == {
        "labs/demo:case": ["--label", "two words"]
    }


def test_target_extra_arg_preflight_rejects_unmatched_target() -> None:
    issues = run_benchmarks_module._preflight_target_coverage_and_assets(
        [run_benchmarks_module.repo_root / "ch01"],
        {"ch01": {"gemm"}},
        only_cuda=False,
        only_python=False,
        target_extra_args={"ch01:not_selected": ["--flag"]},
        enforce_external_assets=False,
    )

    assert any("does not match any selected benchmark" in issue for issue in issues)


def test_target_extra_arg_preflight_rejects_cuda_binary_wrapper() -> None:
    issues = run_benchmarks_module._preflight_target_coverage_and_assets(
        [run_benchmarks_module.repo_root / "ch01"],
        {"ch01": {"gemm"}},
        only_cuda=True,
        only_python=False,
        target_extra_args={"ch01:gemm": ["--m", "64"]},
        enforce_external_assets=False,
    )

    assert any("selects a CUDA binary wrapper" in issue for issue in issues)


class OverrideAwareBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.mode = "forward"
        self.mode_seen_by_get_config: Optional[str] = None
        self.output: Optional[torch.Tensor] = None

    def apply_target_overrides(self, argv: list[str]) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--mode", choices=("forward", "fwd_bwd"), default=None)
        args, _ = parser.parse_known_args(argv)
        if args.mode:
            self.mode = args.mode

    def get_config(self) -> BenchmarkConfig:
        self.mode_seen_by_get_config = self.mode
        return BenchmarkConfig(
            iterations=1,
            warmup=0,
            enable_profiling=False,
            enable_memory_tracking=False,
            timeout_multiplier=1.0,
        )

    def setup(self) -> None:
        self.output = None

    def benchmark_fn(self) -> None:
        self.output = torch.tensor([1.0], dtype=torch.float32)

    def validate_result(self) -> Optional[str]:
        return None

    def get_verify_inputs(self):
        return {"mode": torch.tensor([0 if self.mode == "forward" else 1], dtype=torch.int64)}

    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("Output not produced")
        return self.output

    def get_output_tolerance(self):
        return (0.0, 0.0)

    def get_input_signature(self) -> dict:
        return {"mode": self.mode}


def test_lookup_target_extra_args_matches_slash_and_underscore_chapter_labels() -> None:
    overrides = {
        "labs/moe_cuda_ptx:moe_layer": ["--mode", "fwd_bwd"],
    }

    assert _lookup_target_extra_args(overrides, "labs/moe_cuda_ptx:moe_layer") == ["--mode", "fwd_bwd"]
    assert _lookup_target_extra_args(overrides, "labs_moe_cuda_ptx:moe_layer") == ["--mode", "fwd_bwd"]


def test_harness_applies_target_overrides_before_get_config() -> None:
    benchmark = OverrideAwareBenchmark()
    config = BenchmarkConfig(
        iterations=1,
        warmup=0,
        enable_profiling=False,
        enable_memory_tracking=False,
        allow_foreign_gpu_processes=True,
        enforce_environment_validation=False,
        target_label="labs_moe_cuda_ptx:moe_layer",
        target_extra_args={"labs/moe_cuda_ptx:moe_layer": ["--mode", "fwd_bwd"]},
        timeout_multiplier=1.0,
    )

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)

    assert result.errors == []
    assert benchmark.mode_seen_by_get_config == "fwd_bwd"
    assert benchmark.mode == "fwd_bwd"


def test_subprocess_worker_import_receives_target_extra_args(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "import_arg_benchmark.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import argparse
            import os

            import torch

            from core.harness.benchmark_harness import BaseBenchmark

            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--scale", type=float, default=1.0)
            import_args, _ = parser.parse_known_args()


            class ImportArgBenchmark(BaseBenchmark):
                allow_cpu = True

                def apply_target_overrides(self, argv):
                    # The worker presents the same values as sys.argv while it
                    # imports this module. The hook declares that the override
                    # is accepted and checks that both readings agree.
                    hook_args, unknown = parser.parse_known_args(argv)
                    if unknown or hook_args.scale != import_args.scale:
                        raise ValueError(f"override {argv} does not match the import-time arguments")

                def setup(self):
                    self.input = torch.arange(4, dtype=torch.float32)
                    self.output = None

                def benchmark_fn(self):
                    os.write(1, b"native worker diagnostic before JSON\\n")
                    self.output = self.input * import_args.scale

                def get_verify_inputs(self):
                    return {"input": self.input}

                def get_verify_output(self):
                    if self.output is None:
                        raise RuntimeError("CPU benchmark did not execute")
                    return self.output

                def get_input_signature(self):
                    return {"scale": import_args.scale}

                def get_output_tolerance(self):
                    return (0.0, 0.0)

                def validate_result(self):
                    torch.testing.assert_close(
                        self.output,
                        self.input * import_args.scale,
                        rtol=0.0,
                        atol=0.0,
                    )
            """
        ),
        encoding="utf-8",
    )
    module_name = "target_override_import_arg_benchmark"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    benchmark = module.ImportArgBenchmark()
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=0,
        use_subprocess=False,
        enable_profiling=False,
        enable_memory_tracking=False,
        enforce_environment_validation=False,
        target_label="labs/example:import_arg",
        target_extra_args={"labs/example:import_arg": ["--scale", "3"]},
        subprocess_stderr_dir=str(tmp_path / "worker-logs"),
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    harness._ensure_runtime_initialized()

    result = harness._benchmark_with_subprocess(benchmark, config)

    assert result.errors == []
    logs = list((tmp_path / "worker-logs").glob("*_subprocess.stdout.log"))
    assert len(logs) == 1
    assert "native worker diagnostic before JSON" in logs[0].read_text()
    torch.testing.assert_close(
        benchmark._subprocess_verify_output,
        torch.tensor([0.0, 3.0, 6.0, 9.0]),
        rtol=0.0,
        atol=0.0,
    )
