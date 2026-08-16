"""Regression tests for harness immutability guarantees.

These tests ensure benchmarks cannot mutate harness config at runtime and cannot
reseed RNGs during perf runs.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch


from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    LaunchVia,
    TorchrunLaunchSpec,
)


class _CpuBenchmarkBase(BaseBenchmark):
    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        # Force CPU execution even when CUDA is available in the test environment.
        self.device = torch.device("cpu")
        self.input_tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def validate_result(self) -> Optional[str]:
        return None

    def get_verify_inputs(self) -> dict:
        if self.input_tensor is None:
            raise RuntimeError("setup() must set input_tensor")
        return {"input": self.input_tensor}

    def get_verify_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_input_signature(self) -> dict:
        if self.input_tensor is None:
            raise RuntimeError("setup() must be called before get_input_signature()")
        return {"shape": tuple(self.input_tensor.shape), "dtype": str(self.input_tensor.dtype)}


class MutateConfigBenchmark(_CpuBenchmarkBase):
    def setup(self) -> None:
        cfg = self.get_config()
        if cfg is None:
            raise RuntimeError("Missing config in setup()")
        # Benchmarks must not mutate harness config; this should fail fast.
        cfg.iterations = 1  # type: ignore[attr-defined]

    def benchmark_fn(self) -> None:
        self.input_tensor = torch.zeros(8, device=self.device)
        self.output = self.input_tensor


class MutateSeedBenchmark(_CpuBenchmarkBase):
    def setup(self) -> None:
        torch.manual_seed(123)
        self.input_tensor = torch.randn(8, device=self.device)

    def benchmark_fn(self) -> None:
        self.output = self.input_tensor + 1


def _make_cpu_harness() -> BenchmarkHarness:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=2,
        warmup=5,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
        enforce_environment_validation=False,
    )
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)


def test_benchmark_config_is_read_only_during_run():
    harness = _make_cpu_harness()
    result = harness.benchmark(MutateConfigBenchmark())
    assert result.errors, "Expected config mutation to fail"
    assert any("read-only" in err.lower() for err in result.errors)


def test_seed_mutation_is_detected_in_perf_runs():
    harness = _make_cpu_harness()
    result = harness.benchmark(MutateSeedBenchmark())
    assert result.errors, "Expected seed mutation to fail"
    assert any("seed mutation detected" in err.lower() for err in result.errors)


class PersistentConfigBenchmark(_CpuBenchmarkBase):
    def __init__(self) -> None:
        super().__init__()
        self._cfg = BenchmarkConfig(
            device=torch.device("cpu"),
            iterations=7,
            warmup=9,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=False,
        )

    def get_config(self) -> BenchmarkConfig:
        return self._cfg

    def setup(self) -> None:
        self.input_tensor = torch.ones(4, device=self.device)

    def benchmark_fn(self) -> None:
        if self.input_tensor is None:
            raise RuntimeError("setup() must set input_tensor")
        self.output = self.input_tensor + 1


def test_chapter_style_config_normalization_does_not_mutate_benchmark_config(monkeypatch: pytest.MonkeyPatch):
    harness = _make_cpu_harness()
    benchmark = PersistentConfigBenchmark()

    import core.harness.benchmark_harness as benchmark_harness_module

    monkeypatch.setattr(
        benchmark_harness_module,
        "_is_chapter_or_labs_benchmark",
        lambda _benchmark: True,
    )

    result = harness.benchmark(benchmark)

    assert not result.errors
    cfg = benchmark.get_config()
    assert cfg.iterations == 7
    assert cfg.warmup == 9
    assert cfg.timing.timeout_for("torch") == 10800


class TorchrunSeedMutateBenchmark(_CpuBenchmarkBase):
    def __init__(self, script_path: Path) -> None:
        super().__init__()
        self._script_path = script_path

    def setup(self) -> None:  # pragma: no cover - not executed in torchrun mode
        self.input_tensor = torch.zeros(1, device=self.device)

    def benchmark_fn(self) -> None:  # pragma: no cover - not executed in torchrun mode
        if self.input_tensor is None:
            raise RuntimeError("setup() must set input_tensor")
        self.output = self.input_tensor

    def get_torchrun_spec(self, config: BenchmarkConfig) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=self._script_path,
            script_args=[],
            env={},
            parse_rank0_only=True,
            multi_gpu_required=False,
            name="torchrun_seed_mutate",
            config_arg_map={},
        )


def test_seed_mutation_is_detected_in_torchrun_runs(tmp_path: Path):
    if not sys.platform.startswith("linux"):
        pytest.skip("torchrun benchmark validity requires Linux")
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available in PATH")

    script = tmp_path / "seed_mutate_script.py"
    script.write_text(
        "import torch\n"
        "torch.manual_seed(123)\n"
        "print('done')\n",
        encoding="utf-8",
    )

    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=1,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=1,
        multi_gpu_required=False,
        enforce_environment_validation=False,
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(TorchrunSeedMutateBenchmark(script))

    assert result.errors, "Expected torchrun seed mutation to fail"
    joined = "\n".join(result.errors).lower()
    assert "seed mutation detected" in joined


class TorchrunSeedMutateModuleBenchmark(_CpuBenchmarkBase):
    def __init__(self, module_name: str, pythonpath_root: Path) -> None:
        super().__init__()
        self._module_name = module_name
        self._pythonpath_root = pythonpath_root

    def setup(self) -> None:  # pragma: no cover - not executed in torchrun mode
        self.input_tensor = torch.zeros(1, device=self.device)

    def benchmark_fn(self) -> None:  # pragma: no cover - not executed in torchrun mode
        if self.input_tensor is None:
            raise RuntimeError("setup() must set input_tensor")
        self.output = self.input_tensor

    def get_torchrun_spec(self, config: BenchmarkConfig) -> TorchrunLaunchSpec:
        existing = os.environ.get("PYTHONPATH", "")
        entries = [str(self._pythonpath_root)]
        if existing:
            entries.append(existing)
        return TorchrunLaunchSpec(
            module_name=self._module_name,
            script_args=[],
            env={"PYTHONPATH": os.pathsep.join(entries)},
            parse_rank0_only=True,
            multi_gpu_required=False,
            name="torchrun_seed_mutate_module",
            config_arg_map={},
        )


def test_seed_mutation_is_detected_in_torchrun_module_runs(tmp_path: Path):
    if not sys.platform.startswith("linux"):
        pytest.skip("torchrun benchmark validity requires Linux")
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available in PATH")

    pkg = tmp_path / "benchpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "seed_mutate_module.py").write_text(
        "import torch\n"
        "torch.manual_seed(123)\n"
        "print('done')\n",
        encoding="utf-8",
    )

    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=1,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=1,
        multi_gpu_required=False,
        enforce_environment_validation=False,
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(TorchrunSeedMutateModuleBenchmark("benchpkg.seed_mutate_module", tmp_path))

    assert result.errors, "Expected torchrun module seed mutation to fail"
    joined = "\n".join(result.errors).lower()
    assert "seed mutation detected" in joined
