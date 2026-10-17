"""Regression coverage for audit finding W1-112.

The warp specialization CI wrapper must import every comparison pair through
the real wrapper and fail loudly when a pair or CUDA is missing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import tests.test_warp_specialization as warp_compare
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.benchmark_harness import BaseBenchmark

CODE_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = CODE_ROOT / "core/scripts/run_warp_specialization_ci.sh"


def test_wrapper_imports_every_pair_then_fails_loudly_without_cuda() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHON"] = sys.executable
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=CODE_ROOT.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    labels = ", ".join(pair.label for pair in warp_compare.COMPARISON_PAIRS)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        f"SKIPPED: warp specialization comparison ({labels}) requires a CUDA GPU"
        in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert "Error:" not in result.stdout


def test_comparison_pairs_resolve_to_in_process_benchmarks() -> None:
    resolved = warp_compare.resolve_comparison_pairs()

    assert resolved
    for pair, baseline_class, optimized_class in resolved:
        for target, benchmark_class in (
            (pair.baseline, baseline_class),
            (pair.optimized, optimized_class),
        ):
            assert issubclass(benchmark_class, BaseBenchmark)
            assert not issubclass(benchmark_class, CudaBinaryBenchmark)
            assert target == f"{benchmark_class.__module__}:{benchmark_class.__name__}"


@pytest.mark.parametrize(
    ("target", "error"),
    (
        ("ch13.no_such_warp_specialization_module:Missing", ModuleNotFoundError),
        ("ch13.baseline_warp_specialization_training:NoSuchBenchmark", AttributeError),
    ),
)
def test_missing_pair_target_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, target: str, error: type[Exception]
) -> None:
    monkeypatch.setattr(warp_compare, "_cli_iterations", None)
    monkeypatch.setattr(warp_compare, "_cli_warmup", None)
    real_pair = warp_compare.COMPARISON_PAIRS[0]
    monkeypatch.setattr(
        warp_compare,
        "COMPARISON_PAIRS",
        (real_pair._replace(label="broken", optimized=target),),
    )

    with pytest.raises(error):
        warp_compare.main(["--iterations", "1", "--warmup", "0"])


def test_benchmark_config_failure_is_not_swallowed() -> None:
    class _BrokenConfigBenchmark:
        def get_config(self) -> None:
            raise RuntimeError("config load failed")

    with pytest.raises(RuntimeError, match="config load failed"):
        warp_compare._resolve_loop_counts(_BrokenConfigBenchmark(), 10, 5)


def test_removed_test_mode_flag_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(warp_compare, "_cli_iterations", None)
    monkeypatch.setattr(warp_compare, "_cli_warmup", None)

    with pytest.raises(SystemExit) as exc_info:
        warp_compare.main(["--test-mode", "triton"])

    assert exc_info.value.code == 2
