"""Exercise chapter verification on timed CPU outputs from isolated workers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.harness import run_benchmarks
from tests.test_e2e_failure_isolation_resume import _write_real_cpu_pair


@pytest.mark.parametrize(
    ("chapter_name", "optimized_tolerance", "offset", "matching_map", "status"),
    [
        ("ch94", (0.0, 0.0), 0.5, False, "failed_verification"),
        ("ch95", (0.0, 2.0), 0.0, False, "failed_verification"),
        ("ch96", (1e-2, 0.0), 0.0, True, "failed_verification"),
        ("ch97", (0.0, 1.0), 0.5, False, "succeeded"),
        ("ch98", (0.0, 0.0), 0.5, True, "succeeded"),
    ],
    ids=["stricter", "looser-absolute", "looser-relative-with-map", "equal", "matching-map"],
)
def test_chapter_applies_pair_output_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chapter_name: str,
    optimized_tolerance: tuple[float, float], offset: float, matching_map: bool, status: str,
) -> None:
    chapter = tmp_path / chapter_name
    chapter.mkdir()
    _write_real_cpu_pair(chapter)
    baseline = chapter / "baseline_real_cpu.py"
    optimized = chapter / "optimized_real_cpu.py"
    per_output = ', output_tolerances={"output": (0.0, 1.0)}' if matching_map else ""
    for path, tolerance, arm_offset in (
        (baseline, (0.0, 1.0), 0.0),
        (optimized, optimized_tolerance, offset),
    ):
        path.write_text(
            path.read_text()
            .replace("(self.input + 1.0).square()", f"(self.input + 1.0).square() + {arm_offset}")
            .replace("output_tolerance=(0.0, 0.0)", f"output_tolerance={tolerance}{per_output}")
        )

    # Allow the chapter control plane to reach its CPU-enabled fixture.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    actual_torch = run_benchmarks.torch

    class ChapterTorch:
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)

        def __getattr__(self, name):
            return getattr(actual_torch, name)

    monkeypatch.setattr(run_benchmarks, "torch", ChapterTorch())
    monkeypatch.setattr(run_benchmarks, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(run_benchmarks, "detect_expectation_key", lambda: "cpu")
    monkeypatch.setattr(run_benchmarks, "get_gpu_state", lambda **_kwargs: {})
    monkeypatch.setattr(run_benchmarks, "reset_cuda_state", lambda **_kwargs: None)
    monkeypatch.setattr(run_benchmarks, "reset_gpu_state", lambda: None)
    monkeypatch.setattr(run_benchmarks, "clean_build_directories", lambda _path: None)

    result = run_benchmarks.test_chapter(
        chapter,
        only_examples=["real_cpu"],
        iterations=1,
        warmup=5,
        enforce_environment_validation=False,
        validity_profile="portable",
    )

    entry, = result["benchmarks"]
    assert entry["status"] == status, entry
    variant, = entry["optimizations"]
    receipt = variant["verification"]
    optimized_rtol, optimized_atol = optimized_tolerance
    assert receipt["passed"] == (status == "succeeded")
    assert receipt["baseline_tolerance"]["atol"] == 1.0
    assert receipt["optimized_tolerance"]["rtol"] == optimized_rtol
    assert receipt["optimized_tolerance"]["atol"] == optimized_atol
    if optimized_rtol > 0.0 or optimized_atol > 1.0:
        assert "cannot be looser" in variant["error"]
    else:
        assert receipt["atol"] == optimized_atol
        if status == "failed_verification":
            assert receipt["max_diff"] == pytest.approx(offset)
