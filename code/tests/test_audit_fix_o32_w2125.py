"""Regression tests for audit finding W2-125.

The lab README promises that compare_against_baselines.py exits non-zero when
result files are missing records. The comparison used to loop over whatever
records the baseline held. A sweep where every shape failed wrote an empty
"results" list, and a --shapes subset sweep wrote a partial one. Both compared
cleanly, wrote comparison.json and returned 0.
"""

from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path

import pytest

from labs.cutlass_profiler_kernel_selector import (
    compare_against_baselines,
    run_cutlass_profiler_sweep,
)
from labs.cutlass_profiler_kernel_selector.shapes import transformer_gemm_shapes

ALL_SHAPES = [shape.name for shape in transformer_gemm_shapes()]

# Writes the columns run_cutlass_profiler_sweep.parse_best_result requires,
# using the real tool's ".gemm.csv" suffix rule.
PASSING_PROFILER = textwrap.dedent(
    """\
    import sys

    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    m, n, k = int(args["m"]), int(args["n"]), int(args["k"])
    gflops = 2.0 * m * n * k / 1.0e6
    with open(args["output"] + ".gemm.csv", "w") as f:
        f.write(
            "Provider,Operation,Status,Disposition,raster_order,swizzle_size,cluster_m,cluster_n,cluster_k,"
            "cluster_m_fallback,cluster_n_fallback,cluster_k_fallback,Runtime,GFLOPs\\n"
        )
        f.write(f"CUTLASS,fake_cutlass_kernel,success,passed,along_n,1,1,1,1,1,1,1,1.0,{gflops}\\n")
    """
)
FAILING_PROFILER = "import sys\nsys.exit(3)\n"


def _fake_profiler(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "fake_cutlass_profiler"
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _run_sweep(tmp_path: Path, source: str, shapes: list[str] | None = None) -> tuple[int, Path]:
    out_dir = tmp_path / "sweep"
    argv = ["--profiler-bin", str(_fake_profiler(tmp_path, source)), "--output-dir", str(out_dir)]
    if shapes is not None:
        argv += ["--shapes", *shapes]
    rc = run_cutlass_profiler_sweep.main(argv)
    return rc, out_dir / "cutlass_profiler_results.json"


def _write_results(path: Path, provider: str, tflops_by_shape: dict, failed: list[str] | None = None) -> Path:
    payload = {
        "provider": provider,
        "results": [{"name": name, "tflops": tflops, "runtime_ms": 1.0} for name, tflops in tflops_by_shape.items()],
    }
    if failed is not None:
        payload["failed"] = failed
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def artifact_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "artifacts"
    monkeypatch.setattr(compare_against_baselines, "ARTIFACT_DIR", out)
    monkeypatch.setattr(compare_against_baselines, "DEFAULT_TRITON", out / "triton_matmul_results.json")
    return out


def test_baseline_from_an_all_failed_sweep_fails_the_comparison(
    tmp_path: Path, artifact_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sweep_rc, baseline = _run_sweep(tmp_path, FAILING_PROFILER)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert sweep_rc == 1
    assert payload["results"] == []
    assert payload["failed"] == ALL_SHAPES
    provider = _write_results(tmp_path / "triton.json", "triton", dict.fromkeys(ALL_SHAPES, 500.0))
    capsys.readouterr()

    rc = compare_against_baselines.main(["--baseline", str(baseline), "--providers", str(provider)])

    assert rc == 1
    assert not (artifact_dir / "comparison.json").exists()
    assert "cutlass_profiler lists these shapes as failed: " + ", ".join(ALL_SHAPES) in capsys.readouterr().err


def test_subset_sweep_needs_the_same_shapes_flag_on_the_comparison(
    tmp_path: Path, artifact_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subset = [ALL_SHAPES[3], ALL_SHAPES[1]]
    sweep_rc, baseline = _run_sweep(tmp_path, PASSING_PROFILER, shapes=subset)
    assert sweep_rc == 0
    provider = _write_results(tmp_path / "triton.json", "triton", dict.fromkeys(subset, 1.0))
    capsys.readouterr()

    rc = compare_against_baselines.main(["--baseline", str(baseline), "--providers", str(provider)])

    assert rc == 1
    assert not (artifact_dir / "comparison.json").exists()
    err = capsys.readouterr().err
    missing = [name for name in ALL_SHAPES if name not in subset]
    assert "cutlass_profiler has no record for these shapes: " + ", ".join(missing) in err
    assert "triton has no record for these shapes: " + ", ".join(missing) in err

    rc = compare_against_baselines.main(
        ["--baseline", str(baseline), "--providers", str(provider), "--shapes", *subset]
    )

    assert rc == 0
    comparison = json.loads((artifact_dir / "comparison.json").read_text(encoding="utf-8"))
    assert list(comparison) == [ALL_SHAPES[1], ALL_SHAPES[3]]
    for name, row in comparison.items():
        shape = next(s for s in transformer_gemm_shapes() if s.name == name)
        expected_tflops = 2.0 * shape.m * shape.n * shape.k / 1.0e6 / 1000.0
        assert row["baseline_kernel"] == "fake_cutlass_kernel"
        assert row["baseline_tflops"] == pytest.approx(expected_tflops)
        assert row["competitors"][0]["speedup_vs_cutlass"] == pytest.approx(1.0 / expected_tflops)


def test_include_default_triton_fails_when_the_triton_file_is_missing(tmp_path: Path, artifact_dir: Path) -> None:
    baseline = _write_results(tmp_path / "baseline.json", "cutlass_profiler", dict.fromkeys(ALL_SHAPES, 400.0))
    provider = _write_results(tmp_path / "custom.json", "custom", dict.fromkeys(ALL_SHAPES, 500.0))
    assert not compare_against_baselines.DEFAULT_TRITON.exists()

    rc = compare_against_baselines.main(
        ["--baseline", str(baseline), "--providers", str(provider), "--include-default-triton"]
    )

    assert rc == 1
    assert not (artifact_dir / "comparison.json").exists()


@pytest.mark.parametrize("bad_tflops", [None, 0.0, -2.0, math.nan, "fast", True])
def test_record_without_a_positive_tflops_value_fails(
    tmp_path: Path, artifact_dir: Path, bad_tflops: object
) -> None:
    baseline = _write_results(tmp_path / "baseline.json", "cutlass_profiler", dict.fromkeys(ALL_SHAPES, 400.0))
    tflops = dict.fromkeys(ALL_SHAPES, 500.0)
    tflops[ALL_SHAPES[2]] = bad_tflops
    provider = _write_results(tmp_path / "triton.json", "triton", tflops)

    rc = compare_against_baselines.main(["--baseline", str(baseline), "--providers", str(provider)])

    assert rc == 1
    assert not (artifact_dir / "comparison.json").exists()


def test_full_coverage_compares_every_shape(tmp_path: Path, artifact_dir: Path) -> None:
    baseline = _write_results(tmp_path / "baseline.json", "cutlass_profiler", dict.fromkeys(ALL_SHAPES, 400.0), [])
    provider = _write_results(tmp_path / "triton.json", "triton", dict.fromkeys(ALL_SHAPES, 500.0))

    rc = compare_against_baselines.main(["--baseline", str(baseline), "--providers", str(provider)])

    assert rc == 0
    comparison = json.loads((artifact_dir / "comparison.json").read_text(encoding="utf-8"))
    assert list(comparison) == ALL_SHAPES
    assert {row["competitors"][0]["speedup_vs_cutlass"] for row in comparison.values()} == {1.25}


def test_unknown_shape_name_fails(tmp_path: Path, artifact_dir: Path) -> None:
    baseline = _write_results(tmp_path / "baseline.json", "cutlass_profiler", dict.fromkeys(ALL_SHAPES, 400.0))
    provider = _write_results(tmp_path / "triton.json", "triton", dict.fromkeys(ALL_SHAPES, 500.0))

    rc = compare_against_baselines.main(
        ["--baseline", str(baseline), "--providers", str(provider), "--shapes", "not_a_shape"]
    )

    assert rc == 1
    assert not (artifact_dir / "comparison.json").exists()


def test_compare_api_rejects_a_partial_baseline(tmp_path: Path) -> None:
    baseline = _write_results(tmp_path / "baseline.json", "cutlass_profiler", {ALL_SHAPES[0]: 400.0})
    provider = _write_results(tmp_path / "triton.json", "triton", dict.fromkeys(ALL_SHAPES, 500.0))

    with pytest.raises(ValueError, match="cutlass_profiler has no record for these shapes"):
        compare_against_baselines.compare(baseline, [provider])

    comparison = compare_against_baselines.compare(baseline, [provider], shape_names=[ALL_SHAPES[0]])
    assert list(comparison) == [ALL_SHAPES[0]]
    assert comparison[ALL_SHAPES[0]]["competitors"][0]["speedup_vs_cutlass"] == 1.25


def test_duplicate_shape_records_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "triton.json"
    path.write_text(
        json.dumps(
            {
                "provider": "triton",
                "results": [{"name": ALL_SHAPES[0], "tflops": 1.0}, {"name": ALL_SHAPES[0], "tflops": 9.0}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="more than one record"):
        compare_against_baselines.load_results(path)
