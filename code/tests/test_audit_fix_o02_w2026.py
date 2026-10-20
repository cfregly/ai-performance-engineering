"""Regression tests for audit finding W2-026 in the CUTLASS profiler kernel selector.

The CSV fixtures use the exact column layout written by CUTLASS
tools/profiler/src/performance_report.cpp for a GEMM problem. That header
carries both "Flops" (FLOPs per launch, equal for every kernel of a problem)
and "GFLOPs" (throughput). Ranking by "Flops" kept the first kernel listed.
It also reported the per-launch FLOP count as GFLOP/s, which overstates
throughput by a factor of runtime_ms x 1e6, for example 1e6 for a 1 ms kernel.

Each fixture row also gets distinct runtime parameters, so a test can tell
which row the recorded raster order, swizzle size and cluster shape came from.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from labs.cutlass_profiler_kernel_selector import run_cutlass_profiler_sweep
from labs.cutlass_profiler_kernel_selector.run_cutlass_profiler_sweep import (
    parse_best_result,
    profiler_binary,
)
from labs.cutlass_profiler_kernel_selector.shapes import transformer_gemm_shapes

CODE_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = CODE_ROOT / "labs" / "cutlass_profiler_kernel_selector" / "run_cutlass_profiler_sweep.py"

# GEMM argument names from gemm_operation_profiler.cu followed by the tile
# description arguments from operation_profiler.cu.
GEMM_ARGUMENT_COLUMNS = (
    "gemm_kind", "m", "n", "k", "A", "B", "C", "D", "alpha", "beta",
    "split_k_mode", "split_k_slices", "batch_count", "raster_order",
    "runtime_input_datatype_a", "runtime_input_datatype_b", "use_pdl",
    "enable_sm90_mixed_dtype_shuffle_test", "swizzle_size", "op_class", "accum",
    "cta_m", "cta_n", "cta_k", "cluster_m", "cluster_n", "cluster_k",
    "cluster_m_fallback", "cluster_n_fallback", "cluster_k_fallback", "stages",
    "warps_m", "warps_n", "warps_k", "inst_m", "inst_n", "inst_k", "min_cc", "max_cc",
)
CUTLASS_GEMM_HEADER = (
    "Problem", "Provider", "OperationKind", "Operation", "Disposition", "Status",
    *GEMM_ARGUMENT_COLUMNS,
    "Bytes", "Flops", "Flops/Byte", "Runtime", "GB/s", "GFLOPs",
)

# Fake cutlass_profiler executable. It follows the real tool's file naming
# (strip ".csv", append ".gemm.csv") and its throughput formulas from
# performance_result.h. Kernels with runtime 0 get blank GB/s and GFLOPs cells,
# as in PerformanceReport::print_result_csv_.
FAKE_PROFILER_SOURCE = textwrap.dedent(
    """\
    import csv
    import json
    import sys

    header = json.loads({header!r})
    kernels = json.loads({kernels!r})
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    m, n, k = int(args["m"]), int(args["n"]), int(args["k"])
    base = args["output"]
    if ".csv" in base:
        base = base[: base.rfind(".csv")]
    flops = 2 * m * n * k
    nbytes = 2 * (m * k + k * n + m * n)
    with open(base + ".gemm.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, restval="1")
        writer.writeheader()
        for operation, disposition, status, runtime, params in kernels:
            timed = runtime > 0
            writer.writerow({{
                "Problem": "0", "Provider": "CUTLASS", "OperationKind": "gemm",
                "Operation": operation, "Disposition": disposition, "Status": status,
                "m": m, "n": n, "k": k,
                "runtime_input_datatype_a": "static", "runtime_input_datatype_b": "static",
                **params,
                "Bytes": nbytes, "Flops": flops, "Flops/Byte": flops / nbytes,
                "Runtime": runtime,
                "GB/s": nbytes / float(1 << 30) / runtime * 1000.0 if timed else "",
                "GFLOPs": flops / runtime / 1.0e6 if timed else "",
            }})
    print("Wrote results to '" + base + ".gemm.csv'")
    """
)

M, N, K = 4096, 4096, 4096
FLOPS = 2 * M * N * K


def _gflops(runtime_ms: float) -> float:
    return FLOPS / runtime_ms / 1.0e6


def _row_runtime_params(index: int) -> dict[str, str]:
    return {
        "raster_order": ("along_n", "along_m")[index % 2],
        "swizzle_size": str(1 << (index % 4)),
        "cluster_m": str(index + 1),
        "cluster_n": "1",
        "cluster_k": "1",
        "cluster_m_fallback": "1",
        "cluster_n_fallback": str(index + 1),
        "cluster_k_fallback": "1",
    }


def _write_cutlass_csv(
    path: Path,
    kernels: list[tuple[str, str, str, float]],
    providers: dict[str, str] | None = None,
) -> None:
    """Write one row per kernel. ``providers`` maps an operation name to a non-CUTLASS provider."""

    nbytes = 2 * (M * K + K * N + M * N)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CUTLASS_GEMM_HEADER, restval="1")
        writer.writeheader()
        for index, (operation, disposition, status, runtime_ms) in enumerate(kernels):
            timed = runtime_ms > 0
            writer.writerow(
                {
                    "Problem": "0",
                    "Provider": (providers or {}).get(operation, "CUTLASS"),
                    "OperationKind": "gemm",
                    "Operation": operation,
                    "Disposition": disposition,
                    "Status": status,
                    "m": M,
                    "n": N,
                    "k": K,
                    **_row_runtime_params(index),
                    "Bytes": nbytes,
                    "Flops": FLOPS,
                    "Flops/Byte": FLOPS / nbytes,
                    "Runtime": runtime_ms,
                    "GB/s": nbytes / float(1 << 30) / runtime_ms * 1000.0 if timed else "",
                    "GFLOPs": _gflops(runtime_ms) if timed else "",
                }
            )


def test_selects_by_gflops_not_by_the_per_launch_flops_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "shape.gemm.csv"
    _write_cutlass_csv(
        csv_path,
        [
            ("slow_kernel", "passed", "success", 2.0),
            ("winning_kernel", "passed", "success", 1.0),
            ("middle_kernel", "passed", "success", 1.5),
        ],
    )

    best = parse_best_result(csv_path)

    assert best["kernel"] == "winning_kernel"
    assert best["disposition"] == "passed"
    assert best["runtime_ms"] == 1.0
    assert best["gflops"] == pytest.approx(_gflops(1.0))
    assert best["tflops"] == pytest.approx(_gflops(1.0) / 1000.0)
    assert best["runtime_params"] == _row_runtime_params(1)


def test_skips_incorrect_and_untimed_kernels(tmp_path: Path) -> None:
    csv_path = tmp_path / "shape.gemm.csv"
    _write_cutlass_csv(
        csv_path,
        [
            ("unsupported_kernel", "not_run", "not_supported", 0.0),
            ("wrong_answer_kernel", "incorrect", "success", 0.5),
            ("failed_kernel", "failed", "success", 0.6),
            ("verified_kernel", "passed", "success", 1.2),
            ("unverified_kernel", "not_verified", "success", 1.1),
        ],
    )

    best = parse_best_result(csv_path)

    assert best["kernel"] == "unverified_kernel"
    assert best["disposition"] == "not_verified"
    assert best["runtime_ms"] == 1.1
    assert best["runtime_params"] == _row_runtime_params(4)


def test_skips_rows_from_a_provider_other_than_cutlass(tmp_path: Path) -> None:
    csv_path = tmp_path / "shape.gemm.csv"
    _write_cutlass_csv(
        csv_path,
        [
            ("cutlass_kernel", "passed", "success", 1.0),
            ("cublas_gemm", "passed", "success", 0.5),
        ],
        providers={"cublas_gemm": "cuBLAS"},
    )

    best = parse_best_result(csv_path)

    assert best["kernel"] == "cutlass_kernel"
    assert best["runtime_ms"] == 1.0
    assert best["runtime_params"] == _row_runtime_params(0)


def test_fails_when_only_other_providers_have_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "shape.gemm.csv"
    _write_cutlass_csv(
        csv_path,
        [("cublas_gemm", "passed", "success", 0.5)],
        providers={"cublas_gemm": "cuBLAS"},
    )

    with pytest.raises(ValueError, match="No selectable kernel") as excinfo:
        parse_best_result(csv_path)
    assert "provider=cuBLAS" in str(excinfo.value)


def test_fails_when_no_kernel_is_selectable(tmp_path: Path) -> None:
    csv_path = tmp_path / "shape.gemm.csv"
    _write_cutlass_csv(
        csv_path,
        [
            ("unsupported_kernel", "not_run", "not_supported", 0.0),
            ("wrong_answer_kernel", "incorrect", "success", 0.5),
        ],
    )

    with pytest.raises(ValueError, match="No selectable kernel") as excinfo:
        parse_best_result(csv_path)
    assert "disposition=incorrect" in str(excinfo.value)


def test_rejects_csv_without_the_cutlass_report_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "shape.gemm.csv"
    csv_path.write_text(
        "Operation,Runtime,Flops\nslow_kernel,2.0,137438953472\nfast_kernel,1.0,137438953472\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a cutlass_profiler CSV report") as excinfo:
        parse_best_result(csv_path)
    assert "'GFLOPs'" in str(excinfo.value)


@pytest.mark.parametrize("column", ["Provider", "swizzle_size", "cluster_k_fallback"])
def test_rejects_csv_without_a_required_column(tmp_path: Path, column: str) -> None:
    csv_path = tmp_path / "shape.gemm.csv"
    _write_cutlass_csv(csv_path, [("winning_kernel", "passed", "success", 1.0)])
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[name for name in CUTLASS_GEMM_HEADER if name != column])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != column} for row in rows)

    with pytest.raises(ValueError, match="not a cutlass_profiler CSV report") as excinfo:
        parse_best_result(csv_path)
    assert f"missing columns ['{column}']" in str(excinfo.value)


def test_rejects_a_winning_row_with_a_blank_runtime_parameter(tmp_path: Path) -> None:
    csv_path = tmp_path / "shape.gemm.csv"
    _write_cutlass_csv(csv_path, [("winning_kernel", "passed", "success", 1.0)])
    with csv_path.open(newline="", encoding="utf-8") as f:
        (row,) = csv.DictReader(f)
    row["raster_order"] = ""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CUTLASS_GEMM_HEADER)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match=r"blank runtime parameter columns \['raster_order'\]"):
        parse_best_result(csv_path)


def test_rejects_a_non_numeric_gflops_cell(tmp_path: Path) -> None:
    csv_path = tmp_path / "shape.gemm.csv"
    _write_cutlass_csv(csv_path, [("winning_kernel", "passed", "success", 1.0)])
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(text.replace(repr(_gflops(1.0)), "n/a"), encoding="utf-8")

    with pytest.raises(ValueError, match="GFLOPs='n/a'"):
        parse_best_result(csv_path)


def test_cli_sweep_records_the_fastest_passing_kernel(tmp_path: Path) -> None:
    shape = next(s for s in transformer_gemm_shapes() if (s.m, s.n, s.k) == (M, N, K))
    kernels = [
        [operation, disposition, status, runtime_ms, _row_runtime_params(index)]
        for index, (operation, disposition, status, runtime_ms) in enumerate(
            [
                ("cutlass3x_sm100_slow", "passed", "success", 2.0),
                ("cutlass3x_sm100_wrong_answer", "incorrect", "success", 0.25),
                ("cutlass3x_sm100_unsupported", "not_run", "not_supported", 0.0),
                ("cutlass3x_sm100_fast", "passed", "success", 0.8),
            ]
        )
    ]
    fake_profiler = tmp_path / "cutlass_profiler"
    fake_profiler.write_text(
        f"#!{sys.executable}\n"
        + FAKE_PROFILER_SOURCE.format(
            header=json.dumps(CUTLASS_GEMM_HEADER),
            kernels=json.dumps(kernels),
        ),
        encoding="utf-8",
    )
    fake_profiler.chmod(0o755)
    output_dir = tmp_path / "artifacts"
    env = {key: value for key, value in os.environ.items() if key != "CUTLASS_PROFILER_BIN"}

    result = subprocess.run(
        [
            sys.executable,
            str(SWEEP_SCRIPT),
            "--profiler-bin",
            str(fake_profiler),
            "--output-dir",
            str(output_dir),
            "--shapes",
            shape.name,
        ],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output_dir / "cutlass_profiler_results.json").read_text(encoding="utf-8"))
    assert summary["failed"] == []
    (row,) = summary["results"]
    assert row["name"] == shape.name
    assert row["kernel"] == "cutlass3x_sm100_fast"
    assert row["disposition"] == "passed"
    assert row["runtime_ms"] == 0.8
    assert row["gflops"] == pytest.approx(_gflops(0.8))
    assert row["runtime_params"] == _row_runtime_params(3)
    assert row["csv"] == str(output_dir / f"{shape.name}.gemm.csv")


def test_profiler_exit_code_is_reported(tmp_path: Path) -> None:
    shape = next(s for s in transformer_gemm_shapes() if (s.m, s.n, s.k) == (M, N, K))
    failing_profiler = tmp_path / "cutlass_profiler"
    failing_profiler.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(3)\n", encoding="utf-8")
    failing_profiler.chmod(0o755)

    with pytest.raises(RuntimeError) as excinfo:
        run_cutlass_profiler_sweep.run_profiler_for_shape(failing_profiler, shape, tmp_path)

    assert str(excinfo.value) == (
        f"cutlass_profiler exited with code 3 for {shape.name}. See {tmp_path / f'{shape.name}.log'}"
    )


def test_missing_profiler_from_environment_fails_instead_of_falling_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUTLASS_PROFILER_BIN", str(tmp_path / "missing_cutlass_profiler"))

    with pytest.raises(FileNotFoundError, match="CUTLASS_PROFILER_BIN does not name a file"):
        profiler_binary()


def test_missing_default_profiler_message_does_not_claim_setup_builds_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUTLASS_PROFILER_BIN", raising=False)
    monkeypatch.setattr(run_cutlass_profiler_sweep, "REPO_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="does not build the profiler"):
        profiler_binary()
