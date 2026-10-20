"""Regression tests for audit residuals W2-102, W2-111, and W2-116 (work unit r05)."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
TMA_COPY_SOURCE = CODE_ROOT / "ch07" / "optimized_tma_copy.cu"


# ---------------------------------------------------------------------------
# W2-111: every parallelism plan assigns each layer, expert, and token once.
# ---------------------------------------------------------------------------


def _parallelism_output(capsys: pytest.CaptureFixture[str], method: str, **config: int) -> list[str]:
    from ch15.disaggregated_inference_multigpu import ParallelismManager

    manager = ParallelismManager(SimpleNamespace(**config))
    getattr(manager, method)()
    return capsys.readouterr().out.splitlines()[1:]


def _owned_indices(lines: list[str], noun: str) -> list[int]:
    singular = noun[:-1]
    owned: list[int] = []
    pattern = re.compile(rf"\b(?:{noun} (\d+)-(\d+)|{singular} (\d+)) \(\d+ of \d+\)")
    for line in lines:
        match = pattern.search(line)
        if match is None:
            assert f"no {noun} (" in line, line
            continue
        if match.group(3) is not None:
            owned.append(int(match.group(3)))
        else:
            owned.extend(range(int(match.group(1)), int(match.group(2)) + 1))
    return owned


@pytest.mark.parametrize("num_gpus", [1, 2, 3, 5, 6, 7, 8, 10, 16])
def test_w2_111_expert_parallelism_assigns_every_expert_exactly_once(
    capsys: pytest.CaptureFixture[str], num_gpus: int
) -> None:
    lines = _parallelism_output(capsys, "setup_expert_parallelism", num_gpus=num_gpus, num_experts=8)

    assert len(lines) == num_gpus
    assert _owned_indices(lines, "experts") == list(range(8))


def test_w2_111_expert_parallelism_reports_idle_gpus_instead_of_empty_ranges(
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = _parallelism_output(capsys, "setup_expert_parallelism", num_gpus=10, num_experts=8)

    assert lines[0] == "GPU 0: Holds expert 0 (1 of 8)"
    assert lines[8:] == [
        "GPU 8: Holds no experts (8 experts across 10 GPUs)",
        "GPU 9: Holds no experts (8 experts across 10 GPUs)",
    ]
    assert not any("Experts 0-0" in line for line in lines)


def test_w2_111_uneven_expert_split_keeps_the_remainder(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _parallelism_output(capsys, "setup_expert_parallelism", num_gpus=3, num_experts=8)

    assert lines == [
        "GPU 0: Holds experts 0-2 (3 of 8)",
        "GPU 1: Holds experts 3-5 (3 of 8)",
        "GPU 2: Holds experts 6-7 (2 of 8)",
    ]


@pytest.mark.parametrize("num_gpus", [1, 3, 6, 7, 8])
def test_w2_111_context_parallelism_assigns_every_token_exactly_once(
    capsys: pytest.CaptureFixture[str], num_gpus: int
) -> None:
    lines = _parallelism_output(
        capsys, "setup_context_parallelism", num_gpus=num_gpus, sequence_length=2048
    )

    assert len(lines) == num_gpus
    assert _owned_indices(lines, "tokens") == list(range(2048))


@pytest.mark.parametrize("num_gpus", [1, 2, 4, 6, 8])
def test_w2_111_pipeline_stages_own_disjoint_whole_layers(
    capsys: pytest.CaptureFixture[str], num_gpus: int
) -> None:
    lines = _parallelism_output(capsys, "setup_pipeline_parallelism", num_gpus=num_gpus)

    assert [line.split(" with ")[0] for line in lines] == [
        f"GPU {gpu}: Pipeline stage {gpu + 1}/{num_gpus}" for gpu in range(num_gpus)
    ]
    assert _owned_indices(lines, "layers") == list(range(6))


def test_w2_111_pipeline_reports_stages_without_layers(capsys: pytest.CaptureFixture[str]) -> None:
    lines = _parallelism_output(capsys, "setup_pipeline_parallelism", num_gpus=8)

    assert lines[0] == "GPU 0: Pipeline stage 1/8 with layer 0 (1 of 6)"
    assert lines[6:] == [
        "GPU 6: Pipeline stage 7/8 with no layers (6 layers across 8 GPUs)",
        "GPU 7: Pipeline stage 8/8 with no layers (6 layers across 8 GPUs)",
    ]


# ---------------------------------------------------------------------------
# W2-116: the rejection analyzer reads the file the harness compared against.
# ---------------------------------------------------------------------------


def _write_expectations(chapter_dir: Path, hardware_key: str, example: str, git_commit: str) -> None:
    chapter_dir.mkdir(parents=True, exist_ok=True)
    (chapter_dir / f"expectations_{hardware_key}.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "hardware_key": hardware_key,
                "examples": {
                    example: {
                        "example": example,
                        "type": "cuda",
                        "provenance": {"hardware_key": hardware_key, "git_commit": git_commit},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _emit_rejected_update(run_dir: Path, *, old_hardware_key: str, new_hardware_key: str) -> None:
    from core.benchmark.expectations import (
        ExpectationEntry,
        RunProvenance,
        UpdateResult,
        ValidationIssue,
    )
    from core.harness.run_benchmarks import BenchmarkEventLogger, log_expectation_delta

    def entry(hardware_key: str, optimized_ms: float) -> ExpectationEntry:
        return ExpectationEntry(
            example="gemm",
            type="cuda",
            optimization_goal="speed",
            baseline_time_ms=10.0,
            best_optimized_time_ms=optimized_ms,
            provenance=RunProvenance(
                git_commit="deadbeef",
                hardware_key=hardware_key,
                profile_name="minimal",
                timestamp="2026-10-12T00:00:00Z",
                iterations=10,
                warmup_iterations=5,
            ),
        )

    logger = logging.getLogger("test.audit_fix_r05")
    logger.addHandler(logging.NullHandler())
    event_logger = BenchmarkEventLogger(
        run_dir / "logs" / "benchmark_events.jsonl", run_id=run_dir.name, logger=logger
    )
    try:
        log_expectation_delta(
            logger,
            example_key="gemm_cuda",
            goal="speed",
            old_entry=entry(old_hardware_key, 5.0),
            new_entry=entry(new_hardware_key, 8.0),
            update_result=UpdateResult(
                status="rejected",
                message="Provenance mismatch: hardware_key differs",
                validation_issues=[
                    ValidationIssue(
                        example_key="gemm_cuda",
                        issue_type="provenance_mismatch",
                        message="Mixed provenance update rejected",
                        stored_value={"hardware_key": old_hardware_key},
                        expected_value={"hardware_key": new_hardware_key},
                    )
                ],
            ),
            event_logger=event_logger,
            chapter="ch09",
        )
    finally:
        event_logger.close()


def test_w2_116_analyzer_reads_the_file_named_by_the_run_hardware_key(tmp_path: Path) -> None:
    from core.analysis.analyze_expectation_rejections import render_expectation_rejection_ledger

    _write_expectations(tmp_path / "ch09", "b200", "gemm_cuda", "b200-file-sha")
    _write_expectations(tmp_path / "ch09", "4x_gb200", "gemm_cuda", "gb200-file-sha")
    run_dir = tmp_path / "run"
    _emit_rejected_update(run_dir, old_hardware_key="b200", new_hardware_key="4x_gb200")

    outputs = render_expectation_rejection_ledger(repo_root=tmp_path, run_dir=run_dir)
    row = json.loads(outputs["json"].read_text(encoding="utf-8"))[0]

    assert row["expectations_file"] == "ch09/expectations_4x_gb200.json"
    assert row["stored_hardware_key"] == "b200"
    assert row["new_hardware_key"] == "4x_gb200"
    assert "hardware_key" in row["provenance_mismatch_fields"]


@pytest.mark.parametrize(
    "provenance",
    [
        {},
        {"old_provenance": {"hardware_key": "b200"}},
        {"new_provenance": {"git_commit": "abc"}},
    ],
    ids=["no_provenance", "old_only", "new_without_key"],
)
def test_w2_116_analyzer_fails_when_the_run_hardware_key_is_missing(
    tmp_path: Path, provenance: dict
) -> None:
    from core.analysis.analyze_expectation_rejections import render_expectation_rejection_ledger

    _write_expectations(tmp_path / "ch09", "b200", "gemm_cuda", "b200-file-sha")
    run_dir = tmp_path / "run"
    events_path = run_dir / "logs" / "benchmark_events.jsonl"
    events_path.parent.mkdir(parents=True)
    event = {
        "event_type": "expectation_update",
        "chapter": "ch09",
        "example": "gemm_cuda",
        "status": "rejected",
        "old_score": 2.0,
        "new_score": 1.0,
        "delta_pct": -50.0,
        **provenance,
    }
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"ch09:gemm_cuda .*new_provenance\.hardware_key"):
        render_expectation_rejection_ledger(repo_root=tmp_path, run_dir=run_dir)

    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.analysis.analyze_expectation_rejections",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            str(run_dir),
        ],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 1
    assert "new_provenance.hardware_key" in cli.stderr
    assert not (run_dir / "analysis").exists()


# ---------------------------------------------------------------------------
# W2-102: the TMA binary checks a non-square, partial-tile shape on device.
# These host checks cannot run CUDA. The B200 run is the device evidence.
# ---------------------------------------------------------------------------


def _cpp_function(source: str, signature: str) -> str:
    start = source.index(signature)
    depth = 0
    for position in range(source.index("{", start), len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[start : position + 1]
    raise AssertionError(f"unterminated C++ function: {signature}")


def _cpp_int_constant(source: str, name: str) -> int:
    match = re.search(rf"^constexpr int {re.escape(name)} = (\d+);", source, flags=re.MULTILINE)
    assert match is not None, name
    return int(match.group(1))


def test_w2_102_benchmark_validates_non_square_shape_before_timing() -> None:
    source = TMA_COPY_SOURCE.read_text(encoding="utf-8")
    rows = _cpp_int_constant(source, "kValidationRows2D")
    columns = _cpp_int_constant(source, "kValidationColumns2D")
    tile_rows = _cpp_int_constant(source, "kTile2D_M")
    tile_columns = _cpp_int_constant(source, "kTile2D_N")

    assert rows != columns
    assert rows > tile_rows and rows % tile_rows != 0
    assert columns > tile_columns and columns % tile_columns != 0
    assert (columns * 4) % 16 == 0, "TMA global strides must be multiples of 16 bytes"

    benchmark = source.split("bool benchmark_tma_2d", 1)[1].split("int main()", 1)[0]
    call = benchmark.index("if (!validate_tma_2d_non_square(encode)) {")
    assert benchmark.index("if (!use_tensor_map) {") < call
    assert call < benchmark.index("descriptor_tma_2d_copy_kernel<kTile2D_M, kTile2D_N><<<grid2d")
    assert call < benchmark.index("CUDA_CHECK(cudaEventRecord(start2d));")
    assert "std::exit(EXIT_FAILURE);" in benchmark[call : call + 300]

    validator = _cpp_function(source, "bool validate_tma_2d_non_square")
    assert "columns, rows, columns,\n            kTile2D_N, kTile2D_M" in validator
    assert "rows, columns, columns);" in validator
    assert "tiled_neighbor_reference_at(input, rows, columns, index)" in validator
    assert "output[index] != reference" in validator
    assert "cudaMemset(d_output, 0xFF, bytes)" in validator


def test_w2_102_non_square_check_detects_a_descriptor_width_height_swap(tmp_path: Path) -> None:
    """Emulate the kernel's TMA box loads through the real layout helper on the host.

    A correct descriptor must match the oracle exactly. A width/height swap must
    produce mismatches on the validation shape, and none on the square timed
    shape, which is why the timed shape alone could not catch it.
    """
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler is unavailable")

    source = TMA_COPY_SOURCE.read_text(encoding="utf-8")
    constants = "\n".join(
        re.search(rf"^constexpr int {name} = \d+;", source, flags=re.MULTILINE).group(0)
        for name in ("kLookahead", "kTile2D_M", "kTile2D_N", "kValidationRows2D", "kValidationColumns2D")
    )
    combine = _cpp_function(source, "__host__ __device__ __forceinline__ float combine_values").replace(
        "__host__ __device__ __forceinline__ ", "", 1
    )
    oracle = _cpp_function(source, "float tiled_neighbor_reference_at")
    validation_input = _cpp_function(source, "float tma_2d_validation_input")
    timed_section = source.split("bool benchmark_tma_2d", 1)[1]
    timed_rows = int(re.search(r"const int M = (\d+);", timed_section).group(1))
    timed_columns = int(re.search(r"const int N = (\d+);", timed_section).group(1))

    driver = tmp_path / "tma_2d_swap_probe.cpp"
    executable = tmp_path / "tma_2d_swap_probe"
    driver.write_text(
        f"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/common/headers/tma_2d_layout.hpp"

{constants}

{combine}

{oracle}

{validation_input}

// Mirrors descriptor_tma_2d_copy_kernel: each block loads one box through the
// descriptor layout, with out-of-bounds elements read as zero.
long long count_mismatches(
    const std::vector<float>& input, int rows, int columns,
    const cuda_tma::TensorMap2DLayout& layout) {{
    const std::uint64_t width = layout.dimensions[0];
    const std::uint64_t height = layout.dimensions[1];
    const std::uint64_t ld = layout.strides_bytes[0] / sizeof(float);
    long long mismatches = 0;
    for (int tile_m = 0; tile_m < rows; tile_m += kTile2D_M) {{
        for (int tile_n = 0; tile_n < columns; tile_n += kTile2D_N) {{
            const int tile_rows = std::min(kTile2D_M, rows - tile_m);
            const int tile_cols = std::min(kTile2D_N, columns - tile_n);
            const int tile_elems = tile_rows * tile_cols;
            const auto load = [&](int linear) {{
                const std::uint64_t row = tile_m + linear / tile_cols;
                const std::uint64_t column = tile_n + linear % tile_cols;
                return (row < height && column < width) ? input[row * ld + column] : 0.0f;
            }};
            for (int i = 0; i < tile_elems; ++i) {{
                const float value = combine_values(
                    load(i),
                    load(std::min(i + 1, tile_elems - 1)),
                    load(std::min(i + kLookahead, tile_elems - 1)));
                const std::size_t index =
                    static_cast<std::size_t>(tile_m + i / tile_cols) * columns + tile_n + i % tile_cols;
                if (value != tiled_neighbor_reference_at(input, rows, columns, index)) {{
                    ++mismatches;
                }}
            }}
        }}
    }}
    return mismatches;
}}

void report(const char* name, int rows, int columns, float (*fill)(int, int)) {{
    std::vector<float> input(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {{
        for (int column = 0; column < columns; ++column) {{
            input[static_cast<std::size_t>(row) * columns + column] = fill(row, column);
        }}
    }}
    const auto correct = cuda_tma::make_2d_tensor_map_layout(
        columns, rows, columns, kTile2D_N, kTile2D_M);
    const auto swapped = cuda_tma::make_2d_tensor_map_layout(
        rows, columns, columns, kTile2D_M, kTile2D_N);
    std::printf("%s correct=%lld swapped=%lld\\n", name,
                count_mismatches(input, rows, columns, correct),
                count_mismatches(input, rows, columns, swapped));
}}

float timed_input(int row, int column) {{
    return static_cast<float>((row * {timed_columns} + column) % 1000) / 1000.0f;
}}

int main() {{
    report("validation", kValidationRows2D, kValidationColumns2D, tma_2d_validation_input);
    report("timed", {timed_rows}, {timed_columns}, timed_input);
    return 0;
}}
""",
        encoding="utf-8",
    )
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-O2", f"-I{CODE_ROOT}", str(driver), "-o", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(executable)], check=False, capture_output=True, text=True, timeout=120)
    assert executed.returncode == 0, executed.stderr

    results = {
        name: (int(correct), int(swapped))
        for name, correct, swapped in re.findall(r"(\w+) correct=(\d+) swapped=(\d+)", executed.stdout)
    }
    assert results["validation"][0] == 0
    assert results["validation"][1] > 0
    assert results["timed"] == (0, 0)
