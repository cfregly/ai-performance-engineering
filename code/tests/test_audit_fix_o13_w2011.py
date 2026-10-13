"""W2-011: both tma_copy arms time the same tile-local transform and check its full output.

The baseline used to time a flat 2^25-element array whose +1 and +64 neighbors
could come from anywhere in it. The optimized binary timed a 4096x4096 matrix
whose neighbors stay inside each 64x64 tile, and its VERIFY checksum covered an
untimed 1D buffer. These tests pin the repaired pair. The host-compiler test
runs the real baseline kernel source on the CPU. The tests at the end build and
run both binaries and need a Blackwell GPU.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import torch

from ch07.baseline_tma_copy import (
    TMA_COPY_BYTES_PER_ITERATION,
    TMA_COPY_WORKLOAD,
    BaselineTMACopyBenchmark,
)
from ch07.optimized_tma_copy import OptimizedTMACopyBenchmark
from core.benchmark.cuda_binary_benchmark import SUPPORTED_CAPABILITY_ARCH, CudaBinaryBenchmark
from core.benchmark.quarantine import QuarantineManager
from core.benchmark.timing_parser import parse_kernel_time_ms
from core.benchmark.verification import compare_workload_metrics
from core.benchmark.verified_comparison import _compare_signatures
from core.benchmark.verify_runner import VerifyRunner
from tests.test_audit_followup_timing_parser import (
    render_printf,
    stdout_printf_calls,
    strip_c_comments,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
CH07 = CODE_ROOT / "ch07"
OPTIMIZED_CUDA = (CH07 / "optimized_tma_copy.cu").read_text(encoding="utf-8")
BASELINE_CUDA = (CH07 / "baseline_tma_copy.cu").read_text(encoding="utf-8")
ARMS = (BaselineTMACopyBenchmark, OptimizedTMACopyBenchmark)
ARM_IDS = ("baseline", "optimized")
FLOAT32_BYTES = 4
STUB_TIME_MS = 0.0291234567


def _cuda_int(source: str, pattern: str) -> int:
    match = re.search(pattern, source, flags=re.MULTILINE)
    assert match is not None, pattern
    return int(match.group(1))


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


def _timed_section(source: str) -> str:
    """Return the code that sets up, times and checks the benchmark kernel."""
    if source is OPTIMIZED_CUDA:
        return source.split("bool benchmark_tma_2d", 1)[1].split("int main()", 1)[0]
    return source.split("int main()", 1)[1]


def _timed_shape(source: str) -> dict[str, int]:
    section = _timed_section(source)
    return {
        "rows": _cuda_int(section, r"const int M = (\d+);"),
        "cols": _cuda_int(section, r"const int N = (\d+);"),
        "tile_rows": _cuda_int(source, r"^constexpr int kTile2D_M = (\d+);"),
        "tile_cols": _cuda_int(source, r"^constexpr int kTile2D_N = (\d+);"),
        "lookahead": _cuda_int(source, r"^constexpr int kLookahead = (\d+);"),
    }


def _source_for(benchmark_type: type[CudaBinaryBenchmark]) -> str:
    return BASELINE_CUDA if benchmark_type is BaselineTMACopyBenchmark else OPTIMIZED_CUDA


def _launched_kernels(source: str) -> set[str]:
    return set(re.findall(r"(\w+)(?:<[^<>;]*>)?<<<", strip_c_comments(source)))


# ---------------------------------------------------------------------------
# Workload declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("benchmark_type", ARMS, ids=ARM_IDS)
def test_wrapper_declares_the_shape_its_binary_times(
    benchmark_type: type[CudaBinaryBenchmark],
) -> None:
    shape = _timed_shape(_source_for(benchmark_type))
    params = benchmark_type()._workload_params

    assert {key: params[key] for key in shape} == shape
    assert params["N"] == shape["rows"] * shape["cols"]
    assert params["source_reads_per_output"] == 3
    assert params["dtype"] == "float32"


def test_both_binaries_time_the_same_shape_input_and_launch_count() -> None:
    assert _timed_shape(BASELINE_CUDA) == _timed_shape(OPTIMIZED_CUDA)
    for source in (BASELINE_CUDA, OPTIMIZED_CUDA):
        section = _timed_section(source)
        assert "h_matrix[i] = static_cast<float>(i % 1000) / 1000.0f;" in section
        assert "dim3 block2d(16, 16);" in section
        assert re.search(r"constexpr int kIterations(?:2D)? = 20;", section)

    baseline = _timed_section(BASELINE_CUDA)
    launch = "scalar_tiled_neighbor_kernel<kTile2D_M, kTile2D_N><<<grid2d, block2d>>>"
    # One warmup launch, then one launch per timed iteration inside the event pair.
    assert baseline.count(launch) == 2
    timed_loop = baseline.split("CUDA_CHECK(cudaEventRecord(start));", 1)[1].split(
        "CUDA_CHECK(cudaEventRecord(stop));", 1
    )[0]
    assert timed_loop.count(launch) == 1
    assert "for (int iter = 0; iter < kIterations; ++iter)" in timed_loop


def test_tma_copy_pair_passes_input_signature_and_workload_checks(tmp_path: Path) -> None:
    baseline = BaselineTMACopyBenchmark()
    optimized = OptimizedTMACopyBenchmark()
    assert baseline._workload_params == optimized._workload_params == TMA_COPY_WORKLOAD

    receipt = _compare_signatures(baseline, optimized)
    assert receipt["passed"] is True

    runner = VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path),
    )
    matched, deltas = compare_workload_metrics(
        runner._extract_workload_metrics(baseline),
        runner._extract_workload_metrics(optimized),
    )
    assert matched
    assert deltas["bytes_per_iteration"] == 0.0
    assert not any(deltas.values())


def test_optimized_bytes_match_the_traffic_the_binary_reports_for_the_timed_kernel() -> None:
    section = _timed_section(OPTIMIZED_CUDA)
    assert "const size_t matrix_bytes = static_cast<size_t>(M) * N * sizeof(float);" in section
    transfers = re.search(r"const double bytes_transferred = ([0-9.]+) \* matrix_bytes;", section)
    assert transfers is not None
    # The timed kernel issues one TMA load per tile and has no global source
    # pointer, so its neighbor taps read only the tile in shared memory.
    kernel = _cpp_function(OPTIMIZED_CUDA, "__global__ void descriptor_tma_2d_copy_kernel(")
    assert kernel.count("cp_async_bulk_tensor_2d_global_to_shared(") == 1
    assert "const float*" not in kernel.split("{", 1)[0]
    shape = _timed_shape(OPTIMIZED_CUDA)
    matrix_bytes = shape["rows"] * shape["cols"] * FLOAT32_BYTES

    declared = OptimizedTMACopyBenchmark().get_workload_metadata().bytes_per_iteration

    assert declared == float(transfers.group(1)) * matrix_bytes == TMA_COPY_BYTES_PER_ITERATION


def test_tma_copy_arms_count_bytes_with_the_same_rule() -> None:
    for benchmark in (BaselineTMACopyBenchmark(), OptimizedTMACopyBenchmark()):
        elements = benchmark._workload_params["N"]
        bytes_per_iteration = benchmark.get_workload_metadata().bytes_per_iteration

        assert bytes_per_iteration == elements * 2 * FLOAT32_BYTES, type(benchmark).__name__


def test_tma_copy_metrics_describe_only_the_timed_kernels() -> None:
    for benchmark in (BaselineTMACopyBenchmark(), OptimizedTMACopyBenchmark()):
        metrics = benchmark.get_custom_metrics()
        assert metrics is not None
        assert not any("1d" in key for key in metrics), type(benchmark).__name__

    assert OptimizedTMACopyBenchmark().get_custom_metrics()["copy.tensor_map_2d_required"] == 1.0
    assert BaselineTMACopyBenchmark().get_custom_metrics()["copy.tensor_map_2d_requested"] == 0.0


# ---------------------------------------------------------------------------
# What each binary times and checks
# ---------------------------------------------------------------------------


def test_optimized_binary_times_only_the_descriptor_tma_kernel() -> None:
    for removed in ("tma_neighbor_copy_kernel", "kElements", "Pipeline-staged", "TMA 1D"):
        assert removed not in OPTIMIZED_CUDA, removed
    assert _launched_kernels(OPTIMIZED_CUDA) == {"descriptor_tma_2d_copy_kernel"}
    main = OPTIMIZED_CUDA.split("int main()", 1)[1]
    assert "<<<" not in main
    assert "if (!benchmark_tma_2d(prop)) {" in main


def test_baseline_kernel_reads_every_tap_from_global_memory() -> None:
    assert _launched_kernels(BASELINE_CUDA) == {"scalar_tiled_neighbor_kernel"}
    for staging in ("__shared__", "memcpy_async", "cp_async", "cuda::pipeline", "tma_helpers.cuh"):
        assert staging not in BASELINE_CUDA, staging
    kernel = _cpp_function(BASELINE_CUDA, "__global__ void scalar_tiled_neighbor_kernel(")
    # Center, near and far taps in the full-tile and partial-tile loops.
    assert kernel.count("tile_src[") == 6
    assert kernel.count("tile_dst[") == 2


def test_baseline_and_optimized_check_against_the_same_reference() -> None:
    for signature in (
        "__host__ __device__ __forceinline__ float combine_values",
        "float tiled_neighbor_reference_at",
    ):
        assert _cpp_function(BASELINE_CUDA, signature) == _cpp_function(OPTIMIZED_CUDA, signature)


@pytest.mark.parametrize("source", [BASELINE_CUDA, OPTIMIZED_CUDA], ids=ARM_IDS)
def test_binary_checks_its_full_timed_output_exactly_before_printing_its_time(source: str) -> None:
    section = _timed_section(source)
    assert "kValidateOutput" not in source
    # NaN bytes in the output make any element the kernel skips fail the check.
    fill = section.index("CUDA_CHECK(cudaMemset(d_mat_dst, 0xFF, matrix_bytes));")
    assert fill < section.index("<<<grid2d, block2d>>>")

    timed_end = section.index("CUDA_CHECK(cudaEventSynchronize(stop")
    check = section.index("for (std::size_t idx = 0; idx < h_matrix_out.size(); ++idx) {")
    failure = section.index("if (!copy_ok) {")
    time_line = section.index('std::printf("TIME_MS: %.6f\\n"')
    assert timed_end < check < failure < time_line
    assert "tiled_neighbor_reference_at(h_matrix, M, N, idx)" in section[check:failure]
    assert "h_matrix_out[idx] != expected" in section[check:failure]
    failure_block = section[failure : section.index("}", failure)]
    assert "return EXIT_FAILURE;" in failure_block or "std::exit(EXIT_FAILURE);" in failure_block


@pytest.mark.parametrize("source", [BASELINE_CUDA, OPTIMIZED_CUDA], ids=ARM_IDS)
def test_binary_prints_one_time_ms_line_and_checksums_the_timed_output(source: str) -> None:
    section = _timed_section(source)
    code = strip_c_comments(source)
    assert code.count("TIME_MS") == 1
    assert code.count("VERIFY_CHECKSUM(") == 1
    verify_block = section.split("#ifdef VERIFY", 1)[1].split("#endif", 1)[0]
    assert "VERIFY_CHECKSUM(h_matrix_out.data(), M * N, &verify_checksum);" in verify_block
    assert "VERIFY_PRINT_CHECKSUM(verify_checksum);" in verify_block
    assert section.index("if (!copy_ok) {") < section.index("#ifdef VERIFY")


def test_optimized_wrapper_uses_the_inherited_timing_and_checksum_paths() -> None:
    # The binary now checksums the timed output, so no override is needed.
    for name in ("benchmark_fn", "get_verify_output"):
        assert name not in OptimizedTMACopyBenchmark.__dict__, name
        assert name not in BaselineTMACopyBenchmark.__dict__, name
    for benchmark_type in ARMS:
        assert benchmark_type().time_regex == r"TIME_MS:\s*([0-9.]+)"


# ---------------------------------------------------------------------------
# The real baseline kernel source, compiled and run on the host
# ---------------------------------------------------------------------------

_HOST_PRELUDE = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define __host__
#define __device__
#define __global__
#define __forceinline__ inline

// Stand-ins for the CUDA built-ins the kernel reads. The driver runs every
// block and thread of the launch in turn. The kernel uses no shared memory
// and no barriers, so running threads one at a time gives the same output.
struct HostDim3 {
    unsigned x, y, z;
};
HostDim3 blockIdx{0, 0, 0};
HostDim3 threadIdx{0, 0, 0};
HostDim3 blockDim{1, 1, 1};
using std::min;
"""

_HOST_DRIVER = r"""
using Kernel = void (*)(const float*, float*, int, int);

std::vector<float> run_kernel(Kernel kernel, const std::vector<float>& input, int rows, int columns) {
    std::vector<float> output(input.size(), std::numeric_limits<float>::quiet_NaN());
    blockDim = HostDim3{BLOCK_X, BLOCK_Y, 1};
    const unsigned grid_x = (columns + baseline_cu::kTile2D_N - 1) / baseline_cu::kTile2D_N;
    const unsigned grid_y = (rows + baseline_cu::kTile2D_M - 1) / baseline_cu::kTile2D_M;
    for (unsigned by = 0; by < grid_y; ++by) {
        for (unsigned bx = 0; bx < grid_x; ++bx) {
            blockIdx = HostDim3{bx, by, 0};
            for (unsigned ty = 0; ty < BLOCK_Y; ++ty) {
                for (unsigned tx = 0; tx < BLOCK_X; ++tx) {
                    threadIdx = HostDim3{tx, ty, 0};
                    kernel(input.data(), output.data(), rows, columns);
                }
            }
        }
    }
    return output;
}

long long mismatches_against_reference(
    const std::vector<float>& output, const std::vector<float>& input, int rows, int columns) {
    long long mismatches = 0;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const float expected = optimized_cu::tiled_neighbor_reference_at(input, rows, columns, index);
        if (std::memcmp(&output[index], &expected, sizeof(float)) != 0) {
            ++mismatches;
        }
    }
    return mismatches;
}

std::vector<float> timed_input(int rows, int columns) {
    std::vector<float> input(static_cast<std::size_t>(rows) * columns);
    for (int i = 0; i < rows * columns; ++i) {
        input[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    return input;
}

std::vector<float> asymmetric_input(int rows, int columns) {
    // Unequal row and column weights expose transposed or shifted reads.
    std::vector<float> input(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            input[static_cast<std::size_t>(row) * columns + column] =
                static_cast<float>(row * 1009 + column * 3) + 0.25f;
        }
    }
    return input;
}

int main() {
    const Kernel kernel = baseline_cu::scalar_tiled_neighbor_kernel<
        baseline_cu::kTile2D_M, baseline_cu::kTile2D_N>;
    const Kernel mutated = mutated_cu::scalar_tiled_neighbor_kernel<
        mutated_cu::kTile2D_M, mutated_cu::kTile2D_N>;
    const int shapes[][2] = {
        {TIMED_ROWS, TIMED_COLUMNS}, {200, 328}, {65, 70}, {64, 64}, {1, 1}, {63, 129}, {130, 1}};
    for (const auto& shape : shapes) {
        const int rows = shape[0];
        const int columns = shape[1];
        const bool timed = rows == TIMED_ROWS && columns == TIMED_COLUMNS;
        const std::vector<float> input =
            timed ? timed_input(rows, columns) : asymmetric_input(rows, columns);
        long long oracle_mismatches = 0;
        for (std::size_t index = 0; index < input.size(); ++index) {
            const float a = baseline_cu::tiled_neighbor_reference_at(input, rows, columns, index);
            const float b = optimized_cu::tiled_neighbor_reference_at(input, rows, columns, index);
            oracle_mismatches += std::memcmp(&a, &b, sizeof(float)) != 0;
        }
        std::printf("shape %dx%d kernel=%lld oracle=%lld\n",
                    rows,
                    columns,
                    mismatches_against_reference(run_kernel(kernel, input, rows, columns), input, rows, columns),
                    oracle_mismatches);
    }

    // Positive controls on the timed shape. The pre-repair baseline transform,
    // whose +1 and +64 neighbors ignore tiles, and a kernel whose near tap is
    // off by one must both disagree with the reference.
    const std::vector<float> input = timed_input(TIMED_ROWS, TIMED_COLUMNS);
    std::vector<float> flat(input.size());
    const std::size_t last = input.size() - 1;
    for (std::size_t i = 0; i < input.size(); ++i) {
        flat[i] = optimized_cu::combine_values(
            input[i],
            input[std::min(i + 1, last)],
            input[std::min(i + static_cast<std::size_t>(optimized_cu::kLookahead), last)]);
    }
    std::printf("control flat=%lld mutated=%lld\n",
                mismatches_against_reference(flat, input, TIMED_ROWS, TIMED_COLUMNS),
                mismatches_against_reference(
                    run_kernel(mutated, input, TIMED_ROWS, TIMED_COLUMNS), input, TIMED_ROWS, TIMED_COLUMNS));
    return 0;
}
"""


def _int_constants(source: str) -> str:
    return "\n".join(
        re.search(rf"^constexpr int {name} = \d+;", source, flags=re.MULTILINE).group(0)
        for name in ("kLookahead", "kTile2D_M", "kTile2D_N")
    )


def test_host_compiled_baseline_kernel_matches_the_optimized_reference(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("SKIPPED: a host C++ compiler is required to run the baseline kernel source")

    combine_signature = "__host__ __device__ __forceinline__ float combine_values"
    kernel = _cpp_function(
        BASELINE_CUDA, "template <int TILE_M, int TILE_N>\n__global__ void scalar_tiled_neighbor_kernel("
    )
    mutated_kernel = kernel.replace("min(i + 1, kFull - 1)", "min(i + 2, kFull - 1)")
    assert mutated_kernel.count("min(i + 2, kFull - 1)") == 1
    block = re.search(r"const dim3 block2d\((\d+), (\d+)\);", BASELINE_CUDA)
    assert block is not None
    assert (
        "const dim3 grid2d((N + kTile2D_N - 1) / kTile2D_N, (M + kTile2D_M - 1) / kTile2D_M);"
        in BASELINE_CUDA
    )
    shape = _timed_shape(BASELINE_CUDA)
    baseline_parts = [_int_constants(BASELINE_CUDA), _cpp_function(BASELINE_CUDA, combine_signature)]
    program = "\n".join(
        [
            _HOST_PRELUDE,
            f"constexpr unsigned BLOCK_X = {block.group(1)};",
            f"constexpr unsigned BLOCK_Y = {block.group(2)};",
            f"constexpr int TIMED_ROWS = {shape['rows']};",
            f"constexpr int TIMED_COLUMNS = {shape['cols']};",
            "namespace baseline_cu {",
            *baseline_parts,
            _cpp_function(BASELINE_CUDA, "float tiled_neighbor_reference_at"),
            kernel,
            "}  // namespace baseline_cu",
            "namespace optimized_cu {",
            _int_constants(OPTIMIZED_CUDA),
            _cpp_function(OPTIMIZED_CUDA, combine_signature),
            _cpp_function(OPTIMIZED_CUDA, "float tiled_neighbor_reference_at"),
            "}  // namespace optimized_cu",
            "namespace mutated_cu {",
            *baseline_parts,
            mutated_kernel,
            "}  // namespace mutated_cu",
            _HOST_DRIVER,
        ]
    )
    driver = tmp_path / "baseline_tma_copy_host.cpp"
    executable = tmp_path / "baseline_tma_copy_host"
    driver.write_text(program, encoding="utf-8")

    compiled = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-ffp-contract=off", str(driver), "-o", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=300
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr

    results = dict(re.findall(r"^shape (\d+x\d+) (.+)$", executed.stdout, flags=re.MULTILINE))
    assert set(results) == {
        f"{shape['rows']}x{shape['cols']}", "200x328", "65x70", "64x64", "1x1", "63x129", "130x1"
    }
    assert all(result == "kernel=0 oracle=0" for result in results.values()), results
    controls = re.search(r"^control flat=(\d+) mutated=(\d+)$", executed.stdout, flags=re.MULTILINE)
    assert controls is not None, executed.stdout
    assert int(controls.group(1)) > 0
    assert int(controls.group(2)) > 0


# ---------------------------------------------------------------------------
# Wrapper parsing, with a stub executable that replays the binary's printf lines
# ---------------------------------------------------------------------------


def _stub_value(index: int, conversion: str) -> object:
    if conversion in "fFeEgG":
        return STUB_TIME_MS
    if conversion in "diu":
        return 4096
    return "stub"


def _printed_lines(benchmark_type: type[CudaBinaryBenchmark]) -> list[str]:
    calls = stdout_printf_calls(_source_for(benchmark_type))
    return [render_printf(call, _stub_value) for call in calls]


def _run_wrapper_on_stdout(
    tmp_path: Path, benchmark_type: type[CudaBinaryBenchmark], lines: list[str]
) -> CudaBinaryBenchmark:
    transcript = tmp_path / "stdout.txt"
    transcript.write_text("".join(lines), encoding="utf-8")
    executable = tmp_path / f"{benchmark_type.__name__}_stub"
    executable.write_text(f'#!/bin/sh\ncat "{transcript}"\n', encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    benchmark = benchmark_type()
    benchmark.exec_path = executable
    benchmark.benchmark_fn()
    return benchmark


@pytest.mark.parametrize("benchmark_type", ARMS, ids=ARM_IDS)
def test_wrapper_records_the_time_ms_value_of_its_binary(
    tmp_path: Path, benchmark_type: type[CudaBinaryBenchmark]
) -> None:
    lines = _printed_lines(benchmark_type)
    assert sum("TIME_MS" in line for line in lines) == 1

    benchmark = _run_wrapper_on_stdout(tmp_path, benchmark_type, lines)

    # TIME_MS carries 6 decimals. The descriptive line rounds to 3.
    assert benchmark.last_time_ms == pytest.approx(round(STUB_TIME_MS, 6))
    assert benchmark.last_time_ms != pytest.approx(round(STUB_TIME_MS, 3))


@pytest.mark.parametrize("benchmark_type", ARMS, ids=ARM_IDS)
def test_wrapper_fails_when_its_binary_prints_no_time_ms_line(
    tmp_path: Path, benchmark_type: type[CudaBinaryBenchmark]
) -> None:
    lines = [line for line in _printed_lines(benchmark_type) if "TIME_MS" not in line]
    # The default patterns would still find the descriptive line's time.
    assert parse_kernel_time_ms("".join(lines)) is not None

    with pytest.raises(ValueError, match="did not match the binary output"):
        _run_wrapper_on_stdout(tmp_path, benchmark_type, lines)


# ---------------------------------------------------------------------------
# Real binaries. These build with the chapter Makefile and need a Blackwell GPU.
# ---------------------------------------------------------------------------


def _run_real(benchmark_type: type[CudaBinaryBenchmark]) -> CudaBinaryBenchmark:
    if not torch.cuda.is_available():
        pytest.skip("SKIPPED: running the tma_copy binaries needs a CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability not in SUPPORTED_CAPABILITY_ARCH:
        pytest.skip(
            "SKIPPED: CudaBinaryBenchmark has no build target for compute capability "
            f"{capability}"
        )

    benchmark = benchmark_type()
    try:
        benchmark.setup()
        benchmark.benchmark_fn()
    except RuntimeError as exc:
        if str(exc).startswith("SKIPPED:"):
            pytest.skip(str(exc))
        raise
    return benchmark


@pytest.mark.parametrize("benchmark_type", ARMS, ids=ARM_IDS)
def test_real_binary_passes_its_output_check_and_prints_one_time_ms_line(
    benchmark_type: type[CudaBinaryBenchmark],
) -> None:
    benchmark = _run_real(benchmark_type)
    stdout = benchmark.last_stdout
    assert stdout is not None

    printed = re.findall(r"TIME_MS:\s*([0-9.]+)", stdout)
    assert len(printed) == 1, stdout
    assert benchmark.last_time_ms == float(printed[0]) > 0.0


def test_real_optimized_bandwidth_line_matches_the_declared_bytes() -> None:
    benchmark = _run_real(OptimizedTMACopyBenchmark)
    bandwidth = re.search(r"Achieved bandwidth: ([0-9.]+) GB/s", benchmark.last_stdout or "")
    assert bandwidth is not None

    declared_gbps = (
        benchmark.get_workload_metadata().bytes_per_iteration
        / (benchmark.last_time_ms * 1e-3)
        / 1e9
    )
    # TIME_MS rounds to 1e-6 ms, and the binary computes bandwidth from the
    # unrounded time and prints it with one decimal.
    rounding = 5e-7 / benchmark.last_time_ms + 1e-3
    assert declared_gbps == pytest.approx(float(bandwidth.group(1)), rel=rounding)


def test_real_binaries_checksum_bit_identical_timed_outputs() -> None:
    outputs = {}
    for benchmark_type in ARMS:
        benchmark = _run_real(benchmark_type)
        outputs[benchmark_type] = benchmark.get_verify_output()

    baseline = outputs[BaselineTMACopyBenchmark]
    optimized = outputs[OptimizedTMACopyBenchmark]
    # Both binaries pass an exact full-output check against the same reference
    # and sum their outputs in the same order, so the checksums must be equal.
    assert torch.isfinite(baseline).all()
    assert torch.equal(baseline, optimized)
