"""Regression tests for audit findings W2-015 and W2-050 (NVFP4 cuBLASLt operands).

cuBLASLt VEC16_UE4M3 block scaling needs K-major FP4 operands and block scales
in the 128x4 scale-factor tile layout. These tests compile the real host
quantizers from the CUDA sources with a host C++ compiler and check where every
block scale and packed nibble lands. The CUDA FP4 and FP8 conversion intrinsics
are replaced by a small host shim that is exact for the probe values used here.
GPU numerics still need a Blackwell run. The cuBLASLt layout descriptors cannot
run on a host, so they are checked as source contracts.

The follow-up tests cover the sibling files. The ch09 optimized binary uses the
same layoutA as its baseline. The ch19 manual baseline kernel reads B as row-major
B^T [N, K] at any shape, checked by running the real kernel body on the host at a
non-square shape. The ch19 wrappers verify the full timed C matrix that the
binaries write through --dump-output, checked with a stand-in executable because
the real binaries need a Blackwell GPU.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
CH09_BASELINE = CODE_ROOT / "ch09" / "baseline_cublas_gemm_fp4_perchannel.cu"
CH09_OPTIMIZED = CODE_ROOT / "ch09" / "optimized_cublas_gemm_fp4_perchannel.cu"
CH19_BASELINE = CODE_ROOT / "ch19" / "baseline_fp4_hardware_kernel.cu"
CH19_OPTIMIZED = CODE_ROOT / "ch19" / "optimized_fp4_hardware_kernel.cu"

_HOST_SHIM = r"""
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stddef.h>
#include <stdint.h>

#define NVTX_RANGE(name) ((void)0)

using __nv_fp4_storage_t = unsigned char;
enum __nv_fp4_interpretation_t { __NV_E2M1 };
enum cudaRoundMode { cudaRoundNearest };

// Nearest E2M1 code with saturation. The probes only produce exact E2M1 values.
inline __nv_fp4_storage_t __nv_cvt_float_to_fp4(
    float value, __nv_fp4_interpretation_t, cudaRoundMode) {
  static const float kMagnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = std::fabs(value);
  int best = 0;
  for (int code = 1; code < 8; ++code) {
    if (std::fabs(kMagnitudes[code] - magnitude) < std::fabs(kMagnitudes[best] - magnitude)) {
      best = code;
    }
  }
  return static_cast<__nv_fp4_storage_t>((value < 0.0f ? 0x8 : 0x0) | best);
}

// Stores the scale unrounded. The probe scales (0.25 and 1.0) are exact in UE4M3.
struct __nv_fp8_e4m3 {
  float value;
  __nv_fp8_e4m3() = default;
  explicit __nv_fp8_e4m3(float f) : value(f) {}
  explicit operator float() const { return value; }
};
"""

_DRIVER_TEMPLATE = r"""
#include <cstdio>
#include <vector>

#include "host_shim.h"
#include "code/core/common/headers/nvfp4_scale_layout.cuh"

constexpr int FP4_BLOCK_SIZE = 16;

@QUANTIZER@

namespace {

constexpr int kRows = 256;
constexpr int kReduction = 128;
constexpr int kScaleColumns = kReduction / FP4_BLOCK_SIZE;

#define INPUT_INDEX(row, k) (@INPUT_INDEX@)

// The canonical 128x4 scale-factor tile, written out without the shared helper.
std::size_t canonical_scale_offset(int row, int scale_column) {
  const int k_tiles = kScaleColumns / 4;
  return static_cast<std::size_t>(row / 128) * 512 * k_tiles
      + static_cast<std::size_t>(scale_column / 4) * 512
      + static_cast<std::size_t>(row % 32) * 16
      + static_cast<std::size_t>((row % 128) / 32) * 4
      + static_cast<std::size_t>(scale_column % 4);
}

}  // namespace

int main() {
  // 0.25 times signed E2M1 magnitudes gives a block scale of exactly 0.25.
  const float pattern[16] = {6.0f, -6.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f,
                             4.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, 6.0f};
  const int codes[16] = {0x7, 0xF, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5,
                         0x6, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0x7};

  std::vector<float> input(static_cast<std::size_t>(kRows) * kReduction);
  std::vector<std::uint8_t> packed(static_cast<std::size_t>(kRows) * kReduction / 2);
  std::vector<__nv_fp8_e4m3> scales(static_cast<std::size_t>(kRows) * kScaleColumns);

  for (int probe_row = 0; probe_row < kRows; ++probe_row) {
    for (int probe_block = 0; probe_block < kScaleColumns; ++probe_block) {
      std::fill(input.begin(), input.end(), 0.0f);
      for (int i = 0; i < FP4_BLOCK_SIZE; ++i) {
        input[INPUT_INDEX(probe_row, probe_block * FP4_BLOCK_SIZE + i)] = 0.25f * pattern[i];
      }
      std::fill(packed.begin(), packed.end(), static_cast<std::uint8_t>(0xFF));
      std::fill(scales.begin(), scales.end(), __nv_fp8_e4m3(-1.0f));

      @QUANTIZE_CALL@;

      const std::size_t probe_offset = canonical_scale_offset(probe_row, probe_block);
      for (std::size_t offset = 0; offset < scales.size(); ++offset) {
        const float expected = (offset == probe_offset) ? 0.25f : 1.0f;
        const float got = static_cast<float>(scales[offset]);
        if (got != expected) {
          std::printf("scale probe=(%d,%d) offset=%zu got=%g expected=%g\n",
                      probe_row, probe_block, offset, got, expected);
          return 1;
        }
      }

      for (int row = 0; row < kRows; ++row) {
        for (int k = 0; k < kReduction; ++k) {
          const std::uint8_t byte = packed[static_cast<std::size_t>(row) * (kReduction / 2) + k / 2];
          const int code = (k % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
          const bool in_probe = row == probe_row && k / FP4_BLOCK_SIZE == probe_block;
          const int expected = in_probe ? codes[k % FP4_BLOCK_SIZE] : 0x0;
          if (code != expected) {
            std::printf("nibble probe=(%d,%d) row=%d k=%d got=%d expected=%d\n",
                        probe_row, probe_block, row, k, code, expected);
            return 2;
          }
        }
      }
    }
  }
  std::printf("ok\n");
  return 0;
}
"""


def _extract_function(source: str, signature: str) -> str:
    start = source.index(signature)
    line_start = source.rfind("\n", 0, start) + 1
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[line_start : index + 1]
    raise AssertionError(f"unterminated function body for {signature!r}")


def _run_quantizer_probe(
    tmp_path: Path, source_path: Path, input_index: str, quantize_call: str
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("SKIPPED: a host C++ compiler is required for the NVFP4 quantizer probe")

    quantizer = _extract_function(
        source_path.read_text(encoding="utf-8"), "void quantize_to_nvfp4_kmajor("
    )
    (tmp_path / "host_shim.h").write_text(_HOST_SHIM, encoding="utf-8")
    driver = (
        _DRIVER_TEMPLATE.replace("@QUANTIZER@", quantizer)
        .replace("@INPUT_INDEX@", input_index)
        .replace("@QUANTIZE_CALL@", quantize_call)
    )
    driver_path = tmp_path / "quantizer_probe.cc"
    binary_path = tmp_path / "quantizer_probe"
    driver_path.write_text(driver, encoding="utf-8")

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(tmp_path),
            "-I",
            str(REPO_ROOT),
            str(driver_path),
            "-o",
            str(binary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [str(binary_path)], check=False, capture_output=True, text=True, timeout=120
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert run_result.stdout.strip() == "ok"


def _layout_descriptors(source: str) -> dict[str, tuple[str, ...]]:
    pattern = re.compile(
        r"cublasLtMatrixLayoutCreate\(&(layout[ABC]),\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\)"
    )
    descriptors: dict[str, tuple[str, ...]] = {}
    for match in pattern.finditer(source):
        name = match.group(1)
        assert name not in descriptors, f"{name} is created more than once"
        descriptors[name] = match.groups()[1:]
    return descriptors


def test_ch19_quantizer_writes_kmajor_values_and_swizzled_vec16_scales(tmp_path: Path) -> None:
    _run_quantizer_probe(
        tmp_path,
        CH19_OPTIMIZED,
        input_index="static_cast<std::size_t>(row) * kReduction + (k)",
        quantize_call=(
            "quantize_to_nvfp4_kmajor(input.data(), packed.data(), scales.data(), "
            "kRows, kReduction)"
        ),
    )


def test_ch09_quantizer_reads_row_major_b_as_kmajor_b_transpose(tmp_path: Path) -> None:
    # The baseline passes row-major B[K, N] with row stride 1 and reduction stride N.
    _run_quantizer_probe(
        tmp_path,
        CH09_BASELINE,
        input_index="static_cast<std::size_t>(k) * kRows + (row)",
        quantize_call=(
            "quantize_to_nvfp4_kmajor(input.data(), packed.data(), scales.data(), "
            "kRows, kReduction, 1, kRows)"
        ),
    )


def test_ch19_cublaslt_operands_match_the_quantized_kmajor_storage() -> None:
    source = CH19_OPTIMIZED.read_text(encoding="utf-8")

    assert '#include "../core/common/headers/nvfp4_scale_layout.cuh"' in source
    assert "aisp::nvfp4_vec16_scale_offset(r, block, reduction_dim)" in source
    assert "r * num_scale_cols + block" not in source
    assert source.count("CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3") == 1
    assert "CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode" in source
    assert "CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode" in source

    assert re.search(
        r"quantize_to_nvfp4_kmajor\(h_A_fp32\.data\(\), h_A_packed\.data\(\), "
        r"h_A_scales\.data\(\), M, K\)",
        source,
    )
    assert re.search(
        r"quantize_to_nvfp4_kmajor\(h_B_fp32\.data\(\), h_B_packed\.data\(\), "
        r"h_B_scales\.data\(\), N, K\)",
        source,
    )
    assert "elements_B_packed = static_cast<size_t>(N) * packed_K" in source
    assert "num_scales_B = static_cast<size_t>(N) * (K / FP4_BLOCK_SIZE)" in source
    for assertion in (
        "static_assert(M % 128 == 0",
        "static_assert(N % 128 == 0",
        "static_assert((K / FP4_BLOCK_SIZE) % 4 == 0",
    ):
        assert assertion in source

    assert "transa = CUBLAS_OP_T" in source
    assert "transb = CUBLAS_OP_N" in source
    assert _layout_descriptors(source) == {
        "layoutA": ("CUDA_R_4F_E2M1", "K", "M", "K"),
        "layoutB": ("CUDA_R_4F_E2M1", "K", "N", "K"),
        "layoutC": ("CUDA_R_16F", "M", "N", "M"),
    }


def _harness_shape(benchmark_type: type) -> tuple[int, int, int]:
    params = benchmark_type()._workload_params
    return int(params["M"]), int(params["N"]), int(params["K"])


def test_ch19_cuda_shape_matches_the_harness_workload() -> None:
    from ch19.baseline_fp4_hardware_kernel import BaselineFP4HardwareKernelBenchmark
    from ch19.optimized_fp4_hardware_kernel import OptimizedFP4HardwareKernelBenchmark

    shape = _harness_shape(OptimizedFP4HardwareKernelBenchmark)
    assert _harness_shape(BaselineFP4HardwareKernelBenchmark) == shape

    source = CH19_OPTIMIZED.read_text(encoding="utf-8")
    match = re.search(r"constexpr int M = (\d+), N = (\d+), K = (\d+);", source)
    assert match is not None
    assert tuple(int(value) for value in match.groups()) == shape

    baseline_source = CH19_BASELINE.read_text(encoding="utf-8")
    baseline_shape = tuple(
        int(re.search(rf"const int {name} = (\d+);", baseline_source).group(1))
        for name in ("M", "N", "K")
    )
    assert baseline_shape == shape

    m, n, k = shape
    assert m % 128 == 0
    assert n % 128 == 0
    assert (k // 16) % 4 == 0


@pytest.mark.parametrize("source_path", (CH09_BASELINE, CH09_OPTIMIZED), ids=("baseline", "optimized"))
def test_ch09_layout_a_does_not_depend_on_square_m_and_k(source_path: Path) -> None:
    source = source_path.read_text(encoding="utf-8")

    assert "transa = CUBLAS_OP_T" in source
    assert "transb = CUBLAS_OP_N" in source
    assert _layout_descriptors(source) == {
        "layoutA": ("CUDA_R_4F_E2M1", "kK", "kM", "kK"),
        "layoutB": ("CUDA_R_4F_E2M1", "kK", "kN", "kK"),
        "layoutC": ("CUDA_R_16F", "kM", "kN", "kM"),
    }


# Host stand-ins for the CUDA launch built-ins and FP16 conversions that the ch19
# manual kernel uses. Every value in the kernel probe is exact in FP32, so the
# float-backed __half matches what the GPU computes before the final FP16 store.
_KERNEL_SHIM = (
    _HOST_SHIM
    + r"""
#define __global__
#define __device__
#define __forceinline__ inline

struct HostDim3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};
inline HostDim3 blockIdx{0, 0, 0};
inline HostDim3 blockDim{1, 1, 1};
inline HostDim3 threadIdx{0, 0, 0};

struct __half_raw {
  float value;
};
struct __half {
  float value;
  __half() = default;
  explicit __half(__half_raw raw) : value(raw.value) {}
};

inline __half_raw __nv_cvt_fp4_to_halfraw(__nv_fp4_storage_t code, __nv_fp4_interpretation_t) {
  static const float kMagnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = kMagnitudes[code & 0x7];
  return __half_raw{(code & 0x8) ? -magnitude : magnitude};
}
inline float __half2float(__half value) { return value.value; }
inline __half __float2half(float value) {
  __half result;
  result.value = value;
  return result;
}
"""
)

_KERNEL_DRIVER = r"""
#include <cmath>
#include <cstdio>
#include <vector>

#include "host_shim.h"

constexpr int FP4_BLOCK_SIZE = 16;

@FUNCTIONS@

int main() {
  // N != K. A kernel that strides B by N / 2 bytes and N / 16 scales reads the
  // wrong elements of the [N, K] buffer at this shape, while staying in bounds.
  constexpr int M = 32;
  constexpr int N = 48;
  constexpr int K = 64;
  constexpr int kBlocks = K / FP4_BLOCK_SIZE;
  const float pattern[16] = {6.0f, -6.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f,
                             4.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, 6.0f};

  // Every 16-value block holds all pattern values times a power-of-two scale,
  // so the block scale and every quantized value are exact.
  auto fill = [&](std::vector<float>& values, int rows, int salt) {
    for (int r = 0; r < rows; ++r) {
      for (int k = 0; k < K; ++k) {
        const int block = k / FP4_BLOCK_SIZE;
        const float scale = std::ldexp(1.0f, (r + 3 * block + salt) % 4 - 2);
        values[static_cast<std::size_t>(r) * K + k] = scale * pattern[(k + r + salt) % 16];
      }
    }
  };

  std::vector<float> a(static_cast<std::size_t>(M) * K);
  std::vector<float> b_t(static_cast<std::size_t>(N) * K);
  fill(a, M, 0);
  fill(b_t, N, 5);

  std::vector<std::uint8_t> a_packed(static_cast<std::size_t>(M) * K / 2);
  std::vector<std::uint8_t> b_t_packed(static_cast<std::size_t>(N) * K / 2);
  std::vector<__nv_fp8_e4m3> a_scales(static_cast<std::size_t>(M) * kBlocks);
  std::vector<__nv_fp8_e4m3> b_t_scales(static_cast<std::size_t>(N) * kBlocks);
  quantize_to_nvfp4(a.data(), a_packed.data(), a_scales.data(), M, K);
  quantize_to_nvfp4(b_t.data(), b_t_packed.data(), b_t_scales.data(), N, K);

  std::vector<__half> c(static_cast<std::size_t>(M) * N, __float2half(std::nanf("")));
  blockDim = HostDim3{16, 16, 1};
  for (unsigned int by = 0; by < (M + 15) / 16; ++by) {
    for (unsigned int bx = 0; bx < (N + 15) / 16; ++bx) {
      for (unsigned int ty = 0; ty < 16; ++ty) {
        for (unsigned int tx = 0; tx < 16; ++tx) {
          blockIdx = HostDim3{bx, by, 0};
          threadIdx = HostDim3{tx, ty, 0};
          nvfp4_gemm_manual(a_packed.data(), b_t_packed.data(), a_scales.data(),
                            b_t_scales.data(), c.data(), M, N, K);
        }
      }
    }
  }

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float expected = 0.0f;
      for (int k = 0; k < K; ++k) {
        expected += a[static_cast<std::size_t>(row) * K + k] *
                    b_t[static_cast<std::size_t>(col) * K + k];
      }
      // The kernel stores C column-major.
      const float got = __half2float(c[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * M]);
      if (got != expected) {
        std::printf("C mismatch row=%d col=%d got=%g expected=%g\n", row, col, got, expected);
        return 1;
      }
    }
  }
  std::printf("ok\n");
  return 0;
}
"""


def test_ch19_manual_baseline_kernel_reads_b_transpose_at_a_non_square_shape(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("SKIPPED: a host C++ compiler is required for the NVFP4 kernel probe")

    source = CH19_BASELINE.read_text(encoding="utf-8")
    functions = "\n\n".join(
        _extract_function(source, signature)
        for signature in (
            "static void quantize_to_nvfp4(",
            "float fp4_to_float(",
            "float fp8_to_float(",
            "void nvfp4_gemm_manual(",
        )
    )
    (tmp_path / "host_shim.h").write_text(_KERNEL_SHIM, encoding="utf-8")
    driver_path = tmp_path / "kernel_probe.cc"
    binary_path = tmp_path / "kernel_probe"
    driver_path.write_text(_KERNEL_DRIVER.replace("@FUNCTIONS@", functions), encoding="utf-8")

    compile_result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(tmp_path), str(driver_path), "-o", str(binary_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [str(binary_path)], check=False, capture_output=True, text=True, timeout=120
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert run_result.stdout.strip() == "ok"


def test_ch19_manual_baseline_builds_b_as_kmajor_b_transpose() -> None:
    source = CH19_BASELINE.read_text(encoding="utf-8")

    assert "std::vector<float> h_B_fp32(static_cast<size_t>(N) * K);" in source
    assert (
        "quantize_to_nvfp4(h_B_fp32.data(), h_B_packed.data(), h_B_scales.data(), N, K);"
        in source
    )
    assert "elements_B_packed = static_cast<size_t>(N) * packed_K;" in source
    assert "num_scales_B = static_cast<size_t>(N) * (K / FP4_BLOCK_SIZE);" in source
    assert "B_packed[col * packed_K + (k / 2)]" in source
    assert "B_scales[col * scale_cols_K + (k / FP4_BLOCK_SIZE)]" in source
    for stale in ("packed_N", "scale_cols_B"):
        assert stale not in source


# Stand-in for the compiled ch19 binaries. It honors the same --dump-output
# contract: it prints TIME_MS and writes C as column-major FP16 after timing.
# C[row, col] = row % 16 + 16 * (col % 16), so a transposed read is visible.
_FAKE_FP4_BINARY = """#!{python}
import os
import sys

import numpy as np

args = sys.argv[1:]
dump_path = args[args.index("--dump-output") + 1]
mode = os.environ["AISP_TEST_FAKE_FP4_DUMP"]
rows = np.arange({m}, dtype=np.float32) % 16
cols = (np.arange({n}, dtype=np.float32) % 16) * 16
c = (rows[:, None] + cols[None, :]).astype(np.float16)
if mode == "swap_columns":
    c[:, [0, 1]] = c[:, [1, 0]]
column_major = np.ascontiguousarray(c.T).ravel()
if mode == "short":
    column_major = column_major[:-1]
if mode != "skip":
    column_major.tofile(dump_path)
print("TIME_MS: 1.5")
"""


def _ch19_benchmark_types() -> tuple[type, type]:
    from ch19.baseline_fp4_hardware_kernel import BaselineFP4HardwareKernelBenchmark
    from ch19.optimized_fp4_hardware_kernel import OptimizedFP4HardwareKernelBenchmark

    return BaselineFP4HardwareKernelBenchmark, OptimizedFP4HardwareKernelBenchmark


def _setup_with_fake_binary(benchmark_type: type, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    benchmark = benchmark_type()
    m, n, _ = _harness_shape(benchmark_type)
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake_binary = tmp_path / f"fake_{benchmark.binary_name}"
    fake_binary.write_text(
        _FAKE_FP4_BINARY.format(python=sys.executable, m=m, n=n), encoding="utf-8"
    )
    fake_binary.chmod(0o755)

    def build_fake(verify_mode: bool = False, *, profile_mode: bool = False) -> Path:
        assert not verify_mode and not profile_mode
        benchmark.exec_path = fake_binary
        return fake_binary

    monkeypatch.setattr(benchmark, "_build_binary", build_fake)
    benchmark.setup()
    return benchmark


@pytest.mark.parametrize(
    ("variant", "metrics_label"),
    (("baseline", "baseline_manual_fp4"), ("optimized", "optimized_cublaslt_nvfp4")),
)
def test_ch19_wrappers_verify_the_full_timed_output_matrix(
    variant: str, metrics_label: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_type = dict(zip(("baseline", "optimized"), _ch19_benchmark_types()))[variant]
    monkeypatch.setenv("AISP_TEST_FAKE_FP4_DUMP", "full")
    benchmark = _setup_with_fake_binary(benchmark_type, tmp_path, monkeypatch)
    m, n, _ = _harness_shape(benchmark_type)

    assert benchmark.run_args[0] == "--dump-output"
    dump_dir = Path(benchmark.run_args[1]).parent
    benchmark.benchmark_fn()
    assert benchmark.validate_result() is None
    assert benchmark.last_time_ms == 1.5

    verified = benchmark.get_verify_output()
    assert verified.shape == (m, n)
    assert verified.dtype == torch.float16
    assert verified.data_ptr() != benchmark.output.data_ptr()
    for row, col in ((0, 0), (1, 0), (0, 1), (17, 3), (m - 1, n - 1)):
        assert verified[row, col].item() == row % 16 + 16 * (col % 16)
    assert benchmark.get_custom_metrics()["variant"] == metrics_label

    benchmark.teardown()
    assert benchmark.run_args == []
    assert not dump_dir.exists()


def test_ch19_wrapper_rejects_a_missing_or_short_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_type, _ = _ch19_benchmark_types()
    monkeypatch.setenv("AISP_TEST_FAKE_FP4_DUMP", "full")
    benchmark = _setup_with_fake_binary(baseline_type, tmp_path, monkeypatch)
    benchmark.benchmark_fn()
    assert benchmark.output is not None

    # A run that writes nothing must not reuse the previous run's dump.
    monkeypatch.setenv("AISP_TEST_FAKE_FP4_DUMP", "skip")
    with pytest.raises(RuntimeError, match="did not write --dump-output"):
        benchmark.benchmark_fn()
    assert benchmark.output is None
    with pytest.raises(RuntimeError, match="completed timed execution"):
        benchmark.get_verify_output()

    monkeypatch.setenv("AISP_TEST_FAKE_FP4_DUMP", "short")
    with pytest.raises(RuntimeError, match="bytes, expected 2097152"):
        benchmark.benchmark_fn()
    benchmark.teardown()


def test_ch19_pair_verification_catches_errors_the_mean_checksum_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.benchmark.quarantine import QuarantineManager
    from core.benchmark.verify_runner import VerifyRunner

    baseline_type, optimized_type = _ch19_benchmark_types()
    monkeypatch.setenv("AISP_TEST_FAKE_FP4_DUMP", "full")
    baseline = _setup_with_fake_binary(baseline_type, tmp_path / "baseline", monkeypatch)
    baseline.benchmark_fn()
    # Swapping two columns of C keeps its mean exactly, like scrambled block scales
    # that move the mean by less than the old checksum tolerance.
    monkeypatch.setenv("AISP_TEST_FAKE_FP4_DUMP", "swap_columns")
    optimized = _setup_with_fake_binary(optimized_type, tmp_path / "optimized", monkeypatch)
    optimized.benchmark_fn()

    expected = baseline.get_verify_output()
    actual = optimized.get_verify_output()
    rtol, atol = baseline.get_output_tolerance()
    assert (rtol, atol) == optimized.get_output_tolerance()
    old_checksum = torch.tensor([expected.double().mean().item()])
    new_checksum = torch.tensor([actual.double().mean().item()])
    assert torch.allclose(old_checksum, new_checksum, rtol=rtol, atol=atol)

    runner = VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path / "quarantine"),
    )
    swapped = runner.compare_perf_outputs(expected, actual, (rtol, atol))
    assert not swapped.passed
    assert swapped.location is not None and swapped.location[1] in (0, 1)
    matched = runner.compare_perf_outputs(expected, expected.clone(), (rtol, atol))
    assert matched.passed

    baseline.teardown()
    optimized.teardown()
