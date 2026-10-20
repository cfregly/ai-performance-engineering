"""Regressions for W2-106: the micro-tiling pair must actually micro-tile.

The file names, benchmark key, and the chapter's learning goal all say the
optimized kernel uses register micro-tiles. These tests require each thread to
keep a multi-output register tile. Hosts without nvcc run the real kernel text
under a host C++ emulation of CUDA blocks and threads. On a CUDA device, the
last test runs the harness build and the verify binary's full-output check.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from ch09.baseline_micro_tiling_matmul import get_benchmark as get_baseline_benchmark
from ch09.optimized_micro_tiling_matmul import get_benchmark as get_optimized_benchmark
from core.benchmark.cuda_binary_benchmark import SUPPORTED_CAPABILITY_ARCH

CH09 = Path(__file__).resolve().parents[1] / "ch09"
KERNEL_PATH = CH09 / "optimized_micro_tiling_matmul.cu"
KERNEL_NAME = "matmul_register_tiled"
SM_REGISTERS = 64 * 1024


def _source() -> str:
    return KERNEL_PATH.read_text(encoding="utf-8")


def _constants(source: str) -> dict[str, int]:
    values = dict(re.findall(r"^constexpr int (BM|BN|BK|TM|TN|A_PAD) = (\d+);$", source, re.M))
    assert set(values) == {"BM", "BN", "BK", "TM", "TN", "A_PAD"}, values
    return {name: int(value) for name, value in values.items()}


def _kernel_section(source: str) -> str:
    """Return the constants and the micro-tiled kernel, ending at its closing brace."""
    start = source.index("constexpr int BM = ")
    signature = source.index(f"{KERNEL_NAME}(const float* __restrict__ A,")
    end = source.index("\n}\n", signature) + len("\n}\n")
    return source[start:end]


def _kernel_body(source: str) -> str:
    section = _kernel_section(source)
    return section[section.index(f"{KERNEL_NAME}(") :]


def test_each_thread_accumulates_a_register_micro_tile() -> None:
    source = _source()
    c = _constants(source)
    body = _kernel_body(source)

    assert c["TM"] * c["TN"] > 1, "each thread must own more than one output"
    assert c["BM"] % c["TM"] == 0 and c["BN"] % c["TN"] == 0
    assert "float acc[TM][TN];" in body
    assert "a_frag[i] = As[k][ty + i * THREADS_Y];" in body
    assert "b_frag[j] = Bs[k][tx + j * THREADS_X];" in body
    assert "acc[i][j] += a_frag[i] * b_frag[j];" in body
    assert "C[row * N + col] = acc[i][j];" in body

    launch = re.findall(r"(\w+)<<<(\w+), (\w+)>>>", source.split("int main()", 1)[1])
    assert launch[0] == (KERNEL_NAME, "grid_tiled", "block_tiled")
    assert "dim3 block_tiled(THREADS_X, THREADS_Y);" in source
    assert "dim3 grid_tiled((N + BN - 1) / BN, (N + BM - 1) / BM);" in source


def test_launch_bounds_match_the_register_budget_in_the_comment() -> None:
    source = _source()
    c = _constants(source)
    threads = (c["BM"] // c["TM"]) * (c["BN"] // c["TN"])
    assert f"__global__ void __launch_bounds__(THREADS, 2)\n{KERNEL_NAME}(" in source
    register_cap = SM_REGISTERS // (threads * 2)
    assert threads == 256 and register_cap == 128
    assert c["TM"] * c["TN"] == 64
    assert "The 8x8 accumulator tile alone takes 64 registers." in source
    assert "128 registers lets two 256-thread blocks" in source
    # Accumulators plus one A and one B fragment must fit under the cap.
    assert c["TM"] * c["TN"] + c["TM"] + c["TN"] < register_cap


def test_padded_a_tile_stores_hit_every_bank_once_per_warp() -> None:
    c = _constants(_source())
    threads = (c["BM"] // c["TM"]) * (c["BN"] // c["TN"])
    row_stride = c["BM"] + c["A_PAD"]
    for load in range((c["BM"] * c["BK"]) // threads):
        for warp in range(threads // 32):
            banks = set()
            for lane in range(32):
                idx = warp * 32 + lane + load * threads
                m, k = divmod(idx, c["BK"])
                banks.add((k * row_stride + m) % 32)
            assert len(banks) == 32


def test_verify_build_checks_every_output_and_fails_on_mismatch() -> None:
    verify = _source().split("#ifdef VERIFY", 2)[2].split("#endif", 1)[0]
    assert "matmul_reference<<<grid_ref, block_ref>>>(d_A, d_B, d_ref, N);" in verify
    assert "for (int i = 0; i < N * N; ++i)" in verify
    assert "if (!(err <= 1e-5 * fmax(1.0, fabs(expected))))" in verify
    assert "exit(EXIT_FAILURE);" in verify
    assert verify.index("exit(EXIT_FAILURE);") < verify.index("VERIFY_PRINT_CHECKSUM")


def test_readme_describes_the_register_micro_tile() -> None:
    c = _constants(_source())
    readme = (CH09 / "README.md").read_text(encoding="utf-8")
    row = next(line for line in readme.splitlines() if "`optimized_micro_tiling_matmul.cu`" in line)
    assert (
        f"a {c['BM']}x{c['BN']} shared-memory tiled matmul in which each thread "
        f"accumulates an {c['TM']}x{c['TN']} register micro-tile" in row
    )
    assert "scalar accumulation" not in row
    assert "- Design micro-tiling schedules that balance register pressure with data reuse." in readme


_EMULATION_PRELUDE = r"""
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

struct dim3 { unsigned x = 1, y = 1, z = 1; };
static thread_local dim3 threadIdx;
static dim3 blockIdx;
static std::barrier<>* block_barrier = nullptr;
static void __syncthreads() { block_barrier->arrive_and_wait(); }
#define __global__
// Blocks run one at a time, so one static copy per kernel acts as the block's
// shared memory.
#define __shared__ static
#define __launch_bounds__(...)
"""

_EMULATION_MAIN = r"""
static std::vector<float> read_matrix(const char* path, size_t count) {
    std::vector<float> data(count);
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr || std::fread(data.data(), sizeof(float), count, f) != count) {
        std::fprintf(stderr, "cannot read %s\n", path);
        std::exit(2);
    }
    std::fclose(f);
    return data;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: N A.bin B.bin C.bin\n");
        return 2;
    }
    const int N = std::atoi(argv[1]);
    const size_t count = static_cast<size_t>(N) * N;
    std::vector<float> A = read_matrix(argv[2], count);
    std::vector<float> B = read_matrix(argv[3], count);
    std::vector<float> C(count, NAN);
    std::barrier<> barrier(THREADS);
    block_barrier = &barrier;
    for (int by = 0; by < (N + BM - 1) / BM; ++by) {
        for (int bx = 0; bx < (N + BN - 1) / BN; ++bx) {
            blockIdx = dim3{static_cast<unsigned>(bx), static_cast<unsigned>(by), 1};
            std::vector<std::thread> threads;
            for (unsigned ty = 0; ty < THREADS_Y; ++ty) {
                for (unsigned tx = 0; tx < THREADS_X; ++tx) {
                    threads.emplace_back([&, tx, ty] {
                        threadIdx = dim3{tx, ty, 1};
                        matmul_register_tiled(A.data(), B.data(), C.data(), N);
                    });
                }
            }
            for (std::thread& t : threads) {
                t.join();
            }
        }
    }
    FILE* f = std::fopen(argv[4], "wb");
    if (f == nullptr || std::fwrite(C.data(), sizeof(float), count, f) != count) {
        std::fprintf(stderr, "cannot write %s\n", argv[4]);
        return 2;
    }
    std::fclose(f);
    return 0;
}
"""


@pytest.fixture(scope="module")
def emulated_kernel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("SKIPPED: a host C++20 compiler is required to emulate the CUDA kernel")
    workdir = tmp_path_factory.mktemp("w2106_emulation")
    program = workdir / "emulate.cpp"
    program.write_text(
        _EMULATION_PRELUDE + _kernel_section(_source()) + _EMULATION_MAIN,
        encoding="utf-8",
    )
    binary = workdir / "emulate"
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-pthread", "-Wno-unknown-pragmas", str(program), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


# 256 fills whole 128x128 tiles. 200 leaves partial tiles. 131 also leaves a
# partial 8-wide k-slice.
@pytest.mark.parametrize("n", [256, 200, 131])
def test_kernel_text_matches_reference_under_host_emulation(
    emulated_kernel: Path, tmp_path: Path, n: int
) -> None:
    rng = np.random.default_rng(2106 + n)
    a = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=(n, n)).astype(np.float32)
    a.tofile(tmp_path / "a.bin")
    b.tofile(tmp_path / "b.bin")
    subprocess.run(
        [
            str(emulated_kernel),
            str(n),
            str(tmp_path / "a.bin"),
            str(tmp_path / "b.bin"),
            str(tmp_path / "c.bin"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    c = np.fromfile(tmp_path / "c.bin", dtype=np.float32).reshape(n, n)
    expected = a.astype(np.float64) @ b.astype(np.float64)
    np.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SKIPPED: building and running the CUDA pair needs a CUDA device",
)
def test_harness_builds_runs_and_verifies_the_pair_on_gpu() -> None:
    capability = torch.cuda.get_device_capability()
    if capability not in SUPPORTED_CAPABILITY_ARCH:
        pytest.skip(f"SKIPPED: no CUDA build target for compute capability {capability}")
    baseline = get_baseline_benchmark()
    optimized = get_optimized_benchmark()
    optimized.setup()
    optimized.benchmark_fn()
    assert optimized.validate_result() is None
    assert "Register micro-tiled matmul (8x8 outputs per thread)" in optimized.last_stdout
    # The verify binary exits non-zero if any output differs from its reference.
    optimized_checksum = optimized.run_verify()
    baseline_checksum = baseline.run_verify()
    assert optimized_checksum is not None and baseline_checksum is not None
    rtol, atol = optimized.get_output_tolerance()
    assert optimized_checksum == pytest.approx(baseline_checksum, rel=rtol, abs=atol)
