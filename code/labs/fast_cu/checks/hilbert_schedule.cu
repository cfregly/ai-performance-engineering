#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using std::max;
using bf16 = __nv_bfloat16;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static void schedulerCudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ':' << line << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

#define cudaCheck(err) schedulerCudaCheck((err), __FILE__, __LINE__)

#include "h100/matmul/matmul_11.cuh"
#include "h100/matmul/matmul_12.cuh"

namespace {

constexpr int kCores = 64;
constexpr int kSpaceLen = 128;
constexpr int kBlockM = 128;
constexpr int kBlockN = 256;
constexpr int kClusterM = 2;
constexpr int kClusterN = 1;
constexpr int kPoison = 0x5a5a5a5a;

struct MatrixShape {
    int m;
    int n;
};

[[noreturn]] void fail(const char *variant, MatrixShape shape,
                       const std::string &message) {
    std::cerr << variant << " scheduler failed for " << shape.m << 'x' << shape.n
              << ": " << message << '\n';
    std::exit(1);
}

template <typename CreateHilbert>
std::vector<int> build_and_validate(const char *variant, MatrixShape shape,
                                    CreateHilbert create_hilbert) {
    if (shape.m % (kBlockM * kClusterM) != 0 ||
        shape.n % (kBlockN * kClusterN) != 0) {
        fail(variant, shape, "test shape is not an exact kernel tile multiple");
    }

    const int tiles_m = shape.m / (kBlockM * kClusterM);
    const int tiles_n = shape.n / (kBlockN * kClusterN);
    const int expected = tiles_m * tiles_n;
    std::vector<int> space(kCores * kSpaceLen, kPoison);
    create_hilbert(tiles_m, tiles_n, kCores, space.data());

    std::vector<bool> seen(expected, false);
    int total = 0;
    int min_count = kSpaceLen;
    int max_count = 0;

    for (int core = 0; core < kCores; ++core) {
        bool saw_sentinel = false;
        int core_count = 0;
        for (int slot = 0; slot < kSpaceLen; ++slot) {
            const int value = space[core * kSpaceLen + slot];
            if (value == -1) {
                saw_sentinel = true;
                continue;
            }
            if (value == kPoison) {
                fail(variant, shape, "constructor left an output slot untouched");
            }
            if (saw_sentinel) {
                fail(variant, shape, "non-sentinel entry follows a sentinel");
            }

            const int tile_m = value >> 16;
            const int tile_n = value & 0xffff;
            if (tile_m < 0 || tile_m >= tiles_m || tile_n < 0 || tile_n >= tiles_n) {
                fail(variant, shape, "encoded tile is outside the workload");
            }
            const int linear = tile_m * tiles_n + tile_n;
            if (seen[linear]) {
                fail(variant, shape, "tile appears more than once");
            }
            seen[linear] = true;
            ++core_count;
            ++total;
        }

        if (core_count > kSpaceLen) {
            fail(variant, shape, "per-core schedule exceeds SPACE_LEN");
        }
        if (saw_sentinel != (core_count < kSpaceLen)) {
            fail(variant, shape, "sentinel does not match the per-core entry count");
        }
        min_count = std::min(min_count, core_count);
        max_count = std::max(max_count, core_count);
    }

    if (total != expected ||
        std::find(seen.begin(), seen.end(), false) != seen.end()) {
        fail(variant, shape, "schedule does not cover every tile exactly once");
    }
    if (max_count - min_count > 1) {
        fail(variant, shape, "work differs by more than one tile across cores");
    }

    std::cout << variant << ' ' << shape.m << 'x' << shape.n << ": " << total
              << " tiles, per-core " << min_count << ".." << max_count << '\n';
    return space;
}

}  // namespace

int main() {
    const MatrixShape shapes[] = {
        {8192, 8192},
        {4096, 8192},
        {8192, 4096},
        {16384, 8192},
        {8192, 16384},
        {32768, 16384},
    };

    for (const MatrixShape shape : shapes) {
        const auto schedule11 = build_and_validate(
            "M11", shape,
            [](int m, int n, int cores, int *space) {
                M11::createHilbert(m, n, cores, space);
            });
        const auto schedule12 = build_and_validate(
            "M12", shape,
            [](int m, int n, int cores, int *space) {
                M12::createHilbert(m, n, cores, space);
            });
        if (schedule11 != schedule12) {
            fail("M11/M12", shape, "scheduler variants disagree");
        }
    }

    std::cout << "scheduler checks passed\n";
    return 0;
}
