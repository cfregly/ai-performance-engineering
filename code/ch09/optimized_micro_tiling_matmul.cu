// optimized_micro_tiling_matmul.cu -- Shared-memory tiled matmul with an 8x8
// register micro-tile per thread.
//
// Each 256-thread block computes a 128x128 tile of C. For every 8-wide slice
// of k it stages a 128x8 slice of A and an 8x128 slice of B in shared memory.
// Each thread keeps 8x8 outputs of C in registers. For every k it reads 8
// values of A and 8 values of B from shared memory into registers and issues
// 64 FMAs, so each shared-memory read feeds 4 FMAs. A kernel that computes one
// output per thread needs 2 shared-memory reads for every FMA.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

// Block tile (BM x BN outputs), k-slice depth (BK), and per-thread register
// micro-tile (TM x TN outputs).
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS_X = BN / TN;
constexpr int THREADS_Y = BM / TM;
constexpr int THREADS = THREADS_X * THREADS_Y;
// Padding the transposed A tile spreads each warp's stores across all 32 banks.
constexpr int A_PAD = 4;

static_assert(BM % TM == 0 && BN % TN == 0, "The micro-tile must divide the block tile.");
static_assert((BM * BK) % THREADS == 0, "A-tile loads must divide evenly across the block.");
static_assert((BK * BN) % THREADS == 0, "B-tile loads must divide evenly across the block.");

#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// The 8x8 accumulator tile alone takes 64 registers. Capping each thread at
// 128 registers lets two 256-thread blocks share an SM's 64K-entry register
// file.
__global__ void __launch_bounds__(THREADS, 2)
matmul_register_tiled(const float* __restrict__ A,
                      const float* __restrict__ B,
                      float* __restrict__ C,
                      int N) {
    __shared__ float As[BK][BM + A_PAD];  // A slice stored transposed: As[k][m]
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    float a_frag[TM];
    float b_frag[TN];

    for (int k0 = 0; k0 < N; k0 += BK) {
        #pragma unroll
        for (int l = 0; l < (BM * BK) / THREADS; ++l) {
            const int idx = tid + l * THREADS;
            const int m = idx / BK;
            const int k = idx % BK;
            const int row = block_row + m;
            const int a_col = k0 + k;
            As[k][m] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        }
        #pragma unroll
        for (int l = 0; l < (BK * BN) / THREADS; ++l) {
            const int idx = tid + l * THREADS;
            const int k = idx / BN;
            const int n = idx % BN;
            const int b_row = k0 + k;
            const int col = block_col + n;
            Bs[k][n] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // A thread owns rows ty + i * THREADS_Y and columns
            // tx + j * THREADS_X. A warp then reads 16 consecutive B values
            // and broadcasts 2 A values, so these reads have no bank conflicts.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[k][ty + i * THREADS_Y];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[k][tx + j * THREADS_X];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = block_row + ty + i * THREADS_Y;
        if (row < N) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = block_col + tx + j * THREADS_X;
                if (col < N) {
                    C[row * N + col] = acc[i][j];
                }
            }
        }
    }
}

#ifdef VERIFY
// One output per thread, accumulating k in the same order as
// matmul_register_tiled.
__global__ void matmul_reference(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
#endif

int main() {
    NVTX_RANGE("main");
    const int N = 2048;
    const size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < N * N; ++i) {
        NVTX_RANGE("setup");
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
        h_B[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block_tiled(THREADS_X, THREADS_Y);
    dim3 grid_tiled((N + BN - 1) / BN, (N + BM - 1) / BM);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        NVTX_RANGE("compute_kernel:matmul_register_tiled");
        matmul_register_tiled<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;

    printf("Register micro-tiled matmul (%dx%d outputs per thread): %.4f ms\n", TM, TN, avg_ms);

#ifdef VERIFY
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // A checksum cannot see outputs that land in the wrong place, so compare
    // every element against a one-output-per-thread reference.
    float *d_ref = nullptr;
    float *h_ref = (float*)malloc(bytes);
    CUDA_CHECK(cudaMalloc(&d_ref, bytes));
    dim3 block_ref(16, 16);
    dim3 grid_ref((N + block_ref.x - 1) / block_ref.x,
                  (N + block_ref.y - 1) / block_ref.y);
    matmul_reference<<<grid_ref, block_ref>>>(d_A, d_B, d_ref, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_ref, d_ref, bytes, cudaMemcpyDeviceToHost));

    long long mismatches = 0;
    double max_abs_err = 0.0;
    for (int i = 0; i < N * N; ++i) {
        const double expected = static_cast<double>(h_ref[i]);
        const double err = fabs(static_cast<double>(h_C[i]) - expected);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
        if (!(err <= 1e-5 * fmax(1.0, fabs(expected)))) {
            if (mismatches < 5) {
                fprintf(stderr, "Mismatch at C[%d][%d]: got %.8f, expected %.8f\n",
                        i / N, i % N, h_C[i], h_ref[i]);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        fprintf(stderr,
                "Register micro-tiled matmul failed the full-output check: "
                "%lld of %d outputs differ, max abs error %.3e\n",
                mismatches, N * N, max_abs_err);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaFree(d_ref));
    free(h_ref);

    double checksum = 0.0;
    for (int i = 0; i < N * N; ++i) {
        NVTX_RANGE("verify");
        checksum += static_cast<double>(h_C[i]);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
