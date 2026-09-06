/**
 * NVSHMEM Advanced Pipeline Patterns
 * ==================================
 *
 * Demonstrates two production-inspired NVSHMEM primitives:
 *  1. Lock-free producer/consumer queue shared between GPUs
 *  2. Double-buffered pipeline handoff with overlapping compute/transfer
 *
 * Designed for multi-GPU Blackwell B200 but will run on any multi-GPU system.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "../core/common/nvtx_utils.cuh"

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(status));                             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#ifdef USE_NVSHMEM

static void initialize_nvshmem_device() {
    nvshmem_init();
    const int world_pe = nvshmem_my_pe();
    const int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    if (local_pe < 0) {
        fprintf(stderr, "PE %d has no rank in NVSHMEMX_TEAM_NODE\n", world_pe);
        nvshmem_global_exit(EXIT_FAILURE);
    }

    // nvshmem_init() may only bootstrap the process when no CUDA device has
    // been selected yet. Select by the node-local PE, then use a documented
    // lazy-initialization collective before any allocation or communication.
    CUDA_CHECK(cudaSetDevice(local_pe));
    nvshmem_barrier_all();
    const int status = nvshmemx_init_status();
    if (status != NVSHMEM_STATUS_IS_INITIALIZED &&
        status != NVSHMEM_STATUS_LIMITED_MPG &&
        status != NVSHMEM_STATUS_FULL_MPG) {
        fprintf(stderr, "PE %d NVSHMEM device initialization failed: status %d\n",
                world_pe, status);
        nvshmem_global_exit(EXIT_FAILURE);
    }
}

// --------------------------------------------------------------------------
// Pattern 1: Lock-Free Producer/Consumer Queue
// --------------------------------------------------------------------------

bool lock_free_queue_demo(int my_pe, int n_pes) {
    const int producer = 0;
    const int consumer = 1;
    if (n_pes < 2) {
        if (my_pe == 0) {
            printf("Lock-free queue demo requires at least 2 PEs\n");
        }
        return true;
    }

    const int capacity = 64;
    const int items = 32;

    float *queue = (float *)nvshmem_malloc(capacity * sizeof(float));
    int *tail = (int *)nvshmem_malloc(sizeof(int));
    if (queue == nullptr || tail == nullptr) {
        fprintf(stderr, "PE %d failed to allocate symmetric queue storage\n", my_pe);
        nvshmem_global_exit(EXIT_FAILURE);
    }

    bool correct = true;
    if (my_pe == consumer) {
        nvshmem_int_p(tail, 0, consumer);
    }
    nvshmem_barrier_all();

    if (my_pe == producer) {
        for (int i = 0; i < items; i++) {
            NVTX_RANGE("iteration");
            int slot = i % capacity;
            float value = 1000.0f + i;
            nvshmem_float_p(queue + slot, value, consumer);
            // Publish the new tail only after the corresponding queue slot.
            nvshmem_fence();
            nvshmem_int_p(tail, i + 1, consumer);
        }
        nvshmem_quiet();
        if (my_pe == 0) {
            printf("Lock-free queue: produced %d items\n", items);
        }
    }

    if (my_pe == consumer) {
        int consumed = 0;
        while (consumed < items) {
            NVTX_RANGE("iteration");
            int tail_value = nvshmem_int_g(tail, consumer);
            if (tail_value <= consumed) {
                continue;
            }
            float value = nvshmem_float_g(queue + consumed % capacity, consumer);
            float expected = 1000.0f + consumed;
            if (!std::isfinite(value) || value != expected) {
                correct = false;
            }
            consumed++;
            printf("  Consumer read %.1f from slot %d\n", value, (consumed - 1) % capacity);
        }
    }

    nvshmem_barrier_all();
    nvshmem_free(queue);
    nvshmem_free(tail);
    return correct;
}

// --------------------------------------------------------------------------
// Pattern 2: Double-Buffered Pipeline Handoff
// --------------------------------------------------------------------------

static __global__ void fill_chunk(float *buf, int len, float base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        buf[idx] = base + idx;
    }
}

bool double_buffer_pipeline_demo(int my_pe, int n_pes) {
    const int stage0 = 0;
    const int stage1 = 1;
    if (n_pes < 2) {
        if (my_pe == 0) {
            printf("Double-buffer pipeline demo requires at least 2 PEs\n");
        }
        return true;
    }

    const int chunk_elems = 512;
    const int chunks = 8;
    const int buffers = 2;

    float *buffer = (float *)nvshmem_malloc(buffers * chunk_elems * sizeof(float));
    int *flags = (int *)nvshmem_malloc(buffers * sizeof(int));
    if (buffer == nullptr || flags == nullptr) {
        fprintf(stderr, "PE %d failed to allocate symmetric pipeline storage\n", my_pe);
        nvshmem_global_exit(EXIT_FAILURE);
    }

    // Each PE owns a distinct symmetric instance. Stage 0 uses its flags as
    // buffer-free acknowledgements; stage 1 uses its flags as data-ready
    // notifications.
    CUDA_CHECK(cudaMemset(flags, 0, buffers * sizeof(int)));

    nvshmem_barrier_all();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (my_pe == stage0) {
        float *local_chunk;
        CUDA_CHECK(cudaMalloc(&local_chunk, chunk_elems * sizeof(float)));
        dim3 threads(256);
        dim3 blocks((chunk_elems + threads.x - 1) / threads.x);

        for (int chunk = 0; chunk < chunks; chunk++) {
            NVTX_RANGE("compute_kernel:fill_chunk");
            int buf = chunk % buffers;
            nvshmemx_int_wait_until_on_stream(
                flags + buf, NVSHMEM_CMP_EQ, 0, stream);
            nvshmemx_int_p_on_stream(flags + buf, 1, stage0, stream);
            fill_chunk<<<blocks, threads, 0, stream>>>(
                local_chunk, chunk_elems, 100.0f * chunk);
            CUDA_CHECK(cudaGetLastError());
            nvshmemx_float_put_on_stream(
                buffer + buf * chunk_elems, local_chunk, chunk_elems,
                stage1, stream);
            nvshmemx_quiet_on_stream(stream);
            nvshmemx_int_p_on_stream(flags + buf, 1, stage1, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(local_chunk));
        if (my_pe == 0) {
            printf("Double-buffer pipeline: produced %d chunks\n", chunks);
        }
    }

    bool correct = true;
    if (my_pe == stage1) {
        for (int received = 0; received < chunks; ++received) {
            NVTX_RANGE("iteration");
            int buf = received % buffers;
            nvshmemx_int_wait_until_on_stream(
                flags + buf, NVSHMEM_CMP_EQ, 1, stream);
            float first = 0.0f;
            CUDA_CHECK(cudaMemcpyAsync(
                &first, buffer + buf * chunk_elems, sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            printf("  Stage-1 consumed chunk %d (first value %.1f)\n", received, first);
            if (!std::isfinite(first) || first != 100.0f * received) {
                correct = false;
            }
            nvshmemx_int_p_on_stream(flags + buf, 0, stage1, stream);
            nvshmemx_int_p_on_stream(flags + buf, 0, stage0, stream);
            nvshmemx_quiet_on_stream(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_barrier_all();
    nvshmem_free(buffer);
    nvshmem_free(flags);
    return correct;
}

#else

bool lock_free_queue_demo(int my_pe, int) {
    if (my_pe == 0) {
        printf("[Educational Mode] Lock-free queue pattern requires NVSHMEM\n");
    }
    return true;
}

bool double_buffer_pipeline_demo(int my_pe, int) {
    if (my_pe == 0) {
        printf("[Educational Mode] Double-buffer pipeline pattern requires NVSHMEM\n");
    }
    return true;
}

#endif

int main() {
    NVTX_RANGE("main");
#ifdef USE_NVSHMEM
    initialize_nvshmem_device();
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
#else
    int my_pe = 0;
    int n_pes = 1;
#endif

    if (my_pe == 0) {
        printf("NVSHMEM Pipeline Patterns Demo\n");
    }

    bool all_correct = lock_free_queue_demo(my_pe, n_pes);
#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
#endif
    all_correct = double_buffer_pipeline_demo(my_pe, n_pes) && all_correct;

#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
    nvshmem_finalize();
#endif
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
