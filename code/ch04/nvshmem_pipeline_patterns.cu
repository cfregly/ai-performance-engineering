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

// --------------------------------------------------------------------------
// Pattern 1: Lock-Free Producer/Consumer Queue
// --------------------------------------------------------------------------

void lock_free_queue_demo(int my_pe, int n_pes) {
    const int producer = 0;
    const int consumer = 1;
    if (n_pes < 2) {
        if (my_pe == 0) {
            printf("Lock-free queue demo requires at least 2 PEs\n");
        }
        return;
    }

    const int capacity = 64;
    const int items = 32;

    float *queue = (float *)nvshmem_malloc(capacity * sizeof(float));
    int *tail = (int *)nvshmem_malloc(sizeof(int));

    if (my_pe == consumer) {
        *tail = 0;
    }
    nvshmem_barrier_all();

    if (my_pe == producer) {
        for (int i = 0; i < items; i++) {
            NVTX_RANGE("iteration");
            int slot = nvshmem_int_atomic_fetch_inc(tail, consumer);
            slot %= capacity;
            float value = 1000.0f + i;
            nvshmem_float_p(queue + slot, value, consumer);
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
            int tail_value = *tail;
            if (tail_value <= consumed) {
                tail_value = nvshmem_int_g(tail, consumer);
                if (tail_value <= consumed) {
                    continue;
                }
            }
            float value = queue[consumed % capacity];
            consumed++;
            printf("  Consumer read %.1f from slot %d\n", value, (consumed - 1) % capacity);
        }
    }

    nvshmem_barrier_all();
    nvshmem_free(queue);
    nvshmem_free(tail);
}

// --------------------------------------------------------------------------
// Pattern 2: Double-Buffered Pipeline Handoff
// --------------------------------------------------------------------------

static __device__ void fill_chunk(float *buf, int len, float base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        buf[idx] = base + idx;
    }
}

void double_buffer_pipeline_demo(int my_pe, int n_pes) {
    const int stage0 = 0;
    const int stage1 = 1;
    if (n_pes < 2) {
        if (my_pe == 0) {
            printf("Double-buffer pipeline demo requires at least 2 PEs\n");
        }
        return;
    }

    const int chunk_elems = 512;
    const int chunks = 8;
    const int buffers = 2;

    float *buffer = (float *)nvshmem_malloc(buffers * chunk_elems * sizeof(float));
    int *flags = (int *)nvshmem_malloc(buffers * sizeof(int));

    if (my_pe == stage1) {
        for (int i = 0; i < buffers; i++) {
            NVTX_RANGE("setup");
            flags[i] = 0;
        }
    }

    nvshmem_barrier_all();

    if (my_pe == stage0) {
        float *local_chunk;
        CUDA_CHECK(cudaMalloc(&local_chunk, chunk_elems * sizeof(float)));
        dim3 threads(256);
        dim3 blocks((chunk_elems + threads.x - 1) / threads.x);

        for (int chunk = 0; chunk < chunks; chunk++) {
            NVTX_RANGE("compute_kernel:fill_chunk");
            int buf = chunk % buffers;
            nvshmem_int_wait_until(flags + buf, NVSHMEM_CMP_EQ, 0);
            fill_chunk<<<blocks, threads>>>(local_chunk, chunk_elems, 100.0f * chunk);
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_float_put(buffer + buf * chunk_elems, local_chunk, chunk_elems, stage1);
            nvshmem_fence();
            nvshmem_int_p(flags + buf, 1, stage1);
        }
        CUDA_CHECK(cudaFree(local_chunk));
        if (my_pe == 0) {
            printf("Double-buffer pipeline: produced %d chunks\n", chunks);
        }
    }

    if (my_pe == stage1) {
        int received = 0;
        while (received < chunks) {
            NVTX_RANGE("iteration");
            int buf = received % buffers;
            nvshmem_int_wait_until(flags + buf, NVSHMEM_CMP_EQ, 1);
            float first = buffer[buf * chunk_elems];
            printf("  Stage-1 consumed chunk %d (first value %.1f)\n", received, first);
            nvshmem_int_p(flags + buf, 0, stage1);
            received++;
        }
    }

    nvshmem_barrier_all();
    nvshmem_free(buffer);
    nvshmem_free(flags);
}

#else

void lock_free_queue_demo(int my_pe, int) {
    if (my_pe == 0) {
        printf("[Educational Mode] Lock-free queue pattern requires NVSHMEM\n");
    }
}

void double_buffer_pipeline_demo(int my_pe, int) {
    if (my_pe == 0) {
        printf("[Educational Mode] Double-buffer pipeline pattern requires NVSHMEM\n");
    }
}

#endif

int main() {
    NVTX_RANGE("main");
#ifdef USE_NVSHMEM
    nvshmem_init();
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
#else
    int my_pe = 0;
    int n_pes = 1;
#endif

    if (my_pe == 0) {
        printf("NVSHMEM Pipeline Patterns Demo\n");
    }

    lock_free_queue_demo(my_pe, n_pes);
#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
#endif
    double_buffer_pipeline_demo(my_pe, n_pes);

#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
    nvshmem_finalize();
#endif
    return 0;
}
