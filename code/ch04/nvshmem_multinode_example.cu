/**
 * NVSHMEM Multi-Node Hierarchical Communication (multi-GPU per node)
 * ================================================================
 *
 * Demonstrates how to compose NVSHMEM collectives across multiple nodes
 * using hierarchical patterns. Designed for clusters with multiple Blackwell
 * B200 GPUs per node connected via NVLink 5.0 (intra-node) and InfiniBand
 * HDR/NDR (inter-node).
 *
 * Highlights:
 * 1. Intra-node reductions via NVSHMEM teams (per-node collective)
 * 2. Cross-node aggregation using host-side NVSHMEM atomics
 * 3. Broadcast of global results back to all GPUs
 *
 * Build (with NVSHMEM):
 *   nvcc -O3 -std=c++17 -arch=sm_100 nvshmem_multinode_example.cu \\
 *        -DUSE_NVSHMEM -I$NVSHMEM_HOME/include -L$NVSHMEM_HOME/lib \\
 *        -lnvshmem -o nvshmem_multinode_example
 *
 * Run:
 *   nvshmemrun -np 16 ./nvshmem_multinode_example --gpus-per-node <num_gpus>
 *
 * When NVSHMEM is unavailable this file still compiles and prints the
 * conceptual flow so it can be used for onboarding and documentation.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../core/common/nvtx_utils.cuh"

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t err = (expr);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#ifdef USE_NVSHMEM

struct NodeContext {
    int world_rank;
    int world_size;
    int gpus_per_node;
    int node_id;
    int local_rank;
    int num_nodes;
    nvshmem_team_t node_team;
    bool node_team_valid;
};

int parse_int_flag(const char *flag, int argc, char **argv, int default_value) {
    for (int i = 1; i < argc; ++i) {
        NVTX_RANGE("iteration");
        if (strcmp(argv[i], flag) == 0 && (i + 1) < argc) {
            return std::atoi(argv[i + 1]);
        }
    }
    return default_value;
}

NodeContext build_node_context(int argc, char **argv) {
    NodeContext ctx{};
    ctx.world_rank = nvshmem_my_pe();
    ctx.world_size = nvshmem_n_pes();
    ctx.gpus_per_node = parse_int_flag("--gpus-per-node", argc, argv, 8);
    if (ctx.gpus_per_node <= 0) ctx.gpus_per_node = 8;
    ctx.node_id = ctx.world_rank / ctx.gpus_per_node;
    ctx.local_rank = ctx.world_rank % ctx.gpus_per_node;
    ctx.num_nodes = (ctx.world_size + ctx.gpus_per_node - 1) / ctx.gpus_per_node;
    ctx.node_team_valid = false;

    if (ctx.gpus_per_node > 1 && ctx.world_size >= ctx.gpus_per_node) {
        int start = ctx.node_id * ctx.gpus_per_node;
        int stride = 1;
        int size = std::min(ctx.gpus_per_node, ctx.world_size - start);
        if (size > 0) {
            nvshmem_team_config_t config;
            std::memset(&config, 0, sizeof(config));
            if (nvshmem_team_split_strided(
                    NVSHMEM_TEAM_WORLD, start, stride, size, &config, &ctx.node_team) == 0) {
                ctx.node_team_valid = true;
            }
        }
    }
    return ctx;
}

float hierarchical_reduce(NodeContext &ctx, float local_value, float *scratch) {
    nvshmem_barrier_all();

    int node_leader_rank = ctx.node_id * ctx.gpus_per_node;
    if (ctx.local_rank == 0) {
        scratch[0] = local_value;
    } else {
        nvshmem_float_p(scratch, local_value, node_leader_rank);
    }

    nvshmem_barrier_all();

    float node_sum = 0.0f;
    if (ctx.local_rank == 0) {
        node_sum = scratch[0];
        int node_members = std::min(ctx.gpus_per_node, ctx.world_size - node_leader_rank);
        for (int i = 1; i < node_members; ++i) {
            NVTX_RANGE("iteration");
            float val = nvshmem_float_g(scratch, node_leader_rank + i);
            node_sum += val;
        }
        scratch[0] = node_sum;
    }

    nvshmem_barrier_all();

    int global_leader = 0;
    if (ctx.local_rank == 0 && ctx.world_rank != global_leader) {
        nvshmem_float_atomic_add(scratch, node_sum, global_leader);
    }

    nvshmem_barrier_all();

    float global_sum = 0.0f;
    if (ctx.world_rank == global_leader) {
        global_sum = scratch[0];
        int total_nodes = ctx.num_nodes;
        for (int node = 1; node < total_nodes; ++node) {
            NVTX_RANGE("iteration");
            int leader = node * ctx.gpus_per_node;
            if (leader >= ctx.world_size) {
                leader = ctx.world_size - 1;
            }
            float val = nvshmem_float_g(scratch, leader);
            global_sum += val;
        }
        scratch[0] = global_sum;
    }

    nvshmem_barrier_all();
    float result = nvshmem_float_g(scratch, global_leader);
    nvshmem_barrier_all();
    return result;
}

void run_multinode_demo(int argc, char **argv) {
    NodeContext ctx = build_node_context(argc, argv);

    if (ctx.world_rank == 0) {
        printf("\nNVSHMEM Multi-Node Hierarchical Demo\n");
        printf("  Total PEs: %d\n", ctx.world_size);
        printf("  GPUs per node (assumed): %d\n", ctx.gpus_per_node);
        printf("  Nodes detected: %d\n\n", ctx.num_nodes);
    }

    float *scratch = static_cast<float *>(nvshmem_malloc(sizeof(float)));
    float local_value = (ctx.world_rank + 1) * 1.0f;
    scratch[0] = local_value;

    float global_sum = hierarchical_reduce(ctx, local_value, scratch);

    if (ctx.world_rank == 0) {
        float expected = (ctx.world_size * (ctx.world_size + 1)) / 2.0f;
        printf("Global sum via hierarchical NVSHMEM: %.1f (expected %.1f)\n", global_sum, expected);
    }

    nvshmem_barrier_all();
    if (ctx.local_rank == 0) {
        float avg = global_sum / ctx.world_size;
        for (int peer = 0; peer < ctx.world_size; ++peer) {
            NVTX_RANGE("iteration");
            nvshmem_float_p(scratch, avg, peer);
        }
    }

    nvshmem_barrier_all();
    float avg_value = scratch[0];
    printf("PE %02d (node %d, local %d) average=%.2f\n",
           ctx.world_rank, ctx.node_id, ctx.local_rank, avg_value);

    nvshmem_free(scratch);
}

#else  // USE_NVSHMEM

void run_multinode_demo(int, char **) {
    printf("NVSHMEM not available - conceptual multi-node example:\n");
    printf("1. Split NVSHMEM_TEAM_WORLD into node-level teams (4 GPUs each)\n");
    printf("2. Perform per-node reductions using NVSHMEM team collectives\n");
    printf("3. Aggregate node leaders via NVSHMEM atomics or NCCL\n");
    printf("4. Broadcast final result back to all PEs\n");
    printf("Compile with -DUSE_NVSHMEM and NVSHMEM libraries for execution.\n");
}

#endif  // USE_NVSHMEM

int main(int argc, char **argv) {
    NVTX_RANGE("main");
#ifdef USE_NVSHMEM
    nvshmem_init();
    CUDA_CHECK(cudaSetDevice(nvshmem_my_pe() % std::max(1, parse_int_flag("--gpus-per-node", argc, argv, 4))));
#endif

    run_multinode_demo(argc, argv);

#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
    nvshmem_finalize();
#endif
    return 0;
}
