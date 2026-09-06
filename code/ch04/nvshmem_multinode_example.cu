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
 * 1. Node-local grouping via NVSHMEM teams
 * 2. Hierarchical aggregation using host-side NVSHMEM gets
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
#include <cmath>
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

static void initialize_nvshmem_device() {
    nvshmem_init();
    const int world_pe = nvshmem_my_pe();
    const int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    if (local_pe < 0) {
        fprintf(stderr, "PE %d has no rank in NVSHMEMX_TEAM_NODE\n", world_pe);
        nvshmem_global_exit(EXIT_FAILURE);
    }

    // Team creation does not trigger NVSHMEM's deferred device-initialization
    // path. Select the CUDA device by the node-local PE and force completion
    // with a documented lazy-init collective before splitting teams.
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
    ctx.node_team = NVSHMEM_TEAM_INVALID;
    ctx.node_team_valid = false;

    // Team creation is collective over the parent team. Every PE therefore
    // creates the node subsets in the same order and retains only its team.
    for (int node = 0; node < ctx.num_nodes; ++node) {
        const int start = node * ctx.gpus_per_node;
        const int size = std::min(ctx.gpus_per_node, ctx.world_size - start);
        nvshmem_team_config_t config;
        std::memset(&config, 0, sizeof(config));
        nvshmem_team_t candidate = NVSHMEM_TEAM_INVALID;
        int status = nvshmem_team_split_strided(
            NVSHMEM_TEAM_WORLD, start, 1, size, &config, 0L, &candidate);
        if (status != 0) {
            fprintf(stderr, "PE %d failed to create node team %d: %d\n",
                    ctx.world_rank, node, status);
            nvshmem_global_exit(EXIT_FAILURE);
        }
        if (node == ctx.node_id) {
            ctx.node_team = candidate;
            ctx.node_team_valid = candidate != NVSHMEM_TEAM_INVALID;
        }
    }
    if (!ctx.node_team_valid) {
        fprintf(stderr, "PE %d was not assigned to a node team\n", ctx.world_rank);
        nvshmem_global_exit(EXIT_FAILURE);
    }
    return ctx;
}

float hierarchical_reduce(NodeContext &ctx, float local_value, float *scratch) {
    CUDA_CHECK(cudaMemcpy(scratch, &local_value, sizeof(float),
                          cudaMemcpyHostToDevice));
    nvshmem_barrier_all();

    int node_leader_rank = ctx.node_id * ctx.gpus_per_node;
    float node_sum = 0.0f;
    if (ctx.local_rank == 0) {
        int node_members = std::min(ctx.gpus_per_node, ctx.world_size - node_leader_rank);
        for (int i = 0; i < node_members; ++i) {
            NVTX_RANGE("iteration");
            float val = nvshmem_float_g(scratch, node_leader_rank + i);
            node_sum += val;
        }
        CUDA_CHECK(cudaMemcpy(scratch, &node_sum, sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    nvshmem_barrier_all();

    int global_leader = 0;
    float global_sum = 0.0f;
    if (ctx.world_rank == global_leader) {
        for (int node = 0; node < ctx.num_nodes; ++node) {
            NVTX_RANGE("iteration");
            int leader = node * ctx.gpus_per_node;
            float val = nvshmem_float_g(scratch, leader);
            global_sum += val;
        }
        CUDA_CHECK(cudaMemcpy(scratch, &global_sum, sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    nvshmem_barrier_all();
    float result = nvshmem_float_g(scratch, global_leader);
    nvshmem_barrier_all();
    return result;
}

bool run_multinode_demo(int argc, char **argv) {
    NodeContext ctx = build_node_context(argc, argv);

    if (ctx.world_rank == 0) {
        printf("\nNVSHMEM Multi-Node Hierarchical Demo\n");
        printf("  Total PEs: %d\n", ctx.world_size);
        printf("  GPUs per node (assumed): %d\n", ctx.gpus_per_node);
        printf("  Nodes detected: %d\n\n", ctx.num_nodes);
    }

    float *scratch = static_cast<float *>(nvshmem_malloc(sizeof(float)));
    if (scratch == nullptr) {
        fprintf(stderr, "PE %d failed to allocate symmetric reduction storage\n",
                ctx.world_rank);
        nvshmem_global_exit(EXIT_FAILURE);
    }
    float local_value = (ctx.world_rank + 1) * 1.0f;
    float global_sum = hierarchical_reduce(ctx, local_value, scratch);

    float expected = (ctx.world_size * (ctx.world_size + 1)) / 2.0f;
    bool correct = std::isfinite(global_sum) && fabs(global_sum - expected) < 0.01f;
    if (ctx.world_rank == 0) {
        printf("Global sum via hierarchical NVSHMEM: %.1f (expected %.1f, %s)\n",
               global_sum, expected, correct ? "PASS" : "FAIL");
    }

    nvshmem_barrier_all();
    if (ctx.world_rank == 0) {
        float avg = global_sum / ctx.world_size;
        for (int peer = 0; peer < ctx.world_size; ++peer) {
            NVTX_RANGE("iteration");
            nvshmem_float_p(scratch, avg, peer);
        }
        nvshmem_quiet();
    }

    nvshmem_barrier_all();
    float avg_value = nvshmem_float_g(scratch, ctx.world_rank);
    float expected_avg = expected / ctx.world_size;
    correct = correct && std::isfinite(avg_value) &&
              fabs(avg_value - expected_avg) < 0.01f;
    printf("PE %02d (node %d, local %d) average=%.2f\n",
           ctx.world_rank, ctx.node_id, ctx.local_rank, avg_value);

    nvshmem_barrier_all();
    nvshmem_team_destroy(ctx.node_team);
    nvshmem_barrier_all();
    nvshmem_free(scratch);
    return correct;
}

#else  // USE_NVSHMEM

bool run_multinode_demo(int, char **) {
    printf("NVSHMEM not available - conceptual multi-node example:\n");
    printf("1. Split NVSHMEM_TEAM_WORLD into node-level teams (4 GPUs each)\n");
    printf("2. Aggregate each node using host-side NVSHMEM gets\n");
    printf("3. Aggregate the node leaders at the global leader\n");
    printf("4. Broadcast final result back to all PEs\n");
    printf("Compile with -DUSE_NVSHMEM and NVSHMEM libraries for execution.\n");
    return true;
}

#endif  // USE_NVSHMEM

int main(int argc, char **argv) {
    NVTX_RANGE("main");
#ifdef USE_NVSHMEM
    initialize_nvshmem_device();
#endif

    bool correct = run_multinode_demo(argc, argv);

#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
    nvshmem_finalize();
#endif
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
