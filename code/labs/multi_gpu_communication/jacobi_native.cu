// Native Jacobi halo exchange with explicit CUDA stream dependencies.
#include "jacobi_contract.hpp"
#include "../../core/common/nvtx_utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#ifdef AISP_HAVE_NCCL
#include <nccl.h>
#endif
#ifdef AISP_HAVE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#ifndef NVSHMEMX_INIT_ATTR_INITIALIZER
#error This lab requires NVSHMEM headers with the versioned initialization attribute initializer
#endif
#endif

#include <array>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace lab = multi_gpu_communication;

struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(call) cuda_check((call), #call)

void mpi_check(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING]{};
        int length = 0;
        MPI_Error_string(status, message, &length);
        throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
    }
}
#define MPI_CHECK(call) mpi_check((call), #call)

#ifdef AISP_HAVE_NCCL
void nccl_check(ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}
#define NCCL_CHECK(call) nccl_check((call), #call)
#endif

struct Options {
    std::string backend = "nccl";
    std::string schedule = "overlap";
    int nx = 256;
    int ny = 257;
    int iterations = 20;
    int warmup = 2;
    bool registered = false;
    bool graph = false;
    bool cuda_aware_mpi_asserted = false;
    std::string json_out;
    std::string dump_output;
};

int parse_int(const std::string& text) {
    std::size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size() || value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Expected a nonnegative 32-bit integer, got " + text);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--registered-buffers") result.registered = true;
        else if (key == "--graph") result.graph = true;
        else if (key == "--assert-cuda-aware-mpi") result.cuda_aware_mpi_asserted = true;
        else {
            if (i + 1 == argc) throw std::invalid_argument("Missing value for " + key);
            const std::string value = argv[++i];
            if (key == "--backend") result.backend = value;
            else if (key == "--schedule") result.schedule = value;
            else if (key == "--nx") result.nx = parse_int(value);
            else if (key == "--ny") result.ny = parse_int(value);
            else if (key == "--iterations") result.iterations = parse_int(value);
            else if (key == "--warmup") result.warmup = parse_int(value);
            else if (key == "--json-out") result.json_out = value;
            else if (key == "--dump-output") result.dump_output = value;
            else throw std::invalid_argument("Unknown option " + key);
        }
    }
    lab::validate_shape(result.nx, result.ny, result.iterations);
    if (result.iterations == 0) throw std::invalid_argument("Native timing requires iterations >= 1");
    if (result.backend != "mpi" && result.backend != "nccl" && result.backend != "nvshmem") {
        throw std::invalid_argument("backend must be mpi, nccl, or nvshmem");
    }
    if (result.schedule != "serialized" && result.schedule != "overlap") {
        throw std::invalid_argument("schedule must be serialized or overlap");
    }
    if (result.backend != "nccl" && (result.registered || result.graph)) {
        throw Unsupported("Registered buffers and CUDA graph replay require the NCCL backend");
    }
    return result;
}

std::string json_quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

__global__ void jacobi_rows(float* next, const float* current, int nx, int first, int end) {
    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) + first;
    for (std::size_t row = y; x < static_cast<std::size_t>(nx - 1) && row < static_cast<std::size_t>(end); row += gridDim.y) {
        const std::size_t index = row * nx + x;
        next[index] = 0.25f * (current[index - 1] + current[index + 1] +
                               current[index - nx] + current[index + nx]);
    }
}

struct NativeJacobi {
    Options options;
    lab::Domain domain;
    int rank = 0;
    int ranks = 0;
    int device = 0;
    int nccl_version = 0;
    int cuda_runtime_version = 0;
    int cuda_driver_version = 0;
    int nvshmem_major = 0;
    int nvshmem_minor = 0;
    int nvshmem_patch = 0;
    int openshmem_major = 0;
    int openshmem_minor = 0;
    std::string device_name;
    std::array<float*, 2> fields{nullptr, nullptr};
    std::array<void*, 2> registration{nullptr, nullptr};
    cudaStream_t compute = nullptr;
    cudaStream_t communication = nullptr;
    cudaEvent_t producer_ready = nullptr;
    cudaEvent_t halo_ready = nullptr;
    cudaGraph_t pair_graph = nullptr;
    cudaGraph_t tail_graph = nullptr;
    cudaGraphExec_t pair_exec = nullptr;
    cudaGraphExec_t tail_exec = nullptr;
#ifdef AISP_HAVE_NCCL
    ncclComm_t communicator = nullptr;
#endif
    double registration_ms = 0;
    double allocation_ms = 0;
    double capture_ms = 0;
    std::vector<float> initial;

    explicit NativeJacobi(Options config, int world_rank, int world_size)
        : options(std::move(config)), domain(lab::decompose(options.ny, world_size, world_rank)),
          rank(world_rank), ranks(world_size) {}

    void setup() {
        NVTX_RANGE("setup:jacobi");
#ifndef AISP_HAVE_NCCL
        if (options.backend == "nccl") throw Unsupported("This binary was built WITH_NCCL=0");
#endif
#ifndef AISP_HAVE_NVSHMEM
        if (options.backend == "nvshmem") throw Unsupported("This binary was built WITH_NVSHMEM=0");
#endif
        if (ranks < 2) throw Unsupported("The communication lab requires at least two MPI ranks");
        if (options.backend == "mpi") {
            bool cuda_aware = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
            cuda_aware = MPIX_Query_cuda_support() != 0;
#endif
            if (!cuda_aware && !options.cuda_aware_mpi_asserted) {
                throw Unsupported("CUDA-aware MPI is not confirmed. Use a CUDA-enabled MPI build or explicitly assert documented support with --assert-cuda-aware-mpi");
            }
        }
        MPI_Comm local;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int local_rank = 0;
        int local_size = 0;
        MPI_CHECK(MPI_Comm_rank(local, &local_rank));
        MPI_CHECK(MPI_Comm_size(local, &local_size));
        int device_count = 0;
        const auto count_status = cudaGetDeviceCount(&device_count);
        if (count_status != cudaSuccess || device_count == 0) throw Unsupported("No usable CUDA device is visible");
        if (device_count != 1 && local_size > device_count) {
            throw Unsupported("There are more local ranks than visible CUDA devices");
        }
        device = device_count == 1 ? 0 : local_rank;
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        device_name = properties.name;
        // Validate physical assignment even when every process sees ordinal zero.
        std::vector<unsigned char> uuids(static_cast<std::size_t>(local_size) * sizeof(properties.uuid));
        MPI_CHECK(MPI_Allgather(&properties.uuid, sizeof(properties.uuid), MPI_BYTE,
                               uuids.data(), sizeof(properties.uuid), MPI_BYTE, local));
        for (int other = 0; other < local_size; ++other) {
            if (other != local_rank && std::equal(uuids.begin() + other * sizeof(properties.uuid),
                    uuids.begin() + (other + 1) * sizeof(properties.uuid),
                    uuids.begin() + local_rank * sizeof(properties.uuid))) {
                throw Unsupported("Two local ranks selected the same physical GPU");
            }
        }
        MPI_CHECK(MPI_Comm_free(&local));
        CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));
        int least = 0;
        int greatest = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&compute, cudaStreamNonBlocking, least));
        CUDA_CHECK(cudaStreamCreateWithPriority(&communication, cudaStreamNonBlocking, greatest));
        CUDA_CHECK(cudaEventCreateWithFlags(&producer_ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&halo_ready, cudaEventDisableTiming));
#ifdef AISP_HAVE_NCCL
        if (options.backend == "nccl") {
            NCCL_CHECK(ncclGetVersion(&nccl_version));
            if (nccl_version < NCCL_VERSION(2, 9, 0)) throw Unsupported("This lab requires NCCL 2.9 or later");
            if (options.graph && cuda_runtime_version < 11070) {
                throw Unsupported("This priority-preserving graph exercise requires NCCL 2.9 and CUDA 11.7 or later");
            }
            if (options.registered && nccl_version < NCCL_VERSION(2, 23, 4)) {
                throw Unsupported("This P2P registration exercise requires NCCL 2.23.4 or later");
            }
            ncclUniqueId id;
            if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
            MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
            NCCL_CHECK(ncclCommInitRank(&communicator, ranks, id, rank));
        }
#endif
#ifdef AISP_HAVE_NVSHMEM
        if (options.backend == "nvshmem") {
            std::string visibility = std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES") : "<unset>";
            int root_length = rank == 0 ? static_cast<int>(visibility.size()) : 0;
            MPI_CHECK(MPI_Bcast(&root_length, 1, MPI_INT, 0, MPI_COMM_WORLD));
            std::string root_visibility(root_length, '\0');
            if (rank == 0) root_visibility = visibility;
            MPI_CHECK(MPI_Bcast(root_visibility.data(), root_length, MPI_CHAR, 0, MPI_COMM_WORLD));
            if (visibility != root_visibility) throw Unsupported("This lab follows current NVSHMEM guidance and supports the same CUDA_VISIBLE_DEVICES value on every PE");
            MPI_Comm world = MPI_COMM_WORLD;
            nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
            attributes.mpi_comm = &world;
            const int status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attributes);
            if (status != 0 || nvshmem_n_pes() != ranks || nvshmem_my_pe() != rank) {
                throw std::runtime_error("NVSHMEM initialization or MPI-to-PE mapping failed");
            }
            nvshmem_info_get_version(&openshmem_major, &openshmem_minor);
            nvshmemx_vendor_get_version_info(&nvshmem_major, &nvshmem_minor, &nvshmem_patch);
        }
#endif
        const std::size_t elements = static_cast<std::size_t>(domain.rows + 2) * options.nx;
        const int allocation_rows = options.backend == "nvshmem" ? lab::decompose(options.ny, ranks, 0).rows : domain.rows;
        const std::size_t bytes = static_cast<std::size_t>(allocation_rows + 2) * options.nx * sizeof(float);
        if (options.registered) {
#if defined(AISP_HAVE_NCCL) && NCCL_VERSION_CODE >= 22304
            for (int slot = 0; slot < 2; ++slot) {
                const double allocation_start = MPI_Wtime();
                NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&fields[slot]), bytes));
                allocation_ms += (MPI_Wtime() - allocation_start) * 1000.0;
                const double registration_start = MPI_Wtime();
                NCCL_CHECK(ncclCommRegister(communicator, fields[slot], bytes, &registration[slot]));
                registration_ms += (MPI_Wtime() - registration_start) * 1000.0;
            }
#else
            throw Unsupported("The build-time NCCL headers do not support this registration exercise");
#endif
        } else if (options.backend == "nvshmem") {
#ifdef AISP_HAVE_NVSHMEM
            const double allocation_start = MPI_Wtime();
            for (auto& field : fields) {
                field = static_cast<float*>(nvshmem_malloc(bytes));
                if (!field) throw std::runtime_error("NVSHMEM symmetric allocation failed");
            }
            allocation_ms = (MPI_Wtime() - allocation_start) * 1000.0;
#endif
        } else {
            const double allocation_start = MPI_Wtime();
            for (auto& field : fields) CUDA_CHECK(cudaMalloc(&field, bytes));
            allocation_ms = (MPI_Wtime() - allocation_start) * 1000.0;
        }
        initial.resize(elements);
        for (int y = 0; y < domain.rows + 2; ++y) {
            for (int x = 0; x < options.nx; ++x) {
                initial[static_cast<std::size_t>(y) * options.nx + x] =
                    lab::initial_value(options.nx, options.ny, x, domain.global_start + y - 1);
            }
        }
        reset();
    }

    void reset() {
        for (auto field : fields) {
            CUDA_CHECK(cudaMemcpyAsync(field, initial.data(), initial.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, compute));
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
    }

    void launch_rows(float* next, const float* current, int first, int end, cudaStream_t stream) {
        if (first >= end) return;
        constexpr int threads = 256;
        const dim3 grid((options.nx - 3) / threads + 1, std::min(end - first, 65535));
        jacobi_rows<<<grid, threads, 0, stream>>>(next, current, options.nx, first, end);
        CUDA_CHECK(cudaGetLastError());
    }

    void exchange(float* next) {
        NVTX_RANGE("transfer_async:halo_exchange");
        const int top = (rank + ranks - 1) % ranks;
        const int bottom = (rank + 1) % ranks;
        const int nx = options.nx;
        if (options.backend == "mpi") {
            CUDA_CHECK(cudaStreamSynchronize(communication));
            MPI_CHECK(MPI_Sendrecv(next + nx, nx, MPI_FLOAT, top, 11,
                                  next + static_cast<std::size_t>(domain.rows + 1) * nx,
                                  nx, MPI_FLOAT, bottom, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            MPI_CHECK(MPI_Sendrecv(next + static_cast<std::size_t>(domain.rows) * nx,
                                  nx, MPI_FLOAT, bottom, 12,
                                  next, nx, MPI_FLOAT, top, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        } else if (options.backend == "nvshmem") {
#ifdef AISP_HAVE_NVSHMEM
            const auto top_domain = lab::decompose(options.ny, ranks, top);
            nvshmemx_float_put_on_stream(next + static_cast<std::size_t>(top_domain.rows + 1) * nx,
                                         next + nx, nx, top, communication);
            nvshmemx_float_put_on_stream(next, next + static_cast<std::size_t>(domain.rows) * nx,
                                         nx, bottom, communication);
            nvshmemx_barrier_all_on_stream(communication);
#endif
        } else {
#ifdef AISP_HAVE_NCCL
            NCCL_CHECK(ncclGroupStart());
            NCCL_CHECK(ncclRecv(next, nx, ncclFloat, top, communicator, communication));
            NCCL_CHECK(ncclSend(next + static_cast<std::size_t>(domain.rows) * nx,
                                nx, ncclFloat, bottom, communicator, communication));
            NCCL_CHECK(ncclRecv(next + static_cast<std::size_t>(domain.rows + 1) * nx,
                                nx, ncclFloat, bottom, communicator, communication));
            NCCL_CHECK(ncclSend(next + nx, nx, ncclFloat, top, communicator, communication));
            NCCL_CHECK(ncclGroupEnd());
#endif
        }
    }

    void step(int input_slot) {
        NVTX_RANGE("step:jacobi");
        float* next = fields[1 - input_slot];
        const float* current = fields[input_slot];
        if (options.schedule == "serialized") {
            launch_rows(next, current, 1, domain.rows + 1, compute);
            CUDA_CHECK(cudaEventRecord(producer_ready, compute));
            CUDA_CHECK(cudaStreamWaitEvent(communication, producer_ready, 0));
        } else {
            CUDA_CHECK(cudaEventRecord(producer_ready, compute));
            CUDA_CHECK(cudaStreamWaitEvent(communication, producer_ready, 0));
            launch_rows(next, current, 1, 2, communication);
            if (domain.rows > 1) launch_rows(next, current, domain.rows, domain.rows + 1, communication);
            launch_rows(next, current, 2, domain.rows, compute);
        }
        exchange(next);
        CUDA_CHECK(cudaEventRecord(halo_ready, communication));
        CUDA_CHECK(cudaStreamWaitEvent(compute, halo_ready, 0));
    }

    void capture() {
        if (!options.graph) return;
        NVTX_RANGE("setup:capture");
        const double start = MPI_Wtime();
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeGlobal));
        step(0);
        step(1);
        CUDA_CHECK(cudaStreamEndCapture(compute, &pair_graph));
#if CUDART_VERSION >= 11070
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&pair_exec, pair_graph, cudaGraphInstantiateFlagUseNodePriority));
#else
        throw Unsupported("Graph node priorities require CUDA headers 11.7 or later");
#endif
        CUDA_CHECK(cudaStreamBeginCapture(compute, cudaStreamCaptureModeGlobal));
        step(0);
        CUDA_CHECK(cudaStreamEndCapture(compute, &tail_graph));
#if CUDART_VERSION >= 11070
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&tail_exec, tail_graph, cudaGraphInstantiateFlagUseNodePriority));
#endif
        capture_ms = (MPI_Wtime() - start) * 1000.0;
    }

    void run(int iterations, bool use_graph) {
        if (use_graph) {
            for (int pair = 0; pair < iterations / 2; ++pair) CUDA_CHECK(cudaGraphLaunch(pair_exec, compute));
            if (iterations % 2) CUDA_CHECK(cudaGraphLaunch(tail_exec, compute));
        } else {
            for (int iteration = 0; iteration < iterations; ++iteration) step(iteration % 2);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(cudaStreamSynchronize(communication));
    }

    void cleanup() {
        NVTX_RANGE("cleanup:jacobi");
        CUDA_CHECK(cudaDeviceSynchronize());
        if (tail_exec) CUDA_CHECK(cudaGraphExecDestroy(tail_exec));
        if (pair_exec) CUDA_CHECK(cudaGraphExecDestroy(pair_exec));
        if (tail_graph) CUDA_CHECK(cudaGraphDestroy(tail_graph));
        if (pair_graph) CUDA_CHECK(cudaGraphDestroy(pair_graph));
        if (options.registered) {
#if defined(AISP_HAVE_NCCL) && NCCL_VERSION_CODE >= 22304
            for (int slot = 0; slot < 2; ++slot) {
                NCCL_CHECK(ncclCommDeregister(communicator, registration[slot]));
                NCCL_CHECK(ncclMemFree(fields[slot]));
            }
#endif
        } else if (options.backend == "nvshmem") {
#ifdef AISP_HAVE_NVSHMEM
            for (auto field : fields) nvshmem_free(field);
            nvshmem_finalize();
#endif
        } else {
            for (auto field : fields) CUDA_CHECK(cudaFree(field));
        }
#ifdef AISP_HAVE_NCCL
        if (communicator) NCCL_CHECK(ncclCommDestroy(communicator));
#endif
        CUDA_CHECK(cudaEventDestroy(halo_ready));
        CUDA_CHECK(cudaEventDestroy(producer_ready));
        CUDA_CHECK(cudaStreamDestroy(communication));
        CUDA_CHECK(cudaStreamDestroy(compute));
    }
};

double max_rank(double local) {
    double maximum = 0;
    MPI_CHECK(MPI_Allreduce(&local, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    return maximum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        auto options = parse_options(argc, argv);
        std::array<int, 9> configuration{options.nx, options.ny, options.iterations, options.warmup,
            options.backend == "mpi" ? 0 : (options.backend == "nccl" ? 1 : 2),
            options.schedule == "serialized" ? 0 : 1, options.registered, options.graph,
            options.cuda_aware_mpi_asserted};
        std::array<int, 9> minimum{};
        std::array<int, 9> maximum{};
        MPI_CHECK(MPI_Allreduce(configuration.data(), minimum.data(), configuration.size(), MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allreduce(configuration.data(), maximum.data(), configuration.size(), MPI_INT, MPI_MAX, MPI_COMM_WORLD));
        if (minimum != maximum) throw std::invalid_argument("Every rank must use the same workload and communication options");
        const double setup_start = MPI_Wtime();
        NativeJacobi solver(options, rank, ranks);
        solver.setup();
        const double setup_ms = max_rank((MPI_Wtime() - setup_start) * 1000.0);
        const double registration_ms = max_rank(solver.registration_ms);
        const double allocation_ms = max_rank(solver.allocation_ms);
        const double allocation_registration_ms = max_rank(solver.allocation_ms + solver.registration_ms);
        const double warmup_start = MPI_Wtime();
        { NVTX_RANGE("warmup:jacobi"); solver.run(options.warmup, false); }
        const double warmup_ms = max_rank((MPI_Wtime() - warmup_start) * 1000.0);
        solver.reset();
        solver.capture();
        const double capture_ms = max_rank(solver.capture_ms);
        double graph_warmup_ms = 0;
        if (options.graph && options.warmup > 0) {
            const double graph_warmup_start = MPI_Wtime();
            solver.run(2, true);
            solver.reset();
            solver.run(1, true);
            solver.reset();
            graph_warmup_ms = max_rank((MPI_Wtime() - graph_warmup_start) * 1000.0);
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double start = MPI_Wtime();
        { NVTX_RANGE("compute_graph:jacobi_steady_state"); solver.run(options.iterations, options.graph); }
        const double steady_ms = max_rank((MPI_Wtime() - start) * 1000.0);

        NVTX_RANGE("verify:complete_mesh");
        const double verification_start = MPI_Wtime();
        std::vector<float> local(solver.initial.size());
        CUDA_CHECK(cudaMemcpy(local.data(), solver.fields[options.iterations % 2],
                               local.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const auto reference = lab::reference_jacobi(options.nx, options.ny, options.iterations);
        double max_error = 0;
        int correct = 1;
        for (int y = 0; y < solver.domain.rows + 2; ++y) {
            for (int x = 0; x < options.nx; ++x) {
                const auto i = static_cast<std::size_t>(y) * options.nx + x;
                const auto expected_i = static_cast<std::size_t>(solver.domain.global_start + y - 1) * options.nx + x;
                const double error = std::fabs(static_cast<double>(local[i]) - reference[expected_i]);
                if (!std::isfinite(local[i]) || !std::isfinite(reference[expected_i]) || error > 2e-6) correct = 0;
                if (std::isfinite(error)) max_error = std::max(max_error, error);
            }
        }
        int all_correct = 0;
        MPI_CHECK(MPI_Allreduce(&correct, &all_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        max_error = max_rank(max_error);
        const double verification_ms = max_rank((MPI_Wtime() - verification_start) * 1000.0);

        if (!options.dump_output.empty()) {
            std::vector<int> counts(ranks);
            std::vector<int> offsets(ranks);
            for (int peer = 0; peer < ranks; ++peer) {
                const auto domain = lab::decompose(options.ny, ranks, peer);
                if (static_cast<std::int64_t>(domain.rows) * options.nx > std::numeric_limits<int>::max() ||
                    static_cast<std::int64_t>(domain.global_start) * options.nx > std::numeric_limits<int>::max()) {
                    throw std::invalid_argument("Output gathering exceeds MPI int count limits");
                }
                counts[peer] = domain.rows * options.nx;
                offsets[peer] = domain.global_start * options.nx;
            }
            std::vector<float> output;
            if (rank == 0) output.resize(reference.size());
            MPI_CHECK(MPI_Gatherv(local.data() + options.nx, counts[rank], MPI_FLOAT,
                                  rank == 0 ? output.data() : nullptr, counts.data(), offsets.data(),
                                  MPI_FLOAT, 0, MPI_COMM_WORLD));
            if (rank == 0) {
                std::copy_n(output.data() + static_cast<std::size_t>(options.ny - 2) * options.nx,
                            options.nx, output.data());
                std::copy_n(output.data() + options.nx, options.nx,
                            output.data() + static_cast<std::size_t>(options.ny - 1) * options.nx);
                std::ofstream file(options.dump_output, std::ios::binary);
                file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
                if (!file) throw std::runtime_error("Cannot write complete-output artifact " + options.dump_output);
            }
        }
        std::array<double, 2> mapping{static_cast<double>(solver.domain.rows), static_cast<double>(solver.device)};
        std::vector<double> mappings(static_cast<std::size_t>(ranks) * 2);
        MPI_CHECK(MPI_Gather(mapping.data(), 2, MPI_DOUBLE, mappings.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        const double cleanup_start = MPI_Wtime();
        solver.cleanup();
        const double cleanup_ms = max_rank((MPI_Wtime() - cleanup_start) * 1000.0);
        if (rank == 0) {
            int mpi_major = 0;
            int mpi_minor = 0;
            MPI_CHECK(MPI_Get_version(&mpi_major, &mpi_minor));
            char mpi_library[MPI_MAX_LIBRARY_VERSION_STRING]{};
            int mpi_library_length = 0;
            MPI_CHECK(MPI_Get_library_version(mpi_library, &mpi_library_length));
            std::ostringstream json;
            json << std::setprecision(12) << "{\"schema\":\"multi_gpu_communication.v1\",\"status\":"
                 << json_quote(all_correct ? "passed" : "failed")
                 << ",\"mode\":\"diagnostic\",\"stopping_policy\":\"fixed_iterations\",\"backend\":" << json_quote(options.backend)
                 << ",\"schedule\":" << json_quote(options.schedule)
                 << ",\"registered_buffers\":" << (options.registered ? "true" : "false")
                 << ",\"graph\":" << (options.graph ? "true" : "false")
                 << ",\"graph_node_priority_requested\":" << (options.graph ? "true" : "false")
                 << ",\"nx\":" << options.nx << ",\"ny\":" << options.ny << ",\"ranks\":" << ranks
                 << ",\"iterations\":" << options.iterations << ",\"warmup\":" << options.warmup
                 << ",\"setup_ms\":" << setup_ms << ",\"allocation_registration_ms\":" << allocation_registration_ms
                 << ",\"allocation_ms\":" << allocation_ms << ",\"registration_ms\":" << registration_ms
                 << ",\"capture_ms\":" << capture_ms << ",\"warmup_ms\":" << warmup_ms
                 << ",\"graph_warmup_ms\":" << graph_warmup_ms
                 << ",\"graph_warmup_iterations\":" << (options.graph && options.warmup > 0 ? 3 : 0)
                 << ",\"steady_ms\":" << steady_ms << ",\"time_per_iteration_ms\":" << steady_ms / options.iterations
                 << ",\"verification_ms\":" << verification_ms << ",\"cleanup_ms\":" << cleanup_ms
                 << ",\"max_abs_error\":" << max_error << ",\"checked_owned_elements\":"
                 << static_cast<std::uint64_t>(options.nx) * (options.ny - 2)
                 << ",\"checked_halo_elements\":" << static_cast<std::uint64_t>(2) * ranks * options.nx
                 << ",\"cuda_runtime_version\":" << solver.cuda_runtime_version
                 << ",\"cuda_driver_version\":" << solver.cuda_driver_version
                 << ",\"nccl_version\":" << solver.nccl_version << ",\"device_name\":" << json_quote(solver.device_name)
                 << ",\"mpi_version\":" << json_quote(std::to_string(mpi_major) + "." + std::to_string(mpi_minor))
                 << ",\"mpi_library\":" << json_quote(std::string(mpi_library, mpi_library_length))
                 << ",\"nvshmem_version\":" << (options.backend == "nvshmem" ?
                     json_quote(std::to_string(solver.nvshmem_major) + "." +
                         std::to_string(solver.nvshmem_minor) + "." + std::to_string(solver.nvshmem_patch)) : "null")
#ifdef AISP_HAVE_NVSHMEM
                 << ",\"nvshmem_header_version\":" << (options.backend == "nvshmem" ?
                     json_quote(std::to_string(NVSHMEM_VENDOR_MAJOR_VERSION) + "." +
                         std::to_string(NVSHMEM_VENDOR_MINOR_VERSION) + "." +
                         std::to_string(NVSHMEM_VENDOR_PATCH_VERSION)) : "null")
#else
                 << ",\"nvshmem_header_version\":null"
#endif
                 << ",\"openshmem_api_version\":" << (options.backend == "nvshmem" ?
                     json_quote(std::to_string(solver.openshmem_major) + "." +
                         std::to_string(solver.openshmem_minor)) : "null")
                 << ",\"timing_scope\":\"maximum_rank_full_step_after_both_streams_complete\",\"canonical\":false"
                 << ",\"rank_mapping\":[";
            for (int peer = 0; peer < ranks; ++peer) {
                if (peer) json << ',';
                json << "{\"rank\":" << peer << ",\"owned_rows\":" << int(mappings[peer * 2])
                     << ",\"device_ordinal\":" << int(mappings[peer * 2 + 1]) << '}';
            }
            json << "]}";
            if (!options.json_out.empty()) {
                std::ofstream file(options.json_out);
                file << json.str() << '\n';
                if (!file) throw std::runtime_error("Cannot write result " + options.json_out);
            }
            std::cout << json.str() << '\n';
            if (all_correct) std::cout << "TIME_MS: " << steady_ms / options.iterations << '\n';
        }
        MPI_CHECK(MPI_Finalize());
        return all_correct ? 0 : 1;
    } catch (const Unsupported& error) {
        std::cerr << "SKIPPED: rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 3);
        return 3;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
