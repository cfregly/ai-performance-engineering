// Device-initiated NVSHMEM examples with complete-output verification.
#include "../multi_gpu_communication/jacobi_contract.hpp"
#include "../../core/common/nvtx_utils.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#ifndef NVSHMEMX_INIT_ATTR_INITIALIZER
#error This lab requires the versioned NVSHMEM initialization attribute initializer
#endif

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace jacobi = multi_gpu_communication;

struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };

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

struct Options {
    std::string mode = "scalar-halo";
    int nx = 17;
    int ny = 7;
    int elements = 32;
    int iterations = 4;
    int warmup = 1;
    std::string json_out;
    std::string dump_output;
};

int parse_int(const std::string& text) {
    std::size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size() || value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Expected a nonnegative 32-bit integer: " + text);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help") {
            std::cout << "NVSHMEM device communication diagnostics\n"
                "--mode scalar-halo|persistent-halo|block-signal|persistent-signal|persistent-atomic\n"
                "--nx N --ny N --elements N --iterations N --warmup N\n"
                "--json-out PATH --dump-output PATH\n"
                "Launch with MPI using two or more ranks, one GPU per local PE.\n";
            std::exit(0);
        }
        if (i + 1 == argc) throw std::invalid_argument("Missing value after " + flag);
        const std::string value = argv[++i];
        if (flag == "--mode") options.mode = value;
        else if (flag == "--nx") options.nx = parse_int(value);
        else if (flag == "--ny") options.ny = parse_int(value);
        else if (flag == "--elements") options.elements = parse_int(value);
        else if (flag == "--iterations") options.iterations = parse_int(value);
        else if (flag == "--warmup") options.warmup = parse_int(value);
        else if (flag == "--json-out") options.json_out = value;
        else if (flag == "--dump-output") options.dump_output = value;
        else throw std::invalid_argument("Unknown argument: " + flag);
    }
    if (options.mode != "scalar-halo" && options.mode != "persistent-halo" && options.mode != "block-signal" &&
        options.mode != "persistent-signal" && options.mode != "persistent-atomic") {
        throw std::invalid_argument("Unsupported mode: " + options.mode);
    }
    jacobi::validate_shape(options.nx, options.ny, options.iterations);
    if (options.iterations < 1 || options.elements < 1) {
        throw std::invalid_argument("Require iterations >= 1 and elements >= 1");
    }
    if (!options.json_out.empty() && options.json_out == options.dump_output) {
        throw std::invalid_argument("JSON and float-output paths must differ");
    }
    return options;
}

bool mesh_mode(const std::string& mode) {
    return mode == "scalar-halo" || mode == "persistent-halo";
}

std::size_t checked_product(std::size_t first, std::size_t second) {
    if (second && first > std::numeric_limits<std::size_t>::max() / second) {
        throw std::invalid_argument("Requested allocation exceeds the addressable size");
    }
    return first * second;
}

double max_rank(double value) {
    double result = 0;
    MPI_CHECK(MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    return result;
}

void require_everywhere(bool supported, const std::string& reason) {
    int local = supported ? 1 : 0;
    int all = 0;
    MPI_CHECK(MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (!all) throw Unsupported(reason);
}

// Every value is an exactly representable positive integer after host range checks.
__host__ __device__ float payload_value(int epoch, int pe, int ranks, int edge,
                                       std::size_t index, std::size_t elements) {
    const auto channel = (static_cast<unsigned long long>(epoch - 1) * ranks + pe) * 2 + edge;
    return static_cast<float>(1 + channel * elements + index);
}

__global__ void scalar_halo(float* output, const float* input, int nx, int rows,
                            int previous_rows, int previous, int next) {
    const std::size_t count = std::size_t(rows) * std::size_t(nx - 2);
    for (std::size_t cell = threadIdx.x; cell < count; cell += blockDim.x) {
        const std::size_t y = 1 + cell / std::size_t(nx - 2);
        const std::size_t x = 1 + cell % std::size_t(nx - 2);
        const std::size_t i = y * std::size_t(nx) + x;
        const float value = 0.25f * (input[i - 1] + input[i + 1] + input[i - nx] + input[i + nx]);
        output[i] = value;
        if (y == 1) {
            nvshmem_float_p(output + (std::size_t(previous_rows) + 1) * nx + x, value, previous);
        }
        if (y == std::size_t(rows)) nvshmem_float_p(output + x, value, next);
    }
}

// A single resident CTA contains every local stencil dependency and phase barrier.
__global__ void persistent_halo(float* field_a, float* field_b, uint64_t* ready,
                                uint64_t* consumed, int nx, int rows, int previous_rows,
                                int steps, int previous, int next) {
    float* input = field_a;
    float* output = field_b;
    const std::size_t count = std::size_t(rows) * std::size_t(nx - 2);
    for (int step = 0; step < steps; ++step) {
        for (std::size_t cell = threadIdx.x; cell < count; cell += blockDim.x) {
            const std::size_t y = 1 + cell / std::size_t(nx - 2);
            const std::size_t x = 1 + cell % std::size_t(nx - 2);
            const std::size_t i = y * std::size_t(nx) + x;
            output[i] = 0.25f * (input[i - 1] + input[i + 1] + input[i - nx] + input[i + nx]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            // The CTA has finished all reads of the current halo, whose epoch is step.
            if (step > 0) {
                nvshmemx_signal_op(consumed + 1, uint64_t(step), NVSHMEM_SIGNAL_SET, previous);
                nvshmemx_signal_op(consumed, uint64_t(step), NVSHMEM_SIGNAL_SET, next);
            }
            // The target ping-pong halo last held epoch step-1, not the current epoch.
            if (step >= 2) {
                nvshmem_signal_wait_until(consumed, NVSHMEM_CMP_GE, uint64_t(step - 1));
                nvshmem_signal_wait_until(consumed + 1, NVSHMEM_CMP_GE, uint64_t(step - 1));
            }
        }
        __syncthreads();
        nvshmemx_float_put_nbi_block(output + (std::size_t(previous_rows) + 1) * nx,
                                     output + nx, std::size_t(nx), previous);
        nvshmemx_float_put_nbi_block(output, output + std::size_t(rows) * nx,
                                     std::size_t(nx), next);
        __syncthreads();
        if (threadIdx.x == 0) {
            // Complete both payload sources and earlier acknowledgements before readiness.
            nvshmem_quiet();
            const uint64_t epoch = uint64_t(step) + 1;
            nvshmemx_signal_op(ready + 1, epoch, NVSHMEM_SIGNAL_SET, previous);
            nvshmemx_signal_op(ready, epoch, NVSHMEM_SIGNAL_SET, next);
            nvshmem_signal_wait_until(ready, NVSHMEM_CMP_GE, epoch);
            nvshmem_signal_wait_until(ready + 1, NVSHMEM_CMP_GE, epoch);
        }
        __syncthreads();
        // All threads change roles only after complete remote halos are ready.
        float* temporary = input;
        input = output;
        output = temporary;
    }
}

__device__ void fill_payload(float* outgoing, std::size_t elements, int epoch, int pe, int ranks) {
    for (std::size_t i = threadIdx.x; i < 2 * elements; i += blockDim.x) {
        outgoing[i] = payload_value(epoch, pe, ranks, int(i / elements), i % elements, elements);
    }
}

// This kernel has no communication and is safe to select separately with NCU.
__global__ void prepare_payload(float* outgoing, std::size_t elements, int epoch, int pe, int ranks) {
    fill_payload(outgoing, elements, epoch, pe, ranks);
}

template <bool Atomic>
__device__ void exchange_epoch(float* inbox, float* outgoing, float* history,
                              uint64_t* ready, uint64_t* consumed,
                              int* atomic_ready, int* atomic_consumed,
                              std::size_t elements, int epoch, int previous, int next) {
    // Every CTA thread calls both collective operations with identical arguments.
    nvshmemx_float_put_nbi_block(inbox + elements, outgoing, elements, previous);
    nvshmemx_float_put_nbi_block(inbox, outgoing + elements, elements, next);
    __syncthreads();
    if (threadIdx.x == 0) {
        // CTA ordering lets this elected thread complete its PE's preceding RMA.
        nvshmem_quiet();
        if constexpr (Atomic) {
            nvshmem_int_atomic_inc(atomic_ready + 1, previous);
            nvshmem_int_atomic_inc(atomic_ready, next);
            nvshmem_int_wait_until_all(atomic_ready, 2, nullptr, NVSHMEM_CMP_GE, epoch);
        } else {
            nvshmemx_signal_op(ready + 1, uint64_t(epoch), NVSHMEM_SIGNAL_SET, previous);
            nvshmemx_signal_op(ready, uint64_t(epoch), NVSHMEM_SIGNAL_SET, next);
            nvshmem_signal_wait_until(ready, NVSHMEM_CMP_GE, uint64_t(epoch));
            nvshmem_signal_wait_until(ready + 1, NVSHMEM_CMP_GE, uint64_t(epoch));
        }
    }
    __syncthreads();
    for (std::size_t i = threadIdx.x; i < 2 * elements; i += blockDim.x) {
        history[std::size_t(epoch - 1) * 2 * elements + i] = inbox[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // All local reads have finished before these remote consumed acknowledgements.
        if constexpr (Atomic) {
            nvshmem_int_atomic_inc(atomic_consumed + 1, previous);
            nvshmem_int_atomic_inc(atomic_consumed, next);
            nvshmem_int_wait_until_all(atomic_consumed, 2, nullptr, NVSHMEM_CMP_GE, epoch);
        } else {
            nvshmemx_signal_op(consumed + 1, uint64_t(epoch), NVSHMEM_SIGNAL_SET, previous);
            nvshmemx_signal_op(consumed, uint64_t(epoch), NVSHMEM_SIGNAL_SET, next);
            nvshmem_signal_wait_until(consumed, NVSHMEM_CMP_GE, uint64_t(epoch));
            nvshmem_signal_wait_until(consumed + 1, NVSHMEM_CMP_GE, uint64_t(epoch));
        }
    }
    __syncthreads();
}

__global__ void block_exchange_signal(float* inbox, float* outgoing, float* history,
                                      uint64_t* ready, uint64_t* consumed,
                                      std::size_t elements, int epoch, int previous, int next) {
    exchange_epoch<false>(inbox, outgoing, history, ready, consumed, nullptr, nullptr,
                           elements, epoch, previous, next);
}

template <bool Atomic>
__global__ void persistent_exchange(float* inbox, float* outgoing, float* history,
                                    uint64_t* ready, uint64_t* consumed,
                                    int* atomic_ready, int* atomic_consumed,
                                    std::size_t elements, int steps, int pe, int ranks,
                                    int previous, int next) {
    for (int step = 0; step < steps; ++step) {
        const int epoch = step + 1;
        fill_payload(outgoing, elements, epoch, pe, ranks);
        __syncthreads();
        exchange_epoch<Atomic>(inbox, outgoing, history, ready, consumed, atomic_ready,
                               atomic_consumed, elements, epoch, previous, next);
    }
}

struct DeviceLab {
    Options options;
    int rank;
    int ranks;
    int previous;
    int next;
    int device = 0;
    cudaStream_t stream{};
    float* fields[2]{};
    float* inbox = nullptr;
    float* outgoing = nullptr;
    float* history = nullptr;
    uint64_t* ready = nullptr;
    uint64_t* consumed = nullptr;
    int* atomic_ready = nullptr;
    int* atomic_consumed = nullptr;
    jacobi::Domain domain{};
    std::size_t field_count = 0;
    std::size_t history_count = 0;
    std::vector<float> initial;
    std::string device_name;
    double allocation_ms = 0;
    int cuda_runtime = 0;
    int cuda_driver = 0;
    int nvshmem_major = 0;
    int nvshmem_minor = 0;
    int nvshmem_patch = 0;
    int openshmem_major = 0;
    int openshmem_minor = 0;
    int query_capacity = 0;

    DeviceLab(Options value, int pe, int size)
        : options(std::move(value)), rank(pe), ranks(size), previous((pe + size - 1) % size),
          next((pe + 1) % size) {}

    template <typename T> T* symmetric_allocate(std::size_t count) {
        T* pointer = static_cast<T*>(nvshmem_malloc(checked_product(count, sizeof(T))));
        int local = pointer != nullptr;
        int all = 0;
        MPI_CHECK(MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (!all) throw std::runtime_error("NVSHMEM symmetric allocation failed on a PE");
        return pointer;
    }

    void setup() {
        NVTX_RANGE("setup:nvshmem_device");
        require_everywhere(ranks >= 2, "This lab requires at least two MPI ranks");
        if (mesh_mode(options.mode)) {
            domain = jacobi::decompose(options.ny, ranks, rank);
        } else {
            const auto largest_epoch = std::size_t(std::max(options.iterations, options.warmup));
            const auto values = checked_product(checked_product(largest_epoch, std::size_t(ranks)),
                                               checked_product(2, std::size_t(options.elements)));
            if (values > (1ULL << 24)) {
                throw std::invalid_argument("Payload formula requires max(iterations,warmup)*ranks*2*elements <= 2^24");
            }
            history_count = checked_product(largest_epoch, checked_product(2, std::size_t(options.elements)));
            checked_product(history_count, sizeof(float));
        }
        MPI_Comm local;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int local_rank = 0;
        int local_ranks = 0;
        MPI_CHECK(MPI_Comm_rank(local, &local_rank));
        MPI_CHECK(MPI_Comm_size(local, &local_ranks));
        const char* mask_env = std::getenv("CUDA_VISIBLE_DEVICES");
        const std::string mask = mask_env ? mask_env : "<unset>";
        int root_length = local_rank == 0 ? int(mask.size()) : 0;
        MPI_CHECK(MPI_Bcast(&root_length, 1, MPI_INT, 0, local));
        std::string root_mask(std::size_t(root_length), '\0');
        if (local_rank == 0) root_mask = mask;
        MPI_CHECK(MPI_Bcast(root_mask.data(), root_length, MPI_CHAR, 0, local));
        const bool same_mask = root_mask == mask;
        MPI_CHECK(MPI_Comm_free(&local));
        require_everywhere(same_mask, "Each node's PEs must use the same CUDA_VISIBLE_DEVICES ordering");
        int visible = 0;
        const auto device_status = cudaGetDeviceCount(&visible);
        require_everywhere(device_status == cudaSuccess && visible >= local_ranks,
                           "CUDA runtime needs one distinct visible GPU per local PE");
        device = local_rank;
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        device_name = properties.name;
        if (options.mode != "scalar-halo") {
            require_everywhere(properties.cooperativeLaunch != 0,
                               "Device waits require CUDA cooperative launch support");
        }
        CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));
        CUDA_CHECK(cudaDriverGetVersion(&cuda_driver));
        MPI_Comm world = MPI_COMM_WORLD;
        nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
        attributes.mpi_comm = &world;
        const int init_status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attributes);
        if (init_status != 0) throw std::runtime_error("NVSHMEM MPI initialization failed");
        if (nvshmem_my_pe() != rank || nvshmem_n_pes() != ranks) {
            throw std::runtime_error("MPI rank and NVSHMEM PE mapping differ");
        }
        nvshmem_info_get_version(&openshmem_major, &openshmem_minor);
        nvshmemx_vendor_get_version_info(&nvshmem_major, &nvshmem_minor, &nvshmem_patch);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const double allocation_start = MPI_Wtime();
        if (mesh_mode(options.mode)) {
            const int max_rows = (options.ny - 2) / ranks + ((options.ny - 2) % ranks != 0);
            field_count = checked_product(std::size_t(max_rows) + 2, std::size_t(options.nx));
            fields[0] = symmetric_allocate<float>(field_count);
            fields[1] = symmetric_allocate<float>(field_count);
            if (options.mode == "persistent-halo") {
                ready = symmetric_allocate<uint64_t>(2);
                consumed = symmetric_allocate<uint64_t>(2);
            }
            initial.resize(checked_product(std::size_t(domain.rows) + 2, std::size_t(options.nx)));
            for (int y = 0; y < domain.rows + 2; ++y) {
                for (int x = 0; x < options.nx; ++x) {
                    initial[std::size_t(y) * options.nx + x] =
                        jacobi::initial_value(options.nx, options.ny, x, domain.global_start + y - 1);
                }
            }
        } else {
            inbox = symmetric_allocate<float>(2 * std::size_t(options.elements));
            outgoing = symmetric_allocate<float>(2 * std::size_t(options.elements));
            if (options.mode == "persistent-atomic") {
                atomic_ready = symmetric_allocate<int>(2);
                atomic_consumed = symmetric_allocate<int>(2);
            } else {
                ready = symmetric_allocate<uint64_t>(2);
                consumed = symmetric_allocate<uint64_t>(2);
            }
            CUDA_CHECK(cudaMalloc(&history, checked_product(history_count, sizeof(float))));
        }
        allocation_ms = (MPI_Wtime() - allocation_start) * 1000;
        reset();
        if (options.mode != "scalar-halo") {
            try { query_and_launch(options.iterations, false); }
            catch (const Unsupported&) { cleanup(); throw; }
        }
    }

    void reset() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (mesh_mode(options.mode)) {
            for (auto field : fields) {
                CUDA_CHECK(cudaMemsetAsync(field, 0, field_count * sizeof(float), stream));
                CUDA_CHECK(cudaMemcpyAsync(field, initial.data(), initial.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
            }
        } else {
            CUDA_CHECK(cudaMemsetAsync(inbox, 0, 2 * std::size_t(options.elements) * sizeof(float), stream));
            CUDA_CHECK(cudaMemsetAsync(outgoing, 0, 2 * std::size_t(options.elements) * sizeof(float), stream));
            CUDA_CHECK(cudaMemsetAsync(history, 0, history_count * sizeof(float), stream));
        }
        if (ready) {
            CUDA_CHECK(cudaMemsetAsync(ready, 0, 2 * sizeof(uint64_t), stream));
            CUDA_CHECK(cudaMemsetAsync(consumed, 0, 2 * sizeof(uint64_t), stream));
        } else if (atomic_ready) {
            CUDA_CHECK(cudaMemsetAsync(atomic_ready, 0, 2 * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(atomic_consumed, 0, 2 * sizeof(int), stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvshmem_barrier_all();
    }

    void query_and_launch(int steps, bool launch) {
        constexpr unsigned threads = 256;
        const dim3 block(threads);
        std::size_t elements = std::size_t(options.elements);
        int epoch = steps;
        int previous_rows = options.mode == "persistent-halo" ?
            jacobi::decompose(options.ny, ranks, previous).rows : 0;
        void* arguments_block[] = {&inbox, &outgoing, &history, &ready, &consumed,
                                   &elements, &epoch, &previous, &next};
        void* arguments_persistent[] = {&inbox, &outgoing, &history, &ready, &consumed,
            &atomic_ready, &atomic_consumed, &elements, &steps, &rank, &ranks, &previous, &next};
        void* arguments_halo[] = {&fields[0], &fields[1], &ready, &consumed, &options.nx,
                                  &domain.rows, &previous_rows, &steps, &previous, &next};
        const void* kernel = options.mode == "persistent-halo" ? reinterpret_cast<const void*>(persistent_halo) :
            (options.mode == "block-signal" ?
            reinterpret_cast<const void*>(block_exchange_signal) :
            (options.mode == "persistent-signal" ? reinterpret_cast<const void*>(persistent_exchange<false>) :
                                                     reinterpret_cast<const void*>(persistent_exchange<true>)));
        void** arguments = options.mode == "persistent-halo" ? arguments_halo :
                            (options.mode == "block-signal" ? arguments_block : arguments_persistent);
        if (!launch) {
            int capacity = 0;
            const int query_status = nvshmemx_collective_launch_query_gridsize(kernel, block, arguments, 0, &capacity);
            require_everywhere(query_status == 0 && capacity >= 1,
                               "NVSHMEM collective launch query cannot support the required resident CTA");
            query_capacity = capacity;
        }
        if (launch) {
            if (query_capacity < 1) throw std::runtime_error("Collective kernel was not capacity-checked during setup");
            const int status = nvshmemx_collective_launch(kernel, dim3(1), block, arguments, 0, stream);
            if (status != 0) throw std::runtime_error("NVSHMEM collective launch failed: " + std::to_string(status));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    void run(int steps) {
        if (options.mode == "scalar-halo") {
            const int previous_rows = jacobi::decompose(options.ny, ranks, previous).rows;
            for (int step = 0; step < steps; ++step) {
                scalar_halo<<<1, 256, 0, stream>>>(fields[(step + 1) % 2], fields[step % 2],
                    options.nx, domain.rows, previous_rows, previous, next);
                CUDA_CHECK(cudaGetLastError());
                nvshmemx_barrier_all_on_stream(stream);
            }
        } else if (options.mode == "block-signal") {
            for (int step = 0; step < steps; ++step) {
                prepare_payload<<<1, 256, 0, stream>>>(outgoing, std::size_t(options.elements), step + 1, rank, ranks);
                CUDA_CHECK(cudaGetLastError());
                query_and_launch(step + 1, true);
            }
        } else if (steps > 0) {
            query_and_launch(steps, true);
        }
        nvshmemx_quiet_on_stream(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void cleanup() {
        NVTX_RANGE("cleanup:nvshmem_device");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvshmem_barrier_all();
        if (mesh_mode(options.mode)) {
            nvshmem_free(fields[0]);
            nvshmem_free(fields[1]);
        } else {
            nvshmem_free(inbox);
            nvshmem_free(outgoing);
            CUDA_CHECK(cudaFree(history));
        }
        if (ready) {
            nvshmem_free(ready);
            nvshmem_free(consumed);
        } else if (atomic_ready) {
            nvshmem_free(atomic_ready);
            nvshmem_free(atomic_consumed);
        }
        nvshmem_finalize();
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

std::string json_string(const std::string& text) {
    std::ostringstream escaped;
    escaped << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') escaped << '\\' << c;
        else if (c < 0x20) escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else escaped << c;
    }
    escaped << '"';
    return escaped.str();
}

// Chunk point-to-point gathering avoids MPI's signed-int collective count limit.
std::vector<float> gather_rank_data(const std::vector<float>& local, int rank, int ranks) {
    unsigned long long count = local.size();
    std::vector<unsigned long long> counts(static_cast<std::size_t>(ranks));
    MPI_CHECK(MPI_Allgather(&count, 1, MPI_UNSIGNED_LONG_LONG, counts.data(), 1,
                           MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD));
    std::size_t total = 0;
    for (auto value : counts) {
        if (value > std::numeric_limits<std::size_t>::max() - total) {
            throw std::invalid_argument("Gathered output exceeds the addressable size");
        }
        total += std::size_t(value);
    }
    std::vector<float> result;
    if (rank == 0) result.resize(total);
    std::size_t offset = 0;
    for (int pe = 0; pe < ranks; ++pe) {
        const std::size_t size = std::size_t(counts[std::size_t(pe)]);
        if (pe == 0 && rank == 0) std::copy(local.begin(), local.end(), result.begin());
        else if (pe != 0 && (rank == 0 || rank == pe)) {
            for (std::size_t start = 0; start < size;) {
                const int chunk = int(std::min(size - start, std::size_t(std::numeric_limits<int>::max())));
                if (rank == pe) MPI_CHECK(MPI_Send(local.data() + start, chunk, MPI_FLOAT, 0, 71, MPI_COMM_WORLD));
                else MPI_CHECK(MPI_Recv(result.data() + offset + start, chunk, MPI_FLOAT, pe, 71,
                                       MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                start += std::size_t(chunk);
            }
        }
        offset += size;
    }
    return result;
}

int main(int argc, char** argv) {
    Options options;
    try { options = parse_options(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 2;
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        const int mode = options.mode == "scalar-halo" ? 0 : (options.mode == "block-signal" ? 1 :
                         (options.mode == "persistent-signal" ? 2 : (options.mode == "persistent-atomic" ? 3 : 4)));
        std::array<int, 7> configuration{mode, options.nx, options.ny, options.elements,
                                       options.iterations, options.warmup, AISP_NVTX_ENABLED};
        std::array<int, 7> minimum{}, maximum{};
        MPI_CHECK(MPI_Allreduce(configuration.data(), minimum.data(), int(configuration.size()), MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allreduce(configuration.data(), maximum.data(), int(configuration.size()), MPI_INT, MPI_MAX, MPI_COMM_WORLD));
        if (minimum != maximum) throw std::invalid_argument("All PEs must use matching mode and workload arguments");
        const double setup_start = MPI_Wtime();
        DeviceLab lab(options, rank, ranks);
        lab.setup();
        const double setup_ms = max_rank((MPI_Wtime() - setup_start) * 1000);
        const double allocation_ms = max_rank(lab.allocation_ms);
        const double warmup_start = MPI_Wtime();
        { NVTX_RANGE("warmup:nvshmem_device"); lab.run(options.warmup); }
        const double warmup_ms = max_rank((MPI_Wtime() - warmup_start) * 1000);
        const double reset_start = MPI_Wtime();
        lab.reset();
        const double reset_ms = max_rank((MPI_Wtime() - reset_start) * 1000);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double start = MPI_Wtime();
        { NVTX_RANGE("compute_graph:nvshmem_device_steady_state"); lab.run(options.iterations); }
        const double steady_ms = max_rank((MPI_Wtime() - start) * 1000);
        const double verify_start = MPI_Wtime();
        int correct = 1;
        double max_error = 0;
        std::vector<float> local;
        std::vector<float> gathered;
        {
            NVTX_RANGE("verify:nvshmem_device_complete_output");
            if (mesh_mode(options.mode)) {
                local.resize(lab.initial.size());
                CUDA_CHECK(cudaMemcpy(local.data(), lab.fields[options.iterations % 2],
                                       local.size() * sizeof(float), cudaMemcpyDeviceToHost));
                const auto expected = jacobi::reference_jacobi(options.nx, options.ny, options.iterations);
                for (std::size_t i = 0; i < local.size(); ++i) {
                    const auto global_i = std::size_t(lab.domain.global_start - 1) * options.nx + i;
                    const double error = std::fabs(double(local[i]) - expected[global_i]);
                    if (!std::isfinite(local[i]) || !std::isfinite(expected[global_i]) || error > 2e-6) correct = 0;
                    if (std::isfinite(error)) max_error = std::max(max_error, error);
                }
                std::vector<float> owned(local.begin() + options.nx,
                    local.begin() + (std::size_t(lab.domain.rows) + 1) * options.nx);
                const auto interior = gather_rank_data(owned, rank, ranks);
                if (rank == 0) {
                    gathered.resize(checked_product(std::size_t(options.ny), std::size_t(options.nx)));
                    std::copy(interior.begin(), interior.end(), gathered.begin() + options.nx);
                    std::copy_n(gathered.data() + std::size_t(options.ny - 2) * options.nx, options.nx, gathered.data());
                    std::copy_n(gathered.data() + options.nx, options.nx,
                                gathered.data() + std::size_t(options.ny - 1) * options.nx);
                }
            } else {
                const auto count = checked_product(std::size_t(options.iterations), 2 * std::size_t(options.elements));
                local.resize(count);
                CUDA_CHECK(cudaMemcpy(local.data(), lab.history, count * sizeof(float), cudaMemcpyDeviceToHost));
                for (int step = 0; step < options.iterations; ++step) {
                    for (int edge = 0; edge < 2; ++edge) {
                        const int source = edge == 0 ? lab.previous : lab.next;
                        const int source_edge = 1 - edge;
                        for (std::size_t i = 0; i < std::size_t(options.elements); ++i) {
                            const auto at = (std::size_t(step) * 2 + edge) * options.elements + i;
                            const float expected = payload_value(step + 1, source, ranks, source_edge, i,
                                                                 std::size_t(options.elements));
                            const double error = std::fabs(double(local[at]) - expected);
                            if (!std::isfinite(local[at]) || local[at] != expected) correct = 0;
                            if (std::isfinite(error)) max_error = std::max(max_error, error);
                        }
                    }
                }
                gathered = gather_rank_data(local, rank, ranks);
            }
            if (lab.ready) {
                uint64_t ready[2]{}, consumed[2]{};
                CUDA_CHECK(cudaMemcpy(ready, lab.ready, sizeof(ready), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(consumed, lab.consumed, sizeof(consumed), cudaMemcpyDeviceToHost));
                const uint64_t expected_consumed = options.mode == "persistent-halo" ?
                    uint64_t(options.iterations - 1) : uint64_t(options.iterations);
                for (int edge = 0; edge < 2; ++edge) {
                    if (ready[edge] != uint64_t(options.iterations) || consumed[edge] != expected_consumed) correct = 0;
                }
            } else if (lab.atomic_ready) {
                int ready[2]{}, consumed[2]{};
                CUDA_CHECK(cudaMemcpy(ready, lab.atomic_ready, sizeof(ready), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(consumed, lab.atomic_consumed, sizeof(consumed), cudaMemcpyDeviceToHost));
                for (int edge = 0; edge < 2; ++edge) {
                    if (ready[edge] != options.iterations || consumed[edge] != options.iterations) correct = 0;
                }
            }
        }
        int all_correct = 0;
        MPI_CHECK(MPI_Allreduce(&correct, &all_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        max_error = max_rank(max_error);
        const double verification_ms = max_rank((MPI_Wtime() - verify_start) * 1000);
        if (rank == 0 && !options.dump_output.empty()) {
            std::ofstream output(options.dump_output, std::ios::binary);
            if (!output) throw std::runtime_error("Cannot open float-output path");
            const auto bytes = checked_product(gathered.size(), sizeof(float));
            if (bytes > std::size_t(std::numeric_limits<std::streamsize>::max())) {
                throw std::invalid_argument("Float-output file exceeds stream write size");
            }
            output.write(reinterpret_cast<const char*>(gathered.data()), std::streamsize(bytes));
            if (!output) throw std::runtime_error("Cannot write complete float output");
        }
        char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING]{};
        int mpi_version_length = 0;
        MPI_CHECK(MPI_Get_library_version(mpi_version, &mpi_version_length));
        const double cleanup_start = MPI_Wtime();
        lab.cleanup();
        const double cleanup_ms = max_rank((MPI_Wtime() - cleanup_start) * 1000);
        if (rank == 0) {
            std::ostringstream json;
            json << std::setprecision(12) << "{\"schema\":\"nvshmem_device.v1\",\"canonical\":false,\"status\":"
                 << json_string(all_correct ? "VERIFIED" : "FAILED")
                 << ",\"mode\":" << json_string(options.mode) << ",\"ranks\":" << ranks
                 << ",\"nx\":" << options.nx << ",\"ny\":" << options.ny
                 << ",\"elements\":" << options.elements << ",\"iterations\":" << options.iterations
                 << ",\"warmup\":" << options.warmup << ",\"setup_ms\":" << setup_ms
                 << ",\"allocation_ms\":" << allocation_ms << ",\"warmup_ms\":" << warmup_ms
                 << ",\"reset_ms\":" << reset_ms << ",\"steady_ms\":" << steady_ms
                 << ",\"time_per_iteration_ms\":" << steady_ms / options.iterations
                 << ",\"verification_ms\":" << verification_ms << ",\"cleanup_ms\":" << cleanup_ms
                 << ",\"max_abs_error\":" << max_error << ",\"verified\":" << (all_correct ? "true" : "false")
                 << ",\"verification\":\"all_elements_all_ranks_nonfinite_rejected\""
                 << ",\"output_dtype\":\"float32\",\"output_layout\":"
                 << json_string(mesh_mode(options.mode) ? "ny,nx" : "rank,epoch,incoming_edge,element")
                 << ",\"output_elements\":" << gathered.size()
                 << ",\"checked_output_elements\":" << gathered.size()
                 << ",\"collective_launch_blocks\":" << (options.mode == "scalar-halo" ? 0 : 1)
                 << ",\"expected_ready_epoch\":" << (options.mode == "scalar-halo" ? 0 : options.iterations)
                 << ",\"expected_consumed_epoch\":" << (options.mode == "scalar-halo" ? 0 :
                     (options.mode == "persistent-halo" ? options.iterations - 1 : options.iterations))
                 << ",\"collective_query_capacity_rank0\":" << lab.query_capacity
                 << ",\"threads_per_block\":256,\"nvtx_enabled\":" << (AISP_NVTX_ENABLED ? "true" : "false")
                 << ",\"cuda_runtime_version\":" << lab.cuda_runtime << ",\"cuda_driver_version\":" << lab.cuda_driver
                 << ",\"nvshmem_version\":" << json_string(std::to_string(lab.nvshmem_major) + "." +
                     std::to_string(lab.nvshmem_minor) + "." + std::to_string(lab.nvshmem_patch))
                 << ",\"openshmem_api_version\":" << json_string(std::to_string(lab.openshmem_major) + "." +
                     std::to_string(lab.openshmem_minor))
                 << ",\"nvshmem_header_version\":" << json_string(std::to_string(NVSHMEM_VENDOR_MAJOR_VERSION) + "." +
                     std::to_string(NVSHMEM_VENDOR_MINOR_VERSION) + "." + std::to_string(NVSHMEM_VENDOR_PATCH_VERSION))
                 << ",\"mpi_library_version\":" << json_string(std::string(mpi_version, std::size_t(mpi_version_length)))
                 << ",\"device_rank0\":" << json_string(lab.device_name) << "}";
            std::cout << json.str() << '\n';
            if (!options.json_out.empty()) {
                std::ofstream output(options.json_out);
                if (!output) throw std::runtime_error("Cannot open JSON output path");
                output << json.str() << '\n';
                if (!output) throw std::runtime_error("Cannot write JSON output");
            }
        }
        MPI_CHECK(MPI_Finalize());
        return all_correct ? 0 : 1;
    } catch (const Unsupported& error) {
        if (rank == 0) {
            std::cerr << "SKIPPED: " << error.what() << '\n';
            const std::string json = "{\"schema\":\"nvshmem_device.v1\",\"canonical\":false,"
                "\"status\":\"SKIPPED\",\"verified\":false,\"mode\":" + json_string(options.mode) +
                ",\"reason\":" + json_string(error.what()) + "}";
            std::cout << json << '\n';
            if (!options.json_out.empty()) {
                std::ofstream output(options.json_out);
                output << json << '\n';
                if (!output) std::cerr << "ERROR: Cannot write unsupported-capability JSON\n";
            }
        }
        MPI_Finalize();
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "ERROR rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
