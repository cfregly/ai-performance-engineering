// Four-arm NCCL allocation and registration study with complete output checks.
#include "../core/common/nvtx_utils.cuh"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <mpi.h>
#include <nccl.h>
#include <unistd.h>

#ifndef NCCL_VERSION_CODE
#error This benchmark requires NCCL headers that define NCCL_VERSION_CODE
#endif
#if NCCL_VERSION_CODE < 22703
#error This benchmark requires NCCL headers version 2.27.3 or newer
#endif
#ifndef NCCL_WIN_COLL_SYMMETRIC
#error This benchmark requires NCCL symmetric window registration support
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kMinimumNcclVersion = 22703;
constexpr double kRelativeTolerance = 0.0;
constexpr double kAbsoluteTolerance = 0.0;

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

void nccl_check(ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}
#define NCCL_CHECK(call) nccl_check((call), #call)

enum class Arm {
    kCudaMalloc = 0,
    kNcclMemAlloc = 1,
    kCommRegister = 2,
    kWindowRegister = 3,
};

std::string arm_name(Arm arm) {
    switch (arm) {
        case Arm::kCudaMalloc: return "cuda_malloc";
        case Arm::kNcclMemAlloc: return "nccl_mem_alloc";
        case Arm::kCommRegister: return "comm_register";
        case Arm::kWindowRegister: return "window_register";
    }
    throw std::logic_error("Unknown benchmark arm");
}

Arm parse_arm(const std::string& value) {
    if (value == "cuda_malloc") return Arm::kCudaMalloc;
    if (value == "nccl_mem_alloc") return Arm::kNcclMemAlloc;
    if (value == "comm_register") return Arm::kCommRegister;
    if (value == "window_register") return Arm::kWindowRegister;
    throw std::invalid_argument(
        "arm must be cuda_malloc, nccl_mem_alloc, comm_register, or window_register"
    );
}

struct Options {
    Arm arm = Arm::kWindowRegister;
    std::size_t elements = 16U * 1024U * 1024U;
    int iterations = 20;
    int warmup = 5;
    bool allow_profiler_preload = false;
    std::string json_out;
    std::string dump_output;
    std::string expected_nccl_library;
};

int parse_nonnegative_int(const std::string& text, const std::string& name) {
    std::size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size() || value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(name + " must be a nonnegative 32-bit integer");
    }
    return static_cast<int>(value);
}

std::size_t parse_elements(const std::string& text) {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("elements must be positive and fit the address space");
    }
    return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--allow-profiler-preload") {
            if (result.allow_profiler_preload) {
                throw std::invalid_argument("--allow-profiler-preload may appear only once");
            }
            result.allow_profiler_preload = true;
            continue;
        }
        if (index + 1 == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[++index];
        if (key == "--arm") result.arm = parse_arm(value);
        else if (key == "--elements") result.elements = parse_elements(value);
        else if (key == "--iterations") result.iterations = parse_nonnegative_int(value, "iterations");
        else if (key == "--warmup") result.warmup = parse_nonnegative_int(value, "warmup");
        else if (key == "--json-out") result.json_out = value;
        else if (key == "--dump-output") result.dump_output = value;
        else if (key == "--expected-nccl-library") result.expected_nccl_library = value;
        else throw std::invalid_argument("Unknown option " + key);
    }
    if (result.iterations < 1) throw std::invalid_argument("iterations must be at least one");
    if (result.json_out.empty()) throw std::invalid_argument("--json-out is required");
    if (result.dump_output.empty()) throw std::invalid_argument("--dump-output is required");
    if (result.expected_nccl_library.empty()) {
        throw std::invalid_argument(
            "NCCL runtime binding: --expected-nccl-library is required"
        );
    }
    return result;
}

std::string json_quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') out << '\\' << character;
        else if (character < 32) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(character)
                << std::dec << std::setfill(' ');
        } else {
            out << character;
        }
    }
    out << '"';
    return out.str();
}

double maximum_rank_time(double local) {
    double maximum = 0.0;
    MPI_CHECK(MPI_Allreduce(&local, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    return maximum;
}

void require_collective_nccl_success(ncclResult_t status, const std::string& operation) {
    const int local_success = status == ncclSuccess ? 1 : 0;
    int all_success = 0;
    MPI_CHECK(MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (!all_success) {
        const std::string local_reason = status == ncclSuccess
            ? "failed on at least one rank"
            : ncclGetErrorString(status);
        throw std::runtime_error(operation + " failed: " + local_reason);
    }
}

void require_collective_cuda_success(cudaError_t status, const std::string& operation) {
    const int local_success = status == cudaSuccess ? 1 : 0;
    int all_success = 0;
    MPI_CHECK(MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (!all_success) {
        const std::string local_reason = status == cudaSuccess
            ? "failed on at least one rank"
            : cudaGetErrorString(status);
        throw std::runtime_error(operation + " failed: " + local_reason);
    }
}

int require_collective_handle(
    const void* handle, int ranks, const std::string& operation
) {
    const int local_handle = handle == nullptr ? 0 : 1;
    int handle_count = 0;
    MPI_CHECK(MPI_Allreduce(
        &local_handle, &handle_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD
    ));
    if (handle_count != ranks) {
        throw std::runtime_error(
            operation + " returned no registration handle on at least one rank"
        );
    }
    return handle_count;
}

struct NcclRuntimeBinding {
    std::string canonical_path;
    int verified_ranks = 0;
    int runtime_version = 0;
};

NcclRuntimeBinding verify_nccl_runtime_binding(
    const std::string& expected_path, bool allow_profiler_preload, int rank, int ranks
) {
    try {
        int root_length = rank == 0 ? static_cast<int>(expected_path.size()) : 0;
        MPI_CHECK(MPI_Bcast(&root_length, 1, MPI_INT, 0, MPI_COMM_WORLD));
        if (root_length < 1) {
            throw std::runtime_error("rank zero supplied an empty expected library path");
        }
        std::string root_expected(static_cast<std::size_t>(root_length), '\0');
        if (rank == 0) root_expected = expected_path;
        MPI_CHECK(MPI_Bcast(root_expected.data(), root_length, MPI_CHAR, 0, MPI_COMM_WORLD));

        std::string local_error;
        std::string canonical_loaded_path;
        decltype(&ncclGetVersion) resolved_get_version = nullptr;
        try {
            const char* preload = std::getenv("LD_PRELOAD");
            if (
                preload != nullptr && preload[0] != '\0' &&
                !allow_profiler_preload
            ) {
                throw std::runtime_error(
                    "LD_PRELOAD is allowed only in a runner-owned profiler target"
                );
            }
            if (expected_path != root_expected) {
                throw std::runtime_error(
                    "every rank must receive the same --expected-nccl-library path"
                );
            }
            if (expected_path.empty() || expected_path.front() != '/') {
                throw std::runtime_error(
                    "--expected-nccl-library must be an absolute canonical path"
                );
            }

            const auto canonicalize = [](const std::string& path) {
                errno = 0;
                std::unique_ptr<char, decltype(&std::free)> resolved(
                    ::realpath(path.c_str(), nullptr), &std::free
                );
                if (resolved == nullptr) {
                    throw std::runtime_error(
                        "realpath(" + path + ") failed: " + std::strerror(errno)
                    );
                }
                return std::string(resolved.get());
            };

            const std::string canonical_expected_path = canonicalize(expected_path);
            if (canonical_expected_path != expected_path) {
                throw std::runtime_error(
                    "--expected-nccl-library must equal its absolute realpath"
                );
            }

            constexpr std::array<const char*, 12> required_symbols{
                "ncclGetVersion",
                "ncclGetErrorString",
                "ncclGetUniqueId",
                "ncclCommInitRank",
                "ncclMemAlloc",
                "ncclMemFree",
                "ncclCommRegister",
                "ncclCommDeregister",
                "ncclCommWindowRegister",
                "ncclCommWindowDeregister",
                "ncclAllReduce",
                "ncclCommDestroy",
            };
            for (const char* symbol_name : required_symbols) {
                ::dlerror();
                void* resolved_symbol = ::dlsym(RTLD_DEFAULT, symbol_name);
                const char* lookup_error = ::dlerror();
                if (lookup_error != nullptr || resolved_symbol == nullptr) {
                    throw std::runtime_error(
                        "dlsym could not resolve " + std::string(symbol_name) + ": " +
                        std::string(lookup_error == nullptr ? "symbol not found" : lookup_error)
                    );
                }

                Dl_info symbol_info{};
                if (
                    ::dladdr(resolved_symbol, &symbol_info) == 0 ||
                    symbol_info.dli_fname == nullptr || symbol_info.dli_fname[0] == '\0'
                ) {
                    throw std::runtime_error(
                        "dladdr could not identify resolved " + std::string(symbol_name)
                    );
                }
                const std::string symbol_path = canonicalize(symbol_info.dli_fname);
                if (symbol_path != expected_path) {
                    throw std::runtime_error(
                        "resolved NCCL symbol " + std::string(symbol_name) + " from " +
                        symbol_path + ", expected " + expected_path
                    );
                }
                if (canonical_loaded_path.empty()) canonical_loaded_path = symbol_path;
                if (std::strcmp(symbol_name, "ncclGetVersion") == 0) {
                    resolved_get_version = reinterpret_cast<decltype(resolved_get_version)>(
                        resolved_symbol
                    );
                }
            }
        } catch (const std::exception& error) {
            local_error = error.what();
        }

        int local_runtime_version = -1;
        if (resolved_get_version != nullptr) {
            const ncclResult_t version_status = resolved_get_version(&local_runtime_version);
            if (version_status != ncclSuccess && local_error.empty()) {
                local_error = std::string("ncclGetVersion failed: ") +
                    ncclGetErrorString(version_status);
            }
        }
        int minimum_runtime_version = 0;
        int maximum_runtime_version = 0;
        MPI_CHECK(MPI_Allreduce(
            &local_runtime_version, &minimum_runtime_version, 1, MPI_INT, MPI_MIN,
            MPI_COMM_WORLD
        ));
        MPI_CHECK(MPI_Allreduce(
            &local_runtime_version, &maximum_runtime_version, 1, MPI_INT, MPI_MAX,
            MPI_COMM_WORLD
        ));
        if (minimum_runtime_version != maximum_runtime_version && local_error.empty()) {
            local_error =
                "ranks loaded heterogeneous NCCL runtime versions " +
                std::to_string(minimum_runtime_version) + " and " +
                std::to_string(maximum_runtime_version);
        } else if (
            minimum_runtime_version != NCCL_VERSION_CODE && local_error.empty()
        ) {
            local_error =
                "NCCL runtime version " + std::to_string(minimum_runtime_version) +
                " does not match header version " + std::to_string(NCCL_VERSION_CODE);
        }

        const int local_verified = local_error.empty() ? 1 : 0;
        int verified_ranks = 0;
        MPI_CHECK(MPI_Allreduce(
            &local_verified, &verified_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD
        ));
        if (verified_ranks != ranks) {
            if (local_error.empty()) {
                local_error = "runtime library verification failed on at least one rank";
            }
            throw std::runtime_error(local_error);
        }
        return NcclRuntimeBinding{
            canonical_loaded_path, verified_ranks, local_runtime_version
        };
    } catch (const std::exception& error) {
        throw std::runtime_error("NCCL runtime binding: " + std::string(error.what()));
    }
}

struct Statistics {
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double stddev = 0.0;
};

Statistics summarize(const std::vector<double>& samples) {
    if (samples.empty()) throw std::invalid_argument("At least one timing sample is required");
    Statistics result;
    std::vector<double> ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    result.minimum = ordered.front();
    result.maximum = ordered.back();
    result.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const std::size_t middle = ordered.size() / 2;
    result.median = ordered.size() % 2 == 0
        ? 0.5 * (ordered[middle - 1] + ordered[middle])
        : ordered[middle];
    const std::size_t p95_index = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(0.95 * ordered.size()))
    ) - 1;
    result.p95 = ordered[p95_index];
    double squared_error = 0.0;
    for (const double sample : samples) {
        const double error = sample - result.mean;
        squared_error += error * error;
    }
    result.stddev = std::sqrt(squared_error / samples.size());
    return result;
}

std::uint32_t rank_index_pattern(int rank, std::size_t index) {
    const std::uint64_t rank_key =
        0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(rank + 1);
    std::uint64_t mixed = static_cast<std::uint64_t>(index) + rank_key;
    mixed = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
    mixed ^= mixed >> 31U;
    return 1U + static_cast<std::uint32_t>(mixed & 0xffU);
}

float input_value(int rank, std::size_t index) {
    return static_cast<float>(rank_index_pattern(rank, index));
}

float expected_value(int ranks, std::size_t index) {
    std::uint64_t sum = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        sum += rank_index_pattern(rank, index);
    }
    return static_cast<float>(sum);
}

struct NativeBenchmark {
    Options options;
    int rank = 0;
    int ranks = 0;
    int local_rank = 0;
    int local_size = 0;
    int device = -1;
    int device_count = 0;
    int nccl_runtime_version = 0;
    float* send_buffer = nullptr;
    float* receive_buffer = nullptr;
    void* send_registration = nullptr;
    void* receive_registration = nullptr;
    ncclWindow_t send_window = nullptr;
    ncclWindow_t receive_window = nullptr;
    ncclComm_t communicator = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t sample_start = nullptr;
    cudaEvent_t sample_stop = nullptr;
    double topology_gate_ms = 0.0;
    double communicator_ms = 0.0;
    double allocation_ms = 0.0;
    double registration_ms = 0.0;
    double initialization_ms = 0.0;
    int send_registration_handle_verified_ranks = 0;
    int receive_registration_handle_verified_ranks = 0;

    NativeBenchmark(Options config, int world_rank, int world_size, int runtime_version)
        : options(std::move(config)), rank(world_rank), ranks(world_size),
          nccl_runtime_version(runtime_version) {}

    // Failure unwinding must reach MPI_Abort without collective NCCL teardown.
    ~NativeBenchmark() = default;

    void require_single_node_unique_p2p_mapping() {
        const double started = MPI_Wtime();
        if (ranks < 2) throw Unsupported("This benchmark requires at least two MPI ranks");

        MPI_Comm local = MPI_COMM_NULL;
        MPI_CHECK(MPI_Comm_split_type(
            MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local
        ));
        MPI_CHECK(MPI_Comm_rank(local, &local_rank));
        MPI_CHECK(MPI_Comm_size(local, &local_size));
        if (local_size != ranks) {
            MPI_Comm_free(&local);
            throw Unsupported("This benchmark supports one shared-memory node only");
        }

        const char* visibility_value = std::getenv("CUDA_VISIBLE_DEVICES");
        const std::string visibility = visibility_value == nullptr ? "<unset>" : visibility_value;
        int root_length = rank == 0 ? static_cast<int>(visibility.size()) : 0;
        MPI_CHECK(MPI_Bcast(&root_length, 1, MPI_INT, 0, MPI_COMM_WORLD));
        std::string root_visibility(static_cast<std::size_t>(root_length), '\0');
        if (rank == 0) root_visibility = visibility;
        MPI_CHECK(MPI_Bcast(root_visibility.data(), root_length, MPI_CHAR, 0, MPI_COMM_WORLD));
        const int local_visibility_match = visibility == root_visibility ? 1 : 0;
        int all_visibility_match = 0;
        MPI_CHECK(MPI_Allreduce(
            &local_visibility_match, &all_visibility_match, 1, MPI_INT, MPI_MIN,
            MPI_COMM_WORLD
        ));
        if (!all_visibility_match) {
            MPI_CHECK(MPI_Comm_free(&local));
            throw Unsupported(
                "Every rank must use the same CUDA_VISIBLE_DEVICES ordering for the P2P gate"
            );
        }

        const cudaError_t count_status = cudaGetDeviceCount(&device_count);
        const int local_device_count_valid =
            count_status == cudaSuccess && device_count >= local_size ? 1 : 0;
        int all_device_counts_valid = 0;
        MPI_CHECK(MPI_Allreduce(
            &local_device_count_valid, &all_device_counts_valid, 1, MPI_INT, MPI_MIN,
            MPI_COMM_WORLD
        ));
        if (!all_device_counts_valid) {
            MPI_CHECK(MPI_Comm_free(&local));
            throw Unsupported("Every local rank needs a distinct visible CUDA device");
        }
        int minimum_device_count = 0;
        int maximum_device_count = 0;
        MPI_CHECK(MPI_Allreduce(
            &device_count, &minimum_device_count, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
        ));
        MPI_CHECK(MPI_Allreduce(
            &device_count, &maximum_device_count, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD
        ));
        if (minimum_device_count != maximum_device_count) {
            MPI_CHECK(MPI_Comm_free(&local));
            throw Unsupported("Every rank must see the same CUDA device set");
        }

        device = local_rank;
        require_collective_cuda_success(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp properties{};
        require_collective_cuda_success(
            cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties"
        );
        std::vector<unsigned char> uuids(
            static_cast<std::size_t>(local_size) * sizeof(properties.uuid)
        );
        MPI_CHECK(MPI_Allgather(
            &properties.uuid, static_cast<int>(sizeof(properties.uuid)), MPI_BYTE,
            uuids.data(), static_cast<int>(sizeof(properties.uuid)), MPI_BYTE, local
        ));
        const auto local_uuid = uuids.begin() + local_rank * sizeof(properties.uuid);
        int local_unique = 1;
        for (int peer = 0; peer < local_size; ++peer) {
            if (peer == local_rank) continue;
            const auto peer_uuid = uuids.begin() + peer * sizeof(properties.uuid);
            if (std::equal(peer_uuid, peer_uuid + sizeof(properties.uuid), local_uuid)) {
                local_unique = 0;
            }
        }
        int all_unique = 0;
        MPI_CHECK(MPI_Allreduce(
            &local_unique, &all_unique, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
        ));
        if (!all_unique) {
            MPI_CHECK(MPI_Comm_free(&local));
            throw Unsupported("Two local ranks selected the same physical GPU");
        }

        int local_p2p = 1;
        cudaError_t p2p_status = cudaSuccess;
        for (int peer = 0; peer < local_size; ++peer) {
            if (peer == local_rank) continue;
            int can_access = 0;
            const cudaError_t status = cudaDeviceCanAccessPeer(&can_access, device, peer);
            if (p2p_status == cudaSuccess && status != cudaSuccess) p2p_status = status;
            local_p2p = std::min(local_p2p, can_access);
        }
        require_collective_cuda_success(p2p_status, "cudaDeviceCanAccessPeer");
        int all_p2p = 0;
        MPI_CHECK(MPI_Allreduce(&local_p2p, &all_p2p, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Comm_free(&local));
        if (!all_p2p) {
            throw Unsupported("Every selected GPU pair must support directed CUDA peer access");
        }
        topology_gate_ms = (MPI_Wtime() - started) * 1000.0;
    }

    void initialize_communicator() {
        const double started = MPI_Wtime();
        ncclUniqueId identifier{};
        if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&identifier));
        MPI_CHECK(MPI_Bcast(&identifier, sizeof(identifier), MPI_BYTE, 0, MPI_COMM_WORLD));
        NCCL_CHECK(ncclCommInitRank(&communicator, ranks, identifier, rank));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&sample_start));
        CUDA_CHECK(cudaEventCreate(&sample_stop));
        communicator_ms = (MPI_Wtime() - started) * 1000.0;
    }

    void allocate_buffers() {
        const std::size_t bytes = options.elements * sizeof(float);
        const double allocation_started = MPI_Wtime();
        if (options.arm == Arm::kCudaMalloc) {
            const cudaError_t send_status = cudaMalloc(
                reinterpret_cast<void**>(&send_buffer), bytes
            );
            const cudaError_t receive_status = cudaMalloc(
                reinterpret_cast<void**>(&receive_buffer), bytes
            );
            require_collective_cuda_success(send_status, "cudaMalloc(send)");
            require_collective_cuda_success(receive_status, "cudaMalloc(receive)");
        } else {
            const ncclResult_t send_status = ncclMemAlloc(
                reinterpret_cast<void**>(&send_buffer), bytes
            );
            const ncclResult_t receive_status = ncclMemAlloc(
                reinterpret_cast<void**>(&receive_buffer), bytes
            );
            require_collective_nccl_success(send_status, "ncclMemAlloc(send)");
            require_collective_nccl_success(receive_status, "ncclMemAlloc(receive)");
        }
        allocation_ms = (MPI_Wtime() - allocation_started) * 1000.0;

        const double registration_started = MPI_Wtime();
        if (options.arm == Arm::kCommRegister) {
            const ncclResult_t send_status = ncclCommRegister(
                communicator, send_buffer, bytes, &send_registration
            );
            const ncclResult_t receive_status = ncclCommRegister(
                communicator, receive_buffer, bytes, &receive_registration
            );
            require_collective_nccl_success(send_status, "ncclCommRegister(send)");
            require_collective_nccl_success(receive_status, "ncclCommRegister(receive)");
            send_registration_handle_verified_ranks = require_collective_handle(
                send_registration, ranks, "ncclCommRegister(send)"
            );
            receive_registration_handle_verified_ranks = require_collective_handle(
                receive_registration, ranks, "ncclCommRegister(receive)"
            );
        } else if (options.arm == Arm::kWindowRegister) {
            const ncclResult_t send_status = ncclCommWindowRegister(
                communicator, send_buffer, bytes, &send_window, NCCL_WIN_COLL_SYMMETRIC
            );
            if (send_status != ncclSuccess) {
                throw std::runtime_error(
                    "ncclCommWindowRegister(send, NCCL_WIN_COLL_SYMMETRIC) failed: " +
                    std::string(ncclGetErrorString(send_status))
                );
            }
            send_registration_handle_verified_ranks = require_collective_handle(
                send_window, ranks,
                "ncclCommWindowRegister(send, NCCL_WIN_COLL_SYMMETRIC)"
            );
            const ncclResult_t receive_status = ncclCommWindowRegister(
                communicator, receive_buffer, bytes, &receive_window,
                NCCL_WIN_COLL_SYMMETRIC
            );
            if (receive_status != ncclSuccess) {
                throw std::runtime_error(
                    "ncclCommWindowRegister(receive, NCCL_WIN_COLL_SYMMETRIC) failed: " +
                    std::string(ncclGetErrorString(receive_status))
                );
            }
            receive_registration_handle_verified_ranks = require_collective_handle(
                receive_window, ranks,
                "ncclCommWindowRegister(receive, NCCL_WIN_COLL_SYMMETRIC)"
            );
        }
        registration_ms = (MPI_Wtime() - registration_started) * 1000.0;
    }

    void initialize_inputs() {
        const double started = MPI_Wtime();
        std::vector<float> input(options.elements);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = input_value(rank, index);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            send_buffer, input.data(), options.elements * sizeof(float),
            cudaMemcpyHostToDevice, stream
        ));
        CUDA_CHECK(cudaMemsetAsync(receive_buffer, 0, options.elements * sizeof(float), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        initialization_ms = (MPI_Wtime() - started) * 1000.0;
    }

    void setup() {
        NVTX_RANGE("setup:nccl_symmetric_memory");
        require_single_node_unique_p2p_mapping();
        initialize_communicator();
        allocate_buffers();
        initialize_inputs();
    }

    void enqueue_allreduce() {
        NCCL_CHECK(ncclAllReduce(
            send_buffer, receive_buffer, options.elements, ncclFloat32, ncclSum,
            communicator, stream
        ));
    }

    double warmup() {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double started = MPI_Wtime();
        {
            NVTX_RANGE("warmup:nccl_allreduce");
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                enqueue_allreduce();
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
        return maximum_rank_time((MPI_Wtime() - started) * 1000.0);
    }

    void poison_final_timed_destination() {
        CUDA_CHECK(cudaMemsetAsync(
            receive_buffer, 0xFF, options.elements * sizeof(float), stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<double> measure() {
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(options.iterations));
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            if (iteration + 1 == options.iterations) poison_final_timed_destination();
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            {
                NVTX_RANGE("allreduce:nccl_symmetric_memory_steady");
                CUDA_CHECK(cudaEventRecord(sample_start, stream));
                enqueue_allreduce();
                CUDA_CHECK(cudaEventRecord(sample_stop, stream));
                CUDA_CHECK(cudaEventSynchronize(sample_stop));
            }
            float local_ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&local_ms, sample_start, sample_stop));
            samples.push_back(maximum_rank_time(static_cast<double>(local_ms)));
        }
        return samples;
    }

    struct Verification {
        std::vector<float> output;
        std::uint64_t bad_elements = 0;
        double maximum_error = 0.0;
    };

    Verification verify() {
        NVTX_RANGE("verify:complete_allreduce_output");
        Verification result;
        result.output.resize(options.elements);
        CUDA_CHECK(cudaMemcpy(
            result.output.data(), receive_buffer, options.elements * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
        for (std::size_t index = 0; index < result.output.size(); ++index) {
            const double actual = result.output[index];
            const double expected = expected_value(ranks, index);
            const double error = std::fabs(actual - expected);
            const double tolerance = kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected);
            if (!std::isfinite(actual) || error > tolerance) ++result.bad_elements;
            if (std::isfinite(error)) result.maximum_error = std::max(result.maximum_error, error);
        }
        return result;
    }

    void cleanup_resources() {
        if (stream != nullptr) CUDA_CHECK(cudaStreamSynchronize(stream));
        if (options.arm == Arm::kWindowRegister) {
            if (receive_window != nullptr) {
                NCCL_CHECK(ncclCommWindowDeregister(communicator, receive_window));
                receive_window = nullptr;
            }
            if (send_window != nullptr) {
                NCCL_CHECK(ncclCommWindowDeregister(communicator, send_window));
                send_window = nullptr;
            }
        } else if (options.arm == Arm::kCommRegister) {
            if (receive_registration != nullptr) {
                NCCL_CHECK(ncclCommDeregister(communicator, receive_registration));
                receive_registration = nullptr;
            }
            if (send_registration != nullptr) {
                NCCL_CHECK(ncclCommDeregister(communicator, send_registration));
                send_registration = nullptr;
            }
        }
        if (communicator != nullptr) {
            NCCL_CHECK(ncclCommDestroy(communicator));
            communicator = nullptr;
        }
        if (options.arm == Arm::kCudaMalloc) {
            if (receive_buffer != nullptr) CUDA_CHECK(cudaFree(receive_buffer));
            if (send_buffer != nullptr) CUDA_CHECK(cudaFree(send_buffer));
        } else {
            if (receive_buffer != nullptr) NCCL_CHECK(ncclMemFree(receive_buffer));
            if (send_buffer != nullptr) NCCL_CHECK(ncclMemFree(send_buffer));
        }
        receive_buffer = nullptr;
        send_buffer = nullptr;
        if (sample_stop != nullptr) {
            CUDA_CHECK(cudaEventDestroy(sample_stop));
            sample_stop = nullptr;
        }
        if (sample_start != nullptr) {
            CUDA_CHECK(cudaEventDestroy(sample_start));
            sample_start = nullptr;
        }
        if (stream != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }

    void cleanup() {
        NVTX_RANGE("cleanup:nccl_symmetric_memory");
        cleanup_resources();
    }
};

void write_exclusive_output(
    const std::string& path, const void* data, std::size_t bytes, const std::string& label
) {
    const int descriptor = ::open(
        path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600
    );
    if (descriptor < 0) {
        throw std::runtime_error(
            "Cannot create " + label + " exclusively at " + path + ": " +
            std::strerror(errno)
        );
    }

    const auto* cursor = static_cast<const unsigned char*>(data);
    std::size_t written = 0;
    while (written < bytes) {
        const ssize_t count = ::write(descriptor, cursor + written, bytes - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            const int write_error = count == 0 ? EIO : errno;
            ::close(descriptor);
            throw std::runtime_error(
                "Cannot write " + label + " at " + path + ": " +
                std::strerror(write_error)
            );
        }
        written += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0) {
        const int sync_error = errno;
        ::close(descriptor);
        throw std::runtime_error(
            "Cannot sync " + label + " at " + path + ": " + std::strerror(sync_error)
        );
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error(
            "Cannot close " + label + " at " + path + ": " + std::strerror(errno)
        );
    }
}

void write_float_output(const std::string& path, const std::vector<float>& output) {
    write_exclusive_output(
        path, output.data(), output.size() * sizeof(float), "complete-output artifact"
    );
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        const Options options = parse_options(argc, argv);
        const std::array<unsigned long long, 4> local_configuration{
            static_cast<unsigned long long>(options.arm),
            static_cast<unsigned long long>(options.elements),
            static_cast<unsigned long long>(options.iterations),
            static_cast<unsigned long long>(options.warmup),
        };
        std::array<unsigned long long, 4> minimum_configuration{};
        std::array<unsigned long long, 4> maximum_configuration{};
        MPI_CHECK(MPI_Allreduce(
            local_configuration.data(), minimum_configuration.data(),
            static_cast<int>(local_configuration.size()), MPI_UNSIGNED_LONG_LONG,
            MPI_MIN, MPI_COMM_WORLD
        ));
        MPI_CHECK(MPI_Allreduce(
            local_configuration.data(), maximum_configuration.data(),
            static_cast<int>(local_configuration.size()), MPI_UNSIGNED_LONG_LONG,
            MPI_MAX, MPI_COMM_WORLD
        ));
        if (minimum_configuration != maximum_configuration) {
            throw std::invalid_argument("Every rank must use the same benchmark configuration");
        }

        const NcclRuntimeBinding runtime_binding = verify_nccl_runtime_binding(
            options.expected_nccl_library, options.allow_profiler_preload, rank, ranks
        );

        const double setup_started = MPI_Wtime();
        NativeBenchmark benchmark(
            options, rank, ranks, runtime_binding.runtime_version
        );
        benchmark.setup();
        const double setup_ms = maximum_rank_time((MPI_Wtime() - setup_started) * 1000.0);
        const double topology_gate_ms = maximum_rank_time(benchmark.topology_gate_ms);
        const double communicator_ms = maximum_rank_time(benchmark.communicator_ms);
        const double allocation_ms = maximum_rank_time(benchmark.allocation_ms);
        const double registration_ms = maximum_rank_time(benchmark.registration_ms);
        const double initialization_ms = maximum_rank_time(benchmark.initialization_ms);

        const double warmup_ms = benchmark.warmup();
        const std::vector<double> samples = benchmark.measure();
        const Statistics statistics = summarize(samples);

        const double verification_started = MPI_Wtime();
        NativeBenchmark::Verification verification = benchmark.verify();
        const double verification_ms = maximum_rank_time(
            (MPI_Wtime() - verification_started) * 1000.0
        );
        const int local_correct = verification.bad_elements == 0 ? 1 : 0;
        int all_correct = 0;
        int verified_ranks = 0;
        MPI_CHECK(MPI_Allreduce(
            &local_correct, &all_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD
        ));
        MPI_CHECK(MPI_Allreduce(
            &local_correct, &verified_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD
        ));

        const unsigned long long local_bad = verification.bad_elements;
        std::vector<unsigned long long> bad_by_rank(static_cast<std::size_t>(ranks));
        std::vector<double> error_by_rank(static_cast<std::size_t>(ranks));
        std::vector<int> devices_by_rank(static_cast<std::size_t>(ranks));
        MPI_CHECK(MPI_Gather(
            &local_bad, 1, MPI_UNSIGNED_LONG_LONG, bad_by_rank.data(), 1,
            MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD
        ));
        MPI_CHECK(MPI_Gather(
            &verification.maximum_error, 1, MPI_DOUBLE, error_by_rank.data(), 1,
            MPI_DOUBLE, 0, MPI_COMM_WORLD
        ));
        MPI_CHECK(MPI_Gather(
            &benchmark.device, 1, MPI_INT, devices_by_rank.data(), 1, MPI_INT, 0,
            MPI_COMM_WORLD
        ));
        double maximum_error = 0.0;
        MPI_CHECK(MPI_Allreduce(
            &verification.maximum_error, &maximum_error, 1, MPI_DOUBLE, MPI_MAX,
            MPI_COMM_WORLD
        ));

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double output_started = MPI_Wtime();
        if (rank == 0) write_float_output(options.dump_output, verification.output);
        const double output_write_ms = maximum_rank_time(
            (MPI_Wtime() - output_started) * 1000.0
        );

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double cleanup_started = MPI_Wtime();
        benchmark.cleanup();
        const double cleanup_ms = maximum_rank_time(
            (MPI_Wtime() - cleanup_started) * 1000.0
        );

        if (rank == 0) {
            std::ostringstream json;
            json << std::setprecision(12)
                 << "{\"schema\":\"nccl_symmetric_memory.v1\",\"status\":"
                 << json_quote(all_correct ? "passed" : "failed")
                 << ",\"arm\":" << json_quote(arm_name(options.arm))
                 << ",\"dtype\":\"float32\",\"reduction\":\"sum\""
                 << ",\"out_of_place\":true,\"single_node\":true,\"p2p_gate_passed\":true"
                 << ",\"elements\":" << options.elements
                 << ",\"bytes_per_buffer\":" << options.elements * sizeof(float)
                 << ",\"ranks\":" << ranks << ",\"iterations\":" << options.iterations
                 << ",\"warmup\":" << options.warmup
                 << ",\"nccl_header_version_code\":" << NCCL_VERSION_CODE
                 << ",\"nccl_runtime_version_code\":" << benchmark.nccl_runtime_version
                 << ",\"nccl_runtime_library_path\":"
                 << json_quote(runtime_binding.canonical_path)
                 << ",\"nccl_runtime_library_verified_ranks\":"
                 << runtime_binding.verified_ranks
                 << ",\"profiler_preload_allowed\":"
                 << (options.allow_profiler_preload ? "true" : "false")
                 << ",\"minimum_nccl_version_code\":" << kMinimumNcclVersion
                 << ",\"setup_ms\":" << setup_ms
                 << ",\"topology_gate_ms\":" << topology_gate_ms
                 << ",\"communicator_ms\":" << communicator_ms
                 << ",\"allocation_ms\":" << allocation_ms
                 << ",\"registration_ms\":" << registration_ms
                 << ",\"send_registration_handle_verified_ranks\":"
                 << benchmark.send_registration_handle_verified_ranks
                 << ",\"receive_registration_handle_verified_ranks\":"
                 << benchmark.receive_registration_handle_verified_ranks
                 << ",\"initialization_ms\":" << initialization_ms
                 << ",\"warmup_ms\":" << warmup_ms
                 << ",\"time_per_iteration_ms\":" << statistics.mean
                 << ",\"verification_ms\":" << verification_ms
                 << ",\"output_write_ms\":" << output_write_ms
                 << ",\"cleanup_ms\":" << cleanup_ms
                 << ",\"timing_scope\":\"MPI_MAX CUDA event elapsed around one ncclAllReduce\""
                 << ",\"timing_excludes\":[\"setup\",\"warmup\",\"MPI barrier\",\"MPI_MAX reduction\",\"verification\",\"output write\",\"cleanup\"]"
                 << ",\"samples_ms\":[";
            for (std::size_t index = 0; index < samples.size(); ++index) {
                if (index != 0) json << ',';
                json << samples[index];
            }
            json << "]"
                 << ",\"statistics_ms\":{\"min\":" << statistics.minimum
                 << ",\"max\":" << statistics.maximum
                 << ",\"mean\":" << statistics.mean
                 << ",\"median\":" << statistics.median
                 << ",\"p95\":" << statistics.p95
                 << ",\"stddev\":" << statistics.stddev << '}'
                 << ",\"rtol\":" << kRelativeTolerance
                 << ",\"atol\":" << kAbsoluteTolerance
                 << ",\"verified_ranks\":" << verified_ranks
                 << ",\"checked_elements_per_rank\":" << options.elements
                 << ",\"checked_elements_total\":"
                 << static_cast<unsigned long long>(options.elements) * ranks
                 << ",\"max_abs_error\":" << maximum_error
                 << ",\"rank_verification\":[";
            for (int peer = 0; peer < ranks; ++peer) {
                if (peer != 0) json << ',';
                json << "{\"rank\":" << peer
                     << ",\"checked_elements\":" << options.elements
                     << ",\"bad_elements\":" << bad_by_rank[peer]
                     << ",\"max_abs_error\":" << error_by_rank[peer] << '}';
            }
            json << "]"
                 << ",\"rank_mapping\":[";
            for (int peer = 0; peer < ranks; ++peer) {
                if (peer != 0) json << ',';
                json << "{\"rank\":" << peer
                     << ",\"device_ordinal\":" << devices_by_rank[peer] << '}';
            }
            json << "]}"
                 << '\n';
            const std::string serialized = json.str();
            write_exclusive_output(
                options.json_out, serialized.data(), serialized.size(), "result"
            );
            std::cout << serialized;
            if (all_correct) std::cout << "TIME_MS: " << statistics.mean << '\n';
        }
        MPI_CHECK(MPI_Finalize());
        return all_correct ? 0 : 2;
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
