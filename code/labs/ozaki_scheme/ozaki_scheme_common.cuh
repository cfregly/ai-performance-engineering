#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../core/common/headers/cuda_verify.cuh"
#include "accuracy.h"

namespace ozaki_scheme {

enum class Variant {
    kNative,
    kDynamic,
    kFixed,
};

enum class EmulationStrategy {
    kLibraryDefault,
    kPerformant,
    kEager,
};

enum class ReferenceMode {
    kNativeFp64,
    kCpuLongDouble,
};

enum class InputPattern {
    kUniform,
    kAlternating,
    kDynamicRange,
};

struct Options {
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int warmup = 3;
    int iters = 10;
    int seed = 2026;
    int dynamic_max_bits = 16;
    int dynamic_offset = -56;
    int fixed_bits = 12;
    double input_scale = 0.001;
    double relative_l2_limit = std::numeric_limits<double>::quiet_NaN();
    double normalized_max_abs_limit = std::numeric_limits<double>::quiet_NaN();
    bool accuracy_measure_only = false;
    std::size_t workspace_bytes = 64ull << 20;
    EmulationStrategy emulation_strategy = EmulationStrategy::kEager;
    ReferenceMode reference_mode = ReferenceMode::kNativeFp64;
    InputPattern input_pattern = InputPattern::kUniform;
};

struct Metrics {
    double time_ms = 0.0;
    double tflops = 0.0;
    double checksum = 0.0;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double relative_l2_error = 0.0;
    double normalized_max_abs_error = 0.0;
    int retained_bits = -1;
    int emulation_used = 0;
    int cuda_runtime_version = 0;
    int cublas_version = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    std::string gpu_name;
};

inline const char* variant_name(Variant variant) {
    switch (variant) {
        case Variant::kNative:
            return "native_fp64";
        case Variant::kDynamic:
            return "ozaki_dynamic";
        case Variant::kFixed:
            return "ozaki_fixed";
    }
    return "unknown";
}

inline const char* emulation_strategy_name(EmulationStrategy strategy) {
    switch (strategy) {
        case EmulationStrategy::kLibraryDefault:
            return "default";
        case EmulationStrategy::kPerformant:
            return "performant";
        case EmulationStrategy::kEager:
            return "eager";
    }
    return "unknown";
}

inline const char* reference_mode_name(ReferenceMode mode) {
    switch (mode) {
        case ReferenceMode::kNativeFp64:
            return "native_fp64";
        case ReferenceMode::kCpuLongDouble:
            return "cpu_long_double";
    }
    return "unknown";
}

inline const char* input_pattern_name(InputPattern pattern) {
    switch (pattern) {
        case InputPattern::kUniform:
            return "uniform";
        case InputPattern::kAlternating:
            return "alternating";
        case InputPattern::kDynamicRange:
            return "dynamic_range";
    }
    return "unknown";
}

inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << file << ":" << line << " CUDA call failed for " << expr << ": "
            << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

inline void check_cublas(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << file << ":" << line << " cuBLAS call failed for " << expr
            << ": status=" << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

#define OZAKI_CHECK_CUDA(expr) ::ozaki_scheme::check_cuda((expr), #expr, __FILE__, __LINE__)
#define OZAKI_CHECK_CUBLAS(expr) ::ozaki_scheme::check_cublas((expr), #expr, __FILE__, __LINE__)

template <typename T>
inline void parse_numeric_arg(const char* raw, T* out, const char* name) {
    std::istringstream stream(raw);
    T value{};
    stream >> value;
    if (!stream || !stream.eof()) {
        std::ostringstream oss;
        oss << "Invalid value for " << name << ": " << raw;
        throw std::runtime_error(oss.str());
    }
    *out = value;
}

inline void validate_options(const Options& options) {
    if (options.m <= 0 || options.n <= 0 || options.k <= 0) {
        throw std::runtime_error("M, N, and K must be positive");
    }
    if (options.warmup < 0 || options.iters <= 0) {
        throw std::runtime_error("warmup must be >= 0 and iters must be > 0");
    }
    if (options.dynamic_max_bits <= 0 || options.fixed_bits <= 0) {
        throw std::runtime_error("dynamic_max_bits and fixed_bits must be > 0");
    }
    if (!std::isfinite(options.input_scale) || options.input_scale <= 0.0) {
        throw std::runtime_error("input_scale must be > 0");
    }
}

inline void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --m <int>                    Matrix rows for A/C (default 4096)\n"
        << "  --n <int>                    Matrix cols for B/C (default 4096)\n"
        << "  --k <int>                    Reduction dimension (default 4096)\n"
        << "  --warmup <int>               Warmup matmuls before timing (default 3)\n"
        << "  --iters <int>                Timed matmuls averaged into TIME_MS (default 10)\n"
        << "  --seed <int>                 RNG seed for deterministic inputs (default 2026)\n"
        << "  --input-scale <float>        Uniform input scale (default 0.001)\n"
        << "  --input-pattern <str>        uniform|alternating|dynamic_range (default uniform)\n"
        << "  --reference-mode <str>       native_fp64|cpu_long_double (default native_fp64)\n"
        << "  --dynamic-max-bits <int>     Max retained bits for dynamic Ozaki (default 16)\n"
        << "  --dynamic-offset <int>       Dynamic mantissa bias (default -56)\n"
        << "  --fixed-bits <int>           Retained bits for fixed Ozaki (default 12)\n"
        << "  --emulation-strategy <str>   One of default|performant|eager (default eager)\n"
        << "  --workspace-mb <int>         cuBLAS workspace cap in MiB (default 64)\n"
        << "  --relative-l2-limit <float>  Required reviewed accuracy bound for emulation, [0,1)\n"
        << "  --normalized-max-abs-limit <float> Required reviewed accuracy bound, [0,1)\n"
        << "  --accuracy-measure-only     Collect errors; exit 2, no accepted timing/checksum\n"
        << "  -h, --help                   Show this help text\n";
}

inline EmulationStrategy parse_emulation_strategy(const std::string& raw) {
    if (raw == "default") {
        return EmulationStrategy::kLibraryDefault;
    }
    if (raw == "performant") {
        return EmulationStrategy::kPerformant;
    }
    if (raw == "eager") {
        return EmulationStrategy::kEager;
    }
    throw std::runtime_error(std::string("Invalid value for --emulation-strategy: ") + raw +
        " (expected default|performant|eager)");
}

inline ReferenceMode parse_reference_mode(const std::string& raw) {
    if (raw == "native_fp64") {
        return ReferenceMode::kNativeFp64;
    }
    if (raw == "cpu_long_double") {
        return ReferenceMode::kCpuLongDouble;
    }
    throw std::runtime_error(std::string("Invalid value for --reference-mode: ") + raw +
        " (expected native_fp64|cpu_long_double)");
}

inline InputPattern parse_input_pattern(const std::string& raw) {
    if (raw == "uniform") {
        return InputPattern::kUniform;
    }
    if (raw == "alternating") {
        return InputPattern::kAlternating;
    }
    if (raw == "dynamic_range") {
        return InputPattern::kDynamicRange;
    }
    throw std::runtime_error(std::string("Invalid value for --input-pattern: ") + raw +
        " (expected uniform|alternating|dynamic_range)");
}

inline Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--accuracy-measure-only") {
            options.accuracy_measure_only = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value after " + arg);
        }
        const char* value = argv[++i];
        if (arg == "--m") {
            parse_numeric_arg(value, &options.m, "--m");
        } else if (arg == "--n") {
            parse_numeric_arg(value, &options.n, "--n");
        } else if (arg == "--k") {
            parse_numeric_arg(value, &options.k, "--k");
        } else if (arg == "--warmup") {
            parse_numeric_arg(value, &options.warmup, "--warmup");
        } else if (arg == "--iters") {
            parse_numeric_arg(value, &options.iters, "--iters");
        } else if (arg == "--seed") {
            parse_numeric_arg(value, &options.seed, "--seed");
        } else if (arg == "--dynamic-max-bits") {
            parse_numeric_arg(value, &options.dynamic_max_bits, "--dynamic-max-bits");
        } else if (arg == "--dynamic-offset") {
            parse_numeric_arg(value, &options.dynamic_offset, "--dynamic-offset");
        } else if (arg == "--fixed-bits") {
            parse_numeric_arg(value, &options.fixed_bits, "--fixed-bits");
        } else if (arg == "--emulation-strategy") {
            options.emulation_strategy = parse_emulation_strategy(value);
        } else if (arg == "--reference-mode") {
            options.reference_mode = parse_reference_mode(value);
        } else if (arg == "--input-pattern") {
            options.input_pattern = parse_input_pattern(value);
        } else if (arg == "--input-scale") {
            parse_numeric_arg(value, &options.input_scale, "--input-scale");
        } else if (arg == "--relative-l2-limit") {
            parse_numeric_arg(value, &options.relative_l2_limit, "--relative-l2-limit");
        } else if (arg == "--normalized-max-abs-limit") {
            parse_numeric_arg(value, &options.normalized_max_abs_limit, "--normalized-max-abs-limit");
        } else if (arg == "--workspace-mb") {
            int workspace_mb = 0;
            parse_numeric_arg(value, &workspace_mb, "--workspace-mb");
            if (workspace_mb <= 0) {
                throw std::runtime_error("--workspace-mb must be > 0");
            }
            options.workspace_bytes = static_cast<std::size_t>(workspace_mb) << 20;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    validate_options(options);
    return options;
}

struct HandleState {
    cublasHandle_t handle = nullptr;
    int* retained_bits_device = nullptr;
};

inline void destroy_handle_state(HandleState* state) {
    if (state->retained_bits_device != nullptr) {
        cudaFree(state->retained_bits_device);
        state->retained_bits_device = nullptr;
    }
    if (state->handle != nullptr) {
        cublasDestroy(state->handle);
        state->handle = nullptr;
    }
}

inline void release_resources(
    HandleState* ref_state,
    HandleState* state,
    cudaEvent_t start,
    cudaEvent_t stop,
    void* workspace,
    double* d_ref,
    double* d_c,
    double* d_b,
    double* d_a,
    cudaStream_t stream) {
    destroy_handle_state(ref_state);
    destroy_handle_state(state);
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (stop != nullptr) {
        cudaEventDestroy(stop);
    }
    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    if (d_ref != nullptr) {
        cudaFree(d_ref);
    }
    if (d_c != nullptr) {
        cudaFree(d_c);
    }
    if (d_b != nullptr) {
        cudaFree(d_b);
    }
    if (d_a != nullptr) {
        cudaFree(d_a);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
}

inline HandleState create_handle_state(
    Variant variant,
    const Options& options,
    cudaStream_t stream,
    void* workspace) {
    HandleState state;
    OZAKI_CHECK_CUBLAS(cublasCreate(&state.handle));
    OZAKI_CHECK_CUBLAS(cublasSetStream(state.handle, stream));
    OZAKI_CHECK_CUBLAS(cublasSetWorkspace(state.handle, workspace, options.workspace_bytes));

    if (variant == Variant::kNative) {
        OZAKI_CHECK_CUBLAS(cublasSetMathMode(state.handle, CUBLAS_DEFAULT_MATH));
        return state;
    }

    const cublasMath_t math_mode = CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH;
    const cudaEmulationMantissaControl mantissa_control =
        variant == Variant::kDynamic
            ? CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC
            : CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
    const int max_bits = variant == Variant::kDynamic ? options.dynamic_max_bits : options.fixed_bits;
    const int offset = variant == Variant::kDynamic ? options.dynamic_offset : 0;

    OZAKI_CHECK_CUBLAS(cublasSetMathMode(state.handle, math_mode));
    if (options.emulation_strategy != EmulationStrategy::kLibraryDefault) {
        const cublasEmulationStrategy_t strategy =
            options.emulation_strategy == EmulationStrategy::kPerformant
                ? CUBLAS_EMULATION_STRATEGY_PERFORMANT
                : CUBLAS_EMULATION_STRATEGY_EAGER;
        OZAKI_CHECK_CUBLAS(cublasSetEmulationStrategy(state.handle, strategy));
    }
    OZAKI_CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaControl(state.handle, mantissa_control));
    OZAKI_CHECK_CUBLAS(cublasSetFixedPointEmulationMaxMantissaBitCount(state.handle, max_bits));
    OZAKI_CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaBitOffset(state.handle, offset));

    OZAKI_CHECK_CUDA(cudaMalloc(&state.retained_bits_device, sizeof(int)));
    OZAKI_CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaBitCountPointer(
        state.handle, state.retained_bits_device));
    return state;
}

inline void launch_matmul(
    const HandleState& state,
    Variant variant,
    const Options& options,
    const double* d_a,
    const double* d_b,
    double* d_c,
    cudaStream_t stream) {
    const double alpha = 1.0;
    const double beta = 0.0;
    if (state.retained_bits_device != nullptr) {
        // Keep the sentinel reset on the GEMM stream. A copy from pageable
        // host memory may synchronize while staging even with the Async API.
        // Filling every byte with 0xff produces the same int32 -1 sentinel.
        OZAKI_CHECK_CUDA(cudaMemsetAsync(
            state.retained_bits_device,
            0xff,
            sizeof(int),
            stream));
    }
    const cublasComputeType_t compute_type =
        variant == Variant::kNative ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
    OZAKI_CHECK_CUBLAS(cublasGemmEx(
        state.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        options.n,
        options.m,
        options.k,
        &alpha,
        d_b,
        CUDA_R_64F,
        options.n,
        d_a,
        CUDA_R_64F,
        options.k,
        &beta,
        d_c,
        CUDA_R_64F,
        options.n,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

inline void fill_host_matrix(
    std::vector<double>* data,
    std::size_t rows,
    std::size_t columns,
    int seed,
    double scale,
    InputPattern pattern,
    bool right_operand) {
    if (data->size() != rows * columns) {
        throw std::runtime_error("Input pattern shape does not match allocated matrix");
    }
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
    std::uniform_real_distribution<double> dist(-scale, scale);
    if (pattern == InputPattern::kUniform) {
        for (double& value : *data) {
            value = dist(rng);
        }
        return;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t index = row * columns + column;
            const bool negative = ((row + column + static_cast<std::size_t>(seed) +
                                    (right_operand ? 1u : 0u)) & 1u) != 0;
            double magnitude = scale;
            if (pattern == InputPattern::kAlternating) {
                magnitude *= 1.0 - static_cast<double>((index + static_cast<std::size_t>(seed)) % 17) / 64.0;
            } else {
                const int exponent = -static_cast<int>((index + static_cast<std::size_t>(seed)) % 13);
                magnitude = std::ldexp(scale, exponent);
            }
            (*data)[index] = negative ? -magnitude : magnitude;
        }
    }
}

inline Metrics benchmark_variant(Variant variant, const Options& options) {
    validate_options(options);
    if (variant != Variant::kNative && !options.accuracy_measure_only) {
        validate_accuracy_limits(options.relative_l2_limit, options.normalized_max_abs_limit);
    }

    int device = 0;
    OZAKI_CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp props{};
    OZAKI_CHECK_CUDA(cudaGetDeviceProperties(&props, device));
    if (props.major < 10) {
        throw std::runtime_error("Ozaki lab requires Blackwell-class hardware (SM100+)");
    }

    const std::size_t a_elements = static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_elements = static_cast<std::size_t>(options.k) * options.n;
    const std::size_t c_elements = static_cast<std::size_t>(options.m) * options.n;
    const std::size_t a_bytes = a_elements * sizeof(double);
    const std::size_t b_bytes = b_elements * sizeof(double);
    const std::size_t c_bytes = c_elements * sizeof(double);

    std::vector<double> h_a(a_elements);
    std::vector<double> h_b(b_elements);
    fill_host_matrix(&h_a, options.m, options.k, options.seed, options.input_scale,
                     options.input_pattern, false);
    fill_host_matrix(&h_b, options.k, options.n, options.seed + 17, options.input_scale,
                     options.input_pattern, true);

    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;
    double* d_ref = nullptr;
    void* workspace = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaStream_t stream = nullptr;
    HandleState state;
    HandleState ref_state;

    try {
        OZAKI_CHECK_CUDA(cudaStreamCreate(&stream));
        OZAKI_CHECK_CUDA(cudaMalloc(&d_a, a_bytes));
        OZAKI_CHECK_CUDA(cudaMalloc(&d_b, b_bytes));
        OZAKI_CHECK_CUDA(cudaMalloc(&d_c, c_bytes));
        OZAKI_CHECK_CUDA(cudaMemcpyAsync(d_a, h_a.data(), a_bytes, cudaMemcpyHostToDevice, stream));
        OZAKI_CHECK_CUDA(cudaMemcpyAsync(d_b, h_b.data(), b_bytes, cudaMemcpyHostToDevice, stream));
        OZAKI_CHECK_CUDA(cudaMalloc(&workspace, options.workspace_bytes));
        OZAKI_CHECK_CUDA(cudaEventCreate(&start));
        OZAKI_CHECK_CUDA(cudaEventCreate(&stop));

        state = create_handle_state(variant, options, stream, workspace);

        if (variant != Variant::kNative && options.reference_mode == ReferenceMode::kNativeFp64) {
            OZAKI_CHECK_CUDA(cudaMalloc(&d_ref, c_bytes));
            ref_state = create_handle_state(Variant::kNative, options, stream, workspace);
            launch_matmul(
                ref_state,
                Variant::kNative,
                options,
                d_a,
                d_b,
                d_ref,
                stream);
        }

        OZAKI_CHECK_CUDA(cudaStreamSynchronize(stream));

        for (int i = 0; i < options.warmup; ++i) {
            launch_matmul(state, variant, options, d_a, d_b, d_c, stream);
        }
        OZAKI_CHECK_CUDA(cudaStreamSynchronize(stream));

        OZAKI_CHECK_CUDA(cudaEventRecord(start, stream));
        for (int i = 0; i < options.iters; ++i) {
            launch_matmul(state, variant, options, d_a, d_b, d_c, stream);
        }
        OZAKI_CHECK_CUDA(cudaEventRecord(stop, stream));
        OZAKI_CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        OZAKI_CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        Metrics metrics;
        metrics.gpu_name = props.name;
        metrics.compute_capability_major = props.major;
        metrics.compute_capability_minor = props.minor;
        OZAKI_CHECK_CUDA(cudaRuntimeGetVersion(&metrics.cuda_runtime_version));
        OZAKI_CHECK_CUBLAS(cublasGetVersion(state.handle, &metrics.cublas_version));
        metrics.time_ms = static_cast<double>(elapsed_ms) / static_cast<double>(options.iters);
        metrics.tflops = (2.0 * static_cast<double>(options.m) * options.n * options.k) /
            (metrics.time_ms * 1.0e9);

        std::vector<double> h_c(c_elements);
        OZAKI_CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

        if (state.retained_bits_device != nullptr) {
            OZAKI_CHECK_CUDA(cudaMemcpy(
                &metrics.retained_bits,
                state.retained_bits_device,
                sizeof(int),
                cudaMemcpyDeviceToHost));
            metrics.emulation_used = metrics.retained_bits >= 0 ? 1 : 0;
        }

        if (variant != Variant::kNative) {
            std::vector<double> h_ref;
            if (options.reference_mode == ReferenceMode::kNativeFp64) {
                h_ref.resize(c_elements);
                OZAKI_CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, c_bytes, cudaMemcpyDeviceToHost));
            } else {
                h_ref = reference_gemm_long_double(
                    h_a.data(), h_b.data(), options.m, options.n, options.k);
            }
            const AccuracyMetrics accuracy = measure_accuracy(h_c.data(), h_ref.data(), c_elements);
            metrics.max_abs_error = accuracy.max_abs_error;
            metrics.mean_abs_error = accuracy.mean_abs_error;
            metrics.relative_l2_error = accuracy.relative_l2;
            metrics.normalized_max_abs_error = accuracy.normalized_max_abs;
            if (!options.accuracy_measure_only) {
                assert_accuracy(accuracy, options.relative_l2_limit, options.normalized_max_abs_limit);
            }
            for (double value : h_c) metrics.checksum += value;
            if (metrics.emulation_used == 0) {
                throw std::runtime_error(
                    "Ozaki emulation descriptor fell back to native FP64; "
                    "adjust retained-bit parameters before claiming a speedup");
            }
        } else {
            for (double value : h_c) {
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite native FP64 result");
                metrics.checksum += value;
            }
        }

        release_resources(&ref_state, &state, start, stop, workspace, d_ref, d_c, d_b, d_a, stream);
        return metrics;
    } catch (...) {
        release_resources(&ref_state, &state, start, stop, workspace, d_ref, d_c, d_b, d_a, stream);
        throw;
    }
}

inline void print_metrics(Variant variant, const Options& options, const Metrics& metrics) {
    std::cout << std::scientific << std::setprecision(17);
    std::cout << "VARIANT: " << variant_name(variant) << "\n";
    std::cout << "M: " << options.m << "\n";
    std::cout << "N: " << options.n << "\n";
    std::cout << "K: " << options.k << "\n";
    std::cout << "WARMUP: " << options.warmup << "\n";
    std::cout << "ITERS: " << options.iters << "\n";
    std::cout << "SEED: " << options.seed << "\n";
    std::cout << "INPUT_SCALE: " << options.input_scale << "\n";
    std::cout << "INPUT_PATTERN: " << input_pattern_name(options.input_pattern) << "\n";
    std::cout << "REFERENCE_MODE: " << reference_mode_name(options.reference_mode) << "\n";
    std::cout << "GPU_NAME: " << metrics.gpu_name << "\n";
    std::cout << "COMPUTE_CAPABILITY: " << metrics.compute_capability_major << "."
              << metrics.compute_capability_minor << "\n";
    std::cout << "CUDA_RUNTIME_VERSION: " << metrics.cuda_runtime_version << "\n";
    std::cout << "CUBLAS_VERSION: " << metrics.cublas_version << "\n";
    if (variant == Variant::kDynamic) {
        std::cout << "DYNAMIC_MAX_BITS: " << options.dynamic_max_bits << "\n";
        std::cout << "DYNAMIC_OFFSET: " << options.dynamic_offset << "\n";
    } else if (variant == Variant::kFixed) {
        std::cout << "FIXED_BITS: " << options.fixed_bits << "\n";
    }
    if (variant != Variant::kNative) {
        std::cout << "EMULATION_STRATEGY: " << emulation_strategy_name(options.emulation_strategy) << "\n";
    }
    std::cout << "EMULATION_USED: " << metrics.emulation_used << "\n";
    std::cout << "RETAINED_BITS: " << metrics.retained_bits << "\n";
    std::cout << "MAX_ABS_ERROR: " << metrics.max_abs_error << "\n";
    std::cout << "MEAN_ABS_ERROR: " << metrics.mean_abs_error << "\n";
    std::cout << "RELATIVE_L2_ERROR: " << metrics.relative_l2_error << "\n";
    std::cout << "NORMALIZED_MAX_ABS_ERROR: " << metrics.normalized_max_abs_error << "\n";
    if (options.accuracy_measure_only) {
        std::cout << "ACCURACY_STATUS: MEASUREMENT_ONLY_NOT_ACCEPTED\n";
        return;
    }
    std::cout << "ACCURACY_STATUS: " << (variant == Variant::kNative ? "NATIVE_REFERENCE" : "CONFIGURED_LIMITS_PASSED") << "\n";
    std::cout << "TFLOPS: " << metrics.tflops << "\n";
    std::cout << "RESULT_CHECKSUM: " << metrics.checksum << "\n";
    VERIFY_PRINT_CHECKSUM(static_cast<float>(metrics.checksum));
    std::cout << "TIME_MS: " << metrics.time_ms << "\n";
}

inline int run_and_report(Variant variant, int argc, char** argv) {
    const Options options = parse_args(argc, argv);
    const Metrics metrics = benchmark_variant(variant, options);
    print_metrics(variant, options, metrics);
    return options.accuracy_measure_only ? 2 : 0;
}

}  // namespace ozaki_scheme
