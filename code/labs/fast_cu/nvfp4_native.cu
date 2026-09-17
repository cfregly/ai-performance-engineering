#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FAST_CU_NVFP4_RUNG
#error "FAST_CU_NVFP4_RUNG must select one upstream optimization rung"
#elif FAST_CU_NVFP4_RUNG == 0
#define NVFP4_GEMM_HEADER "gemm0.cuh"
#elif FAST_CU_NVFP4_RUNG == 1
#define NVFP4_GEMM_HEADER "gemm1.cuh"
#elif FAST_CU_NVFP4_RUNG == 2
#define NVFP4_GEMM_HEADER "gemm2.cuh"
#elif FAST_CU_NVFP4_RUNG == 3
#define NVFP4_GEMM_HEADER "gemm3.cuh"
#elif FAST_CU_NVFP4_RUNG == 4
#define NVFP4_GEMM_HEADER "gemm4.cuh"
#elif FAST_CU_NVFP4_RUNG == 5
#define NVFP4_GEMM_HEADER "gemm5.cuh"
#elif FAST_CU_NVFP4_RUNG == 6
#define NVFP4_GEMM_HEADER "gemm6.cuh"
#elif FAST_CU_NVFP4_RUNG == 7
#define NVFP4_GEMM_HEADER "gemm7.cuh"
#elif FAST_CU_NVFP4_RUNG == 8
#define NVFP4_GEMM_HEADER "gemm8.cuh"
#elif FAST_CU_NVFP4_RUNG == 9
#define NVFP4_GEMM_HEADER "gemm9.cuh"
#else
#error "FAST_CU_NVFP4_RUNG must be in [0, 9]"
#endif

// Reuse the pinned upstream host packing, tensor-map, cuBLASLt, and kernel
// implementation without editing the vendored source. The standalone entrypoint
// is renamed because this translation unit supplies a Python module instead.
#define main fast_cu_nvfp4_upstream_standalone_main
#include "main.cu"
#undef main

namespace py = pybind11;

namespace {

std::mutex small_gate_mutex;
std::vector<int> small_gated_devices;

void check_cuda(cudaError_t status, const char* operation) {
    TORCH_CHECK(
        status == cudaSuccess,
        operation,
        " failed: ",
        cudaGetErrorName(status),
        " (", cudaGetErrorString(status), ")");
}

void check_cublas(cublasStatus_t status, const char* operation) {
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        operation,
        " failed with cuBLASLt status ",
        static_cast<int>(status));
}

cudaStream_t current_stream(int device) {
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
    TORCH_CHECK(
        current_device == device,
        "fast.cu NVFP4 context was created on CUDA device ",
        device,
        " but current device is ",
        current_device);
    return at::cuda::getCurrentCUDAStream(device);
}

void require_exact_runtime(cudaDeviceProp* properties, int* device) {
    int runtime_version = 0;
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    TORCH_CHECK(
        runtime_version >= 13010,
        "SKIPPED: fast.cu NVFP4 requires CUDA runtime 13.1 or newer; found ",
        runtime_version);
    check_cuda(cudaGetDevice(device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(properties, *device), "cudaGetDeviceProperties");
    TORCH_CHECK(
        properties->major == 10 && properties->minor == 3,
        "SKIPPED: fast.cu NVFP4 requires exact SM103 GB300/B300; found sm_",
        properties->major,
        properties->minor,
        " on ",
        properties->name);
    TORCH_CHECK(
        properties->multiProcessorCount > 0 && properties->multiProcessorCount % 2 == 0,
        "fast.cu NVFP4 requires a positive even SM count for two-CTA clusters; found ",
        properties->multiProcessorCount);
}

torch::Tensor cuda_bytes(int64_t count, int device) {
    return torch::empty(
        {count},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));
}

#if NVFP4_HAS_SCHEDULE
// Pybind methods intentionally retain the GIL. Setup synchronizes inside the
// census before replacing these translation-unit-global constants, and every
// launch checks the active device/shape. Native callers bypassing pybind must
// provide their own serialization.
std::mutex schedule_mutex;
int scheduled_device = -1;
std::array<int, 3> scheduled_shape{-1, -1, -1};

void strict_setup_r9_schedule(
    int device,
    const cudaDeviceProp& properties,
    dim3 grid,
    int clusters,
    int M,
    int N,
    int K,
    size_t operand_read_bytes) {
    std::lock_guard<std::mutex> lock(schedule_mutex);
    const int mblocks = (M + nvfp4::CLUSTER_M - 1) / nvfp4::CLUSTER_M;
    const int nblocks = (N + nvfp4::BLOCK_N - 1) / nvfp4::BLOCK_N;
    const int total_work = mblocks * nblocks;
    TORCH_CHECK(
        total_work <= nvfp4::L2A_ROUTE_WORK_CAP,
        "fast.cu r9 schedule has ",
        total_work,
        " tiles but the pinned table holds only ",
        nvfp4::L2A_ROUTE_WORK_CAP);

    const sched::ScheduleMode mode =
        sched::pick_schedule(operand_read_bytes, mblocks, nblocks);
    TORCH_CHECK(
        sched::requires_side_census(mode),
        "fast.cu r9 AUTO schedule unexpectedly disabled L2-side ownership");

    constexpr int64_t probe_bytes = int64_t(2) << 20;
    torch::Tensor probe_storage = cuda_bytes(2 * probe_bytes, device);
    const uintptr_t raw = reinterpret_cast<uintptr_t>(probe_storage.data_ptr());
    char* arena = reinterpret_cast<char*>(
        (raw + probe_bytes - 1) & ~(uintptr_t(probe_bytes) - 1));
    const l2side::RuntimeMap map =
        l2side::probe_stable(arena, probe_bytes, /*repeats=*/3);
    TORCH_CHECK(
        map.nsm == properties.multiProcessorCount,
        "fast.cu r9 L2 map covered ", map.nsm,
        " SMs but the device reports ", properties.multiProcessorCount);
    TORCH_CHECK(
        map.hash == l2side::kExpectedHash && map.model_mismatches == 0,
        "fast.cu r9 L2 address model failed: hash=",
        map.hash,
        " mismatches=",
        map.model_mismatches);

    const size_t census_count = size_t(clusters) * 2;
    torch::Tensor device_smids = torch::empty(
        {static_cast<int64_t>(census_count)},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device));
    cudaStream_t stream = current_stream(device);
    check_cuda(
        cudaMemsetAsync(
            device_smids.data_ptr(),
            0xff,
            census_count * sizeof(unsigned),
            stream),
        "cudaMemsetAsync(cluster census)");
    nvfp4::l2a_cluster_probe<<<grid, nvfp4::TB_SIZE, sizeof(nvfp4::SmemCD), stream>>>(
        reinterpret_cast<unsigned*>(device_smids.data_ptr()));
    check_cuda(cudaGetLastError(), "fast.cu r9 cluster census launch");
    std::vector<unsigned> smids(census_count);
    check_cuda(
        cudaMemcpyAsync(
            smids.data(),
            device_smids.data_ptr(),
            census_count * sizeof(unsigned),
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync(cluster census)");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cluster census)");

    const sched::Census census =
        sched::classify_census(smids, map, properties.multiProcessorCount);
    sched::upload_side_constants(map, census);
    std::vector<int> table = sched::build_schedule(census, mblocks, nblocks, mode);
    TORCH_CHECK(
        !table.empty(),
        "fast.cu r9 could not build a strict L2-owned schedule for ",
        M, "x", N, "x", K,
        "; raster fallback is forbidden");
    sched::upload_table(table, bench::MAIN_TABLE);
    scheduled_device = device;
    scheduled_shape = {M, N, K};
}
#endif

class Nvfp4Context {
public:
    Nvfp4Context(int M, int N, int K, uint64_t seed, bool prepare_fast)
        : M_(M), N_(N), K_(K), seed_(seed) {
        TORCH_CHECK(M > 0 && N > 0 && K > 0, "M, N, and K must be positive");
        TORCH_CHECK(
            K % 32 == 0,
            "cuBLASLt NVFP4 comparison requires K divisible by 32; found ",
            K);

        cudaDeviceProp properties{};
        require_exact_runtime(&properties, &device_);
        c10::cuda::CUDAGuard guard(device_);
        stream_ = current_stream(device_);

        host::Fixture fixture = host::make_fixture(M, N, K, seed, /*keep_deq=*/false);
        TORCH_CHECK(
            fixture.inputs.A == fixture.A_logical &&
                fixture.inputs.B == fixture.B_logical,
            "fast.cu and cuBLASLt input bytes differ");

        a_ = cuda_bytes(static_cast<int64_t>(fixture.inputs.A.size()), device_);
        b_ = cuda_bytes(static_cast<int64_t>(fixture.inputs.B.size()), device_);
        sfa_ = cuda_bytes(static_cast<int64_t>(fixture.inputs.SFA.size()), device_);
        sfb_ = cuda_bytes(static_cast<int64_t>(fixture.inputs.SFB.size()), device_);
        copy_to_device(a_, fixture.inputs.A, "A");
        copy_to_device(b_, fixture.inputs.B, "B");
        copy_to_device(sfa_, fixture.inputs.SFA, "SFA");
        copy_to_device(sfb_, fixture.inputs.SFB, "SFB");
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(input setup)");

        fast_slot_.M = M;
        fast_slot_.N = N;
        fast_slot_.K = K;
        fast_slot_.dA = a_.data_ptr<uint8_t>();
        fast_slot_.dB = b_.data_ptr<uint8_t>();
        fast_slot_.dSFA = sfa_.data_ptr<uint8_t>();
        fast_slot_.dSFB = sfb_.data_ptr<uint8_t>();
        fast_slot_.a_sz = fixture.inputs.A.size();
        fast_slot_.b_sz = fixture.inputs.B.size();
        fast_slot_.sfa_sz = fixture.inputs.SFA.size();
        fast_slot_.sfb_sz = fixture.inputs.SFB.size();
        fast_slot_.A_t = host::make_ab_tmap(
            fast_slot_.dA, M, K, fixture.inputs.ab_row_stride);
        fast_slot_.B_t = host::make_ab_tmap(
            fast_slot_.dB, N, K, fixture.inputs.ab_row_stride);
        fast_slot_.SFA_t = host::make_sf_tmap(fast_slot_.dSFA, M, K);
        fast_slot_.SFB_t = host::make_sf_tmap(fast_slot_.dSFB, N, K);

        sms_ = properties.multiProcessorCount;
        clusters_ = sms_ / 2;
        grid_ = nvfp4::launch_grid(clusters_);
        if (prepare_fast) {
            prepare_fast_kernel(properties, fixture.inputs.A.size() +
                fixture.inputs.B.size() + fixture.inputs.SFA.size() +
                fixture.inputs.SFB.size());
        } else {
            plan_ = std::make_unique<bench::LtPlan>(M, N, K);
            initialize_vendor_descriptor();
            plan_->initialize_algorithm(vendor_);
            if (plan_->workspace_bytes > 0) {
                workspace_ = cuda_bytes(
                    static_cast<int64_t>(plan_->workspace_bytes), device_);
                vendor_.workspace = workspace_.data_ptr();
                TORCH_CHECK(
                    (reinterpret_cast<uintptr_t>(vendor_.workspace) & 255u) == 0,
                    "cuBLASLt workspace is not 256-byte aligned");
            }
        }
    }

    ~Nvfp4Context() {
        if (vendor_.op != nullptr) {
            cublasLtMatmulDescDestroy(vendor_.op);
            vendor_.op = nullptr;
        }
    }

    Nvfp4Context(const Nvfp4Context&) = delete;
    Nvfp4Context& operator=(const Nvfp4Context&) = delete;

    void launch_cublaslt(torch::Tensor output) {
        validate_output(output);
        TORCH_CHECK(plan_ != nullptr, "cuBLASLt was not prepared for this context");
        c10::cuda::CUDAGuard guard(device_);
        cudaStream_t stream = current_stream(device_);
        float alpha = 1.0f;
        float beta = 0.0f;
        void* output_ptr = output.data_ptr<at::Half>();
        check_cublas(
            cublasLtMatmul(
                plan_->handle,
                vendor_.op,
                &alpha,
                vendor_.a,
                plan_->a_layout,
                vendor_.b,
                plan_->b_layout,
                &beta,
                output_ptr,
                plan_->out_layout,
                output_ptr,
                plan_->out_layout,
                &plan_->algo,
                vendor_.workspace,
                plan_->workspace_bytes,
                stream),
            "cublasLtMatmul");
    }

    void launch_fast(torch::Tensor output) {
        validate_output(output);
        TORCH_CHECK(fast_prepared_, "fast.cu kernel was not prepared for this context");
        c10::cuda::CUDAGuard guard(device_);
#if NVFP4_HAS_SCHEDULE
        TORCH_CHECK(
            scheduled_device == device_ && scheduled_shape == std::array<int, 3>{M_, N_, K_},
            "fast.cu r9 schedule no longer matches this context");
#endif
        cudaStream_t stream = current_stream(device_);
        nvfp4::nvfp4_gemm_kernel
            <<<grid_, nvfp4::TB_SIZE, sizeof(nvfp4::SmemCD), stream>>>(
                fast_slot_.A_t,
                fast_slot_.B_t,
                fast_slot_.SFA_t,
                fast_slot_.SFB_t,
                reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
                M_,
                N_,
                K_
#if NVFP4_HAS_SCHEDULE
                , bench::MAIN_TABLE * nvfp4::L2A_ROUTE_WORK_CAP
#endif
            );
        check_cuda(cudaGetLastError(), "fast.cu NVFP4 kernel launch");
    }

    void validate_schedule() {
#if NVFP4_HAS_SCHEDULE
        TORCH_CHECK(fast_prepared_, "fast.cu r9 schedule was not prepared");
        c10::cuda::CUDAGuard guard(device_);
        check_cuda(
            cudaStreamSynchronize(current_stream(device_)),
            "cudaStreamSynchronize(schedule audit)");
        unsigned placement_errors = 0;
        check_cuda(
            cudaMemcpyFromSymbol(
                &placement_errors,
                nvfp4::l2a_placement_errors,
                sizeof(placement_errors)),
            "cudaMemcpyFromSymbol(placement audit)");
        TORCH_CHECK(
            placement_errors == 0,
            "fast.cu r9 placement audit found ",
            placement_errors,
            " cluster/L2 ownership mismatches; result is invalid");
#else
        TORCH_CHECK(fast_prepared_, "fast.cu kernel was not prepared");
#endif
    }

    torch::Tensor a_packed() const { return a_; }
    torch::Tensor b_packed() const { return b_; }
    torch::Tensor sfa_packed() const { return sfa_; }
    torch::Tensor sfb_packed() const { return sfb_; }
    int rung() const { return FAST_CU_NVFP4_RUNG; }
    uint64_t seed() const { return seed_; }

private:
    void copy_to_device(
        torch::Tensor& destination,
        const std::vector<uint8_t>& source,
        const char* label) {
        TORCH_CHECK(
            static_cast<size_t>(destination.numel()) == source.size(),
            label,
            " source/destination size mismatch");
        check_cuda(
            cudaMemcpyAsync(
                destination.data_ptr(),
                source.data(),
                source.size(),
                cudaMemcpyHostToDevice,
                stream_),
            label);
    }

    void initialize_vendor_descriptor() {
        vendor_.a = a_.data_ptr();
        vendor_.b = b_.data_ptr();
        vendor_.sfa = sfa_.data_ptr();
        vendor_.sfb = sfb_.data_ptr();
        check_cublas(
            cublasLtMatmulDescCreate(
                &vendor_.op,
                CUBLAS_COMPUTE_32F,
                CUDA_R_32F),
            "cublasLtMatmulDescCreate");
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        int32_t sf_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        int8_t fast_accum = 0;
        check_cublas(
            cublasLtMatmulDescSetAttribute(
                vendor_.op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)),
            "set cuBLASLt TRANSA");
        check_cublas(
            cublasLtMatmulDescSetAttribute(
                vendor_.op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)),
            "set cuBLASLt TRANSB");
        check_cublas(
            cublasLtMatmulDescSetAttribute(
                vendor_.op,
                CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                &fast_accum,
                sizeof(fast_accum)),
            "set cuBLASLt FAST_ACCUM");
        check_cublas(
            cublasLtMatmulDescSetAttribute(
                vendor_.op,
                CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                &sf_mode,
                sizeof(sf_mode)),
            "set cuBLASLt A scale mode");
        check_cublas(
            cublasLtMatmulDescSetAttribute(
                vendor_.op,
                CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                &sf_mode,
                sizeof(sf_mode)),
            "set cuBLASLt B scale mode");
        void* sfa_pointer = sfa_.data_ptr();
        void* sfb_pointer = sfb_.data_ptr();
        check_cublas(
            cublasLtMatmulDescSetAttribute(
                vendor_.op,
                CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &sfa_pointer,
                sizeof(sfa_pointer)),
            "set cuBLASLt A scale pointer");
        check_cublas(
            cublasLtMatmulDescSetAttribute(
                vendor_.op,
                CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &sfb_pointer,
                sizeof(sfb_pointer)),
            "set cuBLASLt B scale pointer");
    }

    void prepare_fast_kernel(
        const cudaDeviceProp& properties,
        size_t operand_read_bytes) {
        bench::g.sms = sms_;
        bench::g.clusters = clusters_;
        bench::g.l2_bytes = properties.l2CacheSize;
        bench::g.grid = grid_;
        check_cuda(
            cudaFuncSetAttribute(
                nvfp4::nvfp4_gemm_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(sizeof(nvfp4::SmemCD))),
            "cudaFuncSetAttribute(fast.cu kernel)");
#if NVFP4_HAS_SCHEDULE
        check_cuda(
            cudaFuncSetAttribute(
                nvfp4::l2a_cluster_probe,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(sizeof(nvfp4::SmemCD))),
            "cudaFuncSetAttribute(fast.cu census)");
        cudaLaunchConfig_t occupancy_config{};
        cudaLaunchAttribute cluster_attribute{};
        cluster_attribute.id = cudaLaunchAttributeClusterDimension;
        cluster_attribute.val.clusterDim.x = 2;
        cluster_attribute.val.clusterDim.y = 1;
        cluster_attribute.val.clusterDim.z = 1;
        occupancy_config.blockDim = dim3(nvfp4::TB_SIZE, 1, 1);
        occupancy_config.dynamicSmemBytes = sizeof(nvfp4::SmemCD);
        occupancy_config.attrs = &cluster_attribute;
        occupancy_config.numAttrs = 1;
        int active_clusters = 0;
        check_cuda(
            cudaOccupancyMaxActiveClusters(
                &active_clusters,
                nvfp4::nvfp4_gemm_kernel,
                &occupancy_config),
            "cudaOccupancyMaxActiveClusters(fast.cu r9)");
        TORCH_CHECK(
            active_clusters == clusters_,
            "fast.cu r9 requires all ",
            clusters_,
            " two-CTA clusters resident; occupancy reports ",
            active_clusters);
        strict_setup_r9_schedule(
            device_, properties, grid_, clusters_, M_, N_, K_, operand_read_bytes);
#endif
        {
            std::lock_guard<std::mutex> lock(small_gate_mutex);
            const bool already_gated = std::find(
                small_gated_devices.begin(), small_gated_devices.end(), device_)
                != small_gated_devices.end();
            if (!already_gated) {
                TORCH_CHECK(
                    bench::small_correctness_gates(),
                    "fast.cu upstream small host-reference, guard, or determinism gate failed");
                small_gated_devices.push_back(device_);
            }
        }
        fast_prepared_ = true;
    }

    void validate_output(const torch::Tensor& output) const {
        TORCH_CHECK(output.is_cuda(), "NVFP4 output must be a CUDA tensor");
        TORCH_CHECK(
            output.get_device() == device_,
            "NVFP4 output is on CUDA device ",
            output.get_device(),
            " but context is on ",
            device_);
        TORCH_CHECK(output.scalar_type() == torch::kFloat16, "NVFP4 output must be FP16");
        TORCH_CHECK(output.is_contiguous(), "NVFP4 output must be contiguous");
        TORCH_CHECK(
            output.dim() == 2 && output.size(0) == M_ && output.size(1) == N_,
            "NVFP4 output must have shape [", M_, ", ", N_, "]");
    }

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
    uint64_t seed_ = 0;
    int device_ = -1;
    int sms_ = 0;
    int clusters_ = 0;
    dim3 grid_{};
    cudaStream_t stream_ = nullptr;
    bool fast_prepared_ = false;
    std::unique_ptr<bench::LtPlan> plan_;
    bench::LtSlot vendor_{};
    host::DeviceSlot fast_slot_{};
    torch::Tensor a_;
    torch::Tensor b_;
    torch::Tensor sfa_;
    torch::Tensor sfb_;
    torch::Tensor workspace_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    py::class_<Nvfp4Context>(module, "Nvfp4Context")
        .def(py::init<int, int, int, uint64_t, bool>())
        .def("launch_cublaslt", &Nvfp4Context::launch_cublaslt)
        .def("launch_fast", &Nvfp4Context::launch_fast)
        .def("validate_schedule", &Nvfp4Context::validate_schedule)
        .def_property_readonly("a_packed", &Nvfp4Context::a_packed)
        .def_property_readonly("b_packed", &Nvfp4Context::b_packed)
        .def_property_readonly("sfa_packed", &Nvfp4Context::sfa_packed)
        .def_property_readonly("sfb_packed", &Nvfp4Context::sfb_packed)
        .def_property_readonly("rung", &Nvfp4Context::rung)
        .def_property_readonly("seed", &Nvfp4Context::seed);
}
