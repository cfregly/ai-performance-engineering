# Chapter 9 Makefile - Fusion and CUTLASS examples with dual-arch builds

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) $(CUDA_NVTX_CFLAGS)
SUFFIX = $(TARGET_SUFFIX)
CUTLASS_ROOT := ../third_party/cutlass
CUTLASS_INCLUDES := -I$(CUTLASS_ROOT)/include -I$(CUTLASS_ROOT)/tools/util/include -I$(CUTLASS_ROOT)/examples/common

# Build NVCC binaries for all CUDA sources (exclude PyTorch extension sources)
CUDA_SOURCES = $(filter-out tcgen05_basic.cu tcgen05_pipelined.cu, $(wildcard *.cu))
BASE_TARGETS = $(CUDA_SOURCES:.cu=)
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))
VERIFY_TARGETS = $(addsuffix _verify$(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all

all: $(TARGETS)
	@echo "Building Chapter 9 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

$(addsuffix $(SUFFIX), $(BASE_TARGETS)): %$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	@echo "Building $@..."
	@if [ "$*" = "cutlass_gemm_example" ] || [ "$*" = "micro_tiling_matmul" ]; then \
		$(NVCC) $(NVCC_FLAGS) -lcublas -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS); \
	elif [ "$*" = "baseline_cutlass_gemm" ] || [ "$*" = "optimized_cutlass_gemm" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp16" ] || [ "$*" = "optimized_cutlass_gemm_fp16" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp8" ] || [ "$*" = "optimized_cutlass_gemm_fp8" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp4" ] || [ "$*" = "optimized_cutlass_gemm_fp4" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp4_all_concepts" ] || [ "$*" = "optimized_cutlass_gemm_fp4_all_concepts" ] || \
	     [ "$*" = "baseline_cute_dsl_nvfp4_gemm" ] || [ "$*" = "optimized_cute_dsl_nvfp4_gemm" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp4_perchannel" ] || [ "$*" = "optimized_cutlass_gemm_fp4_perchannel" ]; then \
		$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS); \
	elif [ "$*" = "baseline_cublaslt_gemm" ] || [ "$*" = "optimized_cublaslt_gemm" ] || \
	     [ "$*" = "baseline_cublaslt_gemm_fp16" ] || [ "$*" = "optimized_cublaslt_gemm_fp16" ] || \
	     [ "$*" = "baseline_cublaslt_gemm_fp8" ] || [ "$*" = "optimized_cublaslt_gemm_fp8" ] || \
	     [ "$*" = "baseline_cublaslt_gemm_fp4" ] || [ "$*" = "optimized_cublaslt_gemm_fp4" ] || \
	     [ "$*" = "baseline_cublas_gemm_fp4_perchannel" ] || [ "$*" = "optimized_cublas_gemm_fp4_perchannel" ]; then \
		$(NVCC) $(NVCC_FLAGS) -lcublas -lcublasLt -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS); \
	else \
		$(NVCC) $(NVCC_FLAGS) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS); \
	fi

$(VERIFY_TARGETS): %_verify$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	@echo "Building $@ (verify)..."
	@if [ "$*" = "cutlass_gemm_example" ] || [ "$*" = "micro_tiling_matmul" ]; then \
		$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 -lcublas -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS); \
	elif [ "$*" = "baseline_cutlass_gemm" ] || [ "$*" = "optimized_cutlass_gemm" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp16" ] || [ "$*" = "optimized_cutlass_gemm_fp16" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp8" ] || [ "$*" = "optimized_cutlass_gemm_fp8" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp4" ] || [ "$*" = "optimized_cutlass_gemm_fp4" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp4_all_concepts" ] || [ "$*" = "optimized_cutlass_gemm_fp4_all_concepts" ] || \
	     [ "$*" = "baseline_cute_dsl_nvfp4_gemm" ] || [ "$*" = "optimized_cute_dsl_nvfp4_gemm" ] || \
	     [ "$*" = "baseline_cutlass_gemm_fp4_perchannel" ] || [ "$*" = "optimized_cutlass_gemm_fp4_perchannel" ]; then \
		$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 $(CUTLASS_INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS); \
	elif [ "$*" = "baseline_cublaslt_gemm" ] || [ "$*" = "optimized_cublaslt_gemm" ] || \
	     [ "$*" = "baseline_cublaslt_gemm_fp16" ] || [ "$*" = "optimized_cublaslt_gemm_fp16" ] || \
	     [ "$*" = "baseline_cublaslt_gemm_fp8" ] || [ "$*" = "optimized_cublaslt_gemm_fp8" ] || \
	     [ "$*" = "baseline_cublaslt_gemm_fp4" ] || [ "$*" = "optimized_cublaslt_gemm_fp4" ] || \
	     [ "$*" = "baseline_cublas_gemm_fp4_perchannel" ] || [ "$*" = "optimized_cublas_gemm_fp4_perchannel" ]; then \
		$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 -lcublas -lcublasLt -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS); \
	else \
		$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS); \
	fi

compare: clean
	@echo "========================================"
	@echo "Building Chapter 9 samples for both architectures..."
	@echo "========================================"
	@set -e; for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done

b200: ARCH=sm_100
b200: clean all

gb10: ARCH=sm_121
gb10: clean all

b300: ARCH=sm_103
b300: clean all

gb300: ARCH=sm_103
gb300: clean all

profile-nsys: $(TARGETS)
	@echo "Nsight Systems timeline profiling..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_$(ARCH) ./$(firstword $(TARGETS))

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o ncu_profile_$(ARCH) ./$(firstword $(TARGETS))

profile-hta: $(TARGETS)
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$(ARCH) ./$(firstword $(TARGETS))

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	perf record -g -o perf.data_$(ARCH) ./$(firstword $(TARGETS))
	perf report -i perf.data_$(ARCH)

profile-all: $(TARGETS)
	@echo "Comprehensive profiling with all tools..."
	@echo "1. Nsight Systems timeline..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$(ARCH) ./$(firstword $(TARGETS))
	@echo "2. Nsight Compute kernel analysis..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel_$(ARCH) ./$(firstword $(TARGETS))
	@echo "3. Memory profiling..."
	nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory_$(ARCH) ./$(firstword $(TARGETS))
	@echo "4. HTA analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta_$(ARCH) ./$(firstword $(TARGETS))

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
