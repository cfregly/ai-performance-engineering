# Chapter 10 Makefile - Memory Pipelines and GPUDirect Storage
# Auto-builds all *.cu files - no manual BASE_TARGETS list needed

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -lineinfo -Xptxas=-v $(CUDA_NVTX_CFLAGS)
SUFFIX = $(TARGET_SUFFIX)
INCLUDES = -I..
COMMON_HEADERS = ../core/common/headers/arch_detection.cuh ../core/common/headers/tma_helpers.cuh

# CUDA 13.0's bundled CCCL supports the multicast wrapper for SM100a but
# omitted SM103a. The source carries NVIDIA's corrected SM103a PTX path.
# SM120 and SM121 compile the source's explicit non-multicast fallback.
override TMA_MULTICAST_TARGET := 0
ifeq ($(ARCH),sm_100)
override TMA_MULTICAST_TARGET := 100
else ifeq ($(ARCH),sm_103)
override TMA_MULTICAST_TARGET := 103
endif
TMA_MULTICAST_FLAGS := -DTMA_MULTICAST_TARGET=$(TMA_MULTICAST_TARGET)

# Auto-detect all .cu files and derive targets
CUDA_SOURCES := $(wildcard *.cu)
# Filter out files that require external build systems (PyTorch extensions)
EXTENSION_SOURCES := \
	$(wildcard *_extension.cu) \
	matmul_tcgen05.cu \
	tcgen05_cluster.cu \
	tcgen05_warp_specialized.cu \
	tcgen05_warp_specialized_cutlass.cu \
	tcgen05_warpgroup_specialized.cu
STANDALONE_SOURCES := $(filter-out $(EXTENSION_SOURCES),$(CUDA_SOURCES))
BASE_TARGETS := $(STANDALONE_SOURCES:.cu=)
TARGETS := $(addsuffix $(SUFFIX), $(BASE_TARGETS))
VERIFY_TARGETS := $(addsuffix _verify$(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean test compare verify-tma-multicast b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all run test-gds demo help

all: $(TARGETS)
	@echo "Building all Chapter 10 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

verify-tma-multicast: tma_multicast_baseline_verify$(SUFFIX) tma_multicast_cluster_verify$(SUFFIX)
	@echo "✓ Built Chapter 10 TMA multicast verification binaries for $(ARCH)"

# Default pattern rule for most .cu files
%$(SUFFIX): %.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

# Verify binaries build with -DVERIFY=1 for harness checksum parsing
%_verify$(SUFFIX): %.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

# Special rules for files needing -rdc=true (cluster/DSMEM support)
baseline_cluster_group$(SUFFIX): baseline_cluster_group.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

baseline_cluster_group_verify$(SUFFIX): baseline_cluster_group.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_cluster_group$(SUFFIX): optimized_cluster_group.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_cluster_group_verify$(SUFFIX): optimized_cluster_group.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

baseline_cluster_group_no_dsmem$(SUFFIX): baseline_cluster_group_no_dsmem.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

baseline_cluster_group_no_dsmem_verify$(SUFFIX): baseline_cluster_group_no_dsmem.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_cluster_group_no_dsmem$(SUFFIX): optimized_cluster_group_no_dsmem.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_cluster_group_no_dsmem_verify$(SUFFIX): optimized_cluster_group_no_dsmem.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_cluster_group_single_cta$(SUFFIX): optimized_cluster_group_single_cta.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_cluster_group_single_cta_verify$(SUFFIX): optimized_cluster_group_single_cta.cu cluster_group_common.cuh $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

baseline_warp_specialized_cluster_pipeline$(SUFFIX): baseline_warp_specialized_cluster_pipeline.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

baseline_warp_specialized_cluster_pipeline_verify$(SUFFIX): baseline_warp_specialized_cluster_pipeline.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_warp_specialized_cluster_pipeline$(SUFFIX): optimized_warp_specialized_cluster_pipeline.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_warp_specialized_cluster_pipeline_verify$(SUFFIX): optimized_warp_specialized_cluster_pipeline.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

tma_multicast_cluster$(SUFFIX): tma_multicast_cluster.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(TMA_MULTICAST_FLAGS) -std=c++17 -rdc=true $(INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

tma_multicast_cluster_verify$(SUFFIX): tma_multicast_cluster.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(TMA_MULTICAST_FLAGS) -std=c++17 -rdc=true -DVERIFY=1 $(INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

tma_multicast_baseline$(SUFFIX): tma_multicast_baseline.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(TMA_MULTICAST_FLAGS) -std=c++17 -rdc=true $(INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

tma_multicast_baseline_verify$(SUFFIX): tma_multicast_baseline.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(TMA_MULTICAST_FLAGS) -std=c++17 -rdc=true -DVERIFY=1 $(INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_dsmem_reduction$(SUFFIX): optimized_dsmem_reduction.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_dsmem_reduction_verify$(SUFFIX): optimized_dsmem_reduction.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_dsmem_reduction_cluster_atomic$(SUFFIX): optimized_dsmem_reduction_cluster_atomic.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_dsmem_reduction_cluster_atomic_verify$(SUFFIX): optimized_dsmem_reduction_cluster_atomic.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_dsmem_reduction_warp_specialized$(SUFFIX): optimized_dsmem_reduction_warp_specialized.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_dsmem_reduction_warp_specialized_verify$(SUFFIX): optimized_dsmem_reduction_warp_specialized.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

# Special rules for files needing C++20 or -lcuda
baseline_flash_attn_tma_micro_pipeline$(SUFFIX): baseline_flash_attn_tma_micro_pipeline.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

baseline_flash_attn_tma_micro_pipeline_verify$(SUFFIX): baseline_flash_attn_tma_micro_pipeline.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 -DVERIFY=1 $(INCLUDES) -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_flash_attn_tma_micro_pipeline$(SUFFIX): optimized_flash_attn_tma_micro_pipeline.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 $(INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

optimized_flash_attn_tma_micro_pipeline_verify$(SUFFIX): optimized_flash_attn_tma_micro_pipeline.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++20 -DVERIFY=1 $(INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

tma_2d_pipeline_blackwell$(SUFFIX): tma_2d_pipeline_blackwell.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 $(INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

tma_2d_pipeline_blackwell_verify$(SUFFIX): tma_2d_pipeline_blackwell.cu $(COMMON_HEADERS) $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -std=c++17 -DVERIFY=1 $(INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

# Architecture-specific targets
b200: ARCH=sm_100
b200: clean all

gb10: ARCH=sm_121
gb10: clean all

b300: ARCH=sm_103
b300: clean all

gb300: ARCH=sm_103
gb300: clean all

compare: clean
	@set -e; for arch in $(ARCH_LIST); do \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all verify-tma-multicast; \
	done

test: all
	@echo "=== Testing Memory Pipeline (CUDA) ==="
	./baseline_double_buffered_pipeline$(SUFFIX) 128 128 128 || true

test-gds:
	@echo "=== Testing GPUDirect Storage (Python) ==="
	python3 cufile_gds_example.py || true

profile-nsys: all
	@for exe in $(TARGETS); do nsys profile --force-overwrite=true -t cuda,nvtx -o nsys_$$exe ./$$exe 2>/dev/null || true; done

profile-ncu: all
	@for exe in $(TARGETS); do ncu --metrics achieved_occupancy -o ncu_$$exe ./$$exe 2>/dev/null || true; done

profile-hta: all
	@for exe in $(TARGETS); do nsys profile --force-overwrite=true -t cuda,nvtx,nccl -o hta_$$exe ./$$exe 2>/dev/null || true; done

profile-perf: all
	@for exe in $(TARGETS); do perf record -g -o perf_$$exe ./$$exe || true; done

profile-all: profile-nsys profile-ncu profile-hta

help:
	@echo "Chapter 10: Memory Pipelines and GPUDirect Storage"
	@echo ""
	@echo "Targets: make all | make test | make clean | make b200 | make b300"

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
	rm -f nsys_* ncu_* hta_* perf_* comprehensive_*
