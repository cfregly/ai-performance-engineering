# Chapter 7 Makefile - Memory optimization examples with dual-arch support

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) $(CUDA_NVTX_CFLAGS)
SUFFIX = $(TARGET_SUFFIX)

CUDA_SOURCES = $(wildcard *.cu)
BASE_TARGETS = $(CUDA_SOURCES:.cu=)

TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))
VERIFY_TARGETS = $(addsuffix _verify$(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all $(TARGETS) $(VERIFY_TARGETS)

all: $(TARGETS)
	@echo "Building Chapter 7 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

$(TARGETS): %$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	@echo "Compiling $@..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda $(CUDA_NVTX_LDFLAGS)

# Verify binaries build with -DVERIFY=1 for harness checksum parsing
$(VERIFY_TARGETS): %_verify$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	@echo "Compiling verify binary $@..."
	$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 -o $@ $< -lcuda $(CUDA_NVTX_LDFLAGS)

TMA_COPY_TARGETS = async_prefetch_2d_demo optimized_tma_bulk_tensor_2d optimized_tma_copy
TMA_HEADERS = ../core/common/headers/arch_detection.cuh ../core/common/headers/tma_helpers.cuh ../core/common/headers/tma_2d_layout.hpp
$(addsuffix $(SUFFIX),$(TMA_COPY_TARGETS)) $(addsuffix _verify$(SUFFIX),$(TMA_COPY_TARGETS)): $(TMA_HEADERS)

compare: clean
	@echo "========================================"
	@echo "Building Chapter 7 samples for both architectures..."
	@echo "========================================"
	@set -e; for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done

b200:
	$(MAKE) ARCH=sm_100 clean
	$(MAKE) ARCH=sm_100 all

gb10:
	$(MAKE) ARCH=sm_121 clean
	$(MAKE) ARCH=sm_121 all

b300:
	$(MAKE) ARCH=sm_103 clean
	$(MAKE) ARCH=sm_103 all

gb300:
	$(MAKE) ARCH=sm_103 clean
	$(MAKE) ARCH=sm_103 all

profile-nsys: all
	@echo "Nsight Systems timeline profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o nsys_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-ncu: all
	@echo "Nsight Compute kernel profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency -o ncu_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-hta: all
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o hta_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-perf: all
	@echo "Perf system-level profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe with perf..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || echo "perf failed for $$exe"; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

profile-all: all
	@echo "Comprehensive profiling with all tools..."
	@for exe in $(TARGETS); do \
		echo "1. Nsight Systems timeline..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$${exe}_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency -o comprehensive_kernel_$${exe}_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "3. Perf system-level analysis..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || true; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
