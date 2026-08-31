# Chapter 11 Makefile - CUDA Streams and Warp Specialization
# Auto-builds all *.cu files - no manual BASE_TARGETS list needed

include ../core/common/cuda_arch.mk

NVCC_ALLOW_UNSUPPORTED = --allow-unsupported-compiler
NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) $(NVCC_ALLOW_UNSUPPORTED) $(CUDA_NVTX_CFLAGS)
NVCC_LDFLAGS = $(CUDA_NVTX_LDFLAGS)
SUFFIX = $(TARGET_SUFFIX)

# Auto-detect all .cu files and derive targets (no manual list maintenance)
CUDA_SOURCES := $(wildcard *.cu)
# Filter out files that require the PyTorch extension build system.
EXTENSION_SOURCES := $(wildcard *_extension.cu)
EXCLUDE_SOURCES := $(EXTENSION_SOURCES)
STANDALONE_SOURCES := $(filter-out $(EXCLUDE_SOURCES),$(CUDA_SOURCES))
BASE_TARGETS := $(STANDALONE_SOURCES:.cu=)
TARGETS := $(addsuffix $(SUFFIX), $(BASE_TARGETS))
VERIFY_TARGETS := $(addsuffix _verify$(SUFFIX), $(BASE_TARGETS))

# Default target
all: $(TARGETS)
	@echo "Building all examples with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

# Pattern rule for building targets with architecture suffix
%$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LDFLAGS)

# Verify-mode pattern rule (adds -DVERIFY=1)
%_verify$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 -o $@ $< $(NVCC_LDFLAGS)

# Special rules for dual-pipeline executables (use dedicated drivers)
baseline_warp_specialized_two_pipelines_multistream$(SUFFIX): \
	baseline_warp_specialized_two_pipelines_driver.cu baseline_warp_specialized_two_pipelines_common.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ baseline_warp_specialized_two_pipelines_driver.cu $(CUDA_NVTX_LDFLAGS)

optimized_warp_specialized_two_pipelines_multistream$(SUFFIX): \
	optimized_warp_specialized_two_pipelines_driver.cu warp_specialized_two_pipelines_multistream_impl.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(CUDA_NVTX_LDFLAGS)

# Architecture-specific targets
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

compare: clean
	@echo "========================================"
	@echo "Building BOTH architectures for comparison..."
	@echo "========================================"
	@set -e; for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done

profile-nsys: $(TARGETS)
	@echo "Nsight Systems timeline profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o nsys_profile_$$target ./$$target; \
	done

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency -o ncu_profile_$$target ./$$target; \
	done

profile-hta: $(TARGETS)
	@echo "HTA profiling..."
	@for target in $(TARGETS); do \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,nccl -o hta_profile_$$target ./$$target; \
	done

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	@for target in $(TARGETS); do \
		perf record -g -o perf.data_$$target ./$$target || true; \
	done

profile-all: profile-nsys profile-ncu profile-hta

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all b200 gb10 b300 gb300 compare profile-nsys profile-ncu profile-hta profile-perf profile-all clean
