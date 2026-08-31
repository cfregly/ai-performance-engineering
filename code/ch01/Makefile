# Makefile for Chapter 1 - Performance Basics Examples
# Auto-builds all *.cu files

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) --use_fast_math -lineinfo $(CUDA_NVTX_CFLAGS)
NVCC_LIBS = -lcublas -lcublasLt
SUFFIX = $(TARGET_SUFFIX)

# Auto-detect all .cu files and derive targets
CUDA_SOURCES := $(wildcard *.cu)
BASE_TARGETS := $(CUDA_SOURCES:.cu=)
TARGETS := $(addsuffix $(SUFFIX), $(BASE_TARGETS))
VERIFY_TARGETS := $(addsuffix _verify$(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean test compare b200 gb10 b300 gb300

all: $(TARGETS)
	@echo "Building Chapter 1 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

# Default pattern rule
%$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) -o $@ $< $(CUDA_NVTX_LDFLAGS)

# Verify-mode pattern rule (adds -DVERIFY=1)
%_verify$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 $(NVCC_LIBS) -o $@ $< $(CUDA_NVTX_LDFLAGS)

test: all
	@echo "Testing GEMM optimizations..."
	./baseline_gemm$(SUFFIX) || echo "Note: Run baseline_gemm manually"

compare: clean
	@set -e; for arch in $(ARCH_LIST); do \
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

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
	rm -f nsys_* ncu_* hta_* perf_* comprehensive_*
