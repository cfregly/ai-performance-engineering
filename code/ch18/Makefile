# Chapter 18 Makefile - Advanced attention optimizations
# Auto-builds all *.cu files

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -lineinfo $(CUDA_NVTX_CFLAGS)
SUFFIX = $(TARGET_SUFFIX)

# Auto-detect all .cu files and derive targets
CUDA_SOURCES := $(wildcard *.cu)
BASE_TARGETS := $(CUDA_SOURCES:.cu=)
TARGETS := $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean test compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all

all: $(TARGETS)
	@echo "Building all Chapter 18 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

# Default pattern rule
%$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda $(CUDA_NVTX_LDFLAGS)

test: all
	@for exe in $(TARGETS); do timeout 10s ./$$exe >/dev/null 2>&1 && echo "✓ $$exe" || echo "✗ $$exe"; done

compare: clean
	@set -e; for arch in $(ARCH_LIST); do $(MAKE) ARCH=$${arch} all; done

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
	@for exe in $(TARGETS); do nsys profile --force-overwrite=true -t cuda,nvtx -o nsys_$$exe ./$$exe 2>/dev/null || true; done

profile-ncu: all
	@for exe in $(TARGETS); do ncu --metrics achieved_occupancy -o ncu_$$exe ./$$exe 2>/dev/null || true; done

profile-hta: all
	@for exe in $(TARGETS); do nsys profile --force-overwrite=true -t cuda,nvtx,nccl -o hta_$$exe ./$$exe 2>/dev/null || true; done

profile-perf: all
	@for exe in $(TARGETS); do perf record -g -o perf_$$exe ./$$exe || true; done

profile-all: profile-nsys profile-ncu profile-hta

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
	rm -f nsys_* ncu_* hta_* perf_* comprehensive_*
