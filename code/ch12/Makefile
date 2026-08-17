# Chapter 12 Makefile - CUDA graph and dynamic parallelism examples
# Auto-builds all *.cu files

include ../core/common/cuda_arch.mk

NVCC_ALLOW_UNSUPPORTED = --allow-unsupported-compiler
NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) $(CUDA_NVTX_CFLAGS) $(NVCC_ALLOW_UNSUPPORTED)
NVCC_LIBS = -lcuda $(CUDA_NVTX_LDFLAGS)
SUFFIX = $(TARGET_SUFFIX)

# Auto-detect all .cu files and derive targets
CUDA_SOURCES := $(wildcard *.cu)
# Filter out helper files (dependencies, not standalone executables)
HELPER_SOURCES := $(wildcard helper_*.cu)
STANDALONE_SOURCES := $(filter-out $(HELPER_SOURCES),$(CUDA_SOURCES))
BASE_TARGETS := $(STANDALONE_SOURCES:.cu=)
TARGETS := $(addsuffix $(SUFFIX), $(BASE_TARGETS))
VERIFY_TARGETS := $(addsuffix _verify$(SUFFIX), $(BASE_TARGETS))

$(TARGETS): $(CUDA_NVTX_DEPS)
$(VERIFY_TARGETS): $(CUDA_NVTX_DEPS)

.PHONY: all clean compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all

all: $(TARGETS)
	@echo "Building Chapter 12 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

# Default pattern rule
%$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

# Verify binaries build with -DVERIFY=1 for harness checksum parsing
%_verify$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 -o $@ $< $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

# Special rules for files needing -rdc=true (dynamic parallelism / device-side launch)
optimized_uneven_static$(SUFFIX): optimized_uneven_static.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

optimized_uneven_static_verify$(SUFFIX): optimized_uneven_static.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 -o $@ $< -lcudadevrt $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

optimized_dynamic_parallelism_host$(SUFFIX): optimized_dynamic_parallelism_host.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

optimized_dynamic_parallelism_host_verify$(SUFFIX): optimized_dynamic_parallelism_host.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 -o $@ $< -lcudadevrt $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

baseline_dynamic_parallelism_device$(SUFFIX): baseline_dynamic_parallelism_device.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

baseline_dynamic_parallelism_device_verify$(SUFFIX): baseline_dynamic_parallelism_device.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 -o $@ $< -lcudadevrt $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

optimized_dynamic_parallelism_device$(SUFFIX): optimized_dynamic_parallelism_device.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

optimized_dynamic_parallelism_device_verify$(SUFFIX): optimized_dynamic_parallelism_device.cu $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) -rdc=true -DVERIFY=1 -o $@ $< -lcudadevrt $(NVCC_LIBS) $(CUDA_NVTX_LDFLAGS)

# Architecture targets
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
		$(MAKE) ARCH=$${arch} all; \
	done

profile-nsys: $(TARGETS)
	@for t in $(TARGETS); do nsys profile --force-overwrite=true -t cuda,nvtx -o nsys_$$t ./$$t 2>/dev/null || true; done

profile-ncu: $(TARGETS)
	@for t in $(TARGETS); do ncu --metrics achieved_occupancy -o ncu_$$t ./$$t 2>/dev/null || true; done

profile-hta: $(TARGETS)
	@for t in $(TARGETS); do nsys profile --force-overwrite=true -t cuda,nvtx,nccl -o hta_$$t ./$$t 2>/dev/null || true; done

profile-perf: $(TARGETS)
	@for t in $(TARGETS); do perf record -g -o perf_$$t ./$$t || true; done

profile-all: profile-nsys profile-ncu profile-hta

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
	rm -f nsys_* ncu_* hta_* perf_* comprehensive_*
