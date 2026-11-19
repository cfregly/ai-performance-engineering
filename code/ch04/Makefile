# Makefile for Chapter 4: Multi-GPU Communication Examples
# Supports dual-architecture builds for Blackwell B200 and Grace-Blackwell GB10.

include ../core/common/cuda_arch.mk

NVSHMEM_HOME ?= $(if $(wildcard /usr/lib/x86_64-linux-gnu/libnvshmem_host.so),/usr/lib/x86_64-linux-gnu,/usr/lib/aarch64-linux-gnu)
NVSHMEM_INCLUDE = -I/usr/include/nvshmem
NVSHMEM_LIB = -L$(NVSHMEM_HOME) -lnvshmem_host $(NVSHMEM_HOME)/libnvshmem_device.a
HAS_NVSHMEM := $(shell test -f $(NVSHMEM_HOME)/libnvshmem_host.so && echo yes || echo no)

# NCCL 2.27.3 symmetric-window benchmark. This target never builds a fallback.
MPICXX ?= mpicxx
sh_quote = '$(subst ','"'"',$(1))'
MPI_COMPILE_FLAGS ?= $(shell $(call sh_quote,$(MPICXX)) --showme:compile 2>/dev/null)
NCCL_HOME ?= /usr
NCCL_INCLUDE ?= $(NCCL_HOME)/include
NCCL_LIBDIR ?= $(NCCL_HOME)/lib
NCCL_MULTIARCH ?= $(shell $(call sh_quote,$(MPICXX)) -print-multiarch 2>/dev/null)
NCCL_FLAGS ?= -I$(NCCL_INCLUDE)
NCCL_LIBS ?= -L$(NCCL_LIBDIR) $(if $(strip $(NCCL_MULTIARCH)),-L$(NCCL_LIBDIR)/$(NCCL_MULTIARCH)) -L$(NCCL_LIBDIR)64 -lnccl -ldl
NCCL_SYMMETRIC_NVCC_FLAGS ?= $(CUDA_NVCC_ARCH_FLAGS) $(CUDA_NVTX_CFLAGS) -std=c++17 -O3 -lineinfo
NCCL_SYMMETRIC_CUSTODY_DEPS = ../core/common/cuda_arch.mk \
	../core/common/nvtx_utils.cuh ../core/common/nvtx_label_utils.cuh

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -rdc=true $(CUDA_NVTX_CFLAGS)
SUFFIX = $(TARGET_SUFFIX)

NVSHMEM_TARGETS = nvshmem_multigpu_examples nvshmem_advanced_multigpu nvshmem_multinode_example \
                  nvshmem_pipeline_patterns nvshmem_tensor_parallel
ifeq ($(HAS_NVSHMEM),yes)
NVSHMEM_TARGETS += nvshmem_ibgda_microbench
endif
TARGETS = $(addsuffix $(SUFFIX), $(NVSHMEM_TARGETS))

.DELETE_ON_ERROR:
.PHONY: all help nvshmem nccl_symmetric_memory_force compare clean install-help b200 gb10 b300 gb300 $(TARGETS)

all: help
	@echo ""
	@echo "NVSHMEM status: $(HAS_NVSHMEM)"
ifeq ($(HAS_NVSHMEM),no)
	@echo "  → NVSHMEM not found at $(NVSHMEM_HOME)"
	@echo "  → Set NVSHMEM_HOME to the correct path if installed"
endif

help:
	@echo "Available targets:"
	@echo "  make nvshmem         - Compile NVSHMEM examples (requires NVSHMEM)"
	@echo "  make nccl_symmetric_memory_native - Compile the NCCL 2.27.3 symmetric-window benchmark"
	@echo "  make compare         - Build both architectures back-to-back"
	@echo "  make clean           - Remove build outputs"
	@echo "  make install-help    - Show NVSHMEM installation tips"
	@echo ""
	@echo "Current configuration:"
	@echo "  ARCH        = $(ARCH)"
	@echo "  ARCH_NAME   = $(ARCH_NAME)"
	@echo "  Binary suffix = $(SUFFIX)"

nvshmem: $(TARGETS)
ifeq ($(HAS_NVSHMEM),yes)
	@echo "✓ NVSHMEM examples compiled successfully"
else
	@echo "⚠ NVSHMEM not found - compiling in educational mode"
endif

nccl_symmetric_memory_force:

nccl_symmetric_memory_native: nccl_symmetric_memory_force nccl_symmetric_memory_native.cu nccl_symmetric_memory_build_receipt.py Makefile $(NCCL_SYMMETRIC_CUSTODY_DEPS) $(CUDA_NVTX_DEPS)
	@test -n "$(MPI_COMPILE_FLAGS)" || (echo "MPI flag discovery requires Open MPI mpicxx --showme:compile or explicit MPI_COMPILE_FLAGS" >&2; false)
	$(PYTHON) nccl_symmetric_memory_build_receipt.py --binary $@ --compile --arch $(ARCH) --nvcc $(NVCC) --mpicxx $(MPICXX) --nccl-home $(call sh_quote,$(NCCL_HOME)) --build-option $(call sh_quote,nvcc_flags=$(NCCL_SYMMETRIC_NVCC_FLAGS)) --build-option $(call sh_quote,mpi_compile_flags=$(MPI_COMPILE_FLAGS)) --build-option $(call sh_quote,nccl_home=$(NCCL_HOME)) --build-option $(call sh_quote,nccl_flags=$(NCCL_FLAGS)) --build-option $(call sh_quote,nccl_libs=$(NCCL_LIBS)) --build-option $(call sh_quote,nvtx_libs=$(CUDA_NVTX_LDFLAGS))

nvshmem_multigpu_examples$(SUFFIX): nvshmem_multigpu_examples.cu $(CUDA_NVTX_DEPS)
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "✓ Basic examples compiled with NVSHMEM support"
	@echo "  Run: nvshmrun -np 4 ./nvshmem_multigpu_examples$(SUFFIX)"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "Compiled in educational mode (no NVSHMEM library)"
endif

nvshmem_multinode_example$(SUFFIX): nvshmem_multinode_example.cu $(CUDA_NVTX_DEPS)
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "✓ Multi-node example compiled with NVSHMEM support"
	@echo "  Run: nvshmrun -np <world> ./nvshmem_multinode_example$(SUFFIX) --gpus-per-node 4"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "Compiled in educational mode (no NVSHMEM library)"
endif

nvshmem_pipeline_patterns$(SUFFIX): nvshmem_pipeline_patterns.cu $(CUDA_NVTX_DEPS)
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "✓ Pipeline patterns compiled with NVSHMEM support"
	@echo "  Run: nvshmrun -np 2 ./nvshmem_pipeline_patterns$(SUFFIX)"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "Compiled in educational mode (no NVSHMEM library)"
endif

nvshmem_advanced_multigpu$(SUFFIX): nvshmem_advanced_multigpu.cu $(CUDA_NVTX_DEPS)
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "✓ Advanced patterns compiled with NVSHMEM support"
	@echo "  Run: nvshmrun -np 4 ./nvshmem_advanced_multigpu$(SUFFIX)"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "Compiled in educational mode (no NVSHMEM library)"
endif

nvshmem_tensor_parallel$(SUFFIX): nvshmem_tensor_parallel.cu $(CUDA_NVTX_DEPS)
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -lcuda -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "✓ Tensor parallel kernels compiled with NVSHMEM support"
	@echo "  Run: nvshmrun -np 4 ./nvshmem_tensor_parallel$(SUFFIX) --test all"
else
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "Compiled in educational mode (no NVSHMEM library)"
	@echo "This version shows API patterns but won't execute communication"
endif

nvshmem_ibgda_microbench$(SUFFIX): nvshmem_ibgda_microbench.cu $(CUDA_NVTX_DEPS)
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "✓ IBGDA microbenchmark compiled with NVSHMEM support"
	@echo "  Run: nvshmrun -np 2 ./nvshmem_ibgda_microbench$(SUFFIX) --mode=p --bytes=1024 --ctas=8 --iters=1000"
else
	@echo "Skipping nvshmem_ibgda_microbench: NVSHMEM not found (set NVSHMEM_HOME then re-run make)"
	@false
endif

nvshmem_ibgda_microbench_verify$(SUFFIX): nvshmem_ibgda_microbench.cu $(CUDA_NVTX_DEPS)
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(NVCC_FLAGS) -DUSE_NVSHMEM -DVERIFY=1 $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@ $(CUDA_NVTX_LDFLAGS)
	@echo "✓ IBGDA microbenchmark verify binary compiled"
else
	@echo "Skipping nvshmem_ibgda_microbench verify: NVSHMEM not found (set NVSHMEM_HOME then re-run make)"
	@false
endif

compare: clean
	@echo "========================================"
	@echo "Building Chapter 4 NVSHMEM samples for both architectures..."
	@echo "========================================"
	@set -e; for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} nvshmem; \
	done

b200:
	$(MAKE) ARCH=sm_100 clean
	$(MAKE) ARCH=sm_100 nvshmem

gb10:
	$(MAKE) ARCH=sm_121 clean
	$(MAKE) ARCH=sm_121 nvshmem

b300:
	$(MAKE) ARCH=sm_103 clean
	$(MAKE) ARCH=sm_103 nvshmem

gb300:
	$(MAKE) ARCH=sm_103 clean
	$(MAKE) ARCH=sm_103 nvshmem

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123
	rm -f nccl_symmetric_memory_native nccl_symmetric_memory_native.build.json
	rm -f *.o
	@echo "Cleaned Chapter 4 NVSHMEM binaries"

install-help:
	@echo "To install NVSHMEM:"
	@echo "  1. Download from NVIDIA: https://developer.nvidia.com/nvshmem"
	@echo "  2. Extract and install to $(NVSHMEM_HOME)"
	@echo "  3. Set environment variable: export NVSHMEM_HOME=$(NVSHMEM_HOME)"
	@echo "  4. Run: make nvshmem"
