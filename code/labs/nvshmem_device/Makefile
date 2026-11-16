# Native CUDA/NVSHMEM device lab. No communication-free fallback is built.
.DEFAULT_GOAL := all
USE_ARCH_SUFFIX := 0
NVTX_ENABLED ?= 1
include ../../core/common/cuda_arch.mk

MPICXX ?= mpicxx
sh_quote = '$(subst ','"'"',$(1))'
ifeq ($(origin MPI_COMPILE_FLAGS),undefined)
MPI_COMPILE_PROBE := $(shell $(call sh_quote,$(MPICXX)) --showme:compile 2>/dev/null && printf ' %s' AISP_MPI_COMPILE_FLAGS_OK)
ifeq ($(filter AISP_MPI_COMPILE_FLAGS_OK,$(MPI_COMPILE_PROBE)),)
$(error MPI include discovery requires Open MPI mpicxx --showme:compile or explicit MPI_COMPILE_FLAGS)
endif
MPI_COMPILE_FLAGS := $(filter-out AISP_MPI_COMPILE_FLAGS_OK,$(MPI_COMPILE_PROBE))
endif
ifndef NVSHMEM_HOME
$(error This lab requires explicit NVSHMEM_HOME and actual NVSHMEM host/device libraries)
endif
NVSHMEM_INCLUDE ?= $(NVSHMEM_HOME)/include
NVSHMEM_LIBDIR ?= $(NVSHMEM_HOME)/lib
NVCC_FLAGS ?= $(CUDA_NVCC_ARCH_FLAGS) $(CUDA_NVTX_CFLAGS) -std=c++17 -O3 -lineinfo -rdc=true
NVSHMEM_FLAGS ?= -I$(NVSHMEM_INCLUDE)
NVSHMEM_LIBS ?= -L$(NVSHMEM_LIBDIR) -lnvshmem_host $(NVSHMEM_LIBDIR)/libnvshmem_device.a -lcuda

.PHONY: all nvshmem_device_native
all: nvshmem_device_native

# Rebuild on every request so changed compiler/library options cannot reuse a stale binary.
nvshmem_device_native: nvshmem_device_native.cu ../multi_gpu_communication/jacobi_contract.hpp Makefile build_receipt.py $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(MPI_COMPILE_FLAGS) $(NVSHMEM_FLAGS) -ccbin=$(MPICXX) nvshmem_device_native.cu -o $@ $(NVSHMEM_LIBS) $(CUDA_NVTX_LDFLAGS)
	$(PYTHON) build_receipt.py --binary $@ --arch $(ARCH) --nvcc $(NVCC) --mpicxx $(MPICXX) --build-option $(call sh_quote,nvcc_flags=$(NVCC_FLAGS)) --build-option $(call sh_quote,mpi_compile_flags=$(MPI_COMPILE_FLAGS)) --build-option $(call sh_quote,nvshmem_flags=$(NVSHMEM_FLAGS)) --build-option $(call sh_quote,nvshmem_libs=$(NVSHMEM_LIBS)) --build-option $(call sh_quote,nvtx_libs=$(CUDA_NVTX_LDFLAGS))
