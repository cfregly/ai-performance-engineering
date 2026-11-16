# Native CUDA/MPI lab. NCCL can be excluded for an explicit MPI-only build.
.DEFAULT_GOAL := all
USE_ARCH_SUFFIX := 0
include ../../core/common/cuda_arch.mk

MPICXX ?= mpicxx
sh_quote = '$(subst ','"'"',$(1))'
# nvcc preprocesses CUDA before invoking its host compiler wrapper.
# Automatic discovery supports Open MPI. Other wrappers need an explicit override.
ifeq ($(origin MPI_COMPILE_FLAGS),undefined)
MPI_COMPILE_PROBE := $(shell $(call sh_quote,$(MPICXX)) --showme:compile 2>/dev/null && printf ' %s' AISP_MPI_COMPILE_FLAGS_OK)
ifeq ($(filter AISP_MPI_COMPILE_FLAGS_OK,$(MPI_COMPILE_PROBE)),)
$(error MPI flag discovery requires Open MPI mpicxx --showme:compile. Set MPI_COMPILE_FLAGS explicitly for another wrapper)
endif
MPI_COMPILE_FLAGS := $(filter-out AISP_MPI_COMPILE_FLAGS_OK,$(MPI_COMPILE_PROBE))
endif
WITH_NCCL ?= 1
WITH_NVSHMEM ?= 0
NCCL_HOME ?= /usr
NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) $(CUDA_NVTX_CFLAGS) -std=c++17 -lineinfo -O3
NCCL_FLAGS =
NCCL_LIBS =
ifeq ($(WITH_NCCL),1)
NCCL_FLAGS = -DAISP_HAVE_NCCL=1 -I$(NCCL_HOME)/include
NCCL_LIBS = -L$(NCCL_HOME)/lib -L$(NCCL_HOME)/lib64 -lnccl
endif
NVSHMEM_FLAGS =
NVSHMEM_LIBS =
ifeq ($(WITH_NVSHMEM),1)
ifndef NVSHMEM_HOME
$(error WITH_NVSHMEM=1 requires explicit NVSHMEM_HOME)
endif
NVSHMEM_INCLUDE ?= $(NVSHMEM_HOME)/include
NVSHMEM_LIBDIR ?= $(NVSHMEM_HOME)/lib
NVSHMEM_FLAGS = -DAISP_HAVE_NVSHMEM=1 -I$(NVSHMEM_INCLUDE) -rdc=true
NVSHMEM_LIBS = -L$(NVSHMEM_LIBDIR) -lnvshmem_host $(NVSHMEM_LIBDIR)/libnvshmem_device.a -lcuda
endif

.PHONY: all jacobi_native
all: jacobi_native

jacobi_native: jacobi_native.cu jacobi_contract.hpp Makefile build_receipt.py $(CUDA_NVTX_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(MPI_COMPILE_FLAGS) $(NCCL_FLAGS) $(NVSHMEM_FLAGS) -ccbin=$(MPICXX) jacobi_native.cu -o $@ $(NCCL_LIBS) $(NVSHMEM_LIBS) $(CUDA_NVTX_LDFLAGS)
	$(PYTHON) build_receipt.py --binary $@ --arch $(ARCH) --with-nccl $(WITH_NCCL) --with-nvshmem $(WITH_NVSHMEM) --nvcc $(NVCC) --mpicxx $(MPICXX) --build-option $(call sh_quote,nvcc_flags=$(NVCC_FLAGS)) --build-option $(call sh_quote,mpi_compile_flags=$(MPI_COMPILE_FLAGS)) --build-option $(call sh_quote,nccl_flags=$(NCCL_FLAGS)) --build-option $(call sh_quote,nccl_libs=$(NCCL_LIBS)) --build-option $(call sh_quote,nvshmem_flags=$(NVSHMEM_FLAGS)) --build-option $(call sh_quote,nvshmem_libs=$(NVSHMEM_LIBS)) --build-option $(call sh_quote,nvtx_libs=$(CUDA_NVTX_LDFLAGS))
