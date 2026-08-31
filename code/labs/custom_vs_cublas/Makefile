# Makefile for Matching cuBLAS Lab

.DEFAULT_GOAL := all
include ../../core/common/cuda_arch.mk

# This target contains the plain GEMM baseline. The tcgen05 Python extensions
# have their own build path and require architecture-specific 100a/103a code.
NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -lineinfo

TARGETS = kernels.so

all: $(TARGETS)

kernels.so: kernels.cu
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $<

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
