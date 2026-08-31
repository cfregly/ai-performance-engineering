# Lab NVFP4 GEMM Makefile (SM100+)

include ../../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) $(CUDA_NVTX_CFLAGS)
CUTLASS_ROOT := ../../third_party/cutlass
CUTLASS_INCLUDES := -I$(CUTLASS_ROOT)/include -I$(CUTLASS_ROOT)/tools/util/include -I$(CUTLASS_ROOT)/examples/common

CUDA_SOURCES = baseline_nvfp4_gemm.cu optimized_nvfp4_gemm.cu
BASE_TARGETS = $(CUDA_SOURCES:.cu=)
TARGETS = $(addsuffix $(TARGET_SUFFIX), $(BASE_TARGETS))
VERIFY_TARGETS = $(addsuffix _verify$(TARGET_SUFFIX), $(BASE_TARGETS))

.PHONY: all clean b200

all: $(TARGETS)
	@echo "Built NVFP4 GEMM lab binaries for $(ARCH) ($(ARCH_NAME))"

$(addsuffix $(TARGET_SUFFIX), $(BASE_TARGETS)): %$(TARGET_SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	@echo "Building $@..."
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

$(VERIFY_TARGETS): %_verify$(TARGET_SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	@echo "Building $@ (verify)..."
	$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 $(CUTLASS_INCLUDES) -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

b200:
	$(MAKE) ARCH=sm_100 clean
	$(MAKE) ARCH=sm_100 all

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
