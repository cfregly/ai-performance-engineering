# Ozaki scheme lab Makefile (B200 / Blackwell focused)

include ../../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) $(CUDA_NVTX_CFLAGS)

CUDA_SOURCES = \
	baseline_ozaki_scheme.cu \
	optimized_ozaki_scheme_dynamic.cu \
	optimized_ozaki_scheme_fixed.cu
BASE_TARGETS = $(CUDA_SOURCES:.cu=)
TARGETS = $(addsuffix $(TARGET_SUFFIX), $(BASE_TARGETS))
VERIFY_TARGETS = $(addsuffix _verify$(TARGET_SUFFIX), $(BASE_TARGETS))
PROFILE_TARGETS = $(addsuffix _profile$(TARGET_SUFFIX), $(BASE_TARGETS))

.PHONY: all clean b200

all: $(TARGETS)
	@echo "Built Ozaki scheme lab binaries for $(ARCH) ($(ARCH_NAME))"

$(addsuffix $(TARGET_SUFFIX), $(BASE_TARGETS)): %$(TARGET_SUFFIX): %.cu ozaki_scheme_common.cuh accuracy.h $(CUDA_NVTX_DEPS)
	@echo "Building $@..."
	$(NVCC) $(NVCC_FLAGS) -lcublas -lcublasLt -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

$(VERIFY_TARGETS): %_verify$(TARGET_SUFFIX): %.cu ozaki_scheme_common.cuh accuracy.h $(CUDA_NVTX_DEPS)
	@echo "Building $@ (verify)..."
	$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 -lcublas -lcublasLt -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

# Kept separate from ordinary timing binaries so profile instrumentation cannot
# silently replace a cached timing build. The harness supplies NVTX_ENABLED=1.
$(PROFILE_TARGETS): %_profile$(TARGET_SUFFIX): %.cu ozaki_scheme_common.cuh accuracy.h ../../core/common/nvtx_utils.cuh $(CUDA_NVTX_DEPS)
	@test "$(NVTX_ENABLED)" = "1" || (echo "Profiling builds require NVTX_ENABLED=1"; exit 1)
	$(NVCC) $(NVCC_FLAGS) -lcublas -lcublasLt -lcuda -o $@ $< $(CUDA_NVTX_LDFLAGS)

b200:
	$(MAKE) ARCH=sm_100 all

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
