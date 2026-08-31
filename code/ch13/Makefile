# Chapter 13 Makefile - PyTorch training optimizations with architecture-aware builds

include ../core/common/cuda_arch.mk

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) -lineinfo $(CUDA_NVTX_CFLAGS)
SUFFIX = $(TARGET_SUFFIX)

# Python-focused chapter - no CUDA compilation targets currently
BASE_TARGETS =
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean test compare b200 gb10 b300 gb300 profile-baseline profile-tuned

all:
	@echo "Chapter 13: PyTorch training optimizations (Python-focused)"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ No CUDA binaries to build - run Python scripts directly"

test: all
	@echo "Testing Chapter 13 Python scripts..."
	@echo "Run individual Python scripts manually"

compare:
	@echo "Chapter 13 is Python-focused - no architecture-specific builds needed"

b200:
	$(MAKE) ARCH=sm_100 all
	@echo "✓ Configured for B200 (Python scripts use runtime architecture detection)"

gb10:
	$(MAKE) ARCH=sm_121 all
	@echo "✓ Configured for GB10 (Python scripts use runtime architecture detection)"

b300:
	$(MAKE) ARCH=sm_103 all
	@echo "✓ Configured for B300 (Python scripts use runtime architecture detection)"

gb300:
	$(MAKE) ARCH=sm_103 all
	@echo "✓ Configured for GB300 (Python scripts use runtime architecture detection)"

clean:
	rm -f *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*
	rm -f __pycache__/*.pyc
	@echo "Cleaned ch13 artifacts"

PROFILE_CMD_BASELINE ?= python ch13/train.py
PROFILE_CMD_TUNED ?= python ch13/train.py

profile-baseline:
	@cd .. && KERNEL_REGEX=".*" ./profile.sh baseline $(PROFILE_CMD_BASELINE)

profile-tuned:
	@cd .. && KERNEL_REGEX=".*" ./profile.sh tuned $(PROFILE_CMD_TUNED)
