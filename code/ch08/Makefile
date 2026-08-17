# Chapter 8 Makefile - Control-flow and occupancy tuning examples

include ../core/common/cuda_arch.mk

NVCC_LINEINFO ?= -lineinfo
PTXAS_WARN ?= -Xptxas=-warn-spills

ifeq ($(strip $(MAXRREGCOUNT)),)
MAXRREGCOUNT_FLAG :=
else
MAXRREGCOUNT_FLAG := -maxrregcount=$(MAXRREGCOUNT)
endif

NVCC_FLAGS = $(CUDA_NVCC_ARCH_FLAGS) $(NVCC_LINEINFO) --threads=0 $(PTXAS_WARN) $(MAXRREGCOUNT_FLAG) -I../core/common/headers $(CUDA_NVTX_CFLAGS)
SUFFIX = $(TARGET_SUFFIX)

EXTENSION_SOURCES = $(wildcard *_kernels.cu)
CUDA_SOURCES = $(filter-out %_kernels.cu %_tcgen05.cu,$(wildcard *.cu))
BASE_TARGETS = $(CUDA_SOURCES:.cu=)
TARGETS = $(addsuffix $(SUFFIX), $(BASE_TARGETS))
VERIFY_TARGETS = $(addsuffix _verify$(SUFFIX), $(BASE_TARGETS))

.PHONY: all clean compare b200 gb10 b300 gb300 profile-nsys profile-ncu profile-hta profile-perf profile-all $(TARGETS) $(VERIFY_TARGETS)

all: $(TARGETS)
	@echo "Building Chapter 8 examples with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting $(ARCH_NAME)"
	@echo "✓ Built $(words $(BASE_TARGETS)) CUDA examples"

$(TARGETS): %$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	@echo "Compiling $@..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcuda $(CUDA_NVTX_LDFLAGS)

# Verify binaries build with -DVERIFY=1 for harness checksum parsing
$(VERIFY_TARGETS): %_verify$(SUFFIX): %.cu $(CUDA_NVTX_DEPS)
	@echo "Compiling verify binary $@..."
	$(NVCC) $(NVCC_FLAGS) -DVERIFY=1 -o $@ $< -lcuda $(CUDA_NVTX_LDFLAGS)

compare: clean
	@echo "========================================"
	@echo "Building Chapter 8 samples for both architectures..."
	@echo "========================================"
	@set -e; for arch in $(ARCH_LIST); do \
		echo ""; \
		echo ">>> Building $${arch}..."; \
		$(MAKE) ARCH=$${arch} all; \
	done

b200: ARCH=sm_100
b200: clean all

gb10: ARCH=sm_121
gb10: clean all

b300: ARCH=sm_103
b300: clean all

gb300: ARCH=sm_103
gb300: clean all

profile-nsys: all
	@echo "Nsight Systems timeline profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,python --capture-range=nvtx -o nsys_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-ncu: all
	@echo "Nsight Compute kernel profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		ncu --section=SpeedOfLight --metrics breakdown:gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --target-processes all -o ncu_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-hta: all
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,python --capture-range=nvtx -o hta_profile_$${exe}_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-perf: all
	@echo "Perf system-level profiling..."
	@for exe in $(TARGETS); do \
		echo "Profiling $$exe with perf..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || echo "perf failed for $$exe"; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

profile-all: all
	@echo "Comprehensive profiling with all tools..."
	@for exe in $(TARGETS); do \
		echo "1. Nsight Systems timeline..."; \
		nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,python --capture-range=nvtx -o comprehensive_timeline_$${exe}_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --section=SpeedOfLight --metrics breakdown:gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --target-processes all -o comprehensive_kernel_$${exe}_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "3. Perf system-level analysis..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || true; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

clean:
	rm -f *_sm100 *_sm103 *_sm120 *_sm121 *_sm122 *_sm123 *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_* *_verify_sm103 *_verify_sm100 *_verify_sm121
