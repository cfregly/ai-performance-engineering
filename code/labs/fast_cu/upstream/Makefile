NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

NVFP4_NVCC ?= /usr/local/cuda/bin/nvcc
NVFP4_NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG \
	-gencode arch=compute_103a,code=sm_103a
NVFP4_LDFLAGS = -lcublasLt -lcuda
NVFP4_DIR = gb300/nvfp4
NVFP4_SOURCE = $(NVFP4_DIR)/main.cu
NVFP4_RUNGS = 0 1 2 3 4 5 6 7 8 9
NVFP4_TARGETS = $(addprefix nvfp4-r,$(NVFP4_RUNGS))

sum: h100/sum.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

sumprofile: sum
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

matmul: h100/matmul.cu
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

matmulprofile: matmul
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

nvfp4: $(OUT_DIR)/nvfp4

$(OUT_DIR)/nvfp4: $(NVFP4_SOURCE) $(NVFP4_DIR)/gemm9.cuh | $(OUT_DIR)
	$(NVFP4_NVCC) $(NVFP4_NVCC_FLAGS) $< -o $@ $(NVFP4_LDFLAGS)

$(NVFP4_TARGETS): nvfp4-r%: $(OUT_DIR)/nvfp4-r%

$(OUT_DIR)/nvfp4-r%: $(NVFP4_SOURCE) $(NVFP4_DIR)/gemm%.cuh | $(OUT_DIR)
	$(NVFP4_NVCC) $(NVFP4_NVCC_FLAGS) \
		-DNVFP4_GEMM_HEADER='"gemm$*.cuh"' \
		$< -o $@ $(NVFP4_LDFLAGS)

nvfp4-ladder: $(NVFP4_TARGETS)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

clean:
	rm $(OUT_DIR)/*

.PHONY: nvfp4 nvfp4-ladder $(NVFP4_TARGETS)
