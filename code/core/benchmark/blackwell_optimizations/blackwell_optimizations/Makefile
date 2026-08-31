# Makefile for Blackwell Optimization Tests
# Targets: B200 (SM 10.0)

NVCC = nvcc
ARCH = -gencode arch=compute_100,code=sm_100
CXXFLAGS = -std=c++20 -O3
NVCCFLAGS = $(ARCH) $(CXXFLAGS) -lcuda -lcudart -lineinfo

# Blackwell-specific flags
BLACKWELL_FLAGS = -DENABLE_BLACKWELL=1

all: test_tmem test_tma test_clusters test_fp8 test_warp_spec test_dsmem test_all_features

test_tmem: test_tmem.cu
	$(NVCC) $(NVCCFLAGS) $(BLACKWELL_FLAGS) -o test_tmem test_tmem.cu

test_tma: test_tma.cu
	$(NVCC) $(NVCCFLAGS) $(BLACKWELL_FLAGS) -o test_tma test_tma.cu

test_clusters: test_clusters.cu
	$(NVCC) $(NVCCFLAGS) $(BLACKWELL_FLAGS) -o test_clusters test_clusters.cu

test_fp8: test_fp8.cu
	$(NVCC) $(NVCCFLAGS) $(BLACKWELL_FLAGS) -o test_fp8 test_fp8.cu

test_warp_spec: test_warp_spec.cu
	$(NVCC) $(NVCCFLAGS) $(BLACKWELL_FLAGS) -o test_warp_spec test_warp_spec.cu

test_dsmem: test_dsmem.cu
	$(NVCC) $(NVCCFLAGS) $(BLACKWELL_FLAGS) -o test_dsmem test_dsmem.cu

test_all_features: test_all_features.cu
	$(NVCC) $(NVCCFLAGS) $(BLACKWELL_FLAGS) -o test_all_features test_all_features.cu

clean:
	rm -f test_tmem test_tma test_clusters test_fp8 test_warp_spec test_dsmem test_all_features *.o

run_all: all
	@echo "=== Running All Blackwell Optimization Tests ==="
	@echo ""
	@echo "--- TMEM Test ---"
	@./test_tmem
	@echo ""
	@echo "--- Shared-memory scalar control (legacy test_tma) ---"
	@./test_tma
	@echo ""
	@echo "--- CTA Clusters Test ---"
	@./test_clusters
	@echo ""
	@echo "--- FP8 Test ---"
	@./test_fp8
	@echo ""
	@echo "--- Warp Specialization Test ---"
	@./test_warp_spec
	@echo ""
	@echo "--- Distributed Shared Memory Test ---"
	@./test_dsmem
	@echo ""
	@echo "--- Cluster / FP8 storage scalar control ---"
	@./test_all_features

.PHONY: all clean run_all
