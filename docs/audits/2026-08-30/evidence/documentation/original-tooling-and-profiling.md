# Tooling and Profiling Guide

This guide consolidates the repository’s profiling, tooling, and automation
information. It preserves the detailed instructions that previously lived in
the root `README.md` while keeping the top-level README focused on the high-
level story.

## Automated Profiling Harness

All chapter examples can be profiled through a unified harness:

```bash
# List all available examples
python scripts/profile_harness.py --list

# Profile a specific example with multiple tools
python scripts/master_profile.py ch10_warp_specialized_pipeline --profile nsys ncu

# Profile all examples
python scripts/profile_harness.py --profile all

# Shortcut wrappers
./start.sh    # Launch harness across all chapters
./stop.sh     # Terminate active runs
./clean_profiles.sh  # Remove accumulated artifacts
```

## NVIDIA Nsight Systems (Timeline Analysis)

```bash
nsys profile -t cuda,nvtx,osrt,triton -o timeline_profile python script.py
```

**Key Metrics**

- GPU utilization timeline
- Memory transfer patterns
- Kernel launch overhead
- CUDA stream overlap
- Multi-GPU communication patterns

## NVIDIA Nsight Compute (Kernel Analysis)

```bash
ncu --metrics achieved_occupancy,warp_execution_efficiency -o kernel_profile python script.py
```

**Key Metrics**

- Achieved occupancy
- Warp execution efficiency
- Memory throughput
- Compute utilization
- Register usage
- SM % Peak utilization
- DRAM throughput
- L2 hit rate

## Holistic Tracing Analysis (HTA)

```bash
nsys profile -t cuda,nvtx,osrt,cudnn,cublas,nccl,triton -o hta_profile python script.py
```

**Key Metrics**

- Multi-GPU communication patterns
- NCCL collective operation efficiency
- Load balancing across GPUs
- Memory bandwidth distribution
- Inter-GPU synchronization

## PyTorch Profiler (Framework-Level)

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    profile_memory=True,
) as prof:
    # Your code here
```

**Key Metrics**

- Operator execution time
- Memory allocation patterns
- FLOP counts
- Call stack analysis
- Module-level performance

## perf (System-Level Analysis)

```bash
perf record -g -p $(pgrep python) -o perf.data
perf report -i perf.data
```

**Key Metrics**

- CPU utilization
- Cache miss rates
- System call overhead
- Memory access patterns

## Profiling Output Analysis

Extract metrics for book tables and deep dives:

```bash
# Nsight Compute metrics
python tools/extract_ncu_subset.py 'output/reports/*.csv'

# Nsight Systems summary
python tools/extract_nsys_summary.py 'output/traces/*.nsys-rep'

# PyTorch profiler data
python tools/extract_pytorch_profile.py 'profiles/*/pytorch_*/*'

# Or run the automated extraction script
./extract.sh
```

## Example Profiling Workflows

### 1. Basic Performance Analysis

```bash
# Run performance basics
python3 code/ch1/performance_basics.py

# Profile with Nsight Systems
nsys profile -t cuda,nvtx,osrt -o perf_basics python3 code/ch1/performance_basics.py

# Analyze results
nsys stats perf_basics.nsys-rep
```

### 2. Hardware Optimization

```bash
# Check hardware capabilities
python3 code/ch2/hardware_info.py

# Test NUMA binding
python3 code/ch3/bind_numa_affinity.py

# Profile memory access patterns
ncu --metrics memory_throughput -o memory_profile python3 code/ch7/memory_optimization.py
```

### 3. PyTorch Compilation Analysis

```bash
# Test torch.compile performance
python3 code/ch14/torch_compiler_examples.py

# Profile compilation overhead
nsys profile -t cuda,nvtx,osrt -o compile_profile python3 code/ch14/torch_compiler_examples.py

# Analyze kernel performance
ncu --metrics achieved_occupancy -o kernel_profile python3 code/ch14/torch_compiler_examples.py
```

### 4. Triton Kernel Development

```bash
# Test Triton kernels
python3 code/ch14/triton_examples.py

# Profile custom kernels
ncu --metrics triton_kernel_efficiency -o triton_profile python3 code/ch14/triton_examples.py
```

## Tools and Utilities

### Essential Tools Directory (`tools/`)

- `extract_ncu_subset.py`: Collate Nsight Compute CSV metrics for manuscript tables
- `extract_nsys_summary.py`: Extract Nsight Systems timeline summaries
- `extract_pytorch_profile.py`: Process PyTorch profiler output data

### Archive Directory (`archive/`)

- `build_all.sh`: Build CUDA samples and validate Python syntax
- `update_blackwell_requirements.sh`: Update all requirements files
- `update_cuda_versions.sh`: Normalize Makefiles for Blackwell
- `comprehensive_profiling.py`: Demo of all profiling tools
- `generate_example_inventory.py`: Generate chapter-by-chapter catalog
- `run_all_examples.sh`: Execute all chapter examples
- `compare_nsight/`: Nsight Systems vs Compute comparison tools
- `clean_profiles.sh`: Clean accumulated profiling artifacts
- `assert.sh`: Helpful information about extraction tools

### Scripts Directory (`scripts/`)

- `profile_harness.py`: Unified profiling harness for all examples
- `master_profile.py`: Master profiling script with multiple tool support
- `example_registry.py`: Registry of all chapter examples and their metadata
- `ncu_profile.sh`, `nsys_profile.sh`, `perf_profile.sh`: CLI shortcuts for Nsight and perf

## Troubleshooting

### Common Issues

1. **Setup Script Failures**

   ```bash
   # Check permissions
   sudo ./setup.sh

   # Verify GPU detection
   nvidia-smi
   ```

2. **CUDA Version Mismatch**

   ```bash
   # Check CUDA version
   nvcc --version

   # Verify PyTorch CUDA support
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Memory Issues**

   ```bash
   # Check available memory
   free -h

   # Monitor GPU memory
   nvidia-smi
   ```

4. **Profiling Tool Issues**

   ```bash
   # Verify Nsight installation
   nsys --version
   ncu --version

   # Check permissions for profiling
   sudo sysctl kernel.perf_event_paranoid=1
   ```

