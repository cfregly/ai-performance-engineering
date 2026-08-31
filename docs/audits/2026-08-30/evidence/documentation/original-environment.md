# Environment and Configuration

This document consolidates the environment, tooling, and system setup details
that were previously part of the top-level `README.md`. Use it as a reference
for aligning your system with the repository’s NVIDIA Blackwell-focused stack.

## Target Architecture

The repository targets **NVIDIA Blackwell B200/B300 (SM100)**. Every script,
kernel, and configuration assumes:

- CUDA 12.9 toolkit and toolchain
- PyTorch 2.9.0 (cu129 nightly builds)
- Triton 3.5.0

## Core Components

- `arch_config.py` centralizes architecture decisions and normalizes everything
  to Blackwell.
- `build_all.sh` compiles CUDA kernels with `sm_100` and validates Python
  syntax across the chapters.
- Chapter requirements files are harmonized through `requirements_latest.txt`.

## Toolchain Expectations

| Component        | Version / Channel       | Notes                                      |
|------------------|-------------------------|--------------------------------------------|
| CUDA Toolkit     | 12.9 (nvcc 12.9.x)     | `nvcc -arch=sm_100` everywhere             |
| PyTorch          | 2.9.0 (cu129 nightly)  | Install from `https://download.pytorch.org/whl/nightly/cu129` |
| Triton           | 3.4.0                  | Required for Triton kernels in Chapters 14 & 16 |
| Nsight Systems   | 2024.6+                | Used in profiling scripts                   |
| Nsight Compute   | 2024.3+                | Kernel-level profiling                      |

## System Validation and Failure Analysis

The `assert.sh` script performs deep validation:

```bash
./assert.sh
```

It checks:

- System dependencies (Python, CUDA, Nsight tools, `numactl`, `perf`)
- GPU availability and status
- PyTorch and CUDA version compatibility
- Example registry coverage (84 examples)
- Build/smoke/profiling failures across the harness
- Profiling harness dry runs

Sample output:

```
🚨 Recent Profile Session Analysis:
  Latest session: 20250928_182258
  📊 Results Summary:
    build: 83/84 successful (1 failed)
    smoke: 80/83 successful (3 failed)
    nsys: 1/80 successful (79 failed)
    ncu: 15/80 successful (65 failed)
    pytorch_full: 38/38 successful (0 failed)
```

## Environment Variables

```bash
# CUDA optimization
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# NCCL optimization
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# PyTorch optimization
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# CUDA paths
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Hardware Requirements

- **GPU**: NVIDIA B200/B300 (Blackwell) or compatible
- **Memory**: 32GB+ system RAM recommended
- **Storage**: 50GB+ free space
- **Operating System**: Ubuntu 22.04+ (other Linux distros may work)

## Development Environment

Recommended developer tooling:

```bash
# Install dev dependencies
pip3 install black flake8 mypy

# Format code
black code/

# Lint code
flake8 code/

# Type checking
mypy code/
```

## Advanced Utilities

The `archive/` directory carries more advanced orchestration, including:

- `update_blackwell_requirements.sh`: Sync chapter requirements to the latest
  Blackwell stack
- `update_cuda_versions.sh`: Normalize Makefiles
- `comprehensive_profiling.py`: Demonstrate profiling tools in concert
- `clean_profiles.sh`: Remove accumulated profiler artifacts

Refer back to `docs/tooling-and-profiling.md` for more on the profiling suite
and automation workflows.
