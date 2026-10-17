# Environment and Configuration

Run the commands below from the repository's `code/` directory unless a block
explicitly changes into it. The environment sources are
[setup.sh](../code/setup.sh) and
[requirements_latest.txt](../code/requirements_latest.txt), plus the separately
installed ABI-bound vLLM pin in
[vllm_no_deps.pin](../code/vllm_no_deps.pin). The dependency files are not fully
resolved lockfiles.

## Hardware and architecture

B200/GB200 have compute capability 10.0. B300/GB300 have 10.3. RTX Blackwell
devices with 12.0 and GB10 with 12.1 are separate targets. These identities do not
make their feature sets, CPU architecture, memory or performance interchangeable.
See [NVIDIA's capability table](https://developer.nvidia.com/cuda/gpus).

[core/harness/arch_config.py](../code/core/harness/arch_config.py) and
[core/common/cuda_arch.mk](../code/core/common/cuda_arch.mk) retain exact target
identities. Select `ARCH=sm_100`, `sm_103`, `sm_120` or `sm_121` as appropriate.
Architecture-specific tcgen05 code requires its matching `a` target and is not
qualified by a build for SM12x. Unsupported features must fail or report an
explicit skip.

## Repository baseline

| Component | Configured baseline | Source/qualification |
| --- | --- | --- |
| Operating system | Ubuntu 22.04 or 24.04, x86-64 | `setup.sh` stops on other releases. It is tested on 22.04. NVIDIA publishes no CUDA 13.0 packages for 20.04 |
| Python | 3.12 | `setup.sh` |
| CUDA toolkit | 13.0.2 | `setup.sh`. Local toolkit and wheel runtime versions are distinct |
| PyTorch | 2.9.1+cu130 | Exact stable wheel pin in `requirements_latest.txt`, not a nightly |
| Triton | 3.5.1 | Matches the PyTorch 2.9.1 Linux x86-64 wheel dependency. Unsupported targets remain unsupported |
| CMake | 4.4.4 | pip wheel pin in `requirements_latest.txt`. Must be at least the newest first-party `cmake_minimum_required` (3.31.8 in `labs/custom_vs_cublas/cutlass_gemm`) |
| Ninja | 1.13.2 | pip wheel pin in `requirements_latest.txt` |
| Nsight Systems | 2026.3.2 | Installer target in `setup.sh`. Supports drivers from 450 on for CUDA 11+ |
| Nsight Compute | 2026.2.1 | Installer target in `setup.sh` and the only version `core/profiling/ncu_torchrun_capture.py` accepts. Requires a driver compatible with CUDA 13 (r580 or newer) |

This is the B200-oriented dependency baseline, not a claim that every package
installs or every kernel runs on all architectures. Read the separate, historical
[GB300 runbook](../code/docs/gb300-runbook.md) and its correction notice before
using that platform. Its environment and results require their own current
validation. Do not substitute a B200 receipt.

## Installation and inspection

The Ubuntu setup script changes system packages, drivers, profiling permissions
and caches and may require a reboot. Review it and obtain control of the host
before running it. Do not run it as a routine diagnostic on a shared machine.

```bash
cd code
# On an authorized Ubuntu target, after reviewing the host changes:
sudo bash setup.sh
```

Run it from a git checkout. The script installs CMake and ninja from their
`requirements_latest.txt` pins instead of Ubuntu's apt CMake, which is 3.22 on
22.04 and 3.28 on 24.04. It puts the pip CMake first on `PATH`. Before any CMake
build, it reads the newest `cmake_minimum_required` from the tracked first-party
`CMakeLists.txt` files and stops if the CMake on `PATH` is older or is not the
pinned version.

The CUDA 13.0 toolkit packages also install Nsight Systems 2025.3.2 and Nsight
Compute 2025.3.1. The harness runs `nsys` and `ncu` by name, so `setup.sh` puts
the pinned Nsight directories ahead of `/usr/local/cuda-13.0/bin` in the script's
`PATH`, `/etc/environment`, `/etc/profile.d/cuda-13.0.sh` and `~/.bashrc`. It
stops if `nsys --version` or `ncu --version` does not report the pinned version.

For an existing compatible Linux/CUDA installation, create an isolated Python
environment and resolve the pinned dependencies there. Some packages require
native build prerequisites or local wheels. A successful dependency resolution
alone is not GPU qualification. The requirements include the pinned CMake and
ninja wheels, so activate the environment before building a CMake project such as
`labs/custom_vs_cublas/cutlass_gemm`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_latest.txt
# Optional serving wheel. Keep --no-deps: ray[cgraph] otherwise pulls CUDA 12
# CuPy beside the CUDA 13 CuPy distribution required by kvikio-cu13.
python -m pip install --no-deps \
  --index-url https://wheels.vllm.ai/0.16.0/cu130 \
  'vllm==0.16.0+cu130'
```

Record the actual environment before building or benchmarking:

```bash
python --version
nvcc --version
cmake --version
nsys --version
ncu --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m cli.aisp bench list-targets --chapter ch01
make -n -C ch01 ARCH=sm_100
```

The last command previews commands only. It does not compile or validate a CUDA
binary. Use the exact chapter target and architecture for a real build. The old
root-level `assert.sh` and `build_all.sh` workflows are not current entrypoints.

## Environment variables

If a compatible toolkit is installed outside `PATH`, set its actual location:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

Start with framework/NCCL defaults. Record any allocator, transport, affinity,
clock, compiler or profiling overrides with the run. Do not copy a generic set of
"optimization" environment variables into every workload. In particular,
`CUDA_LAUNCH_BLOCKING=1` is a debugging choice that changes timing.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q -ra -o timeout=120
```

The repository's test timeout comes from `conftest.py`. Installing an external
timeout plugin is not required. CPU/source checks and explicit GPU skips do not
establish CUDA correctness. Preserve logs, skipped reasons, build identity and
full-output comparisons, then run applicable CUDA and multi-GPU gates on their
supported targets. Use the [profiling guide](tooling-and-profiling.md) for current
capture commands.
