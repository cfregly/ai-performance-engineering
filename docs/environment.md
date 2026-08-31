# Environment and Configuration

Run the commands below from the repository's `code/` directory unless a block
explicitly changes into it. The environment sources are
[setup.sh](../code/setup.sh) and
[requirements_latest.txt](../code/requirements_latest.txt); the dependency file
is not a fully resolved lockfile.

## Hardware and architecture

B200/GB200 have compute capability 10.0; B300/GB300 have 10.3. RTX Blackwell
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
| Python | 3.12 | `setup.sh` |
| CUDA toolkit | 13.0.2 | `setup.sh`; local toolkit and wheel runtime versions are distinct |
| PyTorch | 2.9.1+cu130 | Exact stable wheel pin in `requirements_latest.txt`, not a nightly |
| Triton | 3.5.1 | Matches the PyTorch 2.9.1 Linux x86-64 wheel dependency; unsupported targets remain unsupported |
| Nsight Systems | 2025.3.2 | Installer target in `setup.sh` |
| Nsight Compute | 2025.3.1 | Installer target in `setup.sh` |

This is the B200-oriented dependency baseline, not a claim that every package
installs or every kernel runs on all architectures. Read the separate, historical
[GB300 runbook](../code/docs/gb300-runbook.md) and its correction notice before
using that platform. Its environment and results require their own current
validation; do not substitute a B200 receipt.

## Installation and inspection

The Ubuntu setup script changes system packages, drivers, profiling permissions
and caches and may require a reboot. Review it and obtain control of the host
before running it. Do not run it as a routine diagnostic on a shared machine.

```bash
cd code
# On an authorized Ubuntu target, after reviewing the host changes:
sudo bash setup.sh
```

For an existing compatible Linux/CUDA installation, create an isolated Python
environment and resolve the pinned dependencies there. Some packages require
native build prerequisites or local wheels; a successful dependency resolution
alone is not GPU qualification.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_latest.txt
```

Record the actual environment before building or benchmarking:

```bash
python --version
nvcc --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m cli.aisp bench list-targets --chapter ch01
make -n -C ch01 ARCH=sm_100
```

The last command previews commands only; it does not compile or validate a CUDA
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
clock, compiler or profiling overrides with the run; do not copy a generic set of
"optimization" environment variables into every workload. In particular,
`CUDA_LAUNCH_BLOCKING=1` is a debugging choice that changes timing.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q -ra -o timeout=120
```

The repository's test timeout comes from `conftest.py`; installing an external
timeout plugin is not required. CPU/source checks and explicit GPU skips do not
establish CUDA correctness. Preserve logs, skipped reasons, build identity and
full-output comparisons, then run applicable CUDA and multi-GPU gates on their
supported targets. Use the [profiling guide](tooling-and-profiling.md) for current
capture commands.
