# Environment and Configuration

Run the commands below from the repository's `code/` directory unless a block
explicitly changes into it. The reviewed dependency source is
[requirements_latest.txt](../code/requirements_latest.txt). Current Linux x86-64
CPython 3.12 installs consume the complete, hash-locked graph in
[requirements_torch2141_linux_x86_64.lock](../code/requirements_torch2141_linux_x86_64.lock)
through [setup.sh](../code/setup.sh). The default serving runtime is built from
the commit in [vllm_source_build.json](../code/vllm_source_build.json) by
[build_vllm_for_torch.py](../code/scripts/build_vllm_for_torch.py).

The lock resolves `torch-c-dlpack-ext` from the repository's
`vendor/python-wheels` directory. Its exact
`0.1.5+torch2141.nobackend1` wheel is reproducibly derived from the verified
upstream 0.1.5 sdist by `scripts/build_torch_dlpack_wheel.py` and
`torch_dlpack_source_build.json`. The adapter keeps the package API and binds it
to `torch==2.14.1`, but removes the unused top-level `build_backend` module that
would collide with FlashInfer's module of the same name. It contains no native
fallback and fails clearly if Torch lacks the built-in
`Tensor.__dlpack_c_exchange_api__` used by this stack.

## Hardware and architecture

B200/GB200 have compute capability 10.0. B300/GB300 have 10.3. RTX Blackwell
devices with 12.0 and GB10 with 12.1 are separate targets. These identities do not
make their feature sets, CPU architecture, memory or performance interchangeable.
See [NVIDIA's capability table](https://developer.nvidia.com/cuda/gpus).

The shared CUDA 13.0 and PyTorch 2.14.1 `setup.sh` native-build contract is
limited to B200/GB200 compute capability 10.0. Full setup rejects B300/GB300
10.3 and every other capability before repository checkout or compiler work.
The historical GB300 runbook and standalone SM103 profiles remain available
for their separately pinned environments and qualification. They are not an
alternate target for this shared setup.

[core/harness/arch_config.py](../code/core/harness/arch_config.py) and
[core/common/cuda_arch.mk](../code/core/common/cuda_arch.mk) retain exact target
identities. Select `ARCH=sm_100`, `sm_103`, `sm_120` or `sm_121` as appropriate.
Architecture-specific tcgen05 code requires its matching `a` target and is not
qualified by a build for SM12x. Unsupported features must fail or report an
explicit skip.

## Repository baseline

| Component | Configured baseline | Source/qualification |
| --- | --- | --- |
| Operating system | Ubuntu 22.04 or 24.04, x86-64 | `setup.sh` stops on other releases. A configured version does not establish a full setup validation |
| Python | 3.12 | `setup.sh` |
| CUDA toolkit | 13.0.2 | `setup.sh`. Local toolkit and wheel runtime versions are distinct |
| PyTorch | 2.14.1+cu130 | Exact stable wheel pin in `requirements_latest.txt` and the Linux hash lock, not a nightly |
| Triton | 3.8.0 | Exact PyTorch 2.14.1 compiler dependency pin. Unsupported targets remain unsupported |
| TorchVision | 0.29.1+cu130 | Exact stable CUDA 13 wheel pin matched to the default PyTorch environment |
| TorchAudio | 2.11.0+cu130 | Exact CUDA 13 stable-ABI wheel pin. Its stable ABI supports PyTorch 2.11 and later |
| TorchAO | 0.18.0+cu130 | Exact stable CUDA 13 wheel pin for the PyTorch 2.14.1 environment |
| TorchCodec | 0.17.0+cu130 | Exact CUDA 13 wheel pin. Media decoding requires FFmpeg shared libraries from Ubuntu's `ffmpeg` package, which `setup.sh` installs |
| TorchTitan | 0.3.0 | Exact package pin in `requirements_latest.txt` and the Linux hash lock |
| FlashInfer Python | 0.7.2 | Exact package pin in `requirements_latest.txt` and the Linux hash lock |
| FlashAttention | `flash-attn-4[cu13]` 4.0.0b36 | Shared default. FA2 2.8.3.post2 uses the same Python/native namespace and is built only in a separate validation environment |
| Transformer Engine | 2.20.2 | Meta-package and CUDA 13 core are hash locked. The PyTorch binding is rebuilt from its verified sdist against the installed Torch 2.14.1 ABI |
| vLLM | 0.33.0.dev0+cu130.torch2141 source build | Source commit and build contract in `vllm_source_build.json`. Built against the installed PyTorch 2.14.1 stack because no published stable vLLM wheel matches this ABI |
| CMake | 4.4.4 | pip wheel pin in the dependency source and Linux hash lock. Must be at least the newest first-party `cmake_minimum_required` (3.31.8 in `labs/custom_vs_cublas/cutlass_gemm`) |
| Ninja | 1.13.2 | pip wheel pin in the dependency source and Linux hash lock |
| Nsight Systems | 2026.5.1 | Exact CLI-only package `nsight-systems-cli-2026.5.1=2026.5.1.161-265138896106v0` from NVIDIA's signed DEVTOOLS repository. Supports drivers from 450 on for CUDA 11+ |
| Nsight Compute | 2026.2.1 | Installer target in `setup.sh` and the only version `core/profiling/ncu_torchrun_capture.py` accepts. Requires a driver compatible with CUDA 13 (r580 or newer) |

This is the B200 compute capability 10.0 dependency baseline. Package and
kernel support on other architectures requires separate validation. Read the
historical [GB300 runbook](../code/docs/gb300-runbook.md) and its correction
notice before using that platform. Its environment and results require their
own current validation. Do not substitute a B200 receipt.

## Installation and inspection

The Ubuntu setup script changes system packages, drivers, profiling permissions
and caches and may require a reboot. Review it and obtain control of the host
before running it. Do not run it as a routine diagnostic on a shared machine.

```bash
cd code
# On an authorized Ubuntu target, after reviewing the host changes:
sudo bash setup.sh
```

Run it from a git checkout. The script installs the current Python graph from
`requirements_torch2141_linux_x86_64.lock`, including the reviewed CMake and
ninja pins, instead of Ubuntu's apt CMake, which is 3.22 on
22.04 and 3.28 on 24.04. It puts the pip CMake first on `PATH`. Before any CMake
build, it reads the newest `cmake_minimum_required` from the tracked first-party
`CMakeLists.txt` files and stops if the CMake on `PATH` is older or is not the
pinned version.

The CUDA 13.0 toolkit also installs older 2025.3.x profiler copies. `setup.sh`
adds NVIDIA's distribution-specific DEVTOOLS repository with its own scoped,
hash- and fingerprint-verified signing key. It installs the exact CLI-only
Nsight Systems 2026.5.1.161 package and keeps Nsight Compute 2026.2.1 from the
CUDA repository. The harness runs `nsys` and `ncu` by name, so `setup.sh` puts
these pinned directories ahead of `/usr/local/cuda-13.0/bin` in
`${AISP_VENV_DIR}/cuda-env.sh`. It composes that file into
`${AISP_VENV_DIR}/runtime-env.sh` with the isolated Python environment, cache
directories and locked wheel-library paths. Source `runtime-env.sh` in a new
shell before building or running the repository. Existing shell profiles and
system-wide environment files remain unchanged. The script stops if
`nsys --version` or `ncu --version` does not report the pinned version.

The hash lock is also the authority for the NCCL, cuDNN and NVSHMEM runtime
and development assets. After installing the graph, `setup.sh` locates each
wheel's headers and shared libraries and stops if any required asset is
missing. `runtime-env.sh` exports `NCCL_HOME`, `CUDNN_HOME`, `NVSHMEM_HOME`,
`NCCL_LIB_PATH`, the wheel include directories in `CPATH`, and the complete
locked wheel library search path in `LD_LIBRARY_PATH`. An `LD_PRELOAD`
containing external NCCL, cuDNN or cuBLAS libraries is rejected because it
would override that graph.

For an existing compatible Linux/CUDA installation, create an isolated Python
environment and install the complete hash-locked graph there. Some packages
require native build prerequisites or local wheels. A successful installation
alone is not GPU qualification. The lock includes the pinned CMake and ninja
wheels, so activate the environment before building a CMake project such as
`labs/custom_vs_cublas/cutlass_gemm`.

```bash
python3.12 -m venv .venv-torch2141
source .venv-torch2141/bin/activate
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --find-links vendor/python-wheels \
  --require-hashes \
  --no-deps \
  -r requirements_torch2141_linux_x86_64.lock
python -m pip check

# Optional serving runtime. Choose a new absolute output path for each build.
export VLLM_BUILD_ROOT=/absolute/path/to/new-vllm-torch2141-build
CUDA_VISIBLE_DEVICES="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTORCH_NVML_BASED_CUDA_CHECK=1 \
  python scripts/build_vllm_for_torch.py \
    --output-dir "${VLLM_BUILD_ROOT}" \
    --cuda-arch-list 10.0 \
    --max-jobs 4 \
    --nvcc-threads 2
VLLM_WHEEL="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["wheel"]["path"])' "${VLLM_BUILD_ROOT}/vllm-source-wheel.manifest.json")"
python -m pip install --no-deps "${VLLM_WHEEL}"
python -m pip check
```

The helper requires the exact build dependencies and Rust 1.95.0 declared in
the contract. It verifies the pinned vLLM source identity and upstream file
hashes, uses the installed PyTorch instead of downloading another build, and
writes the wheel under `${VLLM_BUILD_ROOT}/wheel/`. The output directory must
be new and empty. The helper rejects existing build receipts instead of reusing
them. Its
`vllm-source-wheel.manifest.json` receipt records the source tree, build
environment, wheel metadata, native members and SHA-256. A completed build and
successful native import still require the applicable serving correctness and
GPU gates before use as performance evidence.

The shared environment keeps `flash-attn-4[cu13]==4.0.0b36`. Build its
Transformer Engine PyTorch binding from the hash-verified 2.20.2 sdist without
touching that FlashAttention installation:

```bash
export NATIVE_EXTENSION_BUILD_ROOT=/absolute/path/to/new-te-torch2141-build
CUDA_VISIBLE_DEVICES="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTORCH_NVML_BASED_CUDA_CHECK=1 \
  python scripts/build_torch_extensions.py \
    --package transformer-engine-torch \
    --output-dir "${NATIVE_EXTENSION_BUILD_ROOT}" \
    --cuda-arch-list 10.0 \
    --max-jobs 8 \
    --nvcc-threads 2
```

The same helper can build `flash-attn==2.8.3.post2` with `--package
flash-attn`, but only from a fresh nested environment that contains neither
`flash-attn` nor `flash-attn-4`. Never install that FA2 wheel into the shared
FA4 environment: both distributions own the `flash_attn` Python namespace.
The helper downloads the hash-verified sdist before the build, rejects native
binaries in the extracted source, and runs the build under Linux `unshare
--net` with package-index access disabled. It derives one CUDA toolkit root
from the validated CUDA 13.0 `nvcc`, rejects conflicting CUDA variables, and
replaces inherited compiler, include, library, proxy and Python-path settings
with its explicit build environment. The receipt records the network-isolation
preflight, resolved CUDA tools, sanitized inputs and exact build environment.
The helper builds only, always requires a new output directory, and writes
`native-torch-extensions.manifest.json` with the verified sdist, explicit
C++20 patch (FA2 only), wheel metadata, native ELF members and SHA-256 hashes.
`MAX_JOBS` may be lowered from 8 to 1 and `NVCC_THREADS` from 2 to 1 for a
constrained build host. The receipt records the actual values. These source
builds have no GPU-runtime qualification until the separate hardware gates
complete.

The shared environment rejects `INSTALL_GPT_OSS=1` and `INSTALL_TRTLLM=1`.
Those stacks require separately resolved and qualified runtimes because their
package constraints conflict with the shared Torch 2.14.1 graph.

The former `vllm==0.16.0+cu130` wheel targets an older PyTorch ABI and is not a
default-environment fallback. The retained
[`build_vllm_fa4_compat_wheel.py`](../code/scripts/build_vllm_fa4_compat_wheel.py)
helper belongs to the historical PyTorch 2.9.1, vLLM 0.16.0 and FA4 wheel
reproduction path. Keep that environment separate from the current Torch
2.14.1 source builds; it is unrelated to the shared environment's current FA4
package pin. The
[dated validation checkpoint](reviews/2026-09-06-codebase-repair-checkpoint.md)
retains its measurements and limits.

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
