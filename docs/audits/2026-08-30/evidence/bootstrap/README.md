# Bootstrap remediation evidence

Captured 2026-08-30 for W1-005, W1-124, and W1-125. This evidence also records adjacent prerequisites discovered during resolution; it does not change the 128-finding source audit.

Source changes are ready for review. Full Linux/CUDA bootstrap acceptance remains **HOLD**. No privileged setup, shared-environment package installation, CUDA build, or GPU execution was performed.

## Changes and regression evidence

- `setup.sh` selects stable or nightly sources from each exact package pin after loading `requirements_latest.txt`; initial installs, the PyTorch reinstall used by Transformer Engine, and the Triton retry use the selected source. Stable fallback defaults now agree with the requirements.
- `requirements_latest.txt` exposes the official stable cu130 index while keeping PyPI available for the other dependencies.
- The final cuDNN summary queries the installed system package and reports its actual version alongside the requested pin. A missing or unconfigured package prevents the success summary. It makes no assertion that the system version equals PyTorch's bundled runtime.
- The retired torchao nightly was replaced with `torchao==0.15.0+cu130`, the release built against torch 2.9.1. Setup loads that pin from requirements and defers its installation until after torch.
- On aarch64, torchaudio selects the published `2.9.1` wheel while preserving torch `2.9.1+cu130`. The missing aarch64 torchao CUDA wheel produces an explicit error before bulk requirements or PyTorch replacement. Earlier OS provisioning stages are unchanged; this is not a side-effect-free setup preflight.

The initial source-only slice had eight failing regressions before the fix and ten passing tests afterward (the passing logs are in `initial-source-validation/`). The adjacent torchao/ARM regressions produced **5 failed, 8 passed** before their fixes (`torchao-followup-before.txt`). The final focused run produced **14 passed**, including compatible fallback defaults, with Bash syntax and scoped diff checks also passing. Exact commands, cwd, timestamps, exit codes, and log paths are in `validation-receipts.json`.

The tests execute the real Bash source-selection and summary blocks without running privileged setup. Package-query and Git subprocess seams test orchestration only; they do not represent installed hardware or CUDA runtime evidence.

## Upstream publication and compatibility evidence

- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) documents torch 2.9.1 and torchaudio 2.9.1 using the stable cu130 source.
- The [torchao compatibility table](https://github.com/pytorch/ao/issues/2919) and [v0.15.0 import guard](https://github.com/pytorch/ao/blob/v0.15.0/torchao/__init__.py) pair torchao 0.15.0 with torch 2.9.1. The [release](https://github.com/pytorch/ao/releases/tag/v0.15.0) is therefore an ABI-informed replacement, not a general dependency upgrade.
- The [stable torchao cu130 index](https://download.pytorch.org/whl/cu130/torchao/) publishes `torchao-0.15.0+cu130-cp310-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`, SHA-256 `2d743a43f1f345fd7a6369e37901c5a7a0a929c0f7c8acc52e717f1d70d19155`. No aarch64 wheel for this exact pin was present. The old `0.16.0.dev20251213+cu130` was absent from the [nightly cu130 index](https://download.pytorch.org/whl/nightly/cu130/torchao/).
- The [torchaudio cu130 index](https://download.pytorch.org/whl/cu130/torchaudio/) publishes cp312 x86_64 `2.9.1+cu130` and aarch64 `2.9.1`. This is a local-version suffix difference, not a release downgrade.

`source-observations.json` records the observed URLs, page/metadata hashes, matching wheel URLs and hashes, release source lines, exact tag commit, and owned-source hashes. Index contents are time-sensitive.

## Resolver and isolated import boundaries

The pip metadata checks in `validation-receipts.json` use an isolated venv, explicit Linux platform/Python tags, `--dry-run`, and `--no-deps`. They confirm that pip can select the exact published artifacts; they do **not** install the Linux wheels, resolve the full requirements graph, or validate native ABI loading.

- x86_64: torch 2.9.1+cu130, torchaudio 2.9.1+cu130, torchao 0.15.0+cu130, and Triton 3.5.0 all resolve. Resolver reports retain selected artifact hashes.
- aarch64: torch 2.9.1+cu130 and torchaudio 2.9.1 resolve; torchao 0.15.0+cu130 fails with no matching distribution, as expected.

An actual install into `/tmp/aisp-torchao-compat-zi7v6xmy/venv` used macOS torch 2.9.1, Python-only torchao 0.15.0, and NumPy 2.1.2. `isolated-install-command.json` and `isolated-install.txt` record that installation. `isolated-api-probe.py` imports the FP8 and quantization APIs used by setup, verifies that the version guard accepts the pair, and converts a Linear module into Float8Linear. `isolated-import-receipt.json` records success. This environment has no torchao native `.so` libraries and no CUDA; it proves Python API compatibility only.

The shared CPU test interpreter was `/opt/miniconda3/bin/python` (Python 3.12.2, pytest 8.4.2, torch 2.8.0 without CUDA), not the pinned production GPU stack.

## ARM source-build route and remaining acceptance

The [v0.15.0 README](https://github.com/pytorch/ao/tree/v0.15.0#installation) documents building from source with CUDA and without build isolation. Its [build script](https://github.com/pytorch/ao/blob/v0.15.0/setup.py) enables CUDA extensions when CUDA-enabled torch and a CUDA toolkit are present. This provides a source-build route to investigate, but it is not evidence of a tested aarch64 CUDA 13 build.

For a separate isolated target build, pin source commit `9338966da58ec44b60f0e0b173cabab08f942ed0` (v0.15.0), initialize its pinned submodules, install torch 2.9.1+cu130 first, expose the CUDA 13 toolkit, and build a wheel with the upstream no-build-isolation procedure and dependency replacement disabled. Record the source/submodule commits, compiler/CUDA versions, target SM list, build options, wheel hash, and resulting package metadata. Do not replace CUDA kernels with a CPU-only package or classify an uncompiled source recipe as supported runtime evidence. No such build was launched here, and setup does not yet consume a separately validated ARM source-built artifact.

Before full bootstrap can pass, an isolated supported Linux/NVIDIA target must complete package installation and its final version checks, load torchao native CUDA extensions, pass setup's real FP8 forward/backward check, and preserve the torch pin through the Transformer Engine reinstall. Validate the actual installed cuDNN package and PyTorch-bundled cuDNN separately. Check the full requirements dependency graph and all other setup prerequisites. Aarch64 additionally requires the source-built artifact and target-specific gates above; existing GB300 toolchain constraints remain separate and unchanged.

## Evidence filename migration

`log-migration.json` records byte-identical `.txt` copies of the ignored `.log` evidence. Original logs and pre-migration receipt JSON remain intact. Current receipt log pointers use the durable copies; this migration did not rerun or change the original checks.
