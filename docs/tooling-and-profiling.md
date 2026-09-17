# Tooling and Profiling Guide

Run commands from the repository's `code/` directory. The supported entrypoint is
`python -m cli.aisp`; inspect its current help before selecting a workload or
profiler. Captures below require an available, supported target and permission to
use it. A profiler report alone does not establish correctness or a speedup.

## Discover and profile benchmark pairs

```bash
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --help
python -m cli.aisp bench verify --help

# Execute only on the appropriate available target:
python -m cli.aisp bench run --targets ch01:performance --profile minimal
```

`ch01:performance` is an actual discovered baseline/optimized pair. Choose another
listed target for memory access, compilation or Triton experiments. Keep the
baseline and candidate's workload, dtype, verification policy and timing scope
consistent. Preserve the generated run directory, exact revision, environment,
logs and failed/skipped results. Do not update expectation files from an invalid
or unverified run.

## Check that a profiler works

The small script `tests/fixtures/mcp_torch_profile_target.py` performs a real
matrix multiply. It is a capture smoke test, not a performance benchmark. It uses
CUDA if available and otherwise CPU, so a CPU trace does not prove CUDA capture.

```bash
python -m cli.aisp profile torch --help
python -m cli.aisp profile torch tests/fixtures/mcp_torch_profile_target.py \
  --mode full --output-name docs-smoke --timeout 60
```

This produces a Chrome trace and summary under the reported artifacts directory.
Use a fresh output/run name for each attempt. For your own script, include the
actual forward/backward or serving work in the capture, with warmup and capture
boundaries documented. Profiling overhead must not be reported as ordinary
benchmark latency.

## NVIDIA Nsight Systems

Check the installed binary and supported trace options on the target:

```bash
nsys --version
nsys profile --help
nsys profile --trace=cuda,nvtx,osrt --output=timeline_attempt_01 \
  python tests/fixtures/mcp_torch_profile_target.py
nsys stats timeline_attempt_01.nsys-rep
```

On Linux, `cuda,nvtx,osrt` are supported trace categories in
[Nsight Systems 2025.3](https://archive.docs.nvidia.com/nsight-systems/2025.3/UserGuide/index.html).
`triton` is not a trace category: Triton-generated CUDA kernels appear in the CUDA
timeline. Add library/NCCL tracing only if the installed version advertises it.
The timeline can expose stream dependencies, transfers and idle gaps; verify that
its captured interval covers the intended work.

The repository wrapper accepts a quoted command:

```bash
python -m cli.aisp profile nsys \
  "python tests/fixtures/mcp_torch_profile_target.py" \
  --output-name timeline-smoke --timeout 60
```

## NVIDIA Nsight Compute

Discover metrics on the actual GPU instead of copying legacy nvprof names:

```bash
ncu --version
ncu --list-sections
ncu --query-metrics --query-metrics-mode all
ncu --set basic --export=kernel_attempt_01 \
  python tests/fixtures/mcp_torch_profile_target.py
ncu --import kernel_attempt_01.ncu-rep --page raw --csv > kernel_attempt_01.csv
```

The [Nsight Compute CLI guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
explains section sets and metric suffixes. `achieved_occupancy`,
`warp_execution_efficiency`, `memory_throughput` and
`triton_kernel_efficiency` are not portable metric selectors for these commands.
Use a complete name returned by the installed tool for that device. Kernel
replay can change cache state and add overhead; record replay settings and do not
substitute profiled duration for a separately measured end-to-end baseline.

### Two-rank NCCL captures

Collectives that require concurrent ranks can stall under process-tree replay.
The explicit `core.profiling.ncu_torchrun_capture` command launches one profiler
per rank, with TCP coordination and lockstep limited to named NCCL NVTX ranges.
It currently supports one Linux host, two visible GPUs, Nsight Compute 2026.2.1, kernel
replay, and the repository's five minimal metrics. It runs directly with
`torchrun`; Slurm is not required.

From a clean checkout's `code/` directory, use the Python environment for the
workload and set `NCU` to the absolute path of the 2026.2.1 executable. Choose
unused loopback ports and a new output directory for each attempt:

```bash
AISP_LOCK_GPU_CLOCKS=1 AISP_RAMP_GPU_CLOCKS=1 \
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD" \
python -m core.profiling.ncu_torchrun_capture \
  --label pipeline-nccl --repo-root .. --source "$(git rev-parse HEAD)" \
  --output-dir /tmp/pipeline-nccl-attempt-01 --ncu "$NCU" \
  --tcp-port 29601 --timeout-seconds 180 --all-matching-kernels \
  --nvtx-include 'NCCL@ncclGroupEnd/' --nvtx-include 'NCCL@ncclAllReduce/' \
  -- python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 \
  --rdzv_backend c10d --rdzv_endpoint 127.0.0.1:29600 --max_restarts 0 \
  -m core.harness.torchrun_wrapper --aisp-emit-runtime-provenance \
  --aisp-target-module core.harness.benchmark_worker \
  --aisp-expected-torch-seed 42 --aisp-expected-cuda-seed 42 \
  --module ch04.baseline_pipeline_parallel --callable main \
  -- --iters 3 --warmup 5 --seed 42
```

Omit `--all-matching-kernels` for a faster diagnostic of the first matching
kernel per rank. The selected ranges must be observed in the workload's trace;
globally matching launch ordinals is unsafe for asymmetric pipeline stages.
Neither mode captures non-NCCL compute or changes the default benchmark profiler.

The command retains exact arguments, source/tool hashes, logs, rank reports, and
the owned process-group cleanup result. `CAPTURE_ARTIFACTS_PRESENT` establishes
artifact creation only. Import both reports to check the selected kernels,
finite metrics, PID/device identities and runtime receipts; verify numerical
outputs and ordinary timings separately. See the
[B200 follow-through report](reviews/2026-09-08-b200-remaining-followthrough.md)
for measured scope and retained timeout dispositions.

## Framework traces, HTA and offline analysis

`python -m cli.aisp profile torch` captures framework operators and CPU/CUDA
activities. HTA consumes compatible traces; `python -m cli.aisp profile hta --help`
and `hta-capture --help` describe the current repository interfaces. Do not assume
an arbitrary Nsight report is directly interchangeable with an HTA input.

Current extraction helpers live under `core/profiling/`:

```bash
python core/profiling/extract_nsys_summary.py --help
python core/profiling/extract_pytorch_profile.py --help
# Replace these quoted globs with paths from a completed capture:
python core/profiling/extract_nsys_summary.py 'artifacts/runs/your-run/*.nsys-rep' \
  --output /tmp/your-run-nsys-summary.csv
python core/profiling/extract_pytorch_profile.py 'artifacts/runs/your-run/torch*' \
  --output-prefix /tmp/your-run-torch-summary
```

Inspect each helper's expected schema and read back its result. The legacy
`core/profiling/extract_ncu_subset.py` recognizes a small list of display labels;
it is not a general parser for every modern raw metric name. Keep the original
`.ncu-rep`/CSV if that subset produces no rows. The former root `scripts/`,
`tools/`, `start.sh`, `stop.sh`, `extract.sh` and `clean_profiles.sh` commands in
this guide were stale and are not supported orchestration shortcuts.

## Linux CPU profiling

Select the exact process you own, or launch the workload under `perf`:

```bash
perf record -g -o cpu_attempt_01.data -- \
  python tests/fixtures/mcp_torch_profile_target.py
perf report -i cpu_attempt_01.data
```

Avoid `pgrep python`, which may select unrelated processes. CPU sampling does not
measure CUDA kernel execution. For permission errors, inspect the target's
policy with its administrator; do not change system-wide profiling settings as a
routine troubleshooting step.

## Troubleshooting and evidence boundaries

- Check `python --version`, `nvcc --version`, `nvidia-smi`, `nsys --version` and
  `ncu --version`, then compare them with [the environment guide](environment.md).
- Verify the selected target and script exist; start with `--help` and discovery.
- Missing tools, unsupported metrics/architectures and denied counters remain
  failures or explicit HOLD results. Do not silently substitute a profiler or
  claim an empty report is a successful capture.
- Preserve each attempt in a separate directory. Do not clean a shared artifact
  directory or kill unrelated jobs to make a profiling command run.
- Recheck complete numerical outputs before using new timing or throughput
  claims. A host help check is command validation, not a Linux/CUDA profiler run.
