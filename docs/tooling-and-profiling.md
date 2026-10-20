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

PyTorch 2.9 with CUDA 13.0 can silently omit CUDA activity because its older
Kineto build resets the CUPTI buffer mode before flushing recorded events.
The repository's Torch profiling entrypoints initialize that mode before
profiling to avoid the [upstream ordering bug](https://github.com/pytorch/kineto/pull/1115).
This preserves the default Torch and serving dependencies. Check that the trace
contains CUDA kernel, copy, or memset events with positive durations before
treating it as evidence of GPU activity.

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

For an attached B200 hang investigation, use each diagnostic for its own
evidence:

- Nsight Systems shows the distributed CUDA and NCCL timeline, including which
  ranks reach a collective and where the streams stop making progress.
- Nsight Compute records selected kernel counters. When the selected kernels
  must execute concurrently, use the rank-local TCP topology described below.
  Its application replay starts every generation with a fresh distributed store.
- GDB shows host-native process and thread stacks. Ordinary GDB does not debug a
  running CUDA kernel.
- `strace` shows host system calls and waits. `py-spy` shows Python stacks.

Attach the applicable tools to the actual B200 workload before using their output
as hang evidence. Each attachment is a bounded observation and does not establish
complete execution coverage. Zymtrace can provide supplemental observations, but
it is not the sole diagnostic path.

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
[Nsight Systems 2026.3](https://archive.docs.nvidia.com/nsight-systems/2026.3/UserGuide/index.html),
the version `code/setup.sh` installs.
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
That kernel and range topology supports one Linux host, two visible GPUs,
Nsight Compute 2026.2.1, and the repository's five minimal metrics. It uses
kernel replay by default or explicit range replay. Application replay has two
explicit topologies. The default `parent` topology wraps the original full
two-rank `torchrun` command with one NCU process. The `rank-tcp` topology launches
one NCU process per local rank and uses TCP coordination for the selected
concurrent kernels. Both run without Slurm.

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

The example uses the default kernel replay. Add `--replay-mode range` to collect
range-level metrics for every selected NCCL range. Range replay requires
`--all-matching-kernels` and does not provide per-kernel attribution.

For kernel replay, omit `--all-matching-kernels` for a faster diagnostic of the
first matching kernel per rank. The selected ranges must be observed in the
workload's trace. Globally matching launch ordinals are unsafe for asymmetric
pipeline stages. Neither replay mode captures non-NCCL compute or changes the
default benchmark profiler.

#### Bounded parent application replay

Use application replay only when the launch selector and target visible device
are known in advance. The helper fixes NCU to strict application replay, grid
matching, per-GPU filtering, all target processes, and the repository's five
minimal metrics. Use a `regex:` kernel selector and `--launch-count 1`. This
mode does not use the per-rank TCP communicator or named NVTX range filters.

Run one target visible device at a time. Give each invocation a fresh output
directory and a different torchrun rendezvous port. Set `KERNEL_NAME` and
`LAUNCH_SKIP` from evidence for the exact source, workload, warmup count, and
device. Do not copy a pipeline launch ordinal to another workload or device.

```bash
TARGET_VISIBLE_DEVICE=0
KERNEL_NAME='regex:ncclDevKernel_SendRecv.*'
LAUNCH_SKIP="$WORKLOAD_SPECIFIC_LAUNCH_SKIP"
RDZV_PORT=29610
OUTPUT_DIR=/tmp/pipeline-app-device-0-attempt-01

AISP_LOCK_GPU_CLOCKS=1 AISP_RAMP_GPU_CLOCKS=1 \
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD" \
python -m core.profiling.ncu_torchrun_capture \
  --label pipeline-app-device-0 --repo-root .. \
  --source "$(git rev-parse HEAD)" --output-dir "$OUTPUT_DIR" --ncu "$NCU" \
  --timeout-seconds 300 --replay-mode application --application-topology parent \
  --target-visible-device "$TARGET_VISIBLE_DEVICE" \
  --kernel-name "$KERNEL_NAME" --launch-skip "$LAUNCH_SKIP" --launch-count 1 \
  -- python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 \
  --rdzv_backend c10d --rdzv_endpoint "127.0.0.1:$RDZV_PORT" --max_restarts 0 \
  -m core.harness.torchrun_wrapper --aisp-emit-runtime-provenance \
  --aisp-target-module core.harness.benchmark_worker \
  --aisp-expected-torch-seed 42 --aisp-expected-cuda-seed 42 \
  --module ch04.baseline_pipeline_parallel_multigpu --callable main \
  -- --iters 3 --warmup 5 --seed 42
```

After that command finishes and its artifacts are retained, repeat it serially
with `TARGET_VISIBLE_DEVICE=1`, a device-specific `LAUNCH_SKIP`, a new
`RDZV_PORT`, a new label, and a new `OUTPUT_DIR`. The visible ordinal is not a
pipeline role. Bind it to the physical GPU with the retained CUDA visibility and
report device identity.

Application replay captures real NCU and worker output in one merged stream so
each strict `Application replay pass N` interval contains one complete two-rank
runtime-provenance generation. The helper imports the fresh report with the same
pinned NCU executable and validates the requested device, replay-pass count, and
all five finite metric values for the selected row. Keep the report and imported
CSV for independent audit.

One successful invocation covers one explicitly selected matching launch on one
visible device. Two serial invocations can provide one selected counter row per
actual rank and device. They do not cover every NCCL launch, establish receive
coverage, identify a send or receive direction from the counter row, or prove a
direction-specific root cause. Measure ordinary workload timing separately.
Qualify full-tensor numerical outputs separately because the runtime receipts
prove post-main completion and environment parity, not tensor equality. Run
the public command on the target hardware and review its report, rank receipts,
device mapping, clocks, and cleanup before accepting the capture.

#### Bounded rank-local TCP application replay

Use rank-local TCP application replay when both selected rank kernels must run
concurrently. This topology uses strict application replay, grid matching,
per-GPU filtering, two TCP peers, lockstep coordination, and the repository's
five minimal metrics. Set the exact expected pass count from the selected metric
set, tool version, and workload. Each pass launches the original worker command
in a fresh two-rank worker generation and store. `MASTER_PORT` equals the
rendezvous base plus the zero-based generation number, and
`TORCHELASTIC_USE_AGENT_STORE=False`.

Keep the input torchrun rendezvous port, NCU communicator port, and generated
rendezvous port range disjoint. The example below expects nine
application-replay generations. That count is scoped to this exact two-B200
workload and NCU 2026.2.1:

```bash
KERNEL_NAME='regex:ncclDevKernel_AllReduce.*'
LAUNCH_SKIP=2
NCU_TCP_PORT=29621
INPUT_RDZV_PORT=29622
APPLICATION_RDZV_PORT_BASE=29630
EXPECTED_APPLICATION_PASSES=9
OUTPUT_DIR=/tmp/pipeline-rank-tcp-attempt-01

AISP_LOCK_GPU_CLOCKS=1 AISP_RAMP_GPU_CLOCKS=1 \
AISP_NCU_PROFILE_REQUESTS=4 AISP_NCU_PROFILE_CONTEXT=256 \
AISP_NCU_PROFILE_DECODE=8 AISP_NCU_PROFILE_BATCH=1 \
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD" \
python -m core.profiling.ncu_torchrun_capture \
  --label disagg-rank-tcp --repo-root .. \
  --source "$(git rev-parse HEAD)" --output-dir "$OUTPUT_DIR" --ncu "$NCU" \
  --timeout-seconds 300 --replay-mode application \
  --application-topology rank-tcp --tcp-port "$NCU_TCP_PORT" \
  --expected-application-passes "$EXPECTED_APPLICATION_PASSES" \
  --application-rdzv-port-base "$APPLICATION_RDZV_PORT_BASE" \
  --kernel-name "$KERNEL_NAME" --launch-skip "$LAUNCH_SKIP" --launch-count 1 \
  -- python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 \
  --rdzv_backend c10d --rdzv_endpoint "127.0.0.1:$INPUT_RDZV_PORT" \
  --max_restarts 0 \
  -m core.harness.torchrun_wrapper --aisp-emit-runtime-provenance \
  --aisp-target-module core.harness.benchmark_worker \
  --aisp-expected-torch-seed 42 --aisp-expected-cuda-seed 42 \
  --module ch15.optimized_disaggregated_inference_multigpu --callable main \
  -- --iters 4 --warmup 5 --seed 42
```

For the selected-launch command above, do not pass `--target-visible-device`,
`--all-matching-kernels`, `--expected-kernel-counts`, or `--nvtx-include`.
The helper writes explicit
`rank-0.ncu.log` and `rank-1.ncu.log` paths because NCU does not expand
`%q{LOCAL_RANK}` in `--log-file`. Validate the `[Comm rank X]` prefix in each
log independently of `LOCAL_RANK` because communicator rank and local rank are
different identities.

A successful receipt requires the declared number of complete two-rank
generations, fresh worker process IDs, runtime-provenance parity, two fresh NCU
reports, and one validated raw CSV row per report. Each row must identify the
expected visible device and that local rank's first-generation worker PID,
report the expected replay-pass count, and contain finite positive values for
all five metrics. This proves that the bounded selected kernels produced
auditable counter rows. It does not cover every NCCL launch or direction,
establish numerical correctness, or replace ordinary workload timing.

The running receipt records `requested_visible_devices: [0, 1]` and keeps
`both_devices_covered_by_this_receipt` false. The final receipt sets the coverage
field true only after both native reports were imported and validated without an
artifact error.

#### Full all-matching rank-local application replay

Use the full mode only after a fresh Nsight Systems trace establishes the exact
number of kernels that match the same selector for each worker rank. Collect the
trace from the same source, workload arguments, warmup count, visible-device
order, and software environment intended for NCU. The trace must retain native
NCCL API activity and CUDA GPU activity so the review can map each matching
kernel to its worker PID, device, workload phase, and NCCL `send`, `recv`, or
collective call. Count setup kernels too when they match the selector. A kernel
name such as `ncclDevKernel_SendRecv` does not establish API direction.

Set the two positive counts from that frozen trace inventory. Do not estimate
them and do not copy them between workloads. The full mode requires nine replay
passes, `--launch-skip 0`, and no `--launch-count` argument:

```bash
KERNEL_NAME='regex:ncclDevKernel.*'
EXPECTED_RANK0_KERNELS="$NSYS_RANK0_MATCHING_KERNEL_COUNT"
EXPECTED_RANK1_KERNELS="$NSYS_RANK1_MATCHING_KERNEL_COUNT"
NCU_TCP_PORT=29641
INPUT_RDZV_PORT=29642
APPLICATION_RDZV_PORT_BASE=29650
EXPECTED_APPLICATION_PASSES=9
NCU_TIMEOUT_SECONDS=1800
NCU_REPORT_IMPORT_TIMEOUT_SECONDS=600
OUTPUT_DIR=/tmp/disagg-rank-tcp-all-attempt-01

AISP_LOCK_GPU_CLOCKS=1 AISP_RAMP_GPU_CLOCKS=1 \
AISP_NCU_PROFILE_REQUESTS=4 AISP_NCU_PROFILE_CONTEXT=256 \
AISP_NCU_PROFILE_DECODE=8 AISP_NCU_PROFILE_BATCH=1 \
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD" \
python -m core.profiling.ncu_torchrun_capture \
  --label disagg-rank-tcp-all --repo-root .. \
  --source "$(git rev-parse HEAD)" --output-dir "$OUTPUT_DIR" --ncu "$NCU" \
  --timeout-seconds "$NCU_TIMEOUT_SECONDS" --replay-mode application \
  --report-import-timeout-seconds "$NCU_REPORT_IMPORT_TIMEOUT_SECONDS" \
  --application-topology rank-tcp --tcp-port "$NCU_TCP_PORT" \
  --expected-application-passes "$EXPECTED_APPLICATION_PASSES" \
  --application-rdzv-port-base "$APPLICATION_RDZV_PORT_BASE" \
  --all-matching-kernels \
  --expected-kernel-counts "$EXPECTED_RANK0_KERNELS" "$EXPECTED_RANK1_KERNELS" \
  --kernel-name "$KERNEL_NAME" --launch-skip 0 \
  -- python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 \
  --rdzv_backend c10d --rdzv_endpoint "127.0.0.1:$INPUT_RDZV_PORT" \
  --max_restarts 0 \
  -m core.harness.torchrun_wrapper --aisp-emit-runtime-provenance \
  --aisp-target-module core.harness.benchmark_worker \
  --aisp-expected-torch-seed 42 --aisp-expected-cuda-seed 42 \
  --module ch15.optimized_disaggregated_inference_multigpu --callable main \
  -- --iters 4 --warmup 5 --seed 42
```

The timeout includes worker startup, all replay generations, and native report
creation. The 1,800-second budget above is for this profiling workload. Set a
budget for each workload and retain enough time for report creation after the
last replay generation. After an interrupted capture, use a new output directory
and available ports.

The report-import timeout is a separate per-report budget for each native
`ncu --import` subprocess after capture. Importing the retained 68-row rank-0
report completed naturally in 80.09 seconds, which exceeds the 60-second default
kept for compatibility. The full example therefore allows 600 seconds for each
rank import. Set both timeout budgets for the workload and report size.

The helper requires the imported NCU CSV to contain exactly that many rows for
each rank. It rejects missing, extra, duplicate, or gapped row IDs and any PID,
device, replay-pass, or kernel-selector mismatch. Duration and active-warps
metrics must be positive. Other finite throughput percentages may be zero and
are recorded in the receipt for review. Percentage metrics above 100 are
rejected.

After capture, reconcile every NCU row with the frozen Nsight Systems inventory.
Bind each report to its own worker PID and physical device. Worker PIDs differ
between the Nsys and NCU executions. Compare the stable rank and device mapping,
kernel order, launch geometry, selector, and workload phase across captures, then
use native NCCL API events from Nsight Systems for direction. The full mode also
records NVTX context in NCU so its kernel ranges can be checked independently.
The all-matching receipt records the expected and observed row counts, marker
policy, and zero metric counts. The helper leaves its coverage and qualification
fields false. Record the external reconciliation, numerical validation, clock
review, and cleanup review separately before claiming complete coverage.

Kernel and range modes retain exact arguments, source and tool hashes, logs,
rank reports, and the owned process-group cleanup result.
`CAPTURE_ARTIFACTS_PRESENT` establishes artifact creation only. Import both rank
reports to check the selected kernels, finite metrics, PID and device identities,
and runtime receipts. Verify numerical outputs and ordinary timings separately.
See the
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
