# Collective diagnosis tool

`collective_diagnosis_tool.py` collects rank-level evidence for four common
distributed symptoms: a healthy reference, a rank that reaches a collective
late, GPU work that competes with communication, and a dependency that removes
compute and communication overlap. It is a diagnostic tool. It does not produce
a benchmark speedup or identify a network root cause from duration alone.

The runner requires `torchrun`, NCCL, and at least two CUDA ranks. Unsupported
hosts return a structured `SKIPPED:` artifact and a nonzero exit. The tool does
not substitute a CPU or single-GPU path.

## Run a sweep

Run from `code/`:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m ch04.collective_diagnosis_tool run \
  --output artifacts/collective-diagnosis.json
```

The default sweep covers `all_reduce`, `all_gather`, `reduce_scatter`, and
`all_to_all`. It uses 256 KiB, 4 MiB, and 32 MiB logical payloads per rank. Each
scenario gets three warmups and ten retained rounds. Measured scenarios rotate
their order across rounds to reduce fixed ordering bias.

Use a smaller sweep while checking a launch:

```bash
torchrun --standalone --nproc-per-node=2 \
  -m ch04.collective_diagnosis_tool run \
  --output artifacts/collective-smoke.json \
  --collectives all_reduce,reduce_scatter \
  --message-sizes 256KiB,4MiB \
  --warmups 1 \
  --rounds 3
```

For a multi-node launch, use the site's normal `torchrun` rendezvous settings
and write the rank-zero artifact to durable storage. Keep the same payload,
dtype, rank count, and compute settings across compared scenarios.

## What each scenario changes

| Scenario | Explicit change | What to inspect |
| --- | --- | --- |
| `healthy` | No injected disturbance | Reference step and collective distributions |
| `delayed_rank` | The selected rank sleeps on the host before enqueue | Readiness-to-enqueue time and step tail |
| `competing_gpu_workload` | The selected rank launches extra matrix multiplies on another CUDA stream | Competing CUDA time, collective CUDA time, and step time |
| `forced_dependency` | Every collective stream waits for the base compute event | Step time compared with collective CUDA time |

The default injected rank is the last global rank. Change it with
`--injected-rank`. `--delay-ms`, `--compute-iterations`, and
`--competing-iterations` control the injected work. These labels are stored in
every experiment. The analyzer reports them as known interventions, while it
keeps any deeper transport or fabric explanation unassigned.

The requested message size has the same meaning for every scenario within one
collective. The artifact records requested bytes, logical bytes after alignment,
and actual input and output buffer bytes. `reduce_scatter` needs a larger input
buffer, while `all_gather` needs a larger output buffer. Compare scenarios only
when their `comparison_key` values match.

## Timing fields

Every warmup and measured round is retained. A round contains:

- readiness, collective enqueue, collective completion, and full-step
  completion markers in local monotonic time and Unix wall time
- readiness-to-enqueue and enqueue-to-completion host durations
- CUDA event duration for the collective stream
- CUDA event duration for the base compute stream
- optional CUDA event duration for the competing stream
- full step time from local readiness to completion of all scheduled work
- a correctness result from the completed collective output

These fields answer different questions. CUDA collective duration shows time on
the collective stream. Enqueue-to-completion includes host observation and wait
behavior. Step time also includes delayed enqueue, compute, dependencies, and
the injected competing work. A larger value in any one field does not prove a
fabric fault.

Monotonic timestamps can be compared across ranks on the same host. They cannot
establish arrival skew across hosts. To enable a bounded cross-host wall-clock
comparison, provide an external synchronization receipt to the runner:

```json
{
  "synchronized": true,
  "max_error_ms": 0.25,
  "method": "measured PTP offset receipt"
}
```

```bash
torchrun ... -m ch04.collective_diagnosis_tool run \
  --output artifacts/collective-diagnosis.json \
  --clock-sync-evidence artifacts/clock-sync.json
```

The analyzer also accepts `--max-clock-skew-ms` when an operator has a measured
bound from an external clock check. It records that value as an operator-supplied
bound. Without either form of evidence, cross-host enqueue spread stays null.

## NVTX and profiler traces

Add `--nvtx` to mark each phase, collective, payload, scenario, and round. An
Nsight Systems launch can capture these ranges:

```bash
nsys profile --trace=cuda,nvtx,osrt --output collective-diagnosis \
  torchrun --standalone --nproc-per-node=2 \
  -m ch04.collective_diagnosis_tool run \
  --output artifacts/collective-diagnosis.json \
  --nvtx
```

Add `--profile-dir artifacts/traces` to export one PyTorch profiler trace per
rank. Profiler capture is intrusive. The artifact records that state, and the
captured timing should be used for diagnosis rather than an unprofiled latency
comparison.

## Analyze a retained artifact

Analysis uses only the Python standard library. It works on a CPU host without
PyTorch:

```bash
python -m ch04.collective_diagnosis_tool analyze \
  artifacts/collective-diagnosis.json \
  --format text
```

Write machine-readable analysis for another tool:

```bash
python -m ch04.collective_diagnosis_tool analyze \
  artifacts/collective-diagnosis.json \
  --output artifacts/collective-analysis.json
```

The analysis retains raw evidence in the source artifact and produces grouped
distributions, scenario ratios against the healthy reference, rank coverage,
correctness state, and normalized signals. Slowdown signals compare each host
with its own healthy rounds. Enqueue-spread signals are emitted only when the
clock domain supports the comparison. Every signal retains the synthetic
scenario and its injected condition.

Use `run --workload-id <workload>` when collecting matching fabric evidence.
Pass the same declared ID to `fabric-diagnose --workload-id`. Host signals retain
their hashed host identity. Cross-host aggregate signals require both clock
evidence and a declared workload ID. Without that ID, aggregate timing remains in
the analysis but is not exported as a scoped cross-layer signal. Collective and
payload group labels stay separate from physical path identity.

The analyzer rejects empty groups, missing ranks or rounds, failed correctness,
nonfinite values, out-of-order timestamps, and durations that disagree with
their retained markers. Invalid and rejected artifacts produce a nonzero exit
and no normalized signals.

## Provenance and evidence boundary

The runner records the Git state, Python, PyTorch, CUDA, NCCL, device properties,
selected NCCL settings, visible peer access, and rank-to-host mapping. Rank zero
also performs read-only `nvidia-smi` clock and topology queries when available.
Unavailable metadata is recorded as unavailable. The tool never changes GPU
clocks.

Artifacts always use `diagnostic_only_noncanonical`. An external harness can
attach a JSON receipt through `--harness-gates`, but the tool retains it as
unverified supplemental provenance and does not change the classification. The
receipt does not make the output a benchmark win or prove a fabric cause.
Canonical claims still require the repository's target hardware, clock,
correctness, repeat, profiler, and publication gates outside this tool.
