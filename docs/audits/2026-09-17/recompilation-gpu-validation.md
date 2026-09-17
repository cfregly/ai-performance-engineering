# Recompilation GPU validation

B200 testing exercises the Chapter 14 compiler-policy diagnostic and exposes a
remaining NanoChat decode failure. The rotary tracing defect is fixed. Full-model
compiled serving is not qualified on the tested PyTorch 2.13 runtime.

These are user-approved measurements on a KVM host, classified as non-canonical.
They do not update benchmark expectations or establish a production SLO.

## Compiler-policy measurements

The diagnostic uses an FP32 linear layer with eight input features and four output
features, followed by sigmoid. Each policy ran in three fresh processes with
separate Inductor, Triton, and CUDA cache directories. System and driver caches
are not claimed cold. Each repetition contains 300 attempts across six input
signatures, including empty and singleton inputs, strided inputs, and a signature
omitted from warmup. The order rotates between policies across repetitions.

One B200 was used under an exclusive allocation. The repository clock-lock context
held application clocks at 1,500 MHz SM and 3,996 MHz memory. It restored the entry
state of 1,965 MHz SM, 3,996 MHz memory, and persistence disabled after both the
measurement queue and the final profiler capture. The queue checked for foreign
GPU processes before and during each run.

The table gives the range across three repetitions. Each latency distribution
includes all attempts, including rejected requests and eager fallbacks.

| Policy | p50, ms | p95, ms | p99, ms | Worst request, ms | Serving compile submissions per run | Rejected / eager fallback per run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default, cold in-process compiler state | 0.0641–0.0659 | 0.0824–0.0888 | 133.81–136.30 | 807.11–819.95 | 6 | 0 / 0 |
| Warmup then strict rejection | 0.0681–0.0689 | 0.1456–0.1500 | 0.1597–0.1661 | 0.3250–0.3456 | 0 | 50 / 0 |
| Warmup then explicit eager fallback | 0.0611–0.0628 | 0.0753–0.0775 | 0.1080–0.1132 | 0.1269–0.1534 | 0 | 0 / 50 |

Strict and hybrid runs each compiled five warmup signatures before serving.
Warmup cost was 1.495–1.636 seconds for strict runs and 1.484–1.526 seconds for
hybrid runs. Default runs include compilation in serving latency. Startup cost
has moved outside the serving interval, not disappeared.

All 2,550 successful outputs across 2,700 serving attempts matched the complete
eager reference. The maximum absolute error was `5.96e-8`. The remaining 150
attempts were deliberate strict rejections. There was no implicit fallback.

These values measure host-visible latency with CUDA synchronization. The small
model and synchronization overhead limit generalization. The policies have
different admission and fallback contracts, so this table is not a speedup claim
for equivalent compiled execution or a model-serving load test.

## Matched Transformer comparison

A separate run of `ch14:model_compile_reduced_precision` compares eager and
compiled execution on the same BF16 model, with batch size 24, sequence length
1,536, and 30,212,880 parameters. It uses the existing isolated PyTorch
`2.11.0+cu130` and Triton `3.6.0` environment. No packages were changed. The system
PyTorch 2.13 run was rejected for ambiguous package metadata and contributes no
accepted ratio.

Three fresh paired processes ran eager then compiled, with 50 measured iterations
per variant and 15 harness warmup iterations. Model setup adds 20 eager warmups or
70 compiled warmups. Each pair used separate task cache directories and the same
1,500/3,996 MHz application clocks. Compilation and setup are outside these
steady-state timings.

| Repetition | Eager, ms | Compiled, ms | Eager / compiled |
| --- | ---: | ---: | ---: |
| 1 | 5.7868 | 5.6715 | 1.0203× |
| 2 | 5.7883 | 5.6698 | 1.0209× |
| 3 | 5.7888 | 5.6760 | 1.0199× |

The geometric mean ratio is **1.0204×**, with sample standard deviation 0.00052
across the three ratios. Every run falls below the harness's required 1.05×
threshold and reports `failed_no_speedup`. This is an observed difference of
about 2%, not an accepted performance win.

Input equivalence, runtime parity, and the existing complete-output check passed.
That output check uses permissive tolerances of `rtol=0.5` and `atol=5.0`, so it
does not establish tighter numerical parity. These fixed-shape results do not
measure recompilation tails or NanoChat serving performance.

One earlier attempt was interrupted by the queue's foreign-process detector and
is excluded. The monitor sampled descendants before GPU processes, which could
misclassify a newly spawned worker. The retry snapshots GPU processes first and
recognizes the owned process group. The original interrupted attempt did not
record enough evidence to determine ownership. Its clocks were restored.

Reproduce the pair from `code/` with the isolated runtime:

```bash
python -m cli.aisp bench run --targets ch14:model_compile_reduced_precision \
  --single-gpu --validity-profile portable --gpu-sm-clock-mhz 1500 \
  --iterations 50 --warmup 15 --suite-timeout 1500 --profile none \
  --run-id <unique-run-id>
```

## Profiler evidence

The matched Transformer pair also completed Nsight Systems and Nsight Compute
for both eager and compiled variants. Those four captures are separate from the
three unprofiled repetitions, and their timings are excluded from the ratio.
The receipt records successful profiler statuses and raw artifact digests. The
profiles do not turn the below-threshold result into an accepted performance win.


Nsight Systems 2025.3.2 and Nsight Compute 2025.3.1 both completed successfully.
The final NCU capture used `--clock-control none` so the repository clock lock
remained in control, kernel replay, the basic metric set, and an eight-launch
filter for generated Triton kernels. It captured fused `addmm` and sigmoid
kernels. Instrumented timings are excluded from the latency table.

The profiler capture includes warmup, input creation, eager validation, and serving.
It establishes that generated GPU kernels executed. It does not isolate a
production serving region or establish a NanoChat performance improvement.

## NanoChat findings

The first exact-main B200 run had seven passing tests and one failure. Dynamo
rejected rotary operations that wrote through noncontiguous `out=` views. The fix
uses functional operations while tracing and skips eager-only rotary buffer
caching. Eager buffer reuse and the training path remain covered by regressions.

Standalone compiled CUDA rotary checks pass for sequence lengths 3, 1, and 4.
The complete model passes eager and compiled prefill, then fails during the first
compiled decode. Fresh-cache probes with one and two layers reproduce a native
segmentation fault in Inductor's static launcher during autotuning, without
repository compiler patches or pytest imports.

A diagnostic process with the static launcher disabled also fails, with
`Pointer argument must be either uint64 or have data_ptr method`. That setting
was not adopted. No compiler guards, full-graph requirements, or failure reporting
were weakened. The underlying cause remains unresolved. A vanilla reproduction
alone does not establish an upstream cause.

The Blackwell regression now covers one- and two-layer models. It warms prefill
and the first decode, then checks six subsequent positions under
`fail_on_recompile`. Every step compares full logits and the populated KV cache,
checks the position, and verifies stable preallocated storage. These expanded
tests passed their CPU checks but were not rerun through GPU pytest after the
native abort. The subsequent GPU evidence comes from the standalone probes.

## Clock restoration

The final queue restored GPU 0 to its entry state: 1,965 MHz SM, 3,996 MHz memory,
and persistence disabled. A separate readback at `2026-09-17T02:21:07Z` confirmed
those settings and no compute processes on GPU 0. GPU 1 was then in use by the
B200-port task. Its settings were left under that task's control.

## Reproduction and evidence

From `code/`, the GPU regression is:

```bash
NANOCHAT_RUN_BLACKWELL_COMPILE_TESTS=1 python -m pytest labs/nanochat_fullstack/tests/test_engine_compile_policy.py -q
```

The diagnostic worker command used for each repetition is below. Run it under
`core.harness.benchmark_harness.lock_gpu_clocks`, with fresh task-owned cache
directories and an exclusive GPU allocation. Use `default_compile/strict`,
`guarded_policy/strict`, and `guarded_policy/eager_on_recompile` for the three cases.
The internal worker entrypoint avoids adding a second child-process timeout to
the externally bounded measurement queue.

```bash
python ch14/recompilation_demo.py --_worker-scenario guarded_policy \
  --policy strict --device cuda --backend inductor --iterations 50
```

Source for the measured diagnostic was `b7a09066e73e9b159755ef1ea695d26977468928`.
The diagnostic imported PyTorch `2.13.0+cu130` and Triton `3.7.1`, with CUDA 13.0,
driver 580.173.02, and NVIDIA B200. The system installation has conflicting
package metadata. These diagnostic observations do not satisfy the harness
runtime-provenance gate. The matched Transformer comparison described above
uses a separate, existing virtual environment with unambiguous metadata.
The NanoChat runtime source hashes and the later comment-only edit are recorded
in the receipt.

Local focused checks passed 10 tests, skipped two Blackwell cases, and deselected
28 unrelated tests. Ruff correctness checks passed. Local pytest used
`--noconftest` because the existing macOS `rustbpe` extension failed to link.
The full repository suite was not rerun.

- [Machine-readable receipt](recompilation-gpu-validation.json)
- [All 2,730 warmup and serving records](recompilation-requests.csv)
- [Sanitized profiler summaries](recompilation-profiler-summary.json)
- [Chapter 14 guidance and attribution](../../../code/ch14/recompilation.md)

Raw profiles and logs remain in the task's operator artifacts. Published evidence
omits live infrastructure identifiers and includes digests for those raw files.
