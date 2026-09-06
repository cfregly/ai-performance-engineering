# Codebase repair checkpoint

This change repairs benchmark correctness, profiler lifecycle and reporting,
distributed launch behavior, and environment diagnostics. It is a validated
repair checkpoint, not a claim that every example is maximally fast or that the
complete hardware matrix has passed.

## Main repairs

- Preserve real numerical verification and equivalent workloads in block
  scaling, KV-cache compression, attention, MoE, and distributed examples.
- Fix BF16-to-FP8 double rounding in KV-cache quantization. A one-dimensional
  FP32 scale requests FP32 multiplication while retaining the preallocated FP8
  output. The existing tolerance and timed workload are unchanged.
- Validate complete Nsight Compute application-range captures, bind their
  provenance to the actual report, and keep descriptive range duration separate
  from kernel timing and benchmark speedup across the CLI, MCP and summaries.
- Improve CUDA/NVSHMEM bounds and lifecycle checks, process cleanup, device
  visibility handling, and structured unsupported-environment diagnostics.
- Keep ordinary examples runnable through Python or local `torchrun`.
  Slurm is optional; a scheduler can reserve one allocation for a whole sweep.
  Direct all-reduce runs now use unique artifact names without a Slurm job ID.

## Retained validation

The integrated CPU run before the final small KV/portability corrections
collected 5,084 tests: 4,600 passed, 481 skipped, and three failed because the
local Torch 2.8 runtime lacks the expected precision API. Those failures remain
recorded; this is not a full-suite pass. Static checks covered 2,788 Python files
and 161 shell files. The final KV-focused local suite passed nine tests and
skipped four CUDA-only tests.

Actual execution on a shared, virtualized two-B200 host with Torch 2.9.1 and
CUDA 13 confirmed full block-scaling numerical checks and profiler/report
collection, plus both two-rank serving regressions. The standalone public CLI
application-range capture passed with all five requested metrics, the complete
selected NVTX range, exact report provenance and verified child runtimes.
Every completed stage's owned processes drained.

The original KV output check failed. A reduced real-class B200 diagnostic
reproduced 31,600 tolerance violations with identical inputs; explicit FP32
multiplication removed all violations without changing tolerance. The correction then passed all 13 append-path tests and full default-workload
input/output verification on the B200. That replay still ended as a profiler
failure: sidecar materialization incorrectly applied a 16 MiB source-file
limit to the valid 41,843,422-byte optimized NCU capture. The repair separates
streamed artifact hashing from source-worker limits and prevents failed
required captures from updating saved expectations. Retained-report
reprocessing and the complete merged-revision replay remain separate gates.
The reported memory saving and single-pair timings are not accepted
performance wins while required validation remains incomplete.

## Remaining work

### September 6 direct execution follow-up

PR #15 merged as `c40767b8294d0471082aea9cfb8f5636b8961aec`. The final
candidate's hosted CPU suite completed with 4,598 passed, 492 skipped, and no
failures or errors (5,090 collected). Static, dashboard, and native build checks
also passed. Both retained KV NCU reports were successfully reprocessed after
the large-artifact hashing repair; the original interrupted/failed runs retain
their original status.

The unfiltered GPU test suite ran directly on the two-B200 host on
September 6 starting at 10:05:44 UTC, using the clean merged source, Torch
2.9.1+cu130, and `python -m pytest tests -q --tb=short`. It completed in
1,831.28 seconds: **5,008 passed, 77 skipped, and 5 failed**. Two-rank NCCL
tests executed. One failure selected system CMake 3.28.3 instead of the
already-installed 3.31.10; four process-lifecycle checks encountered exited
children retained as zombies by the outer validation supervisor until final
drain. The supervisor ultimately drained all owned processes. All five tests
then passed on the same merged source in 26.31 seconds after selecting CMake
3.31.10 and reaping the supervisor's adopted, exited children during execution.
Both receipts are retained: the original full run remains failed, and the
targeted rerun passed and drained its processes. No Slurm allocation is required.
The subsequent example
inventory includes benchmark pairs, registered demos, and tools; the pytest
suite alone does not establish that all of those entrypoints executed.

PR #16 (`7b7728a8dd043a1e9a07a29565626dfed89c8984`) adds explicit optional
device-identity and evaluation contracts, and distributed workload metadata.
It merged as `3d7982a8860506505d1ca24b8c3754f3826fdba1` after CPU validation,
dashboard, static analysis, and dual-architecture native build checks passed.
Its 86 focused contract tests passed on the B200 host with no skips in 9.29
seconds. Declared collective algorithms are not profiler observations,
and dataset hashes do not prove semantic freedom from contamination.

The integrated local contract/harness suite passed 172 tests and skipped 24
hardware/platform cases. Evaluation receipts now originate in the actual worker,
survive subprocess serialization on failure, and finalize after teardown; source
or threshold drift rejects timing results. The default 873-file benchmark lint
scan passed without errors or warnings.

Discovery confirmed 486 benchmark targets, 29 demos, and 34 tools. Ten registered
demo/tool paths previously had factories but no invoked workload entrypoint;
they now invoke their actual standalone execution paths. Real CLI tests with
CUDA hidden reject execution with explicit diagnostics rather than exiting
successfully after import. These remain
demo/tool runs, not baseline-versus-optimized performance evidence.

The broad sweep exposed two practical continuation gaps: aggregate preflight can
reject a whole bucket, and an exception escaping a chapter can abort later
chapters. The direct validation queue therefore gives each target a separate
process and result record, continues on nonzero exits, and preserves earlier
results. Native tier-1, cluster, and fabric entrypoints run separately from this
486-target pass. A follow-up repair now isolates explicit filtered targets when
continuation is requested, records preflight failures and ordinary exceptions,
and proceeds to later units. Its 78 focused and existing regression tests passed.
Run-level aborts still propagate. Batch target resolution and contiguous-prefix
native resume accounting remain improvement items.

The first direct demo pass attempted all 29 registered demos: 23 exited
successfully and six required follow-up. One requires GB10 coherent memory;
two need explicit workload arguments; the other three exposed a TMA alignment
failure, a CUTLASS dependency preflight failure, and a distributed launcher
selecting another environment's Python through `PATH`. Tools also began
executing. Completed results were preserved before replacing the coordinator
to resume with corrected source and arguments.

The launcher repair uses the selected Python's `torch.distributed.run` for demos
and profiler workers, matching benchmark timing. A real two-worker identity test
passes even with another environment's `torchrun` first on `PATH`. The combined
launcher and TMA static contract selection passed 82 tests with two skips.
The CUDA alignment repair then passed the actual production kernel's three
full-output and canary shapes under Compute Sanitizer with zero reported errors,
and the original 2048-square demo completed. The focused B200 regression
selection passed 90 tests with no skips. The initial standalone validator
compile exposed a separate flag defect: `-arch=sm_100a` emits generic PTX as well;
the architecture-specific caller requires an explicit `compute_100a` to
`sm_100a` code-generation target. The launcher, alignment, and continuation
repairs are committed in `a738687c28c10161e3000a97ce2ed640795872b6`; they are
included in PR #17, which merged as
`cd35b3bd36970a5dcf3b91f89d5a317b3adb80d7` after all four hosted checks passed.
The validator code-generation flag repair is included in that merge.

CI's broader `--include-unpaired --fail-on-warnings` lint scan also passed:
932 files, zero errors and warnings. The contract checker recognizes exact
registered demo/tool paths as standalone entrypoints while continuing to reject
`__main__` blocks in paired benchmark modules.

### Resumed examples and measured KV graph candidate

The resumed queue uses clean source
`57e754fe5eb87d8669e800a0dca3e05bef31a1aa`, retaining prior source identities
for reused results. All 29 registered demos and 34 tools have been attempted.
The reconciled exit counts are 25 successful demo exits and four nonzero exits,
and 32 successful tool exits and two nonzero exits. These are execution counts,
not inference-quality claims: some tools are calculators, simulations, asset
generators, or dependency probes. The GB10-only demo remains unsupported on B200.

The two-B200 reruns exposed invalid collective keyword arguments in expert
parallelism, unsafe unbatched ring transfers and fully masked softmax blocks,
and an incompatible full-graph compile request around Transformer Engine's
disabled compiler boundary. The DTensor tool also confused visible device IDs
with process ranks. Repairs are committed in `99a365de6` with 57 focused CPU
checks passing and three hardware checks skipped. Their B200 results follow
below. The default dynamic-router tool now fails if requested vLLM execution
cannot initialize; explicit synthetic mode records its provenance. Its earlier
successful exit must not be reused as real vLLM evidence.

The seven-caller TMA validator compiled and found a separate odd-width defect:
the Chapter 7 copy kernel overwrote 387 row-padding values for a 129-by-385
logical tensor with leading dimension 400. The follow-up preserves TMA stores
for full tiles and bounds ordinary stores on partial tiles. Strict full-output
and canary checks are unchanged; the subsequent B200 closure is recorded below.

The standalone KV experiment compared the exact 256-step FP8 body with graph
replay on the same buffers. Eight interleaved AB/BA rounds produced 80 samples
per path, after five warmups per path:

| Diagnostic | Eager FP8 body | CUDA graph replay |
|---|---:|---:|
| Median CUDA-event time | 36.7343 ms | 23.8563 ms |
| 5th–95th percentile | 36.7258–36.7649 ms | 23.8482–23.8694 ms |
| Profiled kernel executions | 8,192 | 8,192 |
| In-range CUDA runtime calls | 16,385 | 3 |

Raw FP8 bytes and FP32 scales matched bitwise before and after measurement;
the full BF16-reference check also passed at its existing tolerance. Graph
preparation cost 175.357 ms including three preruns, with 1,024 additional
allocated bytes and 4 MiB additional reserved memory. This is approximately
35.1% lower latency for the existing FP8 body, with matching kernel work in the
Nsight trace. It is a noncanonical mechanism experiment, not an overall win
against the BF16 baseline. The integrated candidate still needs its full paired
benchmark and required profiler checks.

The first two multi-GPU benchmark targets completed timing and Nsight Systems
capture but retained `failed_profiler` outcomes. The pure-copy Chapter 2 target
produced no NCU kernel report, and Chapter 4 gradient fusion lacked an explicit
timed NVTX range for direct distributed NCU capture. These failures remain
visible while the queue continues.

Practical improvement priorities are consistent standalone launch metadata,
real behavioral tests for distributed paths, explicit synthetic/unsupported
result categories, and profiler requirements that reflect the workload being
measured. The cost calculator now also ran with explicitly illustrative inputs;
those values are not measured B200 power or token throughput.

### Integrated lab reruns and remaining launch repairs

The direct repair wave on `90bc4f720eaf2881213a2a151087aafe05861730`
passed **64 focused GPU tests with no skips**. The expert-parallel, compiled
decode, and DTensor entrypoints completed with two actual B200 workers. The
context-parallel demo completed with `CUDA_LAUNCH_BLOCKING=1`, then passed a
separate normal asynchronous two-rank rerun. All completed stages drained
their owned processes.

The integrated `labs/kv_optimization:kv_standard` pair passed its full default
workload, correctness checks, and both required profilers with graph replay
enabled. This closes the integrated graph candidate's execution gate on this
virtualized host. The paired run used 16,566 MiB versus 32,948 MiB for BF16,
a 49.72% memory reduction, while latency was 22.999 ms versus 17.334 ms.
It passed the declared memory goal; the FP8 path remained slower than BF16.
It does not turn the earlier FP8-versus-FP8 experiment into
an overall performance win against BF16 or into canonical hardware evidence.

The broader sweep had completed 13 multi-GPU target attempts before the repair
wave; none had an overall passing result. Some timing and Nsight Systems
captures completed, but required profiler stages failed. The gradient-fusion
range repair allowed NCU to start, then kernel replay stalled for more than
12 minutes. The owned stage was interrupted with its artifacts retained and
an explicit failed result. Distributed collectives need concurrent execution;
the follow-up selects application-range replay instead of serial kernel replay,
consistent with NVIDIA's [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
That follow-up still requires a fresh successful capture.

The launch audit also found paired distributed modules whose torchrun specs
executed their files without invoking the defined `main()` function. The repair
uses an explicit named-function worker adapter, preserving paired modules as
import-only. CLI tests must demonstrate that the requested worker entrypoint
is invoked; a successful import is insufficient execution evidence.

The seven-caller TMA wave passed cases 0, 5, and 6 under memcheck. Cases 1, 2,
and 4 exposed the same 387 overwritten padding values on an odd-width tensor;
case 3 required the relaxed-constexpr compiler flag already used by native
builds. Commit `284c89346` bounds partial-tile stores in the remaining callers
and adds that flag to the standalone validator. Full tiles retain their TMA
path. Its real host helper regression checks all logical output, padding, and
allocation guards. The subsequent B200 gate on `284c89346` passed all seven
callers and all 42 full-output/canary cases, with Compute Sanitizer enabled.

The same commit removes the NVFP4 tool's FP16 Transformer Engine fallback and
unused graph capture. It requires an explicit single-rank TensorRT-LLM NVFP4
engine with usable assets and actual generation output. Without that engine,
the tool reports an unsupported prerequisite and exits nonzero. A positive
engine-load/generation claim remains gated by a real engine asset. The combined
local TMA, distributed-range, and NVFP4 selection passed 64 tests and skipped
the real-engine-only test. The B200 TMA/NVFP4 selection passed 36 tests with
that same asset-dependent skip. The real NVFP4 CLI returned nonzero with the
explicit missing-engine diagnostic. These reruns drained all owned processes.

The next broad pass prioritizes execution and correctness across the complete
inventory with profiling explicitly disabled, while repaired profiler paths
and optimization candidates receive separate captures. Those broad results are
diagnostic execution evidence and cannot satisfy canonical profiling gates.

### Distributed timing and output follow-up

The execution-first sweep attempted the original 22 multi-GPU inventory entries
and then reached the single-GPU examples. Four distributed targets exited zero
and 18 returned nonzero. These counts do not qualify distributed performance:
the audit found that the harness recorded whole-process wall time even when a
worker emitted iteration latency, and gradient fusion compared an unrelated
parent-side probe. Successful GEMM and batched-GEMM runs are retained separately.

Commit `8490afa84` adds an explicit worker-timing contract. An opted-in worker
must emit one finite positive rank-0 iteration mean and declare how many measured
iterations it represents. Launch duration remains a separate metric; an
aggregate mean does not acquire invented percentiles. Eleven focused timing
tests passed, including real subprocess controls. Existing process-duration
results remain labeled as such and must not be reused as iteration timings.

Commit `9c19a76a6` replaces gradient fusion's repeated in-place FP16 SUM, which
can overflow over 55 reductions, with a stable gradient-average primitive.
The paired workload remains 2,048 tensors of 4 KiB, five warmups, and 50 measured
iterations. Fresh per-rank results carry actual timed output and an independent
FP32 average reference. Two-rank Gloo controls passed; the NCCL and application-
range reruns remain separate B200 gates. Old SUM timings are incompatible with
this corrected operation.

The symmetric-memory pair now transports actual per-rank inputs and receive
outputs and rejects missing or incorrect peer results. Rank-dependent inputs
make a local-copy mistake distinguishable despite identical global seeds.
Pipeline workers also select their own nested stage rather than stage zero,
and the optimized virtual path traverses its nested layer lists correctly.
The pipeline pair still has a separate parent-side verification path; passing
that path does not certify distributed child tensors.

The vLLM tool failed honestly because the prepared FlashAttention 4 environment
exposes `flash_attn` without the older `flash_attn.ops` module expected by vLLM
0.16. An existing isolated environment imports the same Torch 2.9.1+cu130 and
vLLM without that namespace collision; its actual inference rerun is pending.
No shared runtime was changed. The model fetcher now returns absolute paths and
checks the presence of every indexed weight shard before treating a cached
directory as complete. A config file alone is insufficient.

The remaining launch audit identified five more import-only targets in Chapter
17 and the cache-aware disaggregated lab, plus seven targets that need fresh
child-output verification. Thirty distributed-training adapters already report
their missing verification contract explicitly. GPU inventory routing is also
being repaired because imported benchmark classes can require multiple devices
without repeating those requirements in their thin wrapper files.

### Direct B200 wave 6 and corrected sweep routing

All seven wave-6 stages on `9c19a76a6` terminated with confirmed child-process
drain. The focused tests and unprofiled gradient pair exited zero. The other
five stages retained failures; they are not qualified passes.

| Target | Observed result | Interpretation |
| --- | --- | --- |
| Gradient average, 2 B200 | 33.231804 ms baseline; 0.045565 ms fused | Actual full child outputs passed. One aggregate per arm; no repeated interleaved performance claim. |
| Pipeline parallel | 15.577399 ms baseline; 23.287443 ms 1F1B | Candidate slower; performance contract failed. Parent virtual verification remains a separate limitation. |
| Tensor parallel | 9.808 ms baseline; 9.973 ms asynchronous | No measured speedup; performance contract failed. |
| Symmetric-memory performance | Child signature rejected | `float32` and `torch.float32` were compared without canonicalization; `cc72c9cac` repairs the comparison. Full peer-output checks remain mandatory. |
| Gradient profiler bundle | Nsight Systems captured both arms; NCU and Torch failed | NCU application-range capture hit NCCL timeouts; Torch selected the in-process factory path. Neither failure is promoted to a capture pass. |
| Router evaluation | vLLM workers rejected `stdout.fileno()` | `af1708d1b` uses descriptor-backed output capture. Actual model execution is being rerun in the existing isolated vLLM environment. |

The dtype repair passed 49 focused CPU tests, including canonical and
noncanonical dtype spellings, real dtype mismatches, and incorrect output from
either rank. The vLLM capture repair passed 13 model/router tests, including
native descriptor writes and a real subprocess diagnostic on the failure path.
These CPU checks do not substitute for the subsequent GPU runs.

Commit `4d3c568ab` follows benchmark factories and inherited class metadata
statically, without importing workload modules. The inventory remains 486
targets, now correctly classified as 423 single-GPU and 63 multi-GPU targets.
The multi-GPU set contains 49 torchrun targets and 14 with in-process execution.
Routing tests also preserve an explicitly declared four-GPU minimum. The
execution-first sweep resumed on `af1708d1b` with those launch distinctions,
profile mode `none`, and forced reruns of previously invalid distributed
successes. Successful prior stages retain their original source identities.

### Direct B200 wave 7 and real router inference

All eight wave-7 stages on `40a4b9d17` drained their owned processes. The
focused selection passed 106 tests. The symmetric-memory performance pair
passed complete peer-output checks (73.244 ms baseline, 27.442 ms optimized).
The four Chapter 17 pairs now invoke explicit worker entrypoints and validate
their actual complete decode output against an independent prefill reference:

| Pair | Baseline / optimized iteration mean | Result |
| --- | --- | --- |
| Batched | 1031.455 / 4.137 ms | Correctness and timing contract passed |
| Prefill/decode overlap | 1119.273 / 283.281 ms | Correctness and timing contract passed |
| Long-context TPOT | 3385.711 / 831.877 ms | Correctness and timing contract passed |
| TTFT | 848.487 / 209.079 ms | Correctness and timing contract passed |

These are single aggregate measurements per arm on a virtualized host. They
are diagnostic results, not repeated interleaved or canonical speedup claims.

Cache-aware distributed inference passed full child-output checks but failed
its performance contract: 13.672 ms baseline versus 14.592 ms optimized. With
one prefill and one decode rank, both policies select the same decode rank;
this topology cannot demonstrate fewer migrations between decode ranks.
The harness now preserves valid measured ratios below 1.0 rather than
clamping a regression to 1.0.

Both gradient arms passed Nsight Systems and actual Torch worker profiling.
The optimized arm also passed Nsight Compute after restricting its capture
range to rank zero while both ranks execute their collectives. The baseline
NCU capture timed out; its partial report remains a failure. Commit
`52560085c` profiles one representative iteration of the same workload while
leaving the ordinary 50-iteration measurement unchanged. Its fresh B200
capture and the repaired MoE hybrid pairs are the next focused batch.

The isolated vLLM runtime completed real generation for all 16 router-evaluation
prompts after the stdout-descriptor repair. Its result explicitly reports
`vllm_quality_with_synthetic_telemetry` and mixed evidence: quality uses vLLM,
while latency and routing telemetry are synthetic, and throughput is derived
from that synthetic latency. Those printed latency/throughput values are not
measured serving performance or valid cost-per-token inputs.

The breadth sweep was stopped after an owned NVSHMEM pipeline stage stalled;
its failure and confirmed process drain were retained. The pipeline transport
and additional NVSHMEM child-result contracts are being repaired. The next
breadth phase assigns disjoint single-GPU target lists to the two B200s so
coverage can advance while the remaining distributed repairs are checked.

### Direct B200 wave 8 and parallel single-GPU coverage

Wave 8 on `52560085c` completed all four stages with confirmed process drain.
Focused tests passed. Both MoE hybrid variants passed the baseline replay but
rejected the optimized final route assignments. A separate real two-rank,
six-step diagnostic started both paths with identical parameters and input.
Initial routes matched; BF16 output differences of 0.002–0.004 and gradient
differences up to 1.9e-6 became different BF16 AdamW parameter updates. Routes
then diverged on later steps. The mixed-precision training repair remains open;
the output and route checks have not been relaxed.

The one-iteration gradient profiler retry still failed baseline NCU replay.
Both Nsight Systems and Torch captures succeeded, and normal 50-iteration
gradient correctness passed (33.012 ms baseline, 0.045810 ms fused). A failed
profiler bundle remains failed even when ordinary execution succeeds.

Commit `f99426593` preserves all four distributed work-contract fields when
coercing serialized signatures; 34 CPU tests passed, including real Gloo
execution. Commit `4c640451e` fixes foreign-process inspection to query the
selected CUDA device's NVML identity. Its 42 CPU controls passed, and a live
B200 check confirmed that logical device zero under `CUDA_VISIBLE_DEVICES=1`
resolves to physical GPU 1.

The full inventory on `4c640451e` contains 486 targets: 422 single-GPU and 64
multi-GPU. At 13:36 UTC, two direct queues began disjoint sets of 211 single-GPU
targets each. Both initial GEMM targets entered real isolated worker execution.
These queues use profile mode `none`; their timings are diagnostic coverage
results. No two-GPU workload overlaps this phase. The remaining distributed
training manifest contains 61 explicit variant commands with source hashes;
those direct runs and final-source full-suite validation are still pending.

### Direct B200 wave 9 and first breadth results

The two single-GPU queues stopped cleanly after 55 terminal targets: 45
succeeded, seven skipped, and three failed. Their results and original source
identity are retained. One skip requires Grace hardware. Six others exposed
GPU-routing metadata problems. The failures were pinned-prefetch training
correctness and two launch-bounds pairs whose measured ratios were only about
1.003, below their existing performance contract.

Wave 9 on `b587cdde9` completed all ten stages with confirmed process drain.
Focused tests passed, including actual CUDA prefetch training across repeated
setup/teardown cycles. The NVSHMEM training pair passed full child-output and
reference checks on two B200s: 6.203 ms baseline and 0.913 ms optimized. This
single aggregate remains diagnostic timing, not a qualified speedup claim.

The pipeline, training-pattern, broadcast, and ring workers exposed global
rank-specific RNG reseeding after real execution. Commit `d38dfdf15` replaces
those input generators with local generators so rank-distinct data does not
change the harness seed. Measured and reference sequences remain identical.
The affected CPU selection passed 77 tests; fresh GPU checks are pending.

Pinned-prefetch setup previously retained its batch cursor; that restart bug
was fixed and its real CUDA regression passed. Full default execution still
failed because adaptive timing can give the faster stateful training arm more
optimizer updates. Commit `dcee02a05` fixes both arms to the configured update
counts, preserving the ordinary 20 measured iterations and ten warmups.
Focused CPU checks passed; the full B200 pair is being rerun.

Both MoE hybrid variants still rejected optimized final route assignments.
FP32 optimizer state, routing, and accumulation delayed divergence but did not
resolve it. A fresh six-step, two-rank diagnostic on `b587cdde9` found identical
initial routes and output differences of 0.00390625, followed by four and three
route mismatches on the final step. Exact route and complete-output checks
remain intact. This repair is still open.

Commit `1a8958ac1` tests a 512-thread, three-block launch bound for the
register-heavy kernels. Baseline geometry, input size, arithmetic, and repeat
counts remain unchanged. It is an optimization hypothesis until real
correctness, interleaved timing, and counter evidence establish the result.

### Direct B200 wave 10 and resumed coverage

Wave 10 on `1a8958ac1` completed all nine stages and drained their processes.
All 40 focused tests passed on the real CUDA runtime, including architecture
compilation checks. Ordinary benchmark runs used profile mode `none`.

| Pair | Baseline / optimized mean (ms) | Actual result |
| --- | --- | --- |
| Pinned-prefetch MLP | 1.236 / 0.569 | Complete output and performance contract passed |
| NVSHMEM broadcast | 0.04346 / 0.03318 | Two-rank complete output and performance contract passed |
| NVSHMEM training patterns | 24.007 / 22.878 | Complete output passed; 1.04934 ratio missed the unchanged 1.05 threshold |
| Symmetric-memory ring | 0.02337 / 0.09875 | Complete output passed; optimized transport was slower |
| Launch bounds, Python extension | 17.817 / 17.578 | Complete output and performance contract passed |
| Launch bounds, CUDA binary | 0.18485 / 0.18247 | Complete output and performance contract passed |

Both NVSHMEM pipeline variants passed baseline execution and rejected the
optimized full output. About half the output values differed from the serial
reference, with maximum absolute difference 5.0625. The seed repair allowed
the real transport correctness check to run; it did not establish pipeline
correctness. The signal/handoff implementation remains under repair.

A separate launch-bounds experiment on the same source used four ABBA blocks
and five samples per slot, totaling 40 samples per arm. Every slot retained
bitwise equality of all 1,048,576 output elements. Median time for 96 kernel
launches was 17.8022 ms baseline versus 17.5111 ms optimized, a 1.01662 ratio.
Sample standard deviations were 0.01295 and 0.00211 ms. Both Nsight Compute
captures succeeded: the 1024-thread baseline used 36 registers per thread,
while the 512-thread candidate used 39; measured active-warps occupancy rose
from 48.93% to 68.37%. This supports the launch-geometry mechanism. The host
is virtualized and the interleaved probe did not lock clocks; these remain
diagnostic measurements, not a canonical performance claim.

Commit `6cf523280` fixes target-level sweep continuation/resume and GPU
classification. It retains successful verified targets individually, invalidates
stale successes after a newer failed or missing-summary attempt, and preserves
frozen target identities. The focused CPU selection passed 151 tests. The
inventory is now 486 targets: 417 single-GPU and 69 multi-GPU. Five examples
require two visible GPUs in one process; they do not require torchrun. The
disaggregated/reinit examples explicitly retain their one-rank default.

Commit `96027d064` extends the fixed-update correction to 13 other stateful
training targets, changing 22 variant configurations while preserving their
existing iteration/warmup counts. The configuration/lifecycle selection passed
14 CPU tests with two actual-CUDA cases skipped. Full execution of those
targets remains part of the sweep.

At 14:17 UTC, two fresh direct queues resumed disjoint halves of the 417
single-GPU targets on `96027d064`. They share the combined retained ledger,
preserve its original source identities, and force repaired early training
and one-rank metadata targets to rerun. Distributed work will run after this
phase drains. Neither examples nor these queues require Slurm.

### Direct B200 wave 11 and breadth checkpoint

Wave 11 executed on source `433815990`. All nine supervised stages terminated
with confirmed descendant drain, and the focused XML records 37 passed tests
with no failures, errors, or skips. These runs used the portable validity
profile and profile mode `none`; their timings are diagnostic observations.

| Target | Observed result | Interpretation |
| --- | --- | --- |
| MoE six-step diagnostic | 12 rank-steps; zero route mismatches, output delta, gradient delta, and parameter delta | The corrected mixed-precision paths remained exact across both ranks and every step. |
| MoE hybrid EP | Complete input and output verification passed; ratio 0.933636 | Correctness passed, but the unchanged 1.05 performance threshold failed. |
| MoE hybrid EP multi-GPU | Complete input and output verification passed; ratio 1.023102 | Correctness passed, but the unchanged 1.05 performance threshold failed. |
| TMA copy default target | Complete execution and verification passed | This is the current runtime result for the repaired 2D host oracle. |
| Symmetric-memory ring demo | Seven changing-input, full-output checks passed on both ranks; the real default benchmark function also completed | This closes the observed send-before-receive deadlock for this direct two-rank path. |
| Disaggregated communication | Complete input and output verification passed; ratio 0.933842 | Correctness passed, but the unchanged 1.05 performance threshold failed. |
| Both NVSHMEM pipeline aliases | Each baseline passed; each optimized launch exceeded its timeout at about 302 seconds | Neither optimized pipeline run is a runtime pass. |

The first and resumed breadth shards recorded 85 unique terminal targets: 77
passed, six skipped, and two failed. Wave 11 separately reran the previously
failed TMA target successfully; that retest updates its latest evidence but is
not counted as an additional unique breadth target.

Three subsequent source repairs await wave-12 execution. Commit `dfb5be2d3`
eagerly initializes each NVSHMEM pipeline NCCL sideband in deterministic setup
order before asymmetric point-to-point traffic. Commit `e72c0e3af` gives the
communicator lifecycle examples a real one-rank store and direct execution path.
Commit `086f53f07` uses reusable `addmm` outputs for the reduction MLP bias.
Their source and focused CPU checks do not establish B200 runtime results.

The prepared distributed-training manifest still contains 61 commands that
have not run because required dependencies are missing. Manifest validation and
supervisor controls are preparation evidence only. Wave-11 timings, breadth
results, and later source repairs do not support public performance claims.

### Direct B200 waves 12–13 and training retry

Wave 12 executed on `e72c0e3af`. All eight supervised stages drained their
descendants, and the focused XML records 23 passes with no failures, errors, or
skips. The reinitialization pair passed complete verification at 430.622 ms
baseline and 0.031231 ms optimized, a 13,788.29 ratio dominated by local
one-rank process-group setup overhead. It is not distributed performance
evidence. The CPU-reduction and NCCL-labeled reduction targets passed complete
verification at ratios of 4.019 and 4.029. The one-rank disaggregated target
also passed complete verification at 1.104.

The `disaggregated_multigpu` stage exited zero only because it was launched with
one visible GPU and reported `SKIPPED: Distributed benchmark requires multiple
GPUs (found 1 GPU)`. It remains pending a proper two-GPU run. Both NVSHMEM
pipeline aliases again passed their baselines and timed out in optimized
execution; wave 12 did not validate the eager-NCCL-sideband repair.

Wave 13 executed on `feee566fa`. All five stages drained, and the focused XML
records 39 passes with no failures, errors, or skips. Commit `0561c5e76` moves
pipeline ownership tokens to CPU/Gloo while retaining symmetric-memory payload
movement and its completion fences. Both pipeline aliases then passed fresh,
complete child-output verification: `nvshmem_pipeline_parallel` at 3.184846 and
`nvshmem_pipeline_parallel_multigpu` at 3.090132. These are single diagnostic
ratios from the portable, unprofiled run.

Both MoE hybrid targets also passed their complete child-output checks on the
`feee566fa` source. Their ratios, 0.958897 and 0.989798, remained below the
unchanged 1.05 threshold, so both benchmark results correctly remain
performance failures despite their correctness passes.

The isolated training environment preserves Torch 2.9.1+cu130 and Transformers
4.56.0, with Datasets 5.0.1 and Accelerate 1.14.0 added. The TinyLlama snapshot
and GLUE/MRPC splits are materialized in its shared cache. Four actual Linux
controls passed for the private-environment launcher, symlink, shared-cache, and
owned-descendant supervision path. All 61 rows are prepared. The initial
launcher failed before entering the workload because its remote manifest was
missing; that failure is retained. The manifest has since been copied and all
95 bound source hashes reverified, and a retry is in progress. The training
sweep remains incomplete, and no workload execution result is claimed.

Source through `feee566fa` is pushed. At this checkpoint, pull request 18 has
two passing and two still-running CI checks. That transient CI state and all
wave-12/wave-13 timings are operational evidence, not public or canonical
performance claims.

### First complete training pass and direct B200 wave 14

The first 61-row distributed-training pass ran source `feee566fa` to terminal
state for every row and confirmed all 61 descendants drained. Thirty-one rows
passed and 30 failed. The failures formed three concrete groups: six could not
load the required `flash_attn` distribution, 22 pipeline rows attempted an
in-place ReLU on an autograd leaf, and two ZeRO-3 rows modified a backward-hook
view in place. These retained failures are execution evidence; the pass did not
qualify the affected workloads or produce a performance result for them.

Three subsequent source revisions address those groups without relabeling the
failed pass. Commit `db3f03d17` applies padding-aware labels and loss masking to
19 training rows; its focused CPU selection passed 14 tests. Commit `eb7e52763`
preserves autograd leaves at pipeline activation boundaries; six CPU tests ran
the actual schedules successfully. Commit `582ec8c86` protects ZeRO-3
backward-hook views from the in-place activation, with 13 focused Gloo CPU
checks passing. Those CPU checks establish the repaired control and math paths,
not B200 completion.

The separate FlashAttention-2 environment uses the official
`flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE` wheel. Its downloaded SHA-256 is
`a4b43bd016f5d475dc34a87aa91b5a239bd0e8972c13f4fc32839b4032465d21`,
the extension contains an `sm_100` cubin, and the actual import retained Torch
2.9.1+cu130 with CXX11 ABI enabled. A fail-closed B200 probe then passed both
fixed-length `flash_attn_func` and padded, odd-length
`flash_attn_varlen_func` cases. Every output and Q/K/V gradient element was
checked against a forced-math FP32 SDPA reference; the probe exited zero in
2.038 seconds and its supervised process drained. This validates the tested
FA2 APIs and runtime tuple. It does not replace fresh results from the six
training rows that previously failed to import FA2.

The repaired 61-row pass completed on `582ec8c86` inside that isolated FA2
environment: **61 zero exits, zero failures, and all 61 supervised process
trees drained**. The inventory comprises 27 one-rank and 34 two-rank runs.
Every receipt binds manifest SHA-256
`198d92ad61031047c05e026ac0cc012122493ac62257ce32b7d45aa645469066`,
which retains the original 61 commands, workloads and rank topologies and
binds 95 source files. This rerun closes the observed import and autograd
execution failures, including the six FlashAttention rows, 22 pipeline rows
and two ZeRO-3 rows that failed in the first pass. These remain diagnostic
training executions: the generic wrappers still lack a qualified child-result
protocol. Requested step counts are not asserted as completed step counts;
for example, a DDP loader exhausted after 57 steps and reported that actual
count.

The training phase finished at 16:23 UTC, and both B200s immediately started
the resumed 417-target single-GPU inventory in independent shards. The 69
multi-GPU targets, remaining stages and final tests follow afterward. Those
stages remain incomplete. Pull request 18 remains open, with its latest CI
still pending. Full training receipts, commands and logs are retained locally
under the dated `training-all-repaired` artifact bundle alongside the original
failed pass.

Wave 14 ran five supervised stages on the intermediate `eb7e52763` source and
drained all five. The focused XML records ten passes with no failures, errors,
or skips. The ring generation probe checked three message sizes for 45 changing
generations each, or 135 full bitwise comparisons per rank, and passed on both
ranks. The canonical ring benchmark invocation still failed its unchanged
speed contract at a 0.269935 ratio despite complete verification. The
disaggregated benchmark with two visible GPUs in one process also passed
complete verification but failed the speed contract at 0.836775. Its forced
two-rank invocation failed because the worker did not return the required child
verification payload; an explicit child-result contract is under repair. None
of these three failed benchmark results supports a performance claim.

Commit `569e7cf5a` repairs the two-rank disaggregated worker contract: normal
discovery now requests two torchrun ranks, timing comes from worker CUDA
events, and both complete prefill/decode outputs are checked for every rank.
Six focused CPU tests passed, including real two-rank Gloo execution. The
workload still runs both phases and both reductions on each WORLD rank; it
does not claim dedicated prefill/decode GPU groups. A fresh two-B200 run of
this revision remains pending while the broader sweep runs on frozen
`582ec8c86`.

### Direct B200 wave 15 and measured cost-tool inputs

The next breadth segment added 24 successful single-GPU executions on
`582ec8c86`, through the Chapter 10 distributed shared-memory examples. Both
shards stopped after complete stages and their supervisors confirmed drainage.
The parent coordinator then hit an event-field logging collision; its aborted
state is preserved. A focused control now exercises the real event writer with
two stopped child queues and verifies the repaired, distinct lifecycle and
child-terminal fields. No benchmark result was lost or relabeled.

Wave 15 ran 12 stages on `41a609f12` and drained every process tree. All 22
focused tests passed. The repaired two-rank disaggregated target passed full
prefill/decode output verification with 0.198413 ms baseline timing and an
observed 1.106246 ratio. The unchanged default NVSHMEM pipeline also passed
full verification at a 3.184358 ratio after the shared helper gained explicit
CUDA-backend selection.

The CUDA ring passed 135 changing, full bitwise generations per rank on
nondefault streams. Its device barrier waited 500.836 ms for an intentionally
delayed peer, distinguishing it from the NVSHMEM no-op. Four ABBA blocks with
40 samples per arm retained max-rank medians of 0.020847 ms for NCCL and
0.026115 ms for CUDA symmetric memory, a 0.798267 ratio. Both ordinary ring
aliases also passed complete correctness but missed the unchanged speed gate
at 0.833322 and 0.813429. Nsight Systems shows exactly 800 device barrier
kernels and 800 asynchronous copies per rank in each 400-iteration optimized
range, with no NCCL kernel there. The roughly 12.2 ms profiled range contains
4.0–4.4 ms of barrier kernel time, motivating a CUDA graph dispatch experiment.
These unlocked, virtualized-host measurements remain diagnostic.

The four FP16/INT8 compression targets that had previously skipped under
one-GPU routing were run with two ranks. All four exposed missing child
verification payloads; their apparent 6.5-second baseline timings were process
startup, not worker iteration measurements. Their worker/result transport is
under repair. The separate `ch04:no_overlap` pair had the same class of defect;
commit `3526a0b13` adds full training outputs, an independent functional SGD
reference and worker timing. Five focused tests passed, including actual
two-rank Gloo; its B200 rerun remains pending.

The cost calculator now has an actual successful invocation on source
`41a609f12`. A separate two-B200 vLLM run completed all 16 prompts and returned
768 token IDs over a 0.541600-second generation interval. NVML samples bracket
that same interval; their integrated board-power estimate is 534.244 W and
289.347 J. The calculator consumed those measured inputs and exited zero, and
all descendants drained. Its $0.16/kWh electricity and 1.5 PUE values are the
tool's explicitly labeled example assumptions. This short integration check
does not establish serving throughput or production cost. Artifacts are
retained in the dated `wave15`, `cost-tool`, and `continuation-through-ch10`
bundles.

### Direct B200 waves 16–17 and gradient-compression workers

Wave 16 completed seven stages on `8c867e77f`, with all descendants drained
and 20 focused tests passing. The no-overlap pair now passes complete training
output and input verification on two B200s. Its observed ratio was 0.944683,
so the execution defect is closed while the unchanged speed gate remains open.
Both ring aliases also passed correctness but missed their speed gates at
0.848128 and 0.874363. The graph ring passed 135 changing bitwise generations
per rank; four ABBA blocks with 40 samples per arm yielded a max-rank ratio
of 0.846141. Nsight confirms one graph launch, 800 barrier kernels, 400 local
copies and 400 peer copies per rank in each complete 400-iteration range.

Commit `a3d4519a1` removes the ring's redundant local receive copy: the peer
already writes the symmetric receive slot, which is consumed directly after
the same publication and consumption barriers. Wave 17 completed all six
stages and drained every process tree. All 15 focused tests passed, as did
135 changing bitwise generations per rank and a 500.905 ms delayed-peer
barrier control. Nsight independently confirms exactly 400 peer transfers
of 2 MiB and 800 barriers per rank, no local copies or NCCL kernels in the
timed optimized range, and one graph launch. Four ABBA blocks retained
40 samples per arm, with max-rank medians of 0.020765 ms for NCCL and
0.024362 ms for the candidate (0.852351 ratio). The ordinary aliases passed
complete verification but missed the speed gate at 0.870437 and 0.873836.
Removing physical copy work has not established a latency win on this host.
Both waves remain diagnostic measurements from the virtualized, unlocked host.

Commit `03451f9c3` gives all four gradient-compression pairs explicit two-rank
workers, actual worker timing and complete input/output verification. Each
rank retains the original 1 GiB gradient, five warmups and ten iterations.
The full FP16/INT8 variants reduce rewritten compression buffers in place;
baseline buckets require no added assembly copy. Communication-only variants
preserve their constant inputs with the same functional out-of-place collective
in both arms. That process-model allocation differs from the former
same-process preallocated output API, so absolute latency is not continuous
with those older measurements. Eight focused CPU tests passed, including
real two-rank Gloo execution of all eight variants.

The next two-B200 run passed full FP16, FP16 communication-only and INT8
communication-only checks, with observed ratios of 1.895051, 1.881169 and
3.314383 respectively. These are diagnostic harness measurements, not repeated
performance qualification. Full INT8 exposed a parent independent-reference
failure: three values in the first 16,777,216 elements differed across the
CPU/CUDA quantizers by one code per rank near a rounding boundary. The child
reference and timed outputs agree at those positions; the complete inputs,
outputs and child reference are retained. The parent reference is being
corrected to use the execution backend without changing tolerances.
The remaining breadth sweep has resumed on frozen `03451f9c3`.
Its Linux continuation controls passed all four tests, including real normal
descendant drainage and timeout cleanup. Training's 61 successful rows retain
their original `582ec8c86` source identity rather than being relabeled.

That breadth segment added 49 executions through the start of Chapter 13:
46 passed and three Chapter 10 targets failed. The epilogue and pipelined
TCGEN05 logs prove a concurrent cache deletion race: Ninja lost its build
directory and PyTorch's build lock disappeared. Commit `700ed2a46` adds a
per-extension advisory lock outside the deletable directory, covering cache
invalidation, compilation and metadata publication. The same repair keeps
setup buffers private until actual computation produces the epilogue and
cuBLAS comparison outputs. The expanded CPU lane passed 557 tests with one
skip after two stale source-inspection assertions were aligned with the
cached training helper and private output buffers (`00019bed7`).

Both sweep shards stopped after complete stages and drained all descendants.
The new coordinator then exposed another event-field collision while recording
their terminal states. Its aborted result remains intact; a separate recovery
ledger retains the original base and both complete shard ledgers, including
the three failures. This is ledger recovery, not coordinator success. The
coordinator event path is receiving an actual stopped/failed-queue regression.

The corrected coordinator passed five Linux controls, including actual
`wait_queue` completion for both a successful queue and a stopped queue with
retained failures. Wave 19 on `00019bed7` has now passed the complete saved
INT8 input oracle on CUDA and fresh full INT8 runs: 1.852722 observed ratio
for the full pair and 3.322505 for communication-only. All three Chapter 10
targets also passed their normal B200 execution and verification reruns.
The separate concurrent build check passed on `cfcb13bc0`: two processes
rebuilt in an empty private cache, loaded the identical extension hash and
produced exact complete CUDA matmul outputs on the two B200s. The overlapping
load calls completed in 39.13 seconds.

Two further performance candidates preserve workload math: `e77dc2f91` avoids
DDP input routing for tensors already placed on the rank-local device, and
`c4ce8d566` reuses MoE destination counts for both routing metrics and dispatch,
removing a redundant device-to-host readback. Their CPU controls pass; actual
B200 correctness passes for both candidates. The normal DDP run retained a
0.984355 ratio; the MoE aliases retained 0.957508 and 1.196584. These mixed
single-observation outcomes are not a performance conclusion. DDP subsequently
passed all 16 full-output ABBA observations on `00019bed7`, with eight samples
per arm and a 0.990286 ratio between rank-maximum medians. It remains a measured
performance miss. Its baseline Nsight trace contains the expected 30 workload
ranges (five warmups plus ten measured steps on each rank); the private
validator incorrectly searched for the uncanonicalized range name. The failed
receipt is retained, and a profile-only retry uses the actual canonical labels.
Repeated MoE timing and both fresh DDP profiles remain queued behind the full
GPU test suite.

The full suite completed all 5,634 tests on `cfcb13bc0`: 5,552 passed,
78 skipped and four failed in 33 minutes 18 seconds. Every descendant drained,
and the complete logs and JUnit report are retained. Hosted CPU CI completed
with 5,129 passed, 503 skipped and two
stale assertion failures. One still expected the four gradient-compression
targets to run in a single process; the other expected the TMA neighbor
transform to equal its untransformed input. The updated assertions check the
exact per-target launch modes and the full tiled-neighbor reference. Their
owning tests and compiled host-oracle controls pass locally (21 tests).

The GPU-only CPU-control failure exposed a harness bug: CUDA memory accounting
was selected whenever CUDA was available, even for a benchmark on the CPU.
Memory tracking now checks the benchmark device. The fourth GPU-suite failure
came from standalone signature capture on an NVSHMEM wrapper that requires
a distributed worker's complete result callback. That tool now explicitly
reports the required `aisp bench run` path instead of attempting parent-side
signature capture. The focused CPU lane passed 46 tests; two additional controls
cover the distributed-signature boundary and all 1,064,960 gradient elements.
Fresh B200 reruns of these repairs remain required.

The training-patterns candidate batches the 512 gradient-pack and 512 unpack
operations with two foreach calls. It preserves every value, collective,
synchronization fence and optimizer update. CPU controls cover disjoint bucket
views, inactive gradients and replacement gradient objects across steps.
Its previous 1.049336 ratio narrowly missed the 1.05 gate; the candidate still
requires full two-rank verification and repeated timing before any speed claim.

Profiler policy explicitly permits interrupting or terminating owned NCU/NSYS
processes without asking, for direct runs and scheduler jobs. Interrupted
captures retain their artifacts and incomplete status.

Wave 21 completed MoE's 16 full-result timing observations and captured both
arms. Its optimized capture and the DDP baseline capture exposed an Nsys
validator bug: the second statistics query reopened the original report and
could reject the just-exported database as older than that report. Reading the
fresh database directly succeeds on both retained captures and finds the
expected workload ranges. The harness now uses that path; 38 profiler-contract
tests pass with one skip. Original failed receipts remain intact, and the
remaining DDP optimized capture is still required.

Wave 22 on `48ab6ff1a` passed all 87 focused B200 tests, including the four
previous full-suite failures. Both training-patterns targets passed complete
two-rank verification, with initial ratios of 1.512702 and 1.413125. Every
stage drained. Wave 23 then passed all 16 interleaved timing observations and
both Nsys captures. The rank-maximum median fell from 23.785251 to 15.793236 ms
per step (1.506040 ratio); the four block ratios were 1.506725, 1.369103,
1.526312 and 1.508860. The slower optimized observation remains in the data.

The actual 100-step Nsys ranges contain 257,000 baseline kernel launches per
rank and 157,000 optimized launches. The optimized path uses 1,600 batched-copy
kernels for the complete gradient packing/unpacking workload, with 200 NCCL
barrier kernels; the baseline performs 51,200 NCCL per-parameter reductions.
These are unlocked diagnostic measurements on the same two-B200 host, with
full-output checks across every timing and profiler invocation.

The corrected validator also passed all three retained Wave 21 reports.
Fresh baseline and optimized DDP captures on `48ab6ff1a` passed full worker
verification and workload-range checks. Their profiled timings are separate
from the previously retained ABBA measurements; no DDP speed win is claimed.
The breadth continuation then resumed automatically from the original ledger
plus the actual Wave 22 outcomes.

The next breadth segment reached Chapter 15 and exposed further failures.
In the memory-profiling pair, stale expectation metadata labeled the study as
a speed goal despite the executable declaring a memory goal. The harness now
gives the executable's declaration priority. The baseline also accumulated
gradients while the optimized variant cleared them; both now clear gradients
before the same backward pass. Full gradient comparisons pass across repeated
CPU iterations. Previous memory savings do not establish a fair checkpointing
benefit under the corrected gradient lifecycle.

The pooled KV-cache candidate fuses its QKV matrix multiply and bias addition
into `addmm`, reusing the same output buffer and preserving the bias-free path.
Complete projection-output and storage-reuse controls pass. These Chapter 13
repairs passed 28 focused CPU tests and the 553-test hygiene lane (one skip);
their actual B200 reruns are pending while the existing sweep continues.

The cache-aware disaggregation performance miss also has a precise topology
limit: one prefill rank plus one decode rank produces identical baseline and
optimized locality routes. At least two decode ranks are necessary to test
the locality benefit. The smallest useful explicit layout is three GPUs
configured as one prefill plus two decode ranks; the first useful default
layout is four GPUs (two plus two). Existing route tests and documentation
already encode the one-decode-rank control. The two-B200 run remains valid
correctness evidence, and its measured speed failure is preserved.

The current instance exposes neither an InfiniBand device nor `nvshmemrun`.
The IBGDA examples therefore retain their explicit runtime/hardware gate;
Grace/GB10-specific examples still require the named hardware, and NVFP4
TensorRT-LLM execution still requires a compatible engine asset.

### Chapter 13/16 follow-up and main merge

PR #18 merged as `dcd2c55d594dcacf6974b1a179c1eec69bf24153` after all
four exact-head CI checks passed. Its tree matches the tested `48ab6ff1a`
tree. The CPU suite passed 5,136 tests with 503 skips; the core contract lane
passed 751 tests with one skip. Dual-architecture validation also passed.
Runtime receipts retain their actual source commits.

The continued B200 sweep exposed additional correctness defects. Both
Chapter 13 regional-compilation arms retained the first warmup output and
truncated verification to 128 tokens. They now capture the latest input and
complete output and use a fixed iteration count so their rotating sequence
buckets match. The Transformer Engine FP16/FP8 pair now explicitly declares
precision-signature equivalence while retaining its actual precision flags
and numerical tolerances. Obsolete accessors that returned the input as an
output have been removed; the actual payload remains the model output.

Both dense Flash Attention variants now view their preallocated merge buffer
as `[batch, sequence, heads, head_dim]` when copying the transposed attention
heads. Previously that four-dimensional tensor was copied into a
three-dimensional buffer and failed at runtime. Full CPU attention-output
comparisons and storage-reuse checks cover both variants. The focused and
hygiene run passed 561 tests with one skip. GPU reruns of these new repairs
are pending; the original failures remain preserved.

The Chapter 14 regional Triton example also produced a native crash during
isolated-process cleanup. Its complete stderr is retained. Cold-cache
vanilla and repository-path reproductions are required before attributing
that failure to the toolchain or changing a compiler mode.

### Wave 24: Chapter 13/16 B200 reruns

All nine stages on `5a59db088` drained their owned processes. The focused GPU
lane passed 60 tests. Regional compilation now verifies the complete latest
output, with maximum difference `0.03125`, and its diagnostic speed ratio was
`1.0931`. Dense Flash Attention passes full verification with maximum
difference `0.00048828125`; its diagnostic ratio was `12.9137`.

The corrected memory-goal study passes execution and correctness, but does
**not** show a memory benefit: baseline peak was `287.2974 MiB`, candidate
peak `288.2974 MiB`, and the candidate was slower. The pooled KV-cache and
Transformer Engine examples now pass correctness while retaining speed
failures (`0.9478` and `0.6819` respectively). Passing a correctness gate does
not turn these observations into optimization wins.

Chapter 14 regional Triton passed a fresh cold-cache harness run. Separate
cold-cache vanilla main-thread and worker-thread probes used the unchanged
model definitions without importing repository modules. Both compared all
outputs for 18 invocations and exited cleanly, with maximum difference
`0.03125`. The original native crash remains an unreproduced failure; no
compiler downgrade or toolchain attribution was added.

Two further candidates are ready for measurement. The pooled KV cache now
prepares token and prefix views when allocating the pool, avoiding repeated
view construction in the decoding loop. RoPE writes its rotations directly
into each cache slot, removing the scratch copy and final cache copy without
batching future decoding steps. Both RoPE arms now verify every written
cache slot and all step inputs. Full-prefix/cache CPU controls, including
reused requests and untouched cache tails, pass; the focused/hygiene lane
passed 561 tests with one skip. These candidates still require B200 timing
and profiler evidence.

### Waves 25/26: cache measurements and full-output controls

All six Wave 25 stages on `0058fc6e6` drained. Ten focused GPU tests passed;
the pooled KV and RoPE examples passed their normal speed gates at `1.0739`
and `1.1539`. Two more Chapter 14 warm-cache harness runs passed. Its earlier
native crash remains unreproduced, with no compiler workaround introduced.
The dense Blackwell variant's CLI entry was explicitly informational and
executed no benchmark; Wave 26 exercised its real factory and tensor path.

Wave 26 retained eight observations per arm in four ABBA blocks for each
target, with five warmups. Each pooled-cache observation ran the complete
request workload; RoPE and dense-attention observations each contained 50
complete invocations. All observations are retained. CUDA-event medians were:

| Comparison | Baseline | Optimized | Ratio | Four block ratios |
| --- | ---: | ---: | ---: | --- |
| Pooled KV cache | 495.5062 ms | 451.4488 ms | 1.09759 | 1.09878, 1.09752, 1.09507, 1.09775 |
| RoPE direct cache writes | 1.967337 ms | 1.705141 ms | 1.15377 | 1.15341, 1.15383, 1.15379, 1.15373 |
| Dense Flash Attention variant | 7.661654 ms | 0.575625 ms | 13.31015 | 13.32500, 13.30270, 13.33621, 13.29302 |

Verification ran before timing, after timing, and after profiling. The KV
control captured every actual layer output across every token and request:
5,505,024 values matched bitwise. All 2,097,152 written RoPE cache values also
matched bitwise. Dense attention compared all 16,777,216 output values, with
maximum difference `0.00048828125` under the unchanged tolerance.

CPU operator profiles corroborate the mechanisms. Per full KV invocation,
concatenations fell from 21,504 to zero and `as_strided` calls from 225,792
to 161,280, while both arms executed 10,752 Flash Attention operations.
RoPE retained all 64 matrix multiplications and removed the baseline's 64
cache copies. The dense variant used one Flash Attention operation in place
of explicit attention. These profiles establish operator counts, not kernel
timings. The runs used unlocked clocks on a virtualized host and remain
diagnostic evidence rather than canonical performance results.

The remaining breadth sweep resumed automatically on the same immutable
revision. The subsequent decode-compile failure was a speed miss (`0.9451`),
with output verification passing.

### Current follow-up repairs

The monolithic inference stderr identifies why its compiled path was barely
faster: Inductor skipped CUDA graphs because the function mutated an input
buffer. The compiled request now creates its output internally through the
same full prefill and autoregressive decode implementation. A real CPU graph
capture control checks every output and unchanged prompts; B200 capture and
speed verification are still pending.

Memory summaries also clamped negative savings to zero. Aggregation now
retains the best measured candidate's actual signed savings, excluding
unverified candidates and nonfinite values. Reprocessing the retained
memory-study result exposes its approximately `-0.3481%` savings instead of
zero. The original result is preserved, and acceptance policies are unchanged.
The focused and hygiene controls for these follow-ups passed 560 tests with
one skip.

The single-rank hybrid expert-parallel run exposed a signature error: it
declared a distributed collective algorithm despite `world_size=1`. Its
signature now describes local work without a collective; multi-rank runs
retain their all-to-all contract. Thirteen focused CPU controls passed with
one skip. Fresh one- and two-B200 execution remains pending.

CI is deferred to the end of the remaining runtime and repair work, per the
user's updated direction. Code changes continue to be committed and pushed;
final checks and merges remain outstanding.

- Run the complete GPU test suite on the final merged revision.
- Complete all four sweep stages and reconcile the 486-target inventory,
  including unsupported and failed cases rather than silently dropping them.
- Finish missing protection checks and remaining attention/MoE investigations.
- Validate four-GPU, multi-node and unavailable dependency combinations on
  appropriate hardware. Two B200s do not establish those results.
- Use repeated interleaved measurements and profiler evidence before accepting
  a performance claim; virtualized-host measurements remain noncanonical.

Reproducible entrypoints are documented in [the sweep playbook](../../code/FULL_SWEEP.md).
Focused KV checks run with `python -m pytest tests/test_kv_optimization_append_paths.py`
from `code/`. Hardware and dependency requirements still apply independently of
whether an external scheduler is used.
