# Audit remediation plan: waves 1 and 2

Created 2026-08-30. This is the execution plan for the active Codex goal in task
`01a054a4-a0c0-7612-af1a-2df7b0c08851`.

## Goal and completion contract

Resolve every valid finding in the supplied first-wave audit and the same review
session's supplied second wave in this repository. Preserve every original
finding and its disposition. Demonstrate each fix with appropriate regression,
integration, build, and supported-hardware evidence. Finish with zero unresolved
valid findings across both received waves and an independent reconciliation of
the complete inventory against the final source revision.

Creating this plan is not completion of the remediation goal. Wave 2 has now been
received and reconciled, but its valid findings and the applicable first-wave
runtime gates remain open. Missing hardware evidence or a deferred valid finding
keeps the overall goal open.

The current goal was activated by the user's `/goal` request. This document makes
its success criteria concrete; it does not create a second goal or token budget.

## Current state and source of truth

| Item | State |
| --- | --- |
| Checkout at intake | `main`, `b57e4c6a9e261c09ac09208705d040c81b03d35e`; clean before planning artifacts |
| First-wave source | [AI Performance Engineering Audit](https://claude.ai/code/artifact/9a311b78-5dac-4ea7-909f-e52993ab8e0e) |
| Audit revision | `b57e4c6a9`, matching this checkout |
| First-wave inventory | 128 findings: 5 critical, 37 high, 58 medium, 28 low |
| Reviewer confidence | External audit claims; its verification labels are preserved, not treated as our runtime evidence |
| Additional source record | One separately refuted pytest-timeout claim, outside the 128 |
| Second wave | User-supplied full report captured at `cf48c8481`: 141 open findings (3 critical, 31 high, 62 medium, 45 low) across 110 files |
| First-wave external re-review | 120 fixed, 5 need runtime, 2 partial, 1 obsolete; this read-only source verdict does not replace the 76 applicable local runtime acceptance gates |
| Implementation status | First-wave candidate `f49aae73f629f4d90b60d1c6a5b890780e7ef758`, both Wave 2 source batches, and the focused hosted-CI repair through `3316e0efe985040745ffd926c5f76a6bd4436aff` are published on `main`. Exact retained-evidence reconciliation changes the Wave 1 mutable ledger to 115 source-fixed, 5 awaiting runtime, 7 verified, and 1 already fixed with evidence; its 76-row local runtime matrix is 7 verified and 69 pending. All 141 Wave 2 rows remain 89 source-fixed, 4 already fixed with evidence, and 48 awaiting runtime. Hosted run `33391774956` verifies LOCAL-025, W1-006's Linux CPU full-directory CI sub-gate, the seven exact host/configuration rows, and a bounded CPU regression subgate for every Wave 2 runtime row without a rerun. Run `33391774950` supplies bounded four-target CUDA 13 compiler evidence but no device execution. Focused hosted Linux CPU provenance passes for the reviewed 20-direct-pin/56-distribution cell; the full target Linux/CUDA graph, GPU qualification, deployment, and any new speedup claim remain open. |

### Interim main landing checkpoint

The user authorized landing the first-wave source/evidence candidate before B200
custody returned and before Wave 2 was delivered. Commit
`f49aae73f629f4d90b60d1c6a5b890780e7ef758` records the first-wave source and
evidence checkpoint; the following metadata commit binds that revision and
preserves the remaining work. This is a publication checkpoint, not goal closure.

Every known original and adjacent ledger entry has a source disposition: the
original inventory is **52 `source_fixed` and 76 `awaiting_runtime`**, and the 42
adjacent discoveries are **30 `source_fixed` and 12 `awaiting_runtime`**. Closure
is incomplete: all 11 packages remain `in_progress`, 68 original verification
records still contain `requirements_pending_triage`, 55 original entries lack an
independent review record, and no original entry is marked verified at the landed
revision. Per-item `fix_revision` values remain unset until their required closure
evidence exists. The newly received Wave 2 inventory and the supported runtime
matrix remain required scope.

### Earlier implementation checkpoint

The following records the earlier package work. Later integration checkpoints
below supersede its then-pending source work; its hardware limits still apply.

- **Integrated first batch:** **243 passed, 6 explicit skips** across bootstrap,
  harness, comparison, APIs/parsers, ZeRO-2, configuration, and CLI/validation
  entrypoints. No CUDA performance result is included.
  [Integration receipt](docs/audits/2026-08-30/evidence/integration/receipt.json).
- **Harness and comparisons:** timing direction and percentage thresholds repaired;
  summary-only subprocess results preserve unavailable raw samples and percentiles;
  detector flags work independently; token-throughput parsing repaired. Combined
  local integration: **144 passed, 1 Linux-only skip**, with five explicit Darwin
  environment warnings. [Evidence](docs/audits/2026-08-30/evidence/harness/receipt.json).
- **Critical ZeRO-2:** reproduced the no-op optimizer on two actual CPU/Gloo ranks.
  Explicit optimizer updates now match full-batch AdamW for three steps, and local
  sharded state is reported correctly. Focused run: **10 passed, 1 two-GPU skip**.
  Production NCCL acceptance remains open; historical performance is not validated.
  [Evidence](docs/audits/2026-08-30/evidence/zero2/receipt.json).
- **Bootstrap:** stable wheel sources, retired torchao prerequisite, and actual
  cuDNN summary repaired; 14 focused tests pass. Full Linux/CUDA installation is
  still HOLD, including an explicit ARM torchao CUDA-wheel gap.
  [Evidence](docs/audits/2026-08-30/evidence/bootstrap/README.md).
- **APIs/parsers:** seven original findings repaired, plus the retired Anthropic
  default across clients. Real CLI, file-backed readers, subprocess fixtures and
  loopback HTTP checks pass; no paid or hosted model call was made.
  [Evidence](docs/audits/2026-08-30/evidence/api/README.md).
- **Validation coverage:** full-directory CPU and attested B200 test commands
  added to active workflows with JUnit reports. GitHub/GPU execution remains
  pending. All 42 legacy protection smoke bodies have now been replaced. The
  latest protection gate reports **319 passed, 96 skips**: 35 need CUDA and 61
  describe unsupported or obsolete requirements. Failed clock locks cannot pass;
  an unavailable capability fails under the existing attested Tier-1 contract.
  The original test cleanup is source-fixed; CUDA acceptance remains open.
  Unsupported policies are stated explicitly in the 95-row inventory and are not
  fabricated as implemented protections.
  [Latest evidence](docs/audits/2026-08-30/evidence/validation/clock-lock-followup/receipt.json).
- **Critical source repairs ready:** shared TMA descriptor/callers, tcgen05 stage
  lifetime/cluster tails, and nanochat graph/cache replay have source changes and
  bounded target validation runners. Root integration: **36 passed, 20 explicit
  skips**. TMA prepares 42 full-output cases; tcgen05 prepares 1,680 cases. Neither
  CUDA compilation nor those GPU runs occurred. Nanochat graph replay and stream
  correctness also remain GPU HOLD.
- **Additional timing provenance:** Timer retains its actual measured block count;
  torchrun reports one observed process interval, with an explicitly labeled
  amortized metric. Result metadata distinguishes block averages from process
  wall time. Final harness/Blackwell routing integration: **45 passed, 9 skips**.
  The Blackwell suite's original eight CPU false passes now report eight GPU skips.
- **P10 source batch ready:** **88 passed, 1 CUDA/TE skip** across APIs/parsers and
  tooling. Prometheus output passes strict parsing and a loopback HTTP scrape;
  the flame command produced real CPU-trace artifacts; FP8 guidance now names its
  required version and integration context. No hosted model or GPU claim is made.
- **Build entrypoints:** **218 host checks passed** for architecture selection,
  failure propagation, script paths, and selected-Python torch ABI discovery.
  Actual CUDA compile/device-link/extension import remains HOLD. Subsequent CUDA
  chapter dependency edits are a separate source epoch requiring integration.
  [Evidence](docs/audits/2026-08-30/evidence/build-entrypoints/validation-receipts.json).
- **MoE and single-rank ZeRO:** **18 passed, 1 two-GPU skip**. Planner scenarios
  use their intended cluster/model presets and report estimates explicitly;
  expert routing uses each expert's FC2 weights; single-rank ZeRO performs actual
  updates matching dense AdamW. GPU and baseline workload-parity gates remain.
  [MoE evidence](docs/audits/2026-08-30/evidence/moe-planning-router/receipt.json),
  [ZeRO evidence](docs/audits/2026-08-30/evidence/zero2/single-receipt.json).
- **Numerical references and output ownership:** **41 passed, 1 SM100 skip**.
  Full-output KV, FP4 grouped-GEMM and Ozaki checks replace permissive or shared
  references; acceptance without a reviewed accuracy policy is refused. The KV
  README now correctly describes FP8/NVFP4 compute with BF16 cache storage.
  Actual CUDA accuracy calibration and evaluator runs remain HOLD.
  [Evidence](docs/audits/2026-08-30/evidence/numerics/receipt.json).
- **Router telemetry and KV transfers:** **17 passed, 9 runtime skips**. Cumulative
  token payloads are counted once, latency has a separate metric, and a two-rank
  CPU test checks 36 complete batched-cache handoffs. CUDA stream reuse is ordered;
  actual vLLM, CUDA-overlap and NCCL gates remain pending.
  [Evidence](docs/audits/2026-08-30/evidence/routing-kv/receipt.json).
- **Source integration pending at this earlier checkpoint:** shared-revision CPU/source checks,
  withdrawal of the unrelated toy-model verification used by generic torchrun
  wrappers, artifact portability/hash reconciliation, and final ledger readback.
  Exact target compilation, CUDA/NCCL/sanitizer/profile runs and reviewed accuracy
  policies remain separate acceptance gates.
- **At this earlier checkpoint, Wave 2 was unreceived.** The immutable Wave 1
  inventory and this historical checkpoint remain intact; the current Wave 2
  intake below supersedes that wait state. No package or overall goal is closed.

The complete rendered audit, each finding's problem/evidence/suggested fix/verifier
notes, and the unreviewed-area footer are captured in
[`wave-1-source.json`](docs/audits/2026-08-30/wave-1-source.json).
Do not silently replace that capture when the external page changes.

[`remediation-ledger.json`](docs/audits/2026-08-30/remediation-ledger.json) is the
mutable issue inventory. [`wave-1-inventory.md`](docs/audits/2026-08-30/wave-1-inventory.md)
is the readable index. Stable IDs `W1-001` through `W1-128` follow displayed source
order; original wording and severity remain unchanged. Every issue has exactly one
primary work package. Related issues can share a fix, but each needs a disposition
and verification evidence.

## Execution sequence

1. **Intake and revalidation.** Capture sources, keep the audited revision, inspect
   current code, and reproduce each report before editing. The five critical code
   patterns were spot-checked at the matching revision; their runtime effects are
   still unverified by this task. Record contradictions and stale reports rather
   than implementing suggested fixes mechanically.
2. **Restore trustworthy installation and validation.** Begin P01 with the setup
   failure and P02 with anti-cheat, raw-timing provenance, comparison direction,
   and CPU/GPU test selection. Establish behavior-based regression coverage for
   critical fixes before relying on a green test result. P10's CLI and token-budget
   failures can proceed independently when their files are disjoint.
3. **Fix critical correctness paths.** Work P03's shared TMA descriptor and callers,
   P04's asynchronous stage lifetime, P05's graph replay/cache position, and P06's
   optimizer updates. Follow with the high-severity synchronization and output
   errors in the same packages. Do not wait for wave 2 to begin valid wave 1 work.
4. **Close the remaining work packages.** Repair numerical references/tolerances,
   attention/decode paths, hardware metrics, distributed/MoE logic, and APIs.
   Medium and low findings remain required scope. P11 and package-local docs must
   describe the corrected code and evidence, including withdrawn invalid claims.
5. **Integrate the delivered wave 2.** Revalidate it against its reviewed revision
   and current source, link overlaps without dropping IDs, and prioritize the three
   critical and 31 high defects. Use its eight disjoint packages for ownership and
   reopen affected first-wave closures.
6. **Verify the combined final revision.** Run the relevant source/CPU gates, builds,
   real entrypoints, and serialized supported-hardware jobs. Regenerate affected
   reports only from fresh valid measurements. Perform an independent review and
   reconcile every ledger entry before considering the goal complete.

Packages are planning units, not a demand for eleven large commits. Split them
into small changes organized around one mechanism and its regression coverage.
Preserve the actual base/head revision and evidence for each integration step.

## Work packages and acceptance

The inventory contains the exact member IDs. Counts below reconcile to 128.

| Package | Findings | Work and required evidence |
| --- | ---: | --- |
| P01 — Bootstrap and build architecture | 11 | Correct stable/nightly wheel source selection, quickstart dependencies and cuDNN summary; make architecture selection, compare-loop failure propagation, helper paths, CUDA device linking and PyTorch ABI coherent. Use shell/config regression tests, isolated dependency resolution, fresh compile/link logs for applicable architectures, and execution/import on the actual target. Dry-run command text alone does not prove a binary builds or loads. |
| P02 — Harness, anti-cheat and CI | 13 | Repair comparison direction/threshold units, synthetic samples presented as raw measurements, detector initialization and throughput parsing. Replace ineffective tests with tests that invoke protections and fail for deliberately invalid behavior; fix skips/false passes and wire CPU-safe suites into active root CI. Assign GPU cases to an explicit supported runner. A skipped test is never coverage of the behavior. |
| P03 — Chapter CUDA, TMA and stream timing | 20 | Repair descriptor coordinate order and all callers, cp.async tails, tile coverage, occupancy/barriers, copy-compute dependencies, chunk coverage and same-work timing. Compare actual full outputs with independent references on non-square and ragged shapes; run race/memory checks where supported. Measure events on the executing stream or a correctly joined dependency graph. Preserve input/work equivalence and verify every element. |
| P04 — Blackwell kernel and feature correctness | 12 | Fix tcgen05 producer/consumer lifetime and cluster tails, illegal launch sizes, missing bias, multicast/barrier initialization, capability detection, and misleading feature/performance labels. Test repeated stage reuse, odd tile counts, nonzero bias and unsupported targets; compile/link and run on each relevant supported architecture. Claim multicast/TMA only with emitted-instruction or profiler evidence. |
| P05 — Nanochat engine and benchmarks | 6 | Make graph replay update the device-visible KV position, establish side-stream and D2H ordering, exercise flags through the actual Engine, and replace assertion-free validation. Compare repeated eager/graph/persistent decoding with identical inputs and inspect per-step positions/logits/tokens. Show each benchmark mode executes the named implementation and that measured intervals include its work. |
| P06 — Distributed training and MoE | 17 | Restore real optimizer updates, coherent replicated parameters, symmetric collective participation, buffer ownership and transfer dependencies; fix routing units, scenario topology, FP8 arithmetic, expert indexing, partial-math kernels and EP cost modeling. Use independent dense references, distinct expert weights, multiple batches/ranks, zero-token ranks, non-contiguous inputs, step-by-step parameter deltas and timeout-bounded multi-GPU execution. |
| P07 — Attention, decode and async input | 12 | Correct FlashMLA/attention math, score masking and layouts; enforce requested backends; fix persistent-decode reduction/coverage and input/offload lifetimes; correct TFLOP counts and implementation labels. Test non-divisible lengths, distinct head/sequence dimensions, batch larger than program count, allocator pressure and buffer reuse against independent attention/decode outputs. Revalidate or withdraw the affected recorded speedup claim. |
| P08 — FP4/FP8 numerics and verification | 10 | Repair Transformer Engine recipe use, independent output ownership, real group-GEMM reference, executable GEMV entrypoint and scale-cache lifetime. Derive error budgets from actual workload/reference measurements; zero, corrupted, NaN and aliased outputs must not pass. Report actual KV storage dtype/bytes rather than an assumed compression ratio, and keep docs consistent with the implementation. |
| P09 — Hardware specifications and metric math | 8 | Correct bandwidth traffic/units, metric peaks and dense/sparse conventions, architecture identity, FP4 recipes and FP8 benchmark execution. Verify definitions against versioned primary vendor/runtime sources; exercise fixed arithmetic fixtures and negative controls. Keep B200/B300/GB200/GB300/SM12x separate, with precision, sparsity and per-GPU/aggregate conventions explicit. Refresh derived reports only after their inputs and measurement paths are valid. |
| P10 — CLI, LLM, parsing and monitoring | 10 | Restore default CLI behavior, supported token-budget handling, citations and explanation parsing, NVLink/version parsers, truthful flame-graph behavior, Prometheus output and valid what-if examples. Use real CLI/API paths with deterministic fixtures; validate provider payload limits without sending private prompts or making paid calls. Parse a multi-GPU scrape with a standards-compliant parser and test error/unsupported responses. |
| P11 — Documentation and operational commands | 9 | Reconcile environment, architecture, license, NCCL semantics, bandwidth labels and profiling commands with current source and primary documentation. Check paths/options, safely exercise commands on suitable systems, and update README generators where applicable. Do not preserve a performance claim merely by relabeling an invalid run as validated. |

### Five critical findings: required closure evidence

| Finding | First change to investigate | Acceptance |
| --- | --- | --- |
| W1-005 — Setup wheel index | Match the pinned stable PyTorch build to a source that actually serves it; coordinate W1-124/125 | Clean isolated resolution/install on supported Linux/CUDA, import/version verification, a test rejecting stable/nightly mismatch, and agreement between installation and printed summary |
| W1-001 — Shared TMA descriptor | Correct descriptor dimensions, tile box and coordinate conventions together; inspect every helper caller | Full elementwise output comparison with an independent reference on rectangular and asymmetric matrices, tile-boundary/tail cases, both wrapper/asm paths where supported, plus baseline/candidate runtime evidence; checksums alone are insufficient |
| W1-002 — tcgen05 stage reuse | Enforce consumer completion before a producer overwrites a shared-memory stage; coordinate cluster bounds and multicast claims | Multiple pipeline wraps and odd tile grids, independent GEMM output comparison, nonzero bias/epilogue cases, race/memory diagnostics and actual supported-GPU execution |
| W1-003 — Graph KV position | Make position changes visible inside replay rather than relying on a captured Python scalar | Eager versus captured decode over multiple successive replays and prompt lengths; verify actual cache slots and per-step outputs, then establish ordering and trustworthy timing with W1-034/035/036 |
| W1-004 — ZeRO-2 optimizer | Use a supported optimizer/communication integration that performs the intended update | Multi-rank deterministic training compared step-by-step with a reference, nonzero parameter changes, consistent parameters/state and more than one optimizer step; support/version failures remain explicit |

### Dependencies and parallel ownership

- P01's working environment precedes target runtime acceptance. Its CMake/Makefile
  work must integrate with P04's kernel architecture requirements.
- P02's corrected measurements and protections precede accepting benchmark numbers
  from any package. It owns shared `benchmark_harness.py`, `comparison.py`, and
  validation CI changes; other packages propose changes through that owner.
- P09's corrected byte/FLOP accounting and sourced hardware ceilings also precede
  accepting dependent throughput or efficiency claims. Correct kernels do not
  rescue invalid metric definitions.
- P03 owns shared `tma_helpers.cuh` conventions and caller migration. Any P04 kernel
  depending on those conventions integrates after that contract is settled.
- P05 owns `nanochat/engine.py` and its benchmark/validation callers. P06 owns
  `moe_hybrid_ep_common.py` and related distributed state changes. Do not split
  simultaneous writers across the findings in those files.
- P08's recipe/reference decisions and P09's peak/diagnostic implementations must
  agree on pinned Transformer Engine/PyTorch APIs. P09's specification definitions
  must agree with P01 architecture labels and P11 documentation.
- P10 owns `core/perf_core_base.py`, CLI/LLM paths and exporter code. P11 owns common
  documentation; package authors supply technical corrections without concurrent
  writes to the same README, specification table or generated artifact.
- Use at most three independent worker agents plus the coordinating agent. Assign
  exclusive file sets or isolated worktrees before parallel edits. Integration,
  shared tests/artifacts, GPU access, profiler output and expectation publication
  each have one owner and are serialized.

## Verification procedure

For every valid issue, record the current reproduction (or a precise reason it
requires unavailable hardware), expected behavior, smallest safe fix, and an
observable acceptance check before implementation. Use existing regression suites
where they exercise the defect. Prefer a behavior-level failing-then-passing test
over a source-string assertion. Do not add mirror tests for trivial prose edits;
validate those against their authoritative source or working command.

Each closure records: original issue ID, current location, reviewed/fixed source
revision, reproduction, changed files, independent reference/negative control,
exact commands, exit status, artifacts, reviewer disposition and any remaining
hardware requirement. Keep failed attempts and reasons. If later work invalidates
the evidence or changes the path, reopen the entry.

Keep the original severity and verifier notes even when they disagree. Record
locally triaged severity separately, with the reason for any change. Completeness-
critic findings retain their original lack of separate skeptic verification.

Source and CPU gates, run from `code/` in the supported dependency environment:

```bash
python -m pytest <focused tests for the affected behavior> -q
python -m pytest tests/ -q
make lint
make validate
```

The first line is a placeholder to replace with recorded test node IDs before
execution. Full-suite capability skips and collection exclusions must be recorded
and mapped to a runnable Linux/GPU lane, not hidden. macOS is a development gate;
Linux-specific and CUDA/Triton behavior still needs its supported environment.
`make lint` is the repository's curated blocking gate; broad legacy lint debt is
separate and must not be reported as clean merely because that gate passes.

Also run changed-file syntax/import checks, relevant shell checks, and the active
root CI configuration's fallback and contract checks. Existing entrypoints include:

```bash
python -m core.scripts.linting.check_benchmarks --include-unpaired --fail-on-warnings
python core/scripts/audit_silent_fallbacks.py --fail-on-findings \
  --categories global_warning_filter stderr_reassignment stdout_reassignment stdio_dup2_hijack syntax_error read_error
python -m core.verification.review_baseline_optimized_pairs \
  --chapter ch12 --json --markdown --output-dir <fresh-output-directory>
```

Adapt the last command's chapter to touched pairs and record actual arguments.
It is a source/report smoke test, not GPU correctness or performance evidence.
For runtime changes also invoke the relevant real CLI, MCP, dashboard or harness
entrypoint and inspect structured output; tests alone do not establish user-facing
integration. No external message, private-data transmission or paid model call is
required to validate request construction and parsers.

Build and hardware gates:

- Follow `code/AGENTS.md`, the repo-local Dean review skill, and relevant build and
  sweep playbooks. Preserve equivalent workloads, correctness policies and actual
  timed outputs. Never relax tolerance, remove validation, substitute a fallback,
  or fabricate samples to make a test or benchmark green.
- Record clean source revision, build flags/toolchain, exact GPU/model/count,
  driver/runtime/library versions, workload, seed, precision, sparsity, clocks,
  environment validation, and raw output. Use fresh output directories/run IDs.
- Compare actual tensors against independent references, including adversarial
  boundary shapes. Validate synchronization with repeated execution and supported
  sanitizer/profiler tools. Host simulation and compilation cannot close a GPU
  correctness issue by themselves.
- For performance claims, follow harness clock locking and record `app_clock`
  provenance; run repeated interleaved baseline/candidate trials, inspect the
  distribution and relevant `nsys`/`ncu` evidence, and enforce the applicable
  correctness/variance gates. Do not prescribe a fabricated universal speedup.
- Keep hardware-specific receipts separate. Unsupported cases must produce explicit
  unsupported/skip results; they cannot count as passes on supported hardware.
- Invalidate affected derived speedup/roofline/expectation claims before reuse.
  Preserve old evidence as historical with its invalidation reason. Any replacement
  expectation/history must follow existing provenance and acceptance rules.

## Second-wave intake

The complete user-supplied report is captured verbatim in
[`wave-2-source.txt`](docs/audits/2026-08-30/wave-2-source.txt), with a deterministic
parsed inventory in
[`wave-2-source.json`](docs/audits/2026-08-30/wave-2-source.json) and an intake
[receipt](docs/audits/2026-08-30/evidence/intake/wave-2/receipt.json). The parser
reconciles exactly **141 open findings: 3 critical, 31 high, 62 medium, and 45
low**, across **110 unique files**. It also reconciles all 128 first-wave re-review
verdicts: **120 fixed, 5 need runtime, 2 partial, and 1 obsolete**.

The report's 141 rows merge Wave 2, the closure critic, the tail pass, and two
explicit Wave 1 residuals. Stable IDs `W2-001` through `W2-141` preserve source
order and wording; overlaps may share a fix but remain separate historical rows.
The two partial Wave 1 records, W1-069 and W1-098, are reopened. The five
needs-runtime records remain runtime-held. The source-only external review does not erase the 76 runtime acceptance
requirements recorded by the local plan. Current retained-evidence reconciliation
verifies seven exact host/configuration contracts and leaves 69 pending.

The public artifact URL continued to serve its older 128-finding Wave 1 frame at
intake. The user's complete attachment is therefore the authoritative Wave 2
source; the public mismatch, response version, and digests are preserved in the
receipt. This mismatch is an intake limitation, not evidence against the supplied
report.

Implementation proceeds through eight exclusive file-domain packages:

| Package | Findings | Scope |
| --- | ---: | --- |
| W2P01 | 24 | Chapters 1–5 |
| W2P02 | 20 | Chapters 6–11 |
| W2P03 | 8 | Chapters 12–14 |
| W2P04 | 32 | Chapters 15–20 |
| W2P05 | 32 | Labs |
| W2P06 | 12 | MCP, dashboard, and monitoring |
| W2P07 | 9 | Core framework |
| W2P08 | 4 | Scripts, documentation, and residuals |

The first active batch fixed the three critical findings: pointer-sensitive CUDA
graph caching, correctly populated finite conditional graphs, and the CUTLASS
column-major B stride. The published follow-up at `13c4e151f` addresses the high,
medium, and low source findings and incorporates three independent cross-reviews;
`3316e0efe` closes the five bounded integration gaps exposed by hosted CPU CI.
Every row now has a disposition: **89 `source_fixed`, 4
`already_fixed_with_evidence`, and 48 `awaiting_runtime`**. Each batch uses the
smallest sufficient focused tests; the full CPU pass is not repeatedly rerun.

The retained final-source hosted CPU run has also been reconciled against the
remaining first-wave and adjacent contracts. Run `33391774956` explicitly
installed `requests==2.34.2` and `tokenizers==0.22.2` on GitHub-hosted Ubuntu,
ran the complete `tests/` tree, and retained JUnit for **4,346 passed, 461
explicit skips, zero failures, and zero errors**. That existing evidence fully
verifies LOCAL-025 and closes W1-006's Linux CPU full-directory CI sub-gate; it
does not execute W1-006's supported-GPU protection cases. No suite was rerun for
this reconciliation. The heterogeneous skip set also retains 62 known
missing-protection, protection-summary, or absent-route scope skips; none is
converted to passing coverage. [Receipt](docs/audits/2026-08-30/evidence/integration/hosted-cpu-closure/receipt.json).

A row-by-row retained-evidence audit now verifies seven exact Wave 1 host/configuration
contracts (`W1-007`, `W1-052`, `W1-055`, `W1-057`, `W1-067`, `W1-111`, and
`W1-112`), leaving **69 of the 76** local runtime contracts pending. All live
`requirements_pending_triage` placeholders are replaced by explicit pending or
not-required dispositions; historical checkpoints remain unchanged. It also binds
all 48 Wave 2 runtime rows to exact passing final-source hosted CPU regressions,
while keeping all 48 whole rows `awaiting_runtime`. The CUDA 13 compare log supplies
bounded four-target compiler evidence for 17 pending Wave 1 rows and 9 Wave 2 rows;
`W2-078` has only a partial header-through-consumer result. The job had no GPU.
[Non-GPU receipt](docs/audits/2026-08-30/evidence/integration/hosted-non-gpu-runtime-closure/receipt.json)
and [compile receipt](docs/audits/2026-08-30/evidence/integration/hosted-cuda-compile-closure/receipt.json).

### Wave 2 complete source reconciliation

- The frozen combined pass reports **185 passed and 1 CUDA-only skip**; the six
  Ruff-normalized test files then report **53 passed** in a bounded replay.
- The final hosted Benchmark Validation workflow reports **4,346 passed, 461
  explicit skips, and zero failures** in its required CPU job; its 736-test
  core contract selection, dashboard, static analysis, shell, silent-fallback,
  and 932-file benchmark-contract gates also pass.
- The final hosted CUDA 13.0 compare workflow succeeds for all 14 configured
  chapters across `sm_100`, `sm_103`, `sm_120`, and `sm_121`. Exact log mapping
  gives bounded compiler evidence to 17 pending Wave 1 rows and exact-source
  compile/link evidence to 9 Wave 2 rows; `W2-078` receives a partial configured-
  target header result. The runner had no GPU, so this is not device, numerical,
  sanitizer, profiler, or performance acceptance.
- The final epoch contains 123 unique changed Python paths: 122 passed the initial
  compile/fatal-Ruff gate and four passed the CI follow-up gate, with three paths
  overlapping. All 16 new Wave 2 tests pass full Ruff at the final source; 20 JSON
  files parse, 330 expectation entries validate, and 44 changed benchmark
  entrypoints lint with zero errors or warnings. Dashboard Jest and TypeScript
  checks pass.
- The exact final source epoch is `3316e0efe985040745ffd926c5f76a6bd4436aff`,
  with a 174-path content manifest and the full disposition matrix in the
  [complete-source receipt](docs/audits/2026-08-30/evidence/integration/wave-2-complete-source/receipt.json).
- W2P08 is source-complete. W2P01 through W2P07 remain `awaiting_runtime` because
  48 findings require one or more of target CUDA/compiler/numerical execution,
  multi-GPU CUDA/NCCL, pinned vLLM, full-model B200, Grace/non-Grace hardware, or
  live NVML evidence.
- There are no untriaged Wave 2 rows. Source publication does not close those 48
  runtime gates or the applicable first-wave runtime acceptance matrix.

## Known constraints and unsafe shortcuts

- `HANDOFF.md` records B200 ownership yielded to another task and prohibits resuming
  GPU/profiler work until ownership returns. Reverify ownership and runner readiness
  before a job. This task has not reclaimed GPUs, changed clocks, registered runners,
  resumed the paused sweep, or claimed any historical result as new evidence.
- The paused full-sweep campaign is separate. Leave its handoff and artifacts intact.
  Determine the necessary affected-target matrix for these findings; broad shared
  harness changes may require a larger sweep, but do not blindly restart that old
  486-target campaign or inherit its completion credit.
- Suggested fixes are proposals. In particular: a per-CTA early return can break
  cluster synchronization; synchronizing after an already-recorded stop event does
  not repair its interval; checksums do not prove full output correctness; a generic
  async-copy call does not prove native TMA; a partial-content scale-cache fingerprint
  cannot prove lifetime/content identity; and guessed tolerances do not prove
  numerical validity.
- Keep the audit's refuted timeout claim as a refutation. The built-in timeout
  implementation must be checked if relevant, not replaced by installing a plugin
  merely because the original allegation mentioned an absent dependency.
- Scope is remediation, verification, and the user-authorized interim publication
  to `main`. That authorization does not include deployment, external communication,
  new cloud spend, or treating publication as runtime/performance acceptance.
  Existing repository acceptance controls continue to govern later actions.

## Historical targeted source-gate checkpoint

This first-wave checkpoint is preserved as historical evidence. The Wave 2
complete-source reconciliation above supersedes its source counts.

- At that checkpoint, the independent Wave 2 re-review classified the original
  inventory as **120 `source_fixed`, 5 `awaiting_runtime`, 2 `in_progress`, and
  1 `already_fixed_with_evidence`**. Both partial residuals were subsequently
  repaired, and the later retained-evidence reconciliation changes the current
  mutable ledger to **115 `source_fixed`, 5 `awaiting_runtime`, 7 `verified`, and
  1 `already_fixed_with_evidence`**. The separate local acceptance matrix contains
  76 first-wave rows: 7 exact host/configuration contracts are verified and 69
  remain pending. Adjacent discoveries now cover LOCAL-001 through LOCAL-052: **40
  `source_fixed`, 11 `awaiting_runtime`, and 1 `verified`**. The verified row is
  LOCAL-025; W1-006 remains source-fixed pending its supported-GPU sub-gate.
- The final tensor-parallel repair passes **18 focused tests** plus **2 related
  hygiene tests**. It initializes invalid KV padding, applies cache-offset causal
  semantics, validates and honors heterogeneous/zero input lengths, grows cached
  workspaces monotonically, keeps uniform cached layouts on lazy lower-right
  bias, and bounds oversized heterogeneous masks with a correct per-sample lazy
  fallback. Repeated autograd and a directed value oracle are included.
- The repository gate passes all **932 benchmark contracts** with zero errors or
  warnings, fatal repository Ruff, focused Ruff, focused mypy, targeted compile
  and diff checks, and import-edge validation.
- Per the user's explicit direction, the full CPU suite was not rerun after the
  final tensor-parallel fixes. The previous **4,181 passed, 450 skipped, 1
  failed** run is retained only as the diagnostic that exposed those defects.
  `code/AGENTS.md` now requires the smallest sufficient focused verification
  loop and prohibits presenting focused checks as a full-suite pass.
- The focused receipt captured 392 changed/new/deleted source paths and the tracked
  diff against base HEAD `b57e4c6a9e261c09ac09208705d040c81b03d35e`;
  source candidate `f49aae73f629f4d90b60d1c6a5b890780e7ef758` records that exact checkpoint.
  [Receipt](docs/audits/2026-08-30/evidence/integration/source-gate-followup/receipt.json)
  and [manifest](docs/audits/2026-08-30/evidence/integration/source-gate-followup/source-targeted-final.json).
- The complete dependency metadata result is 90 direct specifications to 327
  packages, with a 56-package exact-source CPU union and 49-package first-PyPI
  closure. Broad Darwin imports remain 594/600 because six target-runtime
  Triton/cuda-python/FlashInfer packages are absent; source import edges pass.
- Four ZeRO factories pass final CPU/Gloo child-result verification; the other
  57 generic wrappers continue to refuse unsupported harness qualification.
  Actual two-GPU CUDA/NCCL remains pending.
- Offline profiler/metadata, Grace-provider, extension, and chapter seams have
  focused CPU and independent read-only acceptance. Real Nsight, remaining
  finding-specific CUDA/NVCC target-runtime builds, Grace hardware, supported
  Linux installation, GPU correctness/performance, and the 48 retained
  second-wave runtime dispositions remain open gates.

## Final reconciliation

The goal is complete only when all of the following are true:

- [x] Wave 1's 128 IDs and the separate refutation reconcile exactly to the capture.
- [x] The completed second wave has been captured and reconciled to its actual count.
- [x] Every Wave 2 issue has a reviewed source disposition; none is untriaged or
      in progress at the source layer.
- [ ] Every valid issue is fixed and verified; none is deferred or waiting for
      required runtime evidence.
- [ ] Every stale, duplicate or refuted disposition has current evidence and a reason;
      duplicates point to a verified canonical fix, not a dropped issue.
- [ ] Relevant source, CPU, integration, compile/link and supported-hardware checks
      pass on the final integrated revision, with explicit capability exclusions.
- [ ] Invalid historical claims are withdrawn or corrected; new performance claims
      and expectations point only to valid hardware-specific evidence.
- [ ] An independent review reconciles both waves, shared-file interactions, tests,
      documentation and artifacts. Its new valid findings are also resolved.
- [ ] The final report links each issue to its fix and evidence, states any unsupported
      capability limits, and distinguishes verified local work from merge/deployment.

Until then, report source fixes, passing CPU checks, and remaining hardware/intake
requirements separately. Do not mark the goal complete because the plan is written.

### Integration checkpoint: remaining chapter CUDA, metrics and MoE journey

- Remaining P03 chapter CUDA source repairs passed 180 host/source checks and independent source review. Actual CUDA gates remain HOLD (no nvcc/GPU); 195 chapter outputs, seven GEMMs and four invalid-shape checks are prepared. See `docs/audits/2026-08-30/evidence/chapter-cuda/validation-receipts.json`.
- P09 metrics/specification changes passed 102 CPU checks with seven explicit CUDA skips; old reports remain preserved with correction notices. See `docs/audits/2026-08-30/evidence/validation/peak-metrics-receipt.json`.
- MoE metadata/wrapper changes passed 16 CPU checks with three CUDA skips. Legacy names now describe actual shared BF16 implementations; old timing records are not recertified. See `docs/audits/2026-08-30/evidence/moe-journey/metadata-wrappers-receipt.json`.

### Source integration checkpoint: remaining packages and acceptance limits

- P04 feature/epilogue/runner source corrections: 113 host checks passed; the explicit retirement stub follow-up passed 50 checks. Actual CUDA compile, full-output and sanitizers remain HOLD.
- P07 attention/lifecycle/backend corrections: 65 passed, 25 CUDA skips. Wording identifies actual async-copy/Triton providers; no TMA or speedup is inferred from compatibility names.
- Hybrid expert parallelism: 14 passed, 9 CUDA skips, including real four-rank Gloo failure reproductions and corrected collectives/autograd. NCCL and multi-host evidence remain HOLD.
- Native FP8 scaling/full-reference/policy work: 38 passed, 4 CUDA skips. The original large-value cast generated nonfinite output on CPU; actual FP8 target calibration and reviewed limits remain required.
- MoE CUDA/PTX layer validation: 33 passed, 5 CUDA skips. Zero output passed both old and proposed tolerances; complete independent references and reviewed scale-invariant limits now fail closed without policy.
- Incomplete Triton MoE/FFN/grouped-GEMM experiments were withdrawn; active SiLU helper gradients and layout validation repaired. 27 passed, 32 CUDA skips. The retained elementwise Triton path still needs actual device/stream/numerical/memcheck acceptance.
- ZeRO workload parity: 17 passed, 1 CUDA skip. Real CPU dense/sharded updates match an independent reference; four wrappers now share model/precision/data/clipping/warmup/timing. A separate two-GPU production-training gate is prepared and unexecuted. Its wrapper-verification integration is tracked independently.
- P11: 8 real command/discovery checks, 11 local links, a real CPU profiler capture and 4 CSV-extractor tests passed. All 16 operator rows roundtrip; GPU Nsight capture and environment installation were not run.
- The historical 2026-08-31T03:11:40Z public-artifact readback showed only the
  original 128 findings. The user subsequently supplied the completed Wave 2
  report as an attachment; its intake receipt preserves the continuing public
  mismatch and all original identities.

### Final local integration checkpoint

- Ordinary `code/tests` completed with **3,943 passed, 448 explicit skips, zero failures**. All 4,391 first-attempt test identities remain; no CLI test-file exclusions. The first attempt and its three stale-expectation failures are preserved. The existing benchmark-correctness quick selection remains; this does not run every GPU benchmark target.
- All 320 changed/new source, documentation and test hashes plus the tracked diff stayed unchanged through the final run. Python syntax and fatal-name checks passed for 211 files; generated documentation checks passed.
- Complete prefill/decode payloads, workload parity and stream/input refresh are source-repaired; the current attention gate binds 32 exact CUDA cases and 115 dependencies. Its actual CPU preflight is HOLD with zero CUDA dispatch.
- The unrelated parent-side training verifier is withdrawn. All 61 generic wrappers explicitly refuse harness qualification until actual child results and an independent reference exist. Direct scripts remain available; hardware alone cannot restore wrapper qualification.
- Historical fallback table rows now disclose missing/unverified source lineage. Original evidence and all Nanochat logs are preserved and includable.
- The [local receipt](docs/audits/2026-08-30/evidence/integration/final-local/receipt.json), [current status](docs/audits/2026-08-30/STATUS.md), [Wave 2 complete-source receipt](docs/audits/2026-08-30/evidence/integration/wave-2-complete-source/receipt.json), and [runtime handoff](docs/audits/2026-08-30/evidence/integration/runtime-handoff.md) separate completed local work from pending pinned Linux/build/GPU/numerical-policy gates. Wave 2 source remediation is reconciled; its 48 runtime rows remain open, and 69 of 76 locally required Wave 1 runtime contracts remain pending. **The goal is not complete.**

### Dependency and HTTP follow-up checkpoint

- The pinned Torch 2.9.1 Linux wheel requires Triton 3.5.1. Active setup, shared
  requirements, CPU CI and environment documentation now agree. CI also includes
  the missing requests and Nanochat tokenizers dependencies. The original pin
  conflict and missing-dependency failures are preserved.
- Actual Linux-target metadata resolution rejects the original Triton pin,
  resolves the corrected CUDA core pair (26 packages), and resolves the current
  CPU package set with exact CPU Torch/PyPI sources (58 packages). Independent
  review verified all 121 dependency artifacts. No Linux installation occurred.
- The full CUDA requirements graph remains unqualified. The original uv input
  stopped before solving at its relative find-links parser; a diagnostic retaining
  all 90 package specifications stopped at GPUtil's source-only distribution under
  a binary-only policy. Neither is proof of a normal pip/setup version conflict.
- The HTTP test's original silent success was reproduced. Its obsolete route test
  identity now explicitly skips; separate actual ASGI server and owned-child
  failure checks pass. The legacy routes were not restored or counted as working.
- Fresh ordinary `code/tests`: **3,948 passed, 449 explicit skips, zero failures**,
  456.78 seconds. All 4,391 previous IDs remain, with six new cases. All 321 source
  hashes and the tracked diff stayed unchanged; 212 Python files passed syntax
  and fatal-name checks. This remains the existing macOS CPU environment, not the
  pinned Linux stack.
- Docker's local image pull is blocked by the macOS Keychain credential helper;
  no audit container exists and no credential bypass was attempted. Docker start
  resumed nine existing containers, which this task did not modify or stop.
- At this historical dependency checkpoint, original statuses were 52 source-fixed
  and 76 awaiting runtime, with 26 adjacent discoveries recorded separately. The
  current Wave 2 intake and external re-review supersede those status counts. The
  [current receipt](docs/audits/2026-08-30/evidence/integration/pinned-linux-integration/receipt.json)
  and [remaining acceptance](docs/audits/2026-08-30/evidence/integration/pinned-linux-integration/runtime-update.md)
  supplement the preserved earlier checkpoints. **The goal remains incomplete.**
