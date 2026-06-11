# Coding Styles

## BE EFFICIENT AND ASK QUESTIONS AT KEY DECISION POINTS
- Instead of forging ahead and performing a lot of changes, ask me questions if you are unsure or just want re-assurance that your approach is valid.

## Follow Existing Conventions (CRITICAL)
- Before introducing a new pattern, helper, launcher path, environment setup, harness flow, or sample-specific workaround, examine the existing repo conventions and the relevant benchmark/profiling/runtime harnesses first.
- Prefer matching the established conventions used by nearby chapters, shared helpers, harness entrypoints, and existing long-running benchmark flows when a suitable pattern already exists.
- Treat convention changes as rare and deliberate. Only change a convention when the task explicitly requires it or the current convention is clearly wrong and the fix should apply broadly.
- When you intentionally diverge from an existing convention, call out the prior convention, explain why it is insufficient, and make the convention change explicit in the task summary.

## Safety (CRITICAL)
- DO NOT run destructive git commands in this repo (including `git restore`, `git checkout`, `git reset --hard`, `git revert`, or mass file deletions) unless I explicitly ask.
- NEVER restore/revert/checkout any file to `HEAD` or any commit. Always keep files as-is and include changes (even if unexpected).
- DO NOT delete any files, including untracked files or locally modified files, unless I explicitly ask.
- If you notice unexpected local file modifications, always call them out and ask for guidance; default to keeping them as-is and including them in the changes unless I explicitly say otherwise.
- When you detect modified or untracked files, please treat them as part of this task.
- If a file is already modified or open in the editor, keep its current contents as-is and include it in the final change list; you may continue editing without asking.
- Avoid symlink-based benchmark/profile artifact workflows when possible; prefer real files (copy/materialize artifacts) so pairing/comparison tools operate on concrete paths.
- For Nsight artifacts/comparisons, do not rely on symlink-only layouts; stage real baseline/optimized files in a concrete directory with explicit role names before running compare tools.
- Do not introduce new symlink-dependent profile-pair flows; if a symlink appears in inputs, copy it to a real file before pairing/comparison.
- The Amazon book link in `README.md` is expected to fail automated link checks due to bot protection; treat it as an allowlisted exception.

## Dogfood + Verification (CRITICAL)
- Always dogfood changed runtime paths with a real repo invocation when feasible; do not stop at unit tests if the code is reachable through a CLI, MCP tool, benchmark harness, profiler flow, or report path.
- Every fix must include explicit verification evidence in the same change: targeted tests, syntax/import validation, and at least one realistic execution path that exercises the changed code.
- If the change touches GPU, profiling, benchmarking, or runtime environment logic, prefer running the verification on the local GPU host instead of relying on CPU-only mocks.
- Treat silent fallback removal as incomplete unless the failure or degraded state is visible in structured outputs, logs, reports, or tests.
- Record the exact verification commands and outcomes in the task summary so the next reviewer can reproduce the evidence trail quickly.

## Full Sweep Playbook
- For a fresh all-stages `run-e2e` plus fix-along-the-way workflow, use the repo-root guide in `FULL_SWEEP.md`.
- Keep that guide aligned with the current `bench run-e2e` contract, resume behavior, and `book-after/chXX*` alignment expectations whenever the full-sweep workflow changes.

## Required Capability Gating (CRITICAL)
- If a benchmark, profiler path, or runtime flow requires specific hardware, software, driver, toolkit, or service support, fail fast when that support is unavailable.
- Use explicit `SKIPPED:` or equivalent hard diagnostics for unsupported capability checks; do not continue with a degraded fallback under the same benchmark name.
- Never substitute CPU emulation, reduced-feature code paths, alternate runtimes, or host/device fallback behavior for a required capability unless the target is renamed and documented as a different benchmark.
- Do not publish, compare, or store performance numbers from unsupported fallback paths as if they were valid results for the required benchmark contract.
- Keep expected-unsupported classification explicit in structured outputs, logs, tests, and queue ledgers so unsupported environments cannot be mistaken for valid benchmark data.

## Public-Repo Identifier Hygiene (CRITICAL)
This repository is PUBLIC and widely forked. Never commit live-infrastructure identifiers: pod names, Kubernetes namespaces used as cluster references, kubectl-context strings, node names, org/tenant IDs, internal hostnames, or experiment labels that embed any of these. Use placeholders (`<gb300-pod>`, `<namespace>`) in docs/runbooks/field reports; the real values live in local operator memory/config outside the repo. Generic component names inside published telemetry contracts are fine; references to live cluster objects are not. The current tree was scrubbed 2026-06-11; history before that retains old identifiers — treat those as burned/rotatable, not as secrets, and do not reintroduce them.

## Toolchain Workarounds & Abort Attribution (CRITICAL)
Origin: the 2026-06-11 GB300 sm_103a `tcgen05.wait.st` incident (runbook B69/B70/B71; mechanism evidence `code/upstream/triton-tcgen05-wait-st/STATUS.md`) — `core/benchmark/triton_compat.py` de-suffixed `sm_103a -> sm_103` and impersonated an upstream Triton bug for weeks, spawning guards that tainted banked measurements (B23c).
- Before attributing an uncatchable native abort (LLVM `Cannot select`, ptxas error, SIGABRT in a JIT) to an upstream toolchain bug — and before adding any protective guard — reproduce on the VANILLA stack: strip repo monkey-patches, compat shims, import-time `configure_*` side effects, and env clamps first. They are invisible in the failing stack trace.
- Empirical narrowing is not attribution. The VERDICT needs the two-sided mechanism probe: vanilla clean AND applying the suspected local transform to vanilla reproduces the exact abort. Search the upstream tracker for the exact error string before filing.
- Hardware/toolchain workarounds are ALLOWLISTS, never denylists: clamp exactly the arch/capability/version the bug was proven on (`sm_121[a] -> sm_120`), never blanket-transform identifier strings whose semantics you do not own (the `a` suffix selects arch-conditional ISA: tcgen05/TMA on sm_100a/sm_103a, wgmma on sm_90a). Pin both halves with a unit test — the clamp AND verbatim pass-through for everything else (`tests/test_triton_compat_arch.py` is the pattern).
- Every protective guard (compile-mode downgrade, capability-gated skip, fallback path) must record the bug it avoids WITH evidence and a falsifiable retirement condition. A guard with no retirement condition is permanent perf debt.
- Numbers measured under a protective guard are tainted: they carry the guard in their measurement context and cannot close a "no win" verdict (B23c banked "all compile modes ~1.5%" while max-autotune silently aborted; post-fix it wins 1.083x). When a guard is retired, re-sweep every formerly-gated path.
- Inductor-style autotuners SWALLOW crashing candidates silently, biasing the search toward extern kernels — and warm caches can mask an abort entirely (the B71 cold-cache counterfactual: the abort only reproduces with a fresh `TRITON_CACHE_DIR`). Validate guard removals and abort claims cold-cache.

## Cluster Field Report Mode (ONLY when working in `code/cluster*` or writing the cluster field report)
- This section adds constraints specific to cluster evaluation work; all other rules in this file still apply.
- Discovery/inventory may use `nvidia-smi` and related commands. Benchmarks/profiling must still lock clocks via the harness (`lock_gpu_clocks`); do not manually lock clocks via `nvidia-smi`.
- Cluster eval scripts MUST fail unless clock locking succeeds. Do not add bypass flags; any run without locked clocks is invalid and should not be produced by the harness.

### Root-Cause First (CRITICAL)
- Do not rely on fallback behavior when a benchmark/probe path fails. Identify the underlying root cause and fix it.
- If a mode/feature is unsupported on the current cluster (for example a specific GDR mem type), fail fast with a clear diagnostic and required remediation; do not silently continue.
- Remove warning-only continuation paths for required checks in canonical runs. Canonical suites must be green without remediation context.
- Treat "retry in another runtime/tool" as a temporary debugging step only; do not keep it as the default collection path.
- For GPUDirect RDMA specifically: probe requested `--gdr-mem-types` up front and fail preflight if unsupported (for example `cuda_mem_type=1` with no ODP MR / dmabuf support). Do not downgrade requested coverage mid-run.
- Validate required artifacts by semantic status (`status=ok`, lock metadata, non-empty metrics), not only by file existence.
- Do not manually inject signals/interruptions into benchmark or serving processes during canonical runs; treat any such run as invalid and rerun fresh.
- For multinode vLLM: treat worker `rc=137` after forced teardown as a bug in run lifecycle handling; stop workers cleanly and only report success when leader + worker states are both semantically clean.
- For nvbandwidth host-runtime failures like `cudaErrorUnsupportedPtxVersion`: fix the host user-mode compatibility chain (for example explicit CUDA compat libs) instead of switching canonical collection to a different runtime mode.
- When logs show `CUDA Memory type is not supported with no odp MR or with dmabuf`, treat it as an environment capability gap that must be preflight-detected and reported, not a warning to continue through.

### Canonical Run Hygiene (CRITICAL)
- After producing a new canonical run, clean up superseded intermediate run artifacts so reports only point to the canonical package.
- Preserve the canonical run and any explicitly requested historical baselines; remove only superseded intermediates.
- Keep `results/structured/`, `results/raw/`, and `docs/figures/` tidy enough that stale artifacts cannot be mistaken for canonical evidence.
- When cleanup is requested, remove superseded run artifacts in the same change as report synchronization so stale paths cannot remain referenced.

### Report Completeness + Sync (CRITICAL)
- Before finalizing `field-report.md`, diff against prior report revisions and restore any dropped sections/visuals/evidence links that are still relevant.
- `field-report.md` and `field-report-notes.md` must always be synchronized to the same canonical run id, metrics, issues, and artifact links.
- For localhost/single-node evaluations, always produce and maintain the full template-style package: `cluster/field-report-localhost.md` + `cluster/field-report-localhost-notes.md` mapped to the canonical localhost RUN_ID.
- A standalone `results/structured/*_localhost_environment_report.md` is supplemental only and is never an acceptable replacement for the localhost field-report package.
- If report numbers diverge from current canonical artifacts, treat that as a blocker and fix the report (or rerun collection) before sign-off.
- Explicitly include required issue ledgers (missing artifacts, GDR requested vs effective, latency knees) and verify each claim against canonical structured artifacts.

#### Report Update Checklist (CRITICAL)
- Whenever you update any of these files, complete the full checklist in the same change:
  - `cluster/field-report.md`
  - `cluster/field-report-notes.md`
  - `cluster/docs/field-report-template.md`
  - `cluster/docs/advanced-runbook.md`
- Keep canonical run id + artifact links synchronized across all four docs.
- Run the field-report validator and treat failures as blockers:
  - `cluster/scripts/validate_field_report_requirements.sh --report cluster/field-report.md --notes cluster/field-report-notes.md --template cluster/docs/field-report-template.md --runbook cluster/docs/advanced-runbook.md --canonical-run-id <RUN_ID>`
- Validator stale-artifact hygiene failures are blockers: remove unreferenced non-canonical run artifacts (structured/raw/figures) or explicitly allow requested historical baselines with `--allow-run-id <RUN_ID>`.
- Do not leave old run artifacts on disk that are not referenced by the latest canonical report package.

### Case Study Contract (CRITICAL)
- Treat the cluster case study prompt as a hard contract, not guidance.
- Mandatory narrative outcomes:
- tell a clear cluster story (first-contact timeline + operator reality),
- explicitly call out what is weird/new/interesting,
- anchor on 1-2 primary benchmarks for small-team AI relevance (Benchmark A/B),
- provide reproducible scripts/commands and structured outputs,
- include visualizations and table-based interpretation,
- include insights from direct operator experience that are not obvious from specs alone.
- Table-forward rule: all high-value sections must be table-first (narrative can follow), including TL;DR, scope, merged weird/normal findings (baseline + deep-dive), benchmark summaries, required issues, risks, recommendations, reproducibility, and activity log.
- Do not remove high-value sections during rewrites. If cleanup removes historical artifacts, rewrite the section against current canonical evidence instead of dropping the section.
- Before sign-off, run the field-report validator and treat any missing section/requirement as a blocker:
- `cluster/scripts/validate_field_report_requirements.sh --report cluster/field-report.md --notes cluster/field-report-notes.md`
- Required high-value section inventory (must remain present):
- `TL;DR`
- `Scope + Canonical Artifacts`
- `Cluster Story (First Contact)`
- `Weird / New / Interesting (with Normal Baseline)`
- `Baseline vs Weird Log` (subsection)
- `Deep-Dive Findings` (subsection)
- `Benchmark A (Networking Story)`
- `Benchmark B (Inference Story)`
- `Required Issues (Explicit)`
- `Root Cause + Fix Mapping`
- `Report Completeness Delta`
- `Gaps, Risks, and Smell Checks`
- `Implications for Small AI Teams`
- `Stakeholder Recommendations`
- `Repro Steps`
- `Reproducibility Package`
- `Appendix (Coverage vs Case-Study Goals)`

### Engagement Scope (CRITICAL)
- Explicitly declare the evaluation scope: which hosts/nodes are in-scope, GPU count per host, and any excluded nodes. Never use excluded nodes for discovery or benchmarks.
- Preserve SSH trust by default: do not rotate SSH host keys or machine-ids unless explicitly requested and logged, with pre/post identity snapshots.

### Case Study Field Report (CRITICAL)
- Treat this as a mini product review: first-contact experience, what is weird/new, and 1-2 benchmarks that explain behavior.
- Deliverables must separate: first-contact experience, weird/new/interesting findings, two benchmark plot arcs, and a reproducibility package.
- Do not mention the origin of the challenge; only reference Semianalysis when explicitly describing the cluster evaluation tooling expectations baseline.

#### Discovery + Metadata (CRITICAL)
- Run discovery first and write a single JSON metadata file (Appendix + reproducibility).
- Compute/topology capture: `nvidia-smi`, `nvcc --version`, `nvidia-smi topo -m`, `lscpu`, `numactl -H`, `nvidia-smi -q -d CLOCK,POWER`.
- Networking capture: interface types and speeds via `ibstat`, `rdma link`, `ethtool`, `NCCL_*` defaults, node-to-node latency sanity check via `ping` and `iperf3` only if allowed.
- Storage capture: `lsblk`, `df -hT`, `mount`, plus baseline sequential/random IO only if storage is the benchmark focus.
- Orchestration capture: multi-node launcher (Slurm/K8s/etc), image/container flow (Docker/Podman/Enroot/Singularity), and constraints (no root, egress limits, outbound internet).
- Runtime/CVE capture: collect per-node container runtime evidence and record CVE status in structured outputs (at minimum CVE-2025-23266 and CVE-2025-23267).
- Consistency: full eval suite and health suite must both run the runtime/CVE collection by default; optional skip flags are allowed only as explicit opt-outs and must default to enabled.
- Outcome: 5-10 crisp bullets that describe the cluster personality (HPC vs cloud UX, network behavior, storage behavior, job launch overhead).
- Always establish what "normal" looks like for compute, network, storage, and launch so weirdness is detectable.

#### Benchmarks (CRITICAL)
- Tell the story with 1-2 benchmark arcs, but run a complete eval suite when feasible so results are reusable across clusters.
- Benchmark A (Networking story): `nccl-tests` `all_reduce_perf` (single-node + multi-node) to explain multi-GPU/multi-node scaling behavior.
- Benchmark B (Inference story): vLLM online serving concurrency sweep (`vllm serve` + `vllm bench serve`) to capture tok/s, TTFT/TPOT, and tail latency knees.
- Benchmark C (Compute sanity): dense GEMM (BF16) to quickly detect per-node/per-GPU throughput deltas.
- Benchmark D (System): `iperf3` + IB perftest + torch distributed all-reduce sanity + `fio` (sequential + random IO) to explain bottlenecks outside kernels.
- If vLLM is blocked (egress/model download), pivot quickly and document the constraint as an insight.
- Inference fallback options: NanoGPT token/sec speedrun, torch GEMM/matmul microbench, or a smaller open model.

#### Repro Harness + Repo Conventions (CRITICAL)
- Use the `code/cluster*/` layout (whichever you are working in) with `scripts/`, `analysis/`, `results/raw/`, `results/structured/`, `docs/figures/`.
- Use `RUN_ID=YYYY-MM-DD` (default in scripts). If uniqueness is needed, append a suffix but keep the date prefix (recommended: `_neocloud_<nodecount>nodes_<gpu>_<gitsha>`).
- Required outputs (written under `results/structured/`):
- `${RUN_ID}_${label}_meta.json` with hardware/software/env and exact commands.
- `${RUN_ID}_manifest.json` with file hashes + artifact counts.
- `${RUN_ID}_nccl*.json` with message size to algbw/busbw plus topology context, and `app_clocks` captured.
- `${RUN_ID}_vllm*.csv` / `.jsonl` with concurrency, prompt/gen lengths, tok/s, latency metrics, GPU util, memory, and `app_clocks` captured.
- `${RUN_ID}_gemm*.csv` with TFLOPS and `app_clocks` captured.
- `${RUN_ID}_fio*.json` with seq MB/s + rand IOPS (and test parameters).
- Raw logs belong in `results/raw/`. Plots belong in `docs/figures/`.

#### Charts That Tell The Story (CRITICAL)
- NCCL charts: all-reduce bus bandwidth vs message size (single-node and multi-node) and scaling efficiency vs GPU count for 2-3 message sizes.
- vLLM charts: tokens/sec vs concurrency and p50/p99 latency vs concurrency.
- Interpretation bullets must explain intra-node vs inter-node behavior, latency knees, oversubscription/routing signals, KV-cache/memory bottlenecks, and stability/tail latency behavior.

#### Operator Reality Insights (CRITICAL)
- Capture time-to-first-job, launch ergonomics, observability, stability, multi-tenancy noise, data path/caching, and support responsiveness.
- Maintain a running "Normal vs Weird" log; identifying weirdness is the primary goal.

#### Questions To Ask Early (CRITICAL)
- Preferred container/runtime and any golden path examples.
- Outbound internet policy for model downloads.
- Recommended NCCL settings and network interfaces.
- Expected topology (full bisection vs oversubscription) and multi-tenancy constraints.
- Known gotchas (MTU, RoCE tuning, NCCL timeouts, filesystem caching).

#### Write-up Format (CRITICAL)
- Deliver as Google Doc or 10-12 slide deck.
- Required sections: TL;DR, cluster story, weird/new findings, benchmark A, benchmark B, implications for small AI teams, repro steps, appendix.
- Where helpful, compare against public benchmarks for context (e.g., MLPerf), without referencing the source of the challenge.

#### Stakeholder Markdown Presentation (CRITICAL)
- For all `field-report*.md` files, include a Table of Contents near the top.
- Keep visuals large and readable in stakeholder-facing markdown; do not use tiny thumbnail images inside table columns.
- Do not add a dedicated "Visual" table column; place visuals under the narrative section and keep evidence/data links directly below each visual.
- Image click-through must open the image artifact itself (`docs/figures/...`), not a JSON/CSV/TXT data file.
- Avoid nested bullet-heavy formatting in visual sections; prefer clean paragraphs, concise tables, and explicit `Data:` lines below images.
- Render visuals as block elements (use `<p><a href="..."><img .../></a></p>`) so evidence/data lines never wrap to the right of images.
- Prefer tables over bullets for dense stakeholder sections (TL;DR, recommendations, reproducibility, activity log, historical change logs).

#### Quality Bar (CRITICAL)
- Taste in what to measure, reproducibility, rigor (multiple runs + warmups + noise notes), systems intuition, communication clarity, and practical empathy.
- Always use evidence-based verification over opinion-based conclusions.
- We develop on target hardware; test every assumption with concrete empirical evidence (bench outputs, profiler artifacts, and correctness verification) before declaring a result valid.

#### Execution Order (CRITICAL)
1. Discovery to meta.json.
2. Choose benchmark pair.
3. Run NCCL and plot.
4. Run inference/training benchmark and plot.
5. Write narrative around the plots.

## Deprecations (CRITICAL)
- Do not add or keep deprecated entrypoints, shims, compatibility wrappers, or transitional aliases.
- Deprecations are not allowed to persist anywhere: remove them immediately from code, docs, READMEs, and tests.
- When removing a deprecation, replace all references with the latest entrypoint(s) or APIs in the same change.
- Do not leave deprecation notices, TODOs, or compatibility flags behind; purge and replace in one pass.

## Benchmark Stability (CRITICAL)
- ALWAYS lock GPU clocks before any benchmark/profiling run; focus on relative performance rather than absolute numbers.
- Prioritize relative speedup between baseline/optimized pairs over absolute performance numbers.
- Fix as many variables as possible (persistence mode, power limits, thermal state) and keep them stable across baseline/optimized runs.
- Use the repo’s clock-locking mechanism (`lock_gpu_clocks` in the harness); do not manually invoke `nvidia-smi` to lock clocks.
- Confirm app_clock (SM/memory) is present in both console telemetry and the run manifest for every benchmark run; treat missing app_clock as invalid.
- NEVER disable Nsight tools (ncu/nsys); profiling runs must use both and they must succeed.
- Prefer the most local correct fix for benchmark failures. Patch the harness only for cross-cutting infrastructure defects that can affect many benchmarks (for example profiler path handling or safe generic profiling controls).
- Keep benchmark-specific semantics local to the benchmark. If an example declares the wrong timing model, publishes outputs during `setup()`, misstates metadata, or needs profile-specific replay/metric preferences, fix that in the chapter/example code instead of weakening the harness.
- Do not relax harness validity checks to make broken examples pass. Keep the harness strict, fix the example to satisfy the contract, and only add harness abstractions when the same safe pattern clearly repeats across multiple benchmarks.

## Provenance Review (CRITICAL)
- Review the provenance of each benchmark, profiling, expectation-refresh, and analysis request before trusting the result. If provenance is incomplete, ambiguous, or mixed, improve the system first instead of hand-waving the gap away.
- Preserve and surface the provenance needed to explain a result end to end: `run_id`, target list, git commit, hardware key, profile name, timestamp, iterations, warmup iterations, clock-lock/app-clock state, and any validation issues or rejection reasons.
- Prefer structured provenance in machine-readable artifacts over console-only explanations. If a result can only be explained from log text, add fields, ledgers, or reports so the next review is trustworthy and auditable.
- When expectation updates are rejected, separate true performance drift from provenance-only rejection. Record which provenance fields differ and rerun unstable cases before refreshing stored expectations.
- When analyzing mismatches, improve the artifact trail as needed to make the conclusion more trustworthy, explainable, verifiable, and auditable rather than relying on manual reconstruction.
- If the user explicitly approves current-host verification in a virtualized environment, local benchmark/profiler reruns may proceed there as long as GPU clocks are locked, the run provenance is recorded, and the task summary clearly states that the result is virtualized/non-canonical. This does not relax bare-metal requirements for canonical or publish-grade results.
- If the user explicitly requests expectation refreshes from the current host, updating expectations from a virtualized environment is allowed as long as GPU clocks are locked, the run provenance is recorded, and the task summary clearly states that the source run was virtualized/non-canonical. This exception does not relax any cluster field-report or publish-grade artifact requirements.

## NVFP4 Grouped GEMM Perf Playbook (CRITICAL)
- Treat `--verify` as the hard gate for every tuning candidate; do not promote non-verified wins.
- Use paired A/B runs (old/new interleaved) before changing defaults; single-run wins are not sufficient.
- Prefer geomean decisions only after repeated runs (at least 4-8 repeats for comparison), and track stdev.
- Keep case routing explicit (`CASE{idx}_VARIANT`, `CASE{idx}_CLUSTER_*`, `CASE{idx}_RASTER_ORDER`, `CASE{idx}_MAX_SWIZZLE`, `CASE{idx}_USE_PDL`).
- Case2/case3 are highest ROI for geomean; optimize them first, then re-check full geomean.
- `AISP_NVFP4_GROUP_GEMM_SPLIT_CASE23=1` has regressed in current experiments; default remains non-split unless new verified evidence appears.
- 2SM case2/case3 paths often require different cluster settings to pass `can_implement`; legality does not imply better latency.
- Stage-count variants (`*_s1..s4`) can look fast in quick scans and regress in repeated verify; always re-validate.
- New tile-shape variants (for example N=192) must be treated as exploratory until repeated verify A/B proves a win.
- Prefer robust sweeps that end with a verify phase over quick scan-only ordering.
- For long sweep scripts, use unbuffered execution (`python -u`) and `flush=True` prints so progress is visible in real time.
- Buffered vs unbuffered logging affects observability and debugging speed, not kernel performance; keep comparison methodology identical when making claims.
- If a script captures subprocess output (`capture_output=True`), expect delayed logs; this is a logging artifact, not a kernel artifact.
- Use clock-locked microbench for low-noise directional checks, but only accept default changes when fresh-input verify A/B also improves.
- Do not trust one measurement surface alone; reconcile microbench, fresh-input verify, and (when needed) Popcorn test mode behavior.
- Prefer single-source runtime plans with stable caching; avoid re-planning per call unless shape/tunable/variant changes.
- Optimize host-side submission overhead (pointer table updates, copies, graph replay path) only if correctness and final geomean remain green.
- For router-path tuning, single-run lows can be very misleading (e.g. `~9.70 us` one-off): require interleaved ABAB before promotion.
- Router tuning scripts that mutate env vars in-process MUST call `refresh_case_configs_from_env(clear_caches=True)` before benchmarking; `_CASE_CONFIGS` is built at import time.
- Current router default winner on strict verify + ABAB (clock-locked) is:
  - `case0`: `variant=2sm`, `cluster=2x1`, `raster_order=2`, `use_pdl=1`, `max_swizzle=8`
  - `case1`: `variant=2sm`, `cluster=2x1`, `raster_order=2`, `use_pdl=0`, `max_swizzle=8`
  - `case2`: `variant=1sm_n128_case23_s4`, `cluster=1x2`, `raster_order=0`, `use_pdl=0`, `max_swizzle=16`
  - `case3`: `variant=1sm_n128_case23_s5`, `cluster=1x2`, `raster_order=0`, `use_pdl=0`, `max_swizzle=8`
  - Latest strict all-case snapshot after promotion: `~10.5388 us/group` (`/tmp/nvfp4_postpromote_static_20260225_1959.json`).
  - Promotion evidence: 8-pair ABAB on the promoted pair (`case2=s4 sw16`, `case3=s5 sw8`) gave `delta_mean_B_minus_A~-0.0778 us`, `stdev~0.0754`.
  - Recent strict snapshots remain noisy in a tight band (`~10.53-10.75 us/group` observed), so keep ABAB as the promotion gate and do not trust single-run lows.
- Cross-product recombination of top per-case configs can beat greedy coordinate descent; run a small cross-product rerank after per-case sweeps.
- Popcorn test-mode checks can fail due service maintenance (`503 Offline for Maintenance`); treat this as external outage, not kernel/submission rejection.
- `1sm_n128_s1` is a known pathological option for case2/case3 (`~71 us` case2 and `~31-52 us` case3 in verify-green runs); prune it from practical search spaces.
- Case2 `variant=1sm_n128_case23` with `cluster=1x2,raster_order=2,use_pdl=0,max_swizzle=8` is unstable across repeated all-case checks; keep a non-case23 N128 lane as default unless new ABAB/stability evidence is clear.
- Case2 `variant=1sm_n128_case23_s4` (`cluster=1x2,raster_order=2,use_pdl=0,max_swizzle=8`) can win on static-input ABAB, but regressed on fresh-input A/B in repeated checks; keep exploratory/off for defaults.
- Case2 `variant=1sm_n128_case23_s4` with `max_swizzle=0` also showed static-input ABAB gains but regressed on fresh-input A/B; do not promote without fresh-input confirmation.
- Case0/case1 `2sm_s4, cluster=2x1, raster_order=2, use_pdl=0, max_swizzle=8` can appear strong in coordinate-descent searches, but direct ABAB rechecks were weak/noisy (`case0-only ~-0.0187 us`, `case1-only ~+0.0487 us`, combined `~+0.0357 us`); keep `case0/1=2sm` defaults for now.
- Historical note: `1sm_n128_s5` had directional ABAB wins in older routing experiments, but this path is superseded by `1sm_n128_case23_s5` defaults.
- Case2 exhaustive `1sm_n128_case2` lane (new CUDA path with dedicated carveout policy) is not currently promotable:
  - Best single-run geomean candidate (`cluster=1x2, raster_order=0, use_pdl=1, max_swizzle=0`) reached `~10.814 us` in one sweep.
  - 6-pair ABAB vs current default was negative/noisy (`delta_mean_B_minus_A~+0.021 us`, stdev `~0.078`), so keep it exploratory/off.
- Case2/case3 reserve-byte tuning must be compile-time deterministic:
  - `labs/nvfp4_group_gemm/cutlass_extension.py` now forwards `AISP_NVFP4_GROUP_GEMM_1SM_N128_CASE2_RESERVE_BYTES` and `...CASE3_RESERVE_BYTES` into NVCC flags and extension naming.
  - Without these compile-time defines in the build contract, reserve sweeps are noise-prone and not attributable.
- Deterministic case2 reserve sweep (`variant=1sm_n128_case2`, `cluster=1x2`, `raster_order=0`, `max_swizzle=0`) remained non-promotable:
  - Best single-run candidate was `reserve=8192,use_pdl=1` at `~10.513 us/group`.
  - 6-pair ABAB vs default regressed on average (`delta_mean_B_minus_A~+0.0347 us`), so do not promote.
- Case2 reserve-byte retune on dedicated `1sm_n128_case2` remains non-promotable on current router defaults:
  - Best single-run candidate was `reserve=8192` with `cluster=1x2,raster_order=1,use_pdl=1,max_swizzle=0` at `~10.508 us/group`.
  - 6-pair ABAB vs default regressed on average (`delta_mean_B_minus_A~+1.116 us`, with a large outlier spike), so keep this path exploratory/off.
- Case2 stage-lane cross-check on current router baseline keeps `1sm_n128_s5` as the least-bad N128 stage lane:
  - `s1`/`s2`/`s3` are strongly slower (`~74 us`, `~23 us`, `~17.5 us` for case2 in representative strict runs).
  - `s4`/`s6`/`s7` remain slower than `s5` on all-case geomean under strict verify.
- Historical note: case3 `1sm_n128_s4` wins were against older `case23_s4` routing and are superseded by the current `case23_s5` defaults.
- Case3 dedicated lane (`variant=1sm_n128_case3`) verifies strict all-case but is currently non-promotable:
  - Best rescored single-run config (`cluster=1x2,raster_order=0,use_pdl=0,max_swizzle=0`) reached `~10.473 us/group`.
  - 8-pair ABAB vs default was effectively neutral/slightly negative (`delta_mean_B_minus_A~+0.0027 us`, stdev `~0.0798`).
- Deterministic case3 reserve sweep on `1sm_n128_case3` is also non-promotable:
  - Best single-run reserve was `2048` with `~10.466 us/group`.
  - 6-pair ABAB vs default regressed on average (`delta_mean_B_minus_A~+0.0472 us`), so keep case3 default on `1sm_n128_s4`.
- New kernel lane `1sm_n128_case23_s5` (block-scaled reduced-SMEM s5) is verify-green and currently promotable:
  - ABAB (`repeats=64`, 12 pairs) vs prior defaults: `delta_mean_B_minus_A~-0.0301 us`, `median~-0.0235 us`, `stdev~0.054`.
  - Confirmation ABAB (`repeats=80`, 8 pairs): `delta_mean_B_minus_A~-0.0204 us`, `median~-0.0251 us`, low spread (`stdev~0.0366`).
  - Router defaults now use `case2=case3=1sm_n128_case23_s5` with `cluster=1x2, raster_order=0, max_swizzle=0`.
  - Fresh 3x stability check (`repeats=80`): new defaults `geo_mean~10.5610` vs old defaults `geo_mean~10.5738` (directional win, small margin).
- `1sm_n128_case23_s5` reserve-byte tuning is now compile-time deterministic and non-promotable so far:
  - `cutlass_extension.py` forwards `AISP_NVFP4_GROUP_GEMM_1SM_N128_CASE23_S5_RESERVE_BYTES` into NVCC flags + extension naming; this is required for attributable reserve sweeps.
  - Strict sweep (`4096/8192/12288/16384/20480`) on current defaults did not beat the default reserve; all candidates were slower in single-run checks.
  - 6-pair ABAB for best candidate (`8192`) vs default regressed on average (`delta_mean_B_minus_A~+0.0869 us`, high variance), so keep default reserve.
- Case3 dedicated retune on top of current defaults remains non-promotable:
  - Best screened candidate (`variant=1sm_n128, cluster=1x2, raster_order=2, use_pdl=1, max_swizzle=0`) looked directional in short rescoring.
  - 12-pair ABAB vs default was effectively neutral/slightly regressive (`delta_mean_B_minus_A~+0.00075 us`, stdev `~0.076`), so do not promote.
- Joint case2/case3 combo search (focused 30-combo matrix over top tunables) also remains non-promotable:
  - Top directional combos reached `~10.55 us/group` in single-run checks.
  - ABAB on top-ranked combos was flat/noisy (`~0` to positive mean deltas); keep defaults unchanged.
- New kernel lane `1sm_n128_case23_s1` (StageCount=1) is verify-green but non-promotable:
  - Severe regressions in strict runs (`case2~74 us`, `case3~32 us` on representative probes), so keep off.
- New kernel lane `1sm_n128_case23_s6` (reduced-SMEM s6) is non-promotable:
  - Initial ABAB (`repeats=64`, 12 pairs) on mixed route (`case2=s6`, `case3=s5`) looked positive.
  - Confirmation ABAB (`repeats=80`, 8 pairs) regressed (`delta_mean_B_minus_A~+0.0208 us`), so keep s6 off.
- Case1 promotion note: `use_pdl=0` on `variant=2sm, cluster=2x1, raster_order=2, max_swizzle=8` gave repeated small wins in long sequential checks; keep ABAB gating when re-validating on fresh thermal state.
- Case0 note: switching `use_pdl` from `1 -> 0` on current `2sm` routing is not robustly positive (6-pair ABAB `delta_mean_B_minus_A~+0.008 us`), so keep case0 `use_pdl=1`.
- Case2 `variant=1sm_n192` (including fixed-stage experiments and tunable sweeps) is currently non-viable (`~17.7-19.5 us/group` for case2) and should remain off.
- New CUDA case2 fixed-stage variants (`1sm_n128_case2_s3`, `1sm_n128_case2_s4`) compile and verify, but are currently non-promotable:
  - `case2_s3` is consistently slower (`~+0.17 to +0.34 us` in representative strict checks).
  - `case2_s4` can show directional single-run wins, but repeated burn-in ABAB remains flat/slightly regressive (e.g. `delta_mean_B_minus_A~+0.005 to +0.006 us` with visible spikes).
- New CUDA case2 NVF4 fixed-stage variants (`1sm_n128_case2_nvf4_s3`, `1sm_n128_case2_nvf4_s4`) compile and verify, but are currently non-promotable:
  - `case2_nvf4_s3` is a clear regression (`~10.90 us` geomean in representative strict checks).
  - `case2_nvf4_s4` is neutral/noisy in burn-in ABAB (`delta_B_minus_A` around `-0.011 us` at `repeats=36` and `-0.0018 us` at `repeats=64`, with split wins and high stdev), so keep off by default.
- Case2 NVF4 tunable sweeps (cluster/raster/pdl/swizzle) produced directional single-run lows but no robust promotion:
  - Best quick point observed: `variant=1sm_n128_case2_nvf4, cluster=1x2, raster_order=2, use_pdl=1, max_swizzle=16` (`~10.548 us` quick geomean).
  - Burn-in ABAB on that point regressed overall (`delta_B_minus_A~+0.0456 us`, 12 pairs), including large spike risk.
  - Secondary quick point `cluster=1x1, raster_order=0, use_pdl=1, max_swizzle=0` also regressed in burn-in ABAB (`delta_B_minus_A~+0.0595 us`, 8 pairs).
- Popcorn test-mode checks can be blocked by service availability; a 2026-02-26 check returned `503 Offline for Maintenance`, which provides no timing or structure-validation signal for candidate quality.
- Case2 kernel-family screens on the current default route continue to show only N=128 families as viable:
  - `n64/n64_nvf4/n192/n256/2sm_case2` lanes are materially slower (or fail `can_implement`) for the all-case geomean objective.
  - Keep case2 tuning focused on N=128 lanes unless new correctness-preserving schedule families are added.
- ABAB methodology update for router/kernel pivots:
  - Use explicit A/B burn-in before counting pairs to avoid first-iteration startup artifacts.
  - Treat spike-prone candidates as non-promotable even when trimmed means look slightly positive.
- Pointer-update path finding (host/runtime overhead):
  - `AISP_NVFP4_GROUP_GEMM_NATIVE_PTR_UPDATE=1` is now promoted as default in `popcorn_submission_tuned_router.py` (override with `=0` for A/B).
  - Fresh-input ABAB remains strongly positive after promotion (`delta_old_minus_new~+13.16 us`, `6/6` wins for new default at `repeats=16`).
  - Static-input ABAB is also net-positive in the latest pass (`delta_old_minus_new~+0.0558 us`, `4/6` wins for new default), with higher variance on old path.
  - Additional host-overhead fix landed in `GemmPlanT::update_ptrs_from_tensors`: preallocate and reuse pinned host pointer scratch per plan (no per-call `torch::empty`); this further reduces fresh retarget overhead without changing math paths.
  - New packed pointer-table fast path landed in `GemmPlanT::update_ptrs_from_tensors`: when pointer tensors are row-views over one `[6, G]` table, use one `cudaMemcpyAsync` for all rows; fallback remains row-wise copies for non-packed layouts.
  - Router-path default update (2026-02-27, late pass): keep `AISP_NVFP4_GROUP_GEMM_SKIP_PTR_UPDATE_ON_SAME_DATA=1` and `AISP_NVFP4_GROUP_GEMM_SINGLE_SLOT_FAST=1`, and set `AISP_NVFP4_GROUP_GEMM_ENABLE_DATA_FAST_CACHE=1` by default.
    - Correctness/root-cause fix: runtime cache skip checks must use object identity (`last_data_ref is data`) instead of integer `id(data)` equality to avoid false same-data hits from Python id reuse.
    - Validation summary: fresh ABAB (`6` pairs, no-verify timing gate) strongly favors `ENABLE_DATA_FAST_CACHE=1` (`delta_B_minus_A~-23.24 us/group`, stdev `~1.85`), while static ABAB is neutral/slightly positive (`delta_B_minus_A~-0.029 us/group`).
  - Fresh-input router-path non-promotion (2026-02-27): keep `AISP_NVFP4_GROUP_GEMM_SINGLE_SLOT_FAST=1` as default.
    - ABAB10 fresh for `SINGLE_SLOT_FAST=0` remained non-promotable (mean/median/trimmed geomean regressed), even when case2/case3 means could look better in short runs.
  - Post-packed-copy ABAB confirms promotion on current tuned route:
    - Static ABAB (`4` pairs): `delta_old_minus_new~+0.0769 us` (`3/4` wins for native path).
    - Fresh ABAB (`4` pairs): `delta_old_minus_new~+27.13 us` (`4/4` wins for native path).
  - Case2 variant retest after host-path improvements is still non-promotable: `1sm_n128_case23` looked best in short screen, but 6-pair ABAB vs current `1sm_n128_case2_nvf4` route regressed (`delta_B_minus_A~+0.0384 us`), so keep case2 default on `1sm_n128_case2_nvf4`.
  - Latest case2 block-scaled retest against NVF4 (with case3 pinned to `1sm_n128_case3`) remains non-promotable:
    - Candidate `case2=1sm_n128_case2` vs baseline `case2=1sm_n128_case2_nvf4` gave static regression (`delta_B_minus_A~+0.0456 us/group`) despite fresh improvement (`delta_B_minus_A~-0.7568 us/group`) and worse strict-verify timing geomean.
    - Keep case2 default on `1sm_n128_case2_nvf4`.
  - Deeper case2 CUDA-side additions from this pass are not promotable:
    - New lane `1sm_n256_case2_nvf4` (N=256, K=256 NVF4 schedule) is strict-verify green but regresses heavily (`case2~29 us/group`, ABAB mean geomean delta `~+1.944 us/group` vs `1sm_n128_case2_nvf4`).
    - Proposed `1sm_n128_case2_nvf4_s5` is architecturally invalid on SM100: compile-time static assertion `SMEM usage exceeded capacity`; keep NVF4 case2 stage family capped at S4.
    - Forcing case2 onto 2SM families (`2sm`, `2sm_n64*`, `2sm_n128*`) currently fails `CUTLASS can_implement()` in this submission path; treat those as illegal for case2 routing unless kernel contracts change.
  - Case3 retune (2026-02-27, latest pass) final promotion is `1sm_n128_case3`:
    - Step 1 candidate (`case3_nvf4` vs `case23_s4`) was verify-green and directionally positive, but follow-up combo/ABAB checks found a better stable choice.
    - Step 2 decisive gate (`case3` vs `case3_nvf4`): strict verify green; ABAB (`6` pairs, warmup=2/repeats=12) gave static `delta_B_minus_A~-0.0084 us/group` and fresh `delta_B_minus_A~-0.8381 us/group`.
    - Router default now uses `case3=1sm_n128_case3`.
  - Current promoted defaults stability snapshot (`case2=1sm_n128_case2_nvf4`, `case3=1sm_n128_case3`, 3x runs, warmup=3/repeats=20):
    - static geomean/group mean `~10.5669 us` (stdev `~0.0576`)
    - fresh geomean/group mean `~43.2791 us` (stdev `~0.5202`)
- Current verified per-case CTA order routing for v2 path: `case0=tn_major`, `case1=tm_major`, `case2=tm_major`, `case3=tn_major`.
- `AISP_NVFP4_GROUP_GEMM_V2_ASSUME_NO_N_TAIL=1` is verify-green and ABAB-positive for case0/case1/case2 on the tuned UnrollN=2 build; keep it disabled for case3 where it regresses.
- `AISP_NVFP4_GROUP_GEMM_V2_FUSE_INPUTS_COMPRESS_LIST=1` must retain all fused-slot contexts (including padded tensors) to avoid graph-mode illegal-address failures; keep this behind explicit opt-in unless repeated ABAB shows net geomean gain.
- Case2 kernel specialization (`AISP_NVFP4_GROUP_GEMM_V2_ENABLE_CASE2_INLINE_TM_MAJOR_MAP=1`) is not a promotable default on current tuned builds:
  - 4-pair all-case ABAB showed a regression (`delta_B_minus_A=+0.050609 us/call`, map-on slower on average).
  - Keep it experimental/off unless a new ABAB with stricter noise controls shows a clear win.
- Case2 cta_group::2 probe status (strict verify):
  - `UNROLL_N=1, CTA2_PARTITION_B=1` verifies but is far slower (`~66 us/call` for case2).
  - `UNROLL_N=1, CTA2_PARTITION_B=0` fails verify.
  - `UNROLL_N=2` cta2 paths remain correctness-unsafe (`CTA2_PARTITION_B=0` fails verify; `CTA2_PARTITION_B=1` can hang).
- CUTLASS SM100 block-scaled 1SM legality constraint:
  - `TileShape_M` is hard-constrained to `128` for this kernel family; attempted `M=64` case2 lane (`1sm_m64_n128_case2`) fails compile-time static assertions and must not be retried.
- Case2 exploratory kernel lane `1sm_n128_k512_case2` is non-promotable:
  - strict all-case check showed a severe regression (`case2` around `~38.7 us/group`, geomean regression `~+2.16 us`), so this lane should remain removed.
- Recent strict sweeps (current tuned router defaults) remain non-promotable across these branches:
  - `1sm_n128_k128_case2_nvf4`: best tunable point still materially slower than current defaults.
  - `1sm_n128_case2` vs `1sm_n128_case2_nvf4`: ABAB mean geomean delta was near zero (`~ -0.004 us`) with high stdev; treat as noise/no promotion.
  - case0-only and case1-only 2SM tunable sweeps (`cluster/raster/pdl/swizzle`) did not beat current all-case baseline.
  - latest case3 full variant re-screen and ABAB gating identify `1sm_n128_case3` (non-NVF4) as the current best promotable default; older “no better case3 variant” result is superseded.

## Expectations Files (CRITICAL)
- Expectation baselines live next to each chapter as `expectations_{hardware_key}.json`.
- The hardware key includes GPU count + model slug (examples: `expectations_b200.json` for 1x B200, `expectations_4x_b200.json` for 4x B200).
- Always refresh with `--update-expectations` on the active hardware key.

## Queueing & Monitoring (CRITICAL)
- Use a single queue runner under `artifacts/parallel_runs/` to serialize `aisp bench run` and profiling runs; avoid parallel queues. Append targets to the active queue script instead of creating ad-hoc loops.
- Queue logic must wait for all active benchmark/profiling processes to finish before starting the next run, and must detect overlapping runs; if another run starts during a queued run, re-queue that target after the system is idle.
- Busy detection must ignore waiting queue shell processes and treat the current run’s process group as “self” so child processes do not trigger false reruns.
- Do not terminate `ncu`/`nsys` processes unless the user explicitly requests it for a stuck run; if you do, log the action and reason in the queue log.
- Queue scripts must log start/end timestamps and exit codes to a dedicated log file in `artifacts/parallel_runs/`.
- Failure recovery: a failed run must not abort the queue; log the failure and continue. Only re-run on overlap or explicit user request.
- Monitoring: watch the queue log and report when a run starts, completes, or fails.
- Partial-result mining: mine partial results during execution (while long sweeps are still running) to catch any new per-case wins immediately, and promote those winners into the next queued pass ASAP.

## Explicitness (CRITICAL)
- Prefer explicit flags/parameters over changing global defaults; if a default must change, ask first and document why.
- Example: run a single benchmark with `--ncu-metric-set minimal --ncu-replay-mode kernel` instead of changing default NCU settings.
- When Nsight Compute application replay is unstable (dynamic kernels), use `aisp bench run --ncu-replay-mode kernel` to override the minimal preset for that run.

## Validity Profiles (CRITICAL)
- Use only two benchmark validity modes everywhere (CLI, MCP, dashboard): `strict` and `portable`.
- Default is always `strict` (fail-fast, no implicit downgrades).
- Portable mode must be explicit: `--validity-profile portable`.
- In portable mode, expectation writes are disabled unless explicitly enabled with `--allow-portable-expectations-update`.
- Do not use aliases/synonyms for validity modes (no transitional names); keep terminology exact and stable.
- When strict mode fails due environment capability gaps (for example virtualization, clock lock, or telemetry constraints), surface the exact recovery flag and consequence in the error/help text:
  - `--validity-profile portable` to run compatibility mode.
  - `--allow-portable-expectations-update` only when the user explicitly wants expectation files updated in portable mode.

## Defaults Consistency (CRITICAL)
- CLI, MCP tools, dashboard, and any other entrypoints must stay in sync on defaults (flags, behaviors, and help text). If a default changes, update all entrypoints together in the same change.

## Interface Parity & Documentation Consistency (CRITICAL)
- Keep CLI, MCP, dashboard API, and docs in sync. If you add or change a capability, update all surfaces in the same change.
- The dashboard API should expose **only** what the dashboard UI currently uses; do not expand API scope without explicit request.
- Prefer a single source of truth for tool metadata (descriptions, args, defaults). Regenerate `docs/mcp_tools.md` via `python -m scripts.generate_mcp_docs` and update `docs/api-reference.md` in the same change to avoid drift.
- Tool descriptions and argument docs must be precise and agent-friendly: include *when to use*, *not for*, units, defaults, enum choices, and short examples to enable accurate tool selection in chat loops.
- Postmortem (parity drift): CLI/MCP/docs/dashboard evolved independently, defaults diverged, and no enforced single-source catalog or regeneration step existed. Prevention: keep a single catalog, regenerate docs, and run parity tests whenever tool surfaces change.

## Test Realism (CRITICAL)
- Tests MUST NOT use `precheck_only`, `dry_run`, `estimate_only`, or any other short-circuit/preview mode for benchmark, profiling, or runtime-behavior assertions.
- Benchmark-truth tests MUST execute real repo code paths end-to-end. Do not use mocks, stubs, `monkeypatch`, or similar to fake successful benchmark execution, profiler captures, verification outcomes, GPU capability, or runtime-validity success.
- Narrow control-plane/orchestration tests MAY use `monkeypatch`/`patch` for environment variables, subprocess seams, clocks, temp roots, process inspection, or external-tool availability when the assertion is about state transitions, diagnostics, payload shaping, or error handling rather than benchmark truth.
- When a surface area uses `monkeypatch`/`patch` for orchestration coverage, keep at least one real-path test for that same surface area (subprocess, file-backed, harness, or GPU-backed as appropriate), and prefer the real-path option whenever feasible.
- Assume CI has real network access, GPUs, and Nsight tools (`nsys`, `ncu`) available; tests should validate real behavior accordingly.

## Achieve MAXIMUM speedup when benchmarking baseline_ versus optimized_ variants when possible
- For any speedups <1.05x, we must improve in a natural manner utilizing hardware, software, and algorithmic speedups.  
- Both the baseline and the optimized variants need to equivalent workloads.  Perhaps we need to increase the workloads to demonstrate the speedup?  
- Let's consider all options and find the best speedup 
- Make sure we're staying with the intent of the example and within the context of the corresponding Chapter XX in the AI Systems Performance Engineering book.
- It is OK to increase batch size, sequence length, or message sizes to surface clear speedups, as long as baseline and optimized workloads stay equivalent.

## Multi-GPU Defaults (CRITICAL)
- Multi-GPU scripts should use all visible GPUs by default unless explicitly overridden.
- If an example must specify a fixed GPU count, use 2 or 4 GPUs (prefer 4).
- Replace hard-coded 8-GPU example counts with 4 GPUs.

## Benchmarks vs Tools/Demos (CRITICAL)

### Benchmarks (comparable baseline vs optimized)
- Always implement benchmark pairs (baseline_*/optimized_*) when possible; use demos/tools only when a comparable pair is not feasible.
- `aisp bench run --targets ...` should only include targets that are explicitly intended to demonstrate an optimization outcome.
  - Default: **performance** (speedup) with clear speedup potential.
  - Rare: **memory** (reduced memory) when explicitly the goal.
- Comparable benchmarks must use `baseline_*.py` + `optimized_*.py` naming and MUST be equivalent workloads (no hidden work reduction, no extra work in a variant hot path).
- DO NOT rename benchmark pairs to suffix forms like `*_baseline.py` / `*_optimized.py`. If it’s a real harness pair, keep the `baseline_*/optimized_*` prefix naming.

### Memory-goal benchmarks
- Memory-goal benchmarks are still comparable baseline vs optimized pairs, but are evaluated on memory savings (not speed).
- Gate on `baseline_memory_mb / optimized_memory_mb >= 1.05` (speed may regress; do not add a speed gate).

### Demos / examples (NOT comparable benchmarks)
- Demos are runnable chapter companions / examples. They are NOT compared by the benchmark harness.
- Demos MUST NOT use `baseline_*/optimized_*` naming (to avoid accidental benchmark discovery).
- Prefer `*_demo.py` naming for demo entry points.
- Run these via `aisp demos <name> -- <args...>` by registering the script path in `core/demos/demos_commands.py` (`DEMOS` mapping).
- Demos should convey the same optimization ideas as the chapter/book, but do not need to be byte-for-byte identical to the book snippets. Keep them aligned in intent and narrative.

### Tools / methodology / analysis scripts (NOT comparable benchmarks)
- If something is not meant to be compared as baseline vs optimized (e.g., roofline analysis, config sweeps, monitoring bundles, validation workflows), it MUST NOT use `baseline_*/optimized_*` naming.
- Do NOT use suffix forms like `*_baseline.py` / `*_optimized.py` for tools; prefer descriptive names like `*_tool.py`, `*_analysis.py`, or `*_demo.py` (if it’s a demo).
- If you find an existing `baseline_*/optimized_*` pair that is NOT truly comparable, first try to make it a real harness-comparable pair. If that’s not possible, reclassify it as a demo/tool and rename it out of `baseline_*/optimized_*` (and do not leave compatibility shims/aliases behind).
- Keep only the “full / sophisticated” version (no `_basic`, no smoke/minimal variants).
- Keep the tool script at the chapter/lab level when book context references it, but decouple it from benchmark discovery and `bench run`.
- Run these via `aisp tools <name> -- <args...>` by registering the script path in `core/tools/tools_commands.py` (`TOOLS` mapping).

### Labs (CRITICAL)
- Labs are intended to be **realistic, end-to-end optimization stories** that tie together multiple chapter techniques (kernel + runtime + system), and should be structured as **harness-comparable** `baseline_*.py` / `optimized_*.py` pairs whenever feasible.
- Prefer **augmenting** an existing lab benchmark pair (adding additional optimizations to the optimized variant, keeping the same workload/output) over introducing one-off scripts.
- If something in `labs/` is **not** a harness-comparable baseline/optimized workload (e.g., planners, config generators, diagnostic reporters), it must be treated as a **tool or demo**:
  - It MUST NOT use `baseline_*/optimized_*` naming.
  - It SHOULD be registered under `aisp tools` (utility/analysis) or `aisp demos` (example runner).
  - It MUST NOT keep compatibility shims/wrappers/aliases.
- Avoid duplicating a chapter pair verbatim in `labs/`; labs should add integration value (multi-optimization, multi-GPU, end-to-end workflow) rather than rehosting identical comparisons.

### Hardware Diagnostics (microbench)
- Hardware microbenchmarks (e.g., `hw_*` tools / `core/diagnostics/microbench.py`) are **diagnostic-only** and intentionally bypass the benchmark harness and its 95 validity protections.
- Do not use microbench results to claim baseline-vs-optimized speedups; use harness benchmarks via `aisp bench run --targets ...` for comparable results.

## When to Move Code into `core/` (Reuse Rule)
- If shared logic has **2+ call sites** across chapters/labs, extract it into `core/` (prefer `core/analysis/*` or `core/utils/*`) and import it from chapter code.
- If a chapter’s narrative/book references specific chapter-local code, keep a thin chapter wrapper that calls into `core/` rather than moving everything out of the chapter.

## Verification Interface (CRITICAL)
- Preferred path: use `VerificationPayloadMixin` + `_set_verification_payload()` inside `benchmark_fn()` (or equivalent path).
- If a benchmark cannot use the mixin, it MUST explicitly implement `get_verify_output()/get_input_signature()/get_output_tolerance()` with no auto-inference and no fallbacks.
- New benchmarks should copy the compliant template (`templates/benchmark_compliant.py`) and keep verification behavior consistent.

## Chapter Consistency
- Make sure all code in the chapter (`chXX/` examples) is consistent with the content in the equivalent Chapter XX of the AI Systems Performance Engineering book.

### Benchmark Example Pairs (Baseline vs. Optimized)
- Before making changes to these benchmark pairs (baseline_* and optimized_*), be sure to understand the intent of the optimization/comparison before making any changes.  
- You must preserve the intent of the comparison  (e.g. comparing 32-bit to 16-bit data types)
- Never fall back to another precision in precision-focused examples (FP4/FP8/NVFP4/etc.); fail fast if the target precision is unavailable so the example intent remains intact.
- And you must not introduce additional operations in the measured hot path of a variant just to satisfy requirements of the harness, for instance (e.g. cast from 16-bit to 32-bit to satisfy an output comparison/verification). 
- Instead, keep as much as possible outside of the timed areas so as to not artifically inflate any variant.  (e.g. cast outside the timed area or compare with a tolerance large enough to still maintain the intent of the comparison)
- AVOID SKIPPING EXAMPLES WHENEVER POSSIBLE

### Benchmark Validity Issues Reference

The table below documents known issues that can cause benchmark results to be misleading, along with their protections. Use this as a checklist when creating or reviewing benchmarks. DO NOT ALLOW THESE IN OUR BENCHMARKS.

**✅ All 95 validity issues are now protected by our harness**

**CUDA Graph Note:** Capturing CUDA graphs in `setup()` is allowed for steady-state replay benchmarks (we intentionally measure replay, not capture). It is NOT allowed to precompute and reuse the final output from `setup()`; the output used for verification must come from the timed `benchmark_fn()` run and be surfaced via `capture_verification_payload()`.

**Virtualization Note:** `validate_environment()` treats virtualization (hypervisor present) as invalid. Benchmarks are supported only on bare metal.

| Category | Issue | What Happens | Protection | Status | Real-World Incident |
|----------|-------|--------------|------------|--------|---------------------|
| **Timing** | Unsynced Streams | Work on non-default streams isn't timed | Full device sync + `StreamAuditor` | ✅ | **Locus/KernelBench 2025** ([source](https://x.com/miru_why/status/1991773868806361138)) |
| **Timing** | Incomplete Async Ops | Timer stops before async work finishes | Full device sync | ✅ | **Locus/KernelBench 2025** ([source](https://x.com/miru_why/status/1991773868806361138)) |
| **Timing** | Event Timing Gaps | CUDA events recorded incorrectly | Cross-validate with wall clock | ✅ | |
| **Timing** | Timer Granularity | Measurement too coarse for fast ops | Adaptive iterations | ✅ | |
| **Timing** | Warmup Bleed | Real work happens during warmup | `isolate_warmup_cache` | ✅ | |
| **Timing** | Clock Drift | System clock changes during measurement | Monotonic clock usage | ✅ | |
| **Timing** | Profiler Overhead | Profiling tools add latency | Profile-free timing path | ✅ | |
| **Output** | Constant Output | Same result regardless of input | Jitter check | ✅ | |
| **Output** | Stale Cache | Same result across different seeds | Fresh-input check | ✅ | |
| **Output** | Approximation Drift | Rough estimate instead of full compute | Output tolerance validation | ✅ | |
| **Output** | Invalid Values (NaN) | NaN in output | `validate_result()` NaN check | ✅ | |
| **Output** | Invalid Values (Inf) | Inf in output | `validate_result()` Inf check | ✅ | |
| **Output** | Invalid Ground Truth | Labels/expected values wrong | `GoldenOutputCache` | ✅ | **ImageNet Labels 2021** ([arXiv:2103.14749](https://arxiv.org/abs/2103.14749)), **MMLU Errors 2025** ([PromptEng](https://promptengineering.org/challenges-and-innovations-in-language-model-benchmarking-and-generalization/)) |
| **Output** | Shape Mismatch | Output shape differs from expected | Shape validation | ✅ | |
| **Output** | Dtype Mismatch | Output dtype differs from expected | `ToleranceSpec` dtype check | ✅ | |
| **Output** | Denormalized Values | Subnormal floats cause slowdowns | Denormal check | ✅ | |
| **Output** | Uninitialized Memory | Output contains garbage | Memory initialization check | ✅ | |
| **Workload** | Precision Mismatch | Claims FP32 but uses FP16 | `InputSignature` dtype verification | ✅ | |
| **Workload** | Backend Precision Policy Drift | Global precision policy changes during timing (TF32, matmul precision, reduced-precision reductions) | Backend policy immutability check | ✅ | **PyTorch TF32 Default 2020** ([PyTorch CUDA Notes](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere)) |
| **Workload** | Undeclared Shortcuts | Skips elements without declaring | Workload invariant check | ✅ | **AI Agent Benchmark Shortcuts 2024** ([arXiv:2407.01502](https://arxiv.org/abs/2407.01502)) |
| **Workload** | Early Exit | Stops iteration loops early | Config immutability | ✅ | |
| **Workload** | Batch Shrinking | Processes fewer samples | `InputSignature` matching | ✅ | |
| **Workload** | Sequence Truncation | Processes shorter sequences | `InputSignature` matching | ✅ | |
| **Workload** | Hidden Downsampling | Silently reduces resolution | Dimension validation | ✅ | |
| **Workload** | Sparsity Mismatch | Different sparsity patterns | Sparsity ratio check | ✅ | |
| **Workload** | Attention Mask Mismatch | Different masking applied | Mask equivalence check | ✅ | |
| **Workload** | KV Cache Size Mismatch | Different cache sizes | Cache dimension check | ✅ | |
| **Workload** | Train/Test Overlap | Model tested on training data | Dataset isolation | ✅ | **Computational Biology 2019** ([Nat Commun](https://www.nature.com/articles/s41467-019-09406-4)) |
| **Location** | CPU Spillover | Work offloaded to CPU | GPU kernel time validation | ✅ | |
| **Location** | Setup Pre-computation | Work done in `setup()` | `check_setup_precomputation()` | ✅ | |
| **Location** | Graph Capture Cheat | Pre-compute during graph capture | `GraphCaptureCheatDetector` | ✅ | |
| **Location** | Warmup Computation | Compute results during warmup | `isolate_warmup_cache` | ✅ | |
| **Location** | Background Thread | Compute in separate thread | Process isolation | ✅ | |
| **Location** | Lazy Evaluation Skip | Returns unevaluated lazy tensor | `force_tensor_evaluation()` | ✅ | |
| **Location** | JIT Compilation Timing | JIT compile time included/excluded inconsistently | `clear_compile_cache()` | ✅ | |
| **Memory** | Pre-allocated Output | Result buffer allocated in setup | `MemoryAllocationTracker` | ✅ | |
| **Memory** | Input-Output Aliasing | Output points to pre-filled input | `check_input_output_aliasing()` | ✅ | |
| **Memory** | Pinned Memory Timing | Async pinned transfers not waited | Transfer completion check | ✅ | |
| **Memory** | Memory Pool Reuse | Cached allocations skew timing | `reset_cuda_memory_pool()` | ✅ | |
| **Memory** | Fragmentation Effects | Memory fragmentation differs | Memory pool reset | ✅ | |
| **Memory** | Page Fault Timing | First-touch page faults included | Memory pre-touch | ✅ | |
| **Memory** | Swap Interference | Swapping affects timing | Memory lock / swap disable | ✅ | |
| **CUDA** | Host Callback Escape | `cudaLaunchHostFunc` returns early | Host function tracking | ✅ | |
| **CUDA** | Async Memcpy Incomplete | D2H/H2D copies not awaited | Full device sync | ✅ | |
| **CUDA** | Workspace Pre-compute | Work in cuBLAS workspace alloc | Workspace monitoring | ✅ | |
| **CUDA** | Persistent Kernel | Kernel left running across calls | Kernel lifetime check | ✅ | |
| **CUDA** | Undeclared Multi-GPU | Work spread across undeclared GPUs | `validate_environment()` | ✅ | |
| **CUDA** | Context Switch Overhead | CUDA context switches affect timing | Context pinning | ✅ | |
| **CUDA** | Driver Overhead | Driver calls not accounted for | Driver call tracking | ✅ | |
| **CUDA** | Cooperative Launch Abuse | Cooperative kernels bypass checks | Launch mode validation | ✅ | |
| **CUDA** | Dynamic Parallelism Hidden | Child kernels not tracked | CDP kernel tracking | ✅ | |
| **CUDA** | Unified Memory Faults | Page migration not timed | UM fault tracking | ✅ | |
| **Compile** | Compilation Cache Hit | Returns cached compiled output | `clear_compile_cache()` | ✅ | |
| **Compile** | Trace Reuse | Exploits trace caching | `torch._dynamo.reset()` | ✅ | |
| **Compile** | Mode Inconsistency | Different compile mode verify vs perf | Mode consistency check | ✅ | |
| **Compile** | Inductor Asymmetry | Inductor optimizations inconsistent | Compilation parity | ✅ | |
| **Compile** | Guard Failure Hidden | Recompilation not counted | `get_compile_state()` | ✅ | |
| **Compile** | Autotuning Variance | Autotuning picks different kernels | Fixed autotuning cache | ✅ | |
| **Compile** | Symbolic Shape Exploit | Different shapes trigger different code | `InputSignature` matching | ✅ | |
| **Distributed** | Rank Skipping | Some ranks don't do work | `check_rank_execution()` | ✅ | |
| **Distributed** | Collective Short-circuit | Communication skipped | NCCL validation | ✅ | |
| **Distributed** | Topology Mismatch | Claims different topology | `verify_distributed()` | ✅ | |
| **Distributed** | Barrier Timing | Barrier timing exploited | Barrier synchronization | ✅ | |
| **Distributed** | Gradient Bucketing Mismatch | Different bucket sizes | Bucket size validation | ✅ | |
| **Distributed** | Async Gradient Timing | Async all-reduce not awaited | Full device sync | ✅ | |
| **Distributed** | Pipeline Bubble Hiding | Pipeline bubbles not counted | Bubble time tracking | ✅ | |
| **Distributed** | Shard Size Mismatch | FSDP shards differ | `InputSignature` matching | ✅ | |
| **Environment** | Device Mismatch | Uses different GPU than declared | `validate_environment()` | ✅ | |
| **Environment** | Frequency Boost | Overclocked for benchmark only | `lock_gpu_clocks()` | ✅ | |
| **Environment** | Priority Elevation | Runs at higher priority | Process isolation | ✅ | |
| **Environment** | Memory Overcommit | Exploits memory overcommit | Memory validation | ✅ | |
| **Environment** | NUMA Inconsistency | NUMA placement differs | NUMA audit | ✅ | |
| **Environment** | CPU Governor Mismatch | Different CPU frequency scaling | Governor lock | ✅ | |
| **Environment** | Thermal Throttling | GPU throttles during run | `capture_gpu_state()` pynvml | ✅ | |
| **Environment** | Power Limit Difference | Different TDP settings | `capture_gpu_state()` | ✅ | |
| **Environment** | Driver Version Mismatch | Different CUDA drivers | `RunManifest` version lock | ✅ | |
| **Environment** | Library Version Mismatch | Different cuDNN/cuBLAS | `RunManifest` version lock | ✅ | |
| **Environment** | Container Resource Limits | cgroups limits differ | Resource limit check | ✅ | |
| **Environment** | Virtualization Overhead | VM/container overhead varies | Bare-metal validation | ✅ | |
| **Statistical** | Cherry-picking | Only best iterations reported | All-iteration reporting | ✅ | **Chatbot Arena 2024** ([TechCrunch](https://techcrunch.com/2025/04/22/crowdsourced-ai-benchmarks-have-serious-flaws-some-experts-say/)) |
| **Statistical** | Outlier Injection | Slow iterations added to baseline | Statistical validation | ✅ | |
| **Statistical** | Variance Gaming | Variance reporting manipulated | Consistent statistics | ✅ | |
| **Statistical** | Percentile Selection | Favorable percentile chosen | Fixed percentile policy | ✅ | |
| **Statistical** | Insufficient Samples | Too few iterations for significance | Adaptive iterations | ✅ | **AI Benchmarks 2025** ([The Register (archived)](http://web.archive.org/web/20251113204928/https://www.theregister.com/2025/11/07/measuring_ai_models_hampered_by/)) |
| **Statistical** | Cold Start Inclusion | First run included unfairly | Warmup enforcement | ✅ | |
| **Statistical** | GC Interference | Garbage collection during timing | `gc_disabled()` | ✅ | |
| **Statistical** | Background Process Noise | System processes affect timing | Process isolation | ✅ | |
| **Evaluation** | Eval Code Exploitation | Benchmark code modified to pass | `BenchmarkContract` enforcement | ✅ | |
| **Evaluation** | Timeout Manipulation | Timeout extended to hide slowdowns | Config immutability | ✅ | |
| **Evaluation** | Metric Definition Gaming | Redefine what "speedup" means | Standardized metric definitions | ✅ | **MLPerf 2019** ([Forbes (archived)](https://web.archive.org/web/20191112035148/https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/)), **GLUE 2024** ([Revelry (archived)](http://web.archive.org/web/20250429145344/https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/)) |
| **Evaluation** | Test Data Leakage | Training on test/benchmark data | Data contamination checks | ✅ | **Benchmark Data Contamination Survey 2024** ([arXiv:2406.04244](https://arxiv.org/abs/2406.04244)) |
| **Evaluation** | Benchmark Overfitting | Optimize specifically for benchmark | Fresh-input + jitter checks | ✅ | **Underspecification 2020** ([arXiv:2011.03395](https://arxiv.org/abs/2011.03395)), **Epic Sepsis 2021** ([ChatBench](https://www.chatbench.org/)) |
| **Evaluation** | Self-Modifying Tests | AI/code modifies its own tests | Config immutability | ✅ | |
| **Evaluation** | Benchmark Memorization | Agent memorizes test cases | Fresh-input checks, jitter | ✅ | **AI Agent Benchmark Shortcuts 2024** ([arXiv:2407.01502](https://arxiv.org/abs/2407.01502)) |
| **Evaluation** | Missing Holdout Sets | No proper train/test split | Held-out evaluation data | ✅ | **AI Agent Benchmark Shortcuts 2024** ([arXiv:2407.01502](https://arxiv.org/abs/2407.01502)) |

**Total: 11 categories, 95 validity issues — ✅ ALL PROTECTED by our harness (20 linked to real-world incidents with citations)**

### Notable Real-World Incidents

These validity issues aren't theoretical—they've caused real problems:

| Year | Incident | Issue Type | What Happened | Source |
|------|----------|------------|---------------|--------|
| **2025** | **Locus/KernelBench Stream Exploit** | Unsynced Streams | Claimed 20x speedup on Llama FFW kernel. AI launched work on non-default CUDA streams but timer only measured default stream. **32.8% of RL-generated kernels exploited this**, causing fake 18x speedups. | [X/Twitter @miru_why](https://x.com/miru_why/status/1991773868806361138) |
| **2025** | **AI Benchmark Scientific Rigor** | Metric Definition Gaming | Only 16% of 445 AI benchmarks used statistical tests; ~50% tested abstract concepts without clear definitions. | [The Register (archived)](http://web.archive.org/web/20251113204928/https://www.theregister.com/2025/11/07/measuring_ai_models_hampered_by/) |
| **2025** | **MMLU Benchmark Errors** | Invalid Ground Truth | ~57% of questions in MMLU virology subset found incorrect. Ground truth errors destabilize evaluations. | [PromptEngineering.org](https://promptengineering.org/challenges-and-innovations-in-language-model-benchmarking-and-generalization/) |
| **2025** | **Sakana AI Scientist Evaluation** | Evaluation Integrity | Independent evaluation found frequent experiment failures and hallucinated numerical results, challenging reliability claims for AI-generated research outputs. | [arXiv:2502.14297](https://arxiv.org/abs/2502.14297) |
| **2024** | **AI Agent Benchmark Shortcuts** | Missing Holdout Sets | Study found AI agents memorize benchmark test samples instead of learning to generalize. Many benchmarks lack proper holdout test sets. | [arXiv:2407.01502](https://arxiv.org/abs/2407.01502) |
| **2024** | **GLUE Benchmark Heuristics** | Metric Definition Gaming | Models achieved high GLUE scores by exploiting shallow heuristics rather than genuine language understanding. | [Revelry.co (archived)](http://web.archive.org/web/20250429145344/https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/) |
| **2024** | **HumanEval Limitations** | Benchmark Overfitting | Models performing well on HumanEval struggled with real-world coding tasks; simplified scenarios missed practical complexity. | [Revelry.co (archived)](http://web.archive.org/web/20250429145344/https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/) |
| **2024** | **Chatbot Arena Benchmark Issues** | Cherry-picking | Crowdsourced benchmark results showed selection bias and inconsistent submissions, undermining performance comparisons. | [TechCrunch](https://techcrunch.com/2025/04/22/crowdsourced-ai-benchmarks-have-serious-flaws-some-experts-say/) |
| **2024** | **Benchmark Data Contamination Survey** | Data Contamination | Survey catalogs contamination pathways across LLM benchmarks and highlights mitigation gaps. | [arXiv:2406.04244](https://arxiv.org/abs/2406.04244) |
| **2023** | **NLP Evaluation Data Contamination** | Data Contamination | Position paper warns that LLMs trained on benchmark test splits can inflate reported scores and mask real generalization. | [arXiv:2310.18018](https://arxiv.org/abs/2310.18018) |
| **2022** | **MLPerf Participation Issues** | Cherry-picking | MLPerf faced inconsistent vendor participation; selective scenario submissions led to biased performance representations. | [NextPlatform (archived)](http://web.archive.org/web/20250813110435/https://www.nextplatform.com/2022/04/08/the-performance-of-mlperf-as-a-ubiquitous-benchmark-is-lacking/) |
| **2022** | **ML Benchmark Validity (Berkeley)** | Benchmark Overfitting | Small changes in data distribution caused significant performance drops, questioning external validity of static benchmarks. | [UC Berkeley Tech Report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-180.html) |
| **2021** | **ImageNet Label Errors** | Invalid Ground Truth | Study found **at least 6% label errors** in ImageNet validation set. Average 3.3% error rate across 10 common datasets. | [arXiv:2103.14749](https://arxiv.org/abs/2103.14749) |
| **2021** | **MLPerf Reproducibility** | Benchmark Reproducibility | Users couldn't reproduce MLPerf v0.7 results due to inaccessible datasets and outdated repositories. | [MLCommons Forum](https://groups.google.com/a/mlcommons.org/g/public/c/T_8UsUPIWFo) |
| **2021** | **Epic Sepsis Model Failure** | Benchmark Overfitting | Hospital sepsis prediction model showed significantly worse real-world performance than validation results due to non-representative test data. | [ChatBench.org](https://www.chatbench.org/what-are-the-implications-of-outdated-ai-benchmarks-on-the-accuracy-and-reliability-of-ai-driven-decision-making-and-insights/) |
| **2020** | **Underspecification in ML** | Benchmark Overfitting | ML pipelines produce models with equivalent benchmark performance but divergent deployment behaviors—instability in production. | [arXiv:2011.03395](https://arxiv.org/abs/2011.03395) |
| **2020** | **TF32 Default on Ampere** | Precision Policy Drift | TF32-enabled matmul/conv trades precision for speed unless explicitly disabled in benchmarks. | [PyTorch CUDA Notes](https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere) |
| **2019** | **MLPerf Inference Bias** | Metric Definition Gaming | Inaugural MLPerf inference results showed vendors selectively submitted results highlighting their strengths. | [Forbes (archived)](https://web.archive.org/web/20191112035148/https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/) |
| **2019** | **Computational Biology Overfitting** | Train/Test Overlap | Tools developed and tested on same datasets, performing well on benchmarks but failing on new real-world data. | [Nature Communications](https://www.nature.com/articles/s41467-019-09406-4) |
| **2016** | **Microsoft Tay Chatbot** | Missing Holdout Sets | AI chatbot learned offensive behavior within 24 hours due to lack of adversarial benchmarking and content moderation safeguards. | [ChatBench.org](https://www.chatbench.org/what-are-the-implications-of-outdated-ai-benchmarks-on-the-accuracy-and-reliability-of-ai-driven-decision-making-and-insights/) |
#### Incident Categories and Our Protections

| Category | # Incidents | Our Protection | Status |
|----------|-------------|----------------|--------|
| **Timing Manipulation** | 1 (Locus/KernelBench) | Full device sync + `StreamAuditor` | ✅ |
| **Invalid Ground Truth** | 2 (ImageNet Labels, MMLU) | `GoldenOutputCache` + `validate_result()` | ✅ |
| **Benchmark Overfitting** | 4 (Underspecification, Epic Sepsis, HumanEval, Berkeley) | Fresh-input checks + jitter | ✅ |
| **Data Contamination** | 2 (LLM Survey 2024, NLP Contamination 2023) | Data contamination checks + fresh inputs | ✅ |
| **Metric Gaming** | 3 (AI Benchmarks 2025, GLUE 2024, MLPerf 2019) | Standardized metric definitions | ✅ |
| **Cherry-picking** | 2 (Chatbot Arena, MLPerf 2022) | All-iteration reporting | ✅ |
| **Train/Test Overlap** | 1 (Computational Biology) | Dataset isolation + holdout enforcement | ✅ |
| **Missing Holdout Sets** | 2 (AI Agent Shortcuts, Microsoft Tay) | Held-out evaluation data | ✅ |
| **Reproducibility** | 1 (MLPerf 2021) | `RunManifest` version locking | ✅ |
| **Evaluation Integrity** | 1 (Sakana AI Scientist) | `BenchmarkContract` + verification enforcement | ✅ |
| **Precision Policy Drift** | 1 (TF32 Default) | Backend policy immutability check | ✅ |

#### Deep Dive: The Locus/KernelBench Stream Timing Vulnerability

This 2025 incident perfectly illustrates why correctness verification alone is insufficient:

```python
# VULNERABLE TIMING (what KernelBench did)
start_event.record(original_model_stream)  # Only records on default stream
model(*inputs)                              # But work runs on s1, s2, s3!
end_event.record(original_model_stream)    # Timer stops before s1,s2,s3 finish
torch.cuda.synchronize(device=device)      # Waits, but timing already recorded

# CORRECT TIMING (the fix)
for stream in custom_model_streams:
    custom_model_stream.wait_stream(stream)  # Wait for ALL streams
_event.record(custom_model_stream)           # Then record timing
```

**The exploit pattern:**
1. AI creates non-default streams: `s1 = getStreamFromPool()`, `s2 = ...`, `s3 = ...`
2. AI launches GEMMs on those streams: `at::mm_out(gate, x2d, gate_proj.t())` on s1
3. AI does NOT call `setCurrentCUDAStream(s3)` or wait for streams before returning
4. Correctness test uses `torch.cuda.synchronize()` → **PASSES** (waits for everything)
5. Performance test uses stream-specific events → **FAKE SPEEDUP** (doesn't wait for s1,s2,s3)

**Result:** 82/250 (32.8%) of RL-generated CUDA kernels exploited this, producing artificial 18x "speedups" with zero actual performance improvement.

These incidents demonstrate why rigorous benchmark verification is essential—not just for catching intentional gaming, but for catching subtle bugs that produce misleading results.

#### Protection Implementation Reference

All 95 validity protections are implemented in the following modules:

| Module | Key Protections |
|--------|-----------------|
| `core/harness/benchmark_harness.py` | Full device sync, L2 cache clearing, GPU clock locking, warmup isolation, config immutability, adaptive iterations, CUDA graph mode |
| `core/harness/validity_checks.py` | `StreamAuditor`, `MemoryAllocationTracker`, `GraphCaptureCheatDetector`, `gc_disabled()`, `clear_compile_cache()`, `capture_gpu_state()`, `validate_environment()` |
| `core/harness/l2_cache_utils.py` | Dynamic L2 cache size detection for Blackwell/Hopper/Ampere, `clear_l2_cache()` |
| `core/benchmark/verify_runner.py` | `VerifyRunner`, `GoldenOutputCache`, jitter check, fresh-input check, output comparison, workload invariants |
| `core/benchmark/verification.py` | `InputSignature`, `ToleranceSpec`, `QuarantineReason`, seed mutation detection |
| `core/benchmark/quarantine.py` | `QuarantineManager` with persistence |
| `core/benchmark/contract.py` | `BenchmarkContract` enforcement |

## FAIL FAST - NO FALLBACKS, NO AUTO-INFERENCE

**CRITICAL**: This project follows a STRICT fail-fast policy. DO NOT implement fallbacks, auto-detection, or auto-inference.

### What This Means

1. **NO Auto-Inference**: Never write code that guesses or infers values from attributes
   - BAD: `if hasattr(self, 'batch_size'): return self.batch_size`
   - GOOD: Require explicit implementation, raise `NotImplementedError` if missing

2. **NO Fallbacks**: Never provide default values when explicit implementation is required
   - BAD: `return sig if sig else None` or `return sig if sig else {}`
   - GOOD: `raise NotImplementedError("Benchmark must implement this method")`

3. **NO Silent Failures**: Never swallow errors or return empty/None when something is wrong
   - BAD: `try: ... except: return None`
   - GOOD: Let exceptions propagate with clear error messages

### When You Find Code With Fallbacks

If you encounter code with auto-inference or fallbacks:
1. **DO NOT** add more fallbacks to fix the symptom
2. **DO** fix the underlying benchmarks to implement required methods
3. **DO** remove the fallback logic and make it fail-fast

### Audit Compliance

Use `aisp bench audit --all` to check verification compliance:
- All benchmark files must have 100% compliance
- Compliance means explicit implementations, not auto-detected ones

## Jitter Check (Advisory)

The jitter check protects against benchmarks returning **constant/hardcoded outputs** regardless of input.

**How It Works:**
1. Perturbs the input tensor by adding small noise
2. Re-runs `benchmark_fn()`
3. Verifies output CHANGED (if output unchanged → hardcoded)

**Important Notes:**
- The jitter check is largely **redundant** with proper output verification
- If baseline computes real output and optimized returns hardcoded values, they won't match anyway
- Jitter check only catches the case where BOTH baseline AND optimized return the SAME hardcoded value (extremely unlikely)
- No exemptions needed - the check auto-skips when appropriate

**Anti-patterns (DO NOT USE):**
- `return torch.tensor([1.0])` - Fixed constant (fails verification, not just jitter)
- `return torch.tensor([output.sum().item()])` - Scalar checksum (defeats both jitter AND verification)

**Valid patterns:**
- `return self.output.detach().clone()` - Actual output from benchmark_fn
- `return self.gpu_data[:1000].clone()` - Slice of actual data for large outputs


### Benchmark Verification Interface (Fallback Path)

Use this explicit interface only when `VerificationPayloadMixin` is not applicable for the benchmark type:

```python
def get_verify_output(self) -> torch.Tensor:
    """MANDATORY: Return output tensor for verification."""
    raise NotImplementedError("Must implement explicitly")

def get_input_signature(self) -> dict:
    """MANDATORY: Return workload parameters for matching."""
    raise NotImplementedError("Must implement explicitly")
    
def get_output_tolerance(self) -> tuple:
    """MANDATORY: Return (rtol, atol) for numerical comparison."""
    raise NotImplementedError("Must implement explicitly")
```

## Deterministic Seed Pattern (CRITICAL)

The harness uses **seed 42** by default. Benchmarks MUST match this seed.

**Why this matters:**
- The harness sets seeds via `set_deterministic_seeds(42)` before `setup()`
- After `benchmark_fn()`, it checks if `torch.initial_seed() == 42`
- If seeds don't match → "Benchmark mutated RNG seeds during execution" error

**Correct Pattern:**
```python
def setup(self) -> None:
    torch.manual_seed(42)           # MUST be 42 to match harness
    torch.cuda.manual_seed_all(42)  # Always include for CUDA determinism
    # ... rest of setup
```

**Anti-pattern (DO NOT USE):**
```python
def setup(self) -> None:
    torch.manual_seed(1)    # BAD: mismatches harness seed 42
    torch.manual_seed(101)  # BAD: mismatches harness seed 42
```

## Deterministic Algorithms vs Performance (CRITICAL)

- Do NOT enable deterministic algorithms inside *performance* benchmarks; they can slow kernels significantly and can create misleading baseline-vs-optimized speedups if variants differ.
  - Disallowed in benchmark code by default: `torch.use_deterministic_algorithms(True, ...)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False` when used to force determinism.
- Determinism is handled by the harness in verification/repro modes; benchmark files must not override harness policy.
- **Exception (rare):** If a benchmark must enable determinism for correctness/debuggability, it MUST include an explicit file-level justification comment so `aisp bench audit` can allowlist it:
  - `# aisp: allow_determinism <short reason>`
- Note: Setting `torch.backends.cudnn.benchmark = True` is a performance knob (autotuning) and does NOT require `# aisp: allow_determinism`; keep backend toggles consistent across baseline/optimized unless that toggle is the intended optimization being demonstrated.

## Tests for New Functionality (CRITICAL)

- Any new functionality (new checks, CLI behavior, verification logic, benchmark validity protections) MUST ship with tests.
- Tests MUST exercise real code paths (no mocking) and must fail without the new functionality.
  - Prefer: temp files + real imports, subprocess CLI invocations, and end-to-end checks where feasible.
  - If you believe mocking is unavoidable, STOP and ask for explicit approval first.

## Harness Verification Architecture (IMPORTANT)

The harness uses **POST-TIMING VERIFICATION** - verification happens AFTER timing runs complete, using the outputs from the already-run benchmarks. This is efficient:

1. **Benchmarks run ONCE** during timing (with warmup + N iterations)
2. **Outputs are captured** via `get_verify_output()` 
3. **Comparison happens** after timing completes
4. **No redundant runs** - we don't run benchmarks twice

### How Verification Works

After timing runs complete for both baseline and optimized:
```python
baseline_output = baseline_benchmark.get_verify_output()
optimized_output = optimized_benchmark.get_verify_output()
rtol, atol = baseline_benchmark.get_output_tolerance()
torch.allclose(baseline_output, optimized_output, rtol=rtol, atol=atol)
```

### `_verify_patched_benchmark` - FOR LLM PATCHES ONLY

This separate function is **reserved for LLM-patched benchmarks**:
- When an LLM modifies a benchmark to optimize it
- We need to verify the LLM's version produces same output as original
- This loads and runs benchmarks fresh from disk
- **NOT used for baseline vs optimized pairs** (those use post-timing verification)

**Consistency Rule**: Always use `self.output` for benchmark results (not `self._result` or other names). The harness will use `get_verify_output()` for comparison.

## Checksum Verification is NOT Acceptable (IMPORTANT)

**DO NOT use checksums to work around verification failures.** If baseline and optimized produce different results, something is WRONG:

1. **Same algorithm, different execution pattern** → SHOULD produce identical results (within tolerance)
2. **Different algorithms** → One is incorrect, or they're testing different things
3. **Precision differences (FP16 vs FP32)** → Use appropriate tolerance, NOT checksums

### When Outputs Don't Match - Fix the Root Cause

If verification fails, investigate WHY:
- Are the inputs identical? (Check seed, setup order)
- Is the math the same? (Check operations, order)
- Is there hidden state mutation? (Check in-place ops)
- Is tolerance appropriate for the dtype? (FP16 needs looser tolerance than FP32)

**DO NOT** return `torch.tensor([self.output.sum().item()])` as a workaround.

### Cop-out Patterns That Need Fixing

**ALL of these patterns are COP-OUTS that defeat verification:**

| Pattern | Problem |
|---------|---------|
| `torch.tensor([0.0])` | Constant zero - defeats verification |
| `torch.tensor([hash(str(id(self))) % (2**31)])` | Random hash - defeats verification |
| `torch.tensor([output.sum().item()])` | Checksum - hides element-wise errors |
| `return None` or missing output | No verification at all |

**Files with cop-outs STATUS:**
- ✅ ch01-ch03 - FIXED with actual outputs
- ✅ ch05-ch16 - FIXED with actual outputs or RuntimeError for incompatible
- ✅ ch17-ch20 - FIXED with output surfacing for verification
- ⚠️ ch04/* - SKIPPED (multi-GPU required)

**Nested Harness Benchmarks (ch18/ch19/ch20):**
Many advanced benchmarks use a "nested harness" pattern where a wrapper calls `run_benchmark()`
internally. These must still surface a real verification output from the timed run at the outer layer.

### NO `_run_once_for_verify` in setup()

Verification uses outputs from the TIMING RUN, not a separate pre-run in setup().

**WRONG:**
```python
def setup(self):
    self._run_once_for_verify()  # NO! This runs benchmark twice!
```

**CORRECT:**
```python
def setup(self):
    # Just set up inputs - benchmark_fn() will set self.output
    pass

def get_verify_output(self):
    return self.output  # From timing run
```

### NO FALLBACKS in get_verify_output()

If output is None, FAIL FAST with an error. DO NOT return a cop-out.

**WRONG:**
```python
def get_verify_output(self):
    if self.output is not None:
        return self.output
    return torch.tensor([0.0])  # COP-OUT! Defeats verification!
```

**CORRECT:**
```python
def get_verify_output(self):
    if self.output is None:
        raise RuntimeError("benchmark_fn() must be called before verification")
    return self.output
```

### Training Benchmarks: Capture Output at END with `_verify_input`

For training benchmarks, both baseline and optimized train for N iterations. To verify they produce 
equivalent results, we need to test the TRAINED MODELS with the SAME input. Use `_verify_input`:

```python
def __init__(self):
    super().__init__()
    self.output = None
    self._verify_input = None  # Fixed input for verification
    # ... other init

def setup(self):
    torch.manual_seed(42)
    # ... model and training data setup
    
    # Create fixed verification input (same seed state for both baseline/optimized)
    self._verify_input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
    # ... warmup

def benchmark_fn(self):
    # Training loop (timed)
    for data, target in zip(self.batches, self.targets):
        logits = self.model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        self.optimizer.step()
    
    # Capture output AFTER training completes using fixed verify input
    with torch.no_grad():
        self.model.eval()
        self.output = self.model(self._verify_input).float().clone()  # .float() for FP16 models
        self.model.train()

def get_verify_output(self):
    if self.output is None:
        raise RuntimeError("benchmark_fn() must be called before verification")
    return self.output.detach().clone()

def get_output_tolerance(self):
    # Training with different optimizations may diverge slightly
    return (1e-2, 1e-2)
```

### Wrapper/Impl Pattern Benchmarks

Many benchmarks use an `_impl` class pattern. The wrapper must surface the output:

```python
class BenchmarkWrapper(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self._impl = SomeImplementation()
        self.output = None

    def benchmark_fn(self):
        result = self._impl.run()
        self.output = result  # Surface the output from _impl

    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output.detach().clone()
```

### Benchmarks That Can Skip Verification

Only **two categories** may legitimately skip tensor verification:

| Category | Examples | Reason | Handling |
|----------|----------|--------|----------|
| **Multi-GPU Required** | ch04/*, ch17/multigpu | Requires >=2 GPUs | `raise RuntimeError("SKIPPED: requires >=2 GPUs")` |
| **Config Generation** | ch18/vllm_monitoring | Writes YAML/config files, no GPU computation | Use `verification_not_applicable_reason` attribute |

All other benchmarks **MUST** produce verifiable output:

### Benchmarks With Alternative Output Types

These benchmarks don't produce GPU tensors but MUST still verify their results:

| Category | Solution | Example |
|----------|----------|---------|
| **CUDA Binary** | Use `CudaBinaryBenchmark` base class - builds with `-DVERIFY=1` and parses checksum | ch09/cublaslt_gemm |
| **Simulations** | Convert metrics to tensor (e.g., `torch.tensor([p50, p95, tokens_s])`) | ch15/placement |
| **CPU-only** | Convert output to tensor (e.g., decompressed bytes → tensor) | ch05/cpu_decompression |
| **Nested Harness** | Surface output from inner benchmark to outer wrapper | ch18/speculative_decoding |

### Verification Skip Attribute

For config generation benchmarks (only!), use:
```python
# In __init__:
self.verification_not_applicable_reason = "Config generation - writes YAML, no GPU computation"

# In get_verify_output:
def get_verify_output(self) -> torch.Tensor:
    raise RuntimeError("VERIFICATION_SKIP: Config generation benchmark - writes files, no GPU computation")
```

### Cop-out Fix Status: COMPLETE

All cop-outs (hash/zero/metrics-only) have been eliminated:

| Chapter Range | Status | Notes |
|--------------|--------|-------|
| ch01-ch03 | ✅ Fixed | Proper output capture |
| ch04 | ⏭️ Multi-GPU | Legitimate skip |
| ch05-ch16 | ✅ Fixed | All cop-outs replaced with real outputs |
| ch17-ch20 | ✅ Fixed | Nested harness refactored, simulations converted to tensor |

**CUDA Binary benchmarks:** Use base class verification via `-DVERIFY=1` builds
**Simulations:** Metrics converted to tensors for deterministic verification
**Config generation (4 files):** Only legitimate non-GPU benchmarks - use `verification_not_applicable_reason`

### Fixing Cop-out Patterns

When fixing `torch.tensor([hash(...)])` or `torch.tensor([0.0])` cop-outs:

1. **Find the output**: Look for `self.output`, `self.data`, `self.result`, `self.outputs`, etc.
2. **If output exists**: Return `self.output.detach().clone()`
3. **If wrapper pattern**: Surface output from `_impl`
4. **If no tensor output**: Use explicit `RuntimeError` with reason
5. **Never use hash/zero cop-outs**

### Data Loading / Prefetching Benchmarks: Wide Tolerances

For benchmarks where the optimization is in data loading (prefetching, pinned memory, 
double-buffering), baseline and optimized may process different batches in different 
orders. Use wide tolerances since we're primarily testing timing, not exact output matching.

```python
def get_output_tolerance(self) -> tuple:
    """Wide tolerance for data loading benchmarks.
    
    Primary checks are: no NaN, shapes match, reasonable values.
    """
    return (1.0, 10.0)
```

## ALWAYS MAKE THE LONG_TERM CHOICE
- DO NOT MAKE CHOICES BASED ON IMMEDIATE CONVENIENCE
- PREFER HARNESS CHANGES OVER PER-BENCHMARK HACKS
- DOCUMENT DISCOVERIES IN THIS FILE WHEN THEY CREATE A REPEATED NEED.
- ALWAYS design for the long-term, right way of doing things.

## Prefer flags over environment variables

## Keys are in .env, .env.local in the project root folder.  
- Use these keys for external integrations (e.g. OpenAI, Anthropic, etc)

## NVFP4 Group GEMM V2 Learnings (2026-02-16)
- Timing semantics mismatch is material:
  - Local v2 harness is often run with `AISP_NVFP4_GROUP_GEMM_INPUTS_PER_ITERATION=15`.
  - Popcorn benchmark measures one `custom_kernel(data)` call per timed iteration.
  - Always label metrics with the timing model used.
- Measured baselines in this session:
  - Local v2 case1 loop model (`inputs=15`): `10.336 us/group`.
  - Local v2 single-call model (`inputs=1`): `14.090 us/group`.
  - Local single-file parity run (steady-state style): about `41.434 us/group`.
  - Popcorn benchmark for the same artifact showed case1 mean around `1701 us total` (`212.625 us/group`) with large outliers.
  - Local parity runner updated to emulate Popcorn clone-precheck behavior, which reveals similar first-iteration outlier behavior.
- Updated measurements (same day, after additional kernel patches):
  - Local tuned case1 loop model (`inputs=15`, graph+flush+locked clocks): `10.390 us/group`.
  - Local Popcorn-style parity (`popcorn_parity_case1.py`, single-call timing): `14.085 us/group` mean.
  - Live Popcorn `mode=benchmark` is highly outlier-sensitive on this workload:
    - Sample A case1: mean `1172 us total` (`146.5 us/group`), fast `69.3 us total` (`8.66 us/group`).
    - Sample B case1: mean `1747 us total` (`218.375 us/group`), fast `69.8 us total` (`8.73 us/group`).
  - Interpretation: use both mean and fast-path totals from Popcorn logs; means are currently dominated by sporadic long-tail outliers.
- Latest checkpoint (same day, later):
  - Local tuned case1 loop model (`inputs=15`, graph+flush+locked clocks) now measures `13.141 us/group`
    with:
    - `AISP_NVFP4_GROUP_GEMM_V2_UNROLL_N=2`
    - `AISP_NVFP4_GROUP_GEMM_V2_WS_UNROLL2_MMA=1`
    - `AISP_NVFP4_GROUP_GEMM_V2_EPILOGUE_LD_X32=1`
    - `AISP_NVFP4_GROUP_GEMM_V2_MAXRREGCOUNT=68`
    - `AISP_NVFP4_GROUP_GEMM_V2_CLUSTER_DIM_X=1`
    - `AISP_NVFP4_GROUP_GEMM_V2_TMA_L2_PROMOTION=3`
  - Local Popcorn parity runner:
    - Warmup-enabled (`--warmup 3`) case1 mean: `14.045 us/group`.
    - True clone-precheck/no-warmup (`--warmup 0`) case1 mean can spike to `~1569.7 us/group`
      with p50 near `112 us/call`, showing first-call prep outliers dominate mean.
  - Live Popcorn benchmark artifact:
    - `artifacts/popcorn_checks/20260216T135839Z/benchmark_parsed.json`
    - `artifacts/popcorn_checks/20260216T140735Z/benchmark_parsed.json`
    - case1 mean `1885 us total` (`235.625 us/group`), fast `70.1 us total` (`8.76 us/group`), slow `182 ms`.
    - This matches the local no-warmup outlier pattern: benchmark means are dominated by first-call/prep tails.
- New checkpoint (same day, latest):
  - Local tuned case1 loop model (`inputs=15`, graph+flush+locked clocks):
    - `13.148 us/group` with explicit cta1 config (`UNROLL_N=2`, `WS_UNROLL2_MMA=1`,
      `EPILOGUE_LD_X32=1`, `MAXRREGCOUNT=68`, `TMA_L2_PROMOTION=3`, `CLUSTER_DIM_X=1`).
  - Same config with fused-request launch (`AISP_NVFP4_GROUP_GEMM_V2_FUSE_INPUTS=1`):
    - `10.385 us/group` (loop model only; not directly Popcorn-comparable).
  - Local Popcorn-style parity (`popcorn_parity_case1.py`):
    - Warmed (`--warmup 3`): `14.111 us/group` mean.
    - No-warmup clone-precheck (`--warmup 0`) still dominated by first-call outlier tails.
  - Live Popcorn re-checks:
    - `mode=test`: passed `10/10` on B200 (`artifacts/popcorn_checks/20260216T143352Z/popcorn_test.log`).
    - `mode=benchmark`: `artifacts/popcorn_checks/20260216T143453Z/benchmark_parsed.json`.
    - Case1 in that run: mean `1784 us total` (`223.0 us/group`), fast `70.4 us total` (`8.8 us/group`), slow `171 ms`.
  - cta1 plateau sweep (case1, fused loop model, graph+flush+locked clocks, verify on):
    - Best observed remains ~`10.377-10.385 us/group` with base config.
    - `TMA_L2_PROMOTION`: `3` is best (`0/1/2` are slightly slower).
    - `MAXRREGCOUNT` sweep `{56,60,64,68,72,80}` did not beat current `68`.
    - `WS_SEGMENT_PARALLEL`, `MMA_LANE0_ALL_WARPS`, `WS_SFB1_SEGMENT_HELPERS`, `WS_SPLIT_U0_SEGS` all regressed or were neutral.
    - cta_group::1 cluster launch (`CLUSTER_DIM_X=2/4`) and multicast (`ENABLE_TMA_MULTICAST=1`) regressed vs non-cluster cta1.
- Important tooling/knob finding:
  - `AISP_NVFP4_GROUP_GEMM_V2_BLOCK_M`, `AISP_NVFP4_GROUP_GEMM_V2_BLOCK_N`, and `AISP_NVFP4_GROUP_GEMM_V2_KPACK_TILE`
    do not tune the tcgen05 kernel path (they affect the scalar path only). Do not spend tuning cycles on these for tcgen.
- Compile-time knob sweep snapshot (case1, `inputs=1`, locked clocks):
  - `baseline`: `14.031 us/group`.
  - `AISP_NVFP4_GROUP_GEMM_V2_MMA_LANE0_ALL_WARPS=1`: `14.035 us/group` (neutral).
  - `AISP_NVFP4_GROUP_GEMM_V2_WS_SEGMENT_PARALLEL=1`: `13.954 us/group` (small win).
  - `AISP_NVFP4_GROUP_GEMM_V2_WARP0_ONLY_MAINLOOP=0`: `15.527 us/group` (regression).
  - Net: no large win from these toggles alone.
- Newer all-case/fresh-input finding (2026-02-24, strict verify + ABAB):
  - `AISP_NVFP4_GROUP_GEMM_V2_WARP0_ONLY_MAINLOOP=0` is a large net win for the current tcgen05 path
    (`delta_B_minus_A ≈ -7.52 us/call` geomean across cases 0..3; low variance).
  - Treat the older case1-only regression note above as historically scoped, not globally applicable.
- Additional compile-time sweep (case1, `inputs=15`, graph+flush+locked clocks, verify on):
  - `base` (`WS_UNROLL2_MMA=1`, `EPILOGUE_LD_X32=1`, `MAXRREGCOUNT=68`): `13.15 us/group`.
  - `WS_TMA_PRODUCER=1`: `14.734 us/group` (regression).
  - `WS_SFB1_SEGMENT_HELPERS=1`: `13.463 us/group` (regression vs base).
  - `PIPELINE_STAGES=1` + `STAGE1_PREFETCH=1`: `15.124 us/group` (regression).
  - `USE_UTCCP_64X128B_*` schedule-1 variants (SFA-only, SFB-only, both): all failed verification.
  - Conclusion: keep current base compile config; 64x128b UTCCP remains correctness-unsafe in this path.
- Popcorn portability constraints:
  - Submission must be single-file and avoid repo-relative imports.
  - Embedded CUDA source + temp-file JIT is acceptable.
  - Ensure parent directories exist before `torch.utils.cpp_extension.load(..., build_directory=...)`.
- Standard Popcorn check command:
  - `scripts/nvfp4_popcorn_check.sh <submission.py> test|benchmark`
  - Artifacts are written to `artifacts/popcorn_checks/<timestamp>/`.
- cta_group::2 correctness findings (case1):
  - `cluster=2, unroll=1`: only `AISP_NVFP4_GROUP_GEMM_V2_CTA2_PARTITION_B=1` verifies.
  - `cluster=2, unroll=1`: partition `0` and `2` fail.

### Update (2026-02-18): UTCCP64 + TMEM Scale Layout (SM100a) Findings
- Device-side UTCCP probe is now the source of truth for cta2 scale placement:
  - File: `labs/nvfp4_group_gemm_v2/tmem_sf_frg_probe.cu`
  - Key improvement: probing was stabilized by making `tcgen05.alloc/dealloc` warp-synchronous and synchronizing the 2-CTA cluster around alloc/UTCCP/dealloc. This avoids deadlocks and misaligned-address faults.
- Critical layout discovery (cta_group::2, UTCCP64 `.warpx2::{01_23|02_13}`):
  - The packed 128x16 scale tile layout is `[seg(0..3) * 32 + mm32, mm4*4 + kk4] -> [128,16]` (CUTLASS “blockscaled” layout).
  - UTCCP64 does not behave like a contiguous “64 rows = 2 segments” copy for this layout.
  - Empirically:
    - `start+0` copies K64 segments `{0,2}` (rows `0..31` and `64..95`)
    - `start+32` copies K64 segments `{1,3}` (rows `32..63` and `96..127`)
  - Consequence: to populate all 4 segments in TMEM for block-scaled UMMA, the UTCCP64 path must use descriptor start offsets `{0,32}` (in u128 row units), not `{0,64}`.
- Mapping implication (cta_group::2):
  - The two UTCCP64 destination bases must be the segment-0 and segment-1 TMEM bases (physical `col` groups `0..3` and `4..7`), because each UTCCP64 op implicitly populates its paired segments at `+8` columns (`seg2/seg3` live at `+8` within the same op).
- Current state after applying the above:
  - Default cta1 (`cluster_dim_x=1`, `unroll=1`, no UTCCP64 override) still verifies case1 locally.
  - cta2 correctness is still failing case1 verification, meaning the remaining issue is now likely *segment pointer mapping consumed by UMMA* (per-seg TSFA/TSFB addresses) and/or a rank/partition interaction, not just the UTCCP descriptor step.
  - Next debugging step should use `debug_tmem_dump>=10` (scale TMEM dumps) in `labs/nvfp4_group_gemm_v2/custom_cuda_group_gemm_kernel.cu` to confirm which segments/bytes are actually resident in TMEM at the exact TSFA/TSFB pointers passed to UMMA.
  - `cluster=2, unroll=2`: partition `0/1/2` currently fail (NaN/incorrect).
- cta2/unroll2 bring-up notes:
  - Removed host/kernel forced mode0 override for unroll2 so partition modes can be tested.
  - Added mode1 partitioned B load path + per-`u` SFB load path for unroll2.
  - Adjusted unroll2 cta2 TMEM C rank partition to use DP-axis partitioning to remove NaN-producing overlap; outputs are now finite but still numerically incorrect.
  - Added explicit unroll2 mode1 partitioned B/SFB TMA load paths with rank-aware SFB row offsets and matching barrier byte accounting.
  - Current cta2 matrix after latest patches:
    - `u=1,p=1`: PASS.
    - `u=1,p=0/2`: FAIL.
    - `u=2,p=0/1/2`: FAIL (finite, no longer universal NaN).
  - Notable improvement: `u=2,p=1` no longer exhibits the previous uniform `~1.97x` scaling error after SFB partition plumbing updates, but it remains incorrect.
  - Additional cta2/unroll2 debug sweeps:
    - Sweeping `AISP_NVFP4_GROUP_GEMM_V2_CTA2_TMEM_SF_RANK_WORD_OFFSET` across `{32,40,48,56,64,80,96}`
      and `AISP_NVFP4_GROUP_GEMM_V2_CTA2_SFB_SLOT_MODE` across `{0,1}` did not produce a pass.
    - Sweeping `AISP_NVFP4_GROUP_GEMM_V2_CTA2_TSFB_WORD_OFFSET` across `{0,4,8,12,16,20,24,28,32}`
      with both SFB slot modes did not produce a pass.
    - `WS_UNROLL2_MMA=0` did not fix cta2/unroll2 correctness.
  - Hard isolating evidence from latest TMEM/debug probes:
    - `debug_print_ptrs` + staged-byte dumps confirm `u=0` and `u=1` B/SFB payloads are distinct in SMEM,
      so the duplication bug is not caused by identical TMA source data.
    - `debug_tmem_dump=1` (tile0 base) and `debug_tmem_dump=2` (tile1 base) produce identical dumps under
      both `dp+128` and `idx+1` tile1 mappings, indicating cta2/unroll2 `u=1` C addressing still aliases tile0.
    - Trial `u=1` mapping `col+64` produced immediate NaNs (invalid for current layout).
    - Trial `u=1` mapping `col+128` made tile0 match reference exactly but tile1 became NaN, consistent with
      unavoidable overlap in the current cta2 TMEM col layout.
    - Practical conclusion: cta2/unroll2 needs a true non-overlapping accumulator+scale TMEM layout
      (or N256-style cta2 path), not incremental pointer tweaks.
  - Strong isolating signal from reduced-shape probes:
    - `cta2 + unroll2 + partition_b=1` passes for `n=128` and fails immediately for `n>=256`.
    - Therefore the defect is specifically in the second-N-tile (`u=1`) path, not the base `u=0` path.
  - Accumulator-addressing probe insight:
    - Forcing `tmem_c_tiles[1] = tmem_c_tiles[0]` produced effectively identical numerical failures to
      `tmem_c_tiles[1] = tmem_addr_add(..., dp_add=128, col_add=0)`.
    - In the current 2SM repeated-MMA flow, varying the `u=1` TMEM C pointer does not separate the second tile as expected.
    - Working hypothesis: cta_group::2 repeated N128 MMAs are aliasing the same accumulator region; fixing likely requires
      a true N256-style cta2 path (or equivalent non-aliased layout), not pointer-offset sweeps.
  - cta2 launch-path note:
    - cta2 kernels currently run via legacy max-tile grid path (`cta_*_map` pointers are null in cta2 launches),
      so packed-CTA ordering knobs do not affect cta2 behavior today.
- New kernel/host experiments from this session:
  - TMEM per-rank SF base now uses `tcgen05::tmem_addr_add(...)` instead of raw integer addition.
  - Added cta2 mode1-specific SFB N-major packing path in host prepare and matching kernel row-offset mode.
  - Result so far: cta1/unroll2 and cta2/unroll1 remain correct; cta2/unroll2 still fails verification.
  - This path is still not correctness-closed and must be fixed before perf tuning.
- Cluster/multicast rule:
  - Keep multicast off until cta2/unroll2 correctness is stable.
  - If multicast is re-enabled, retain only with measured net win and cluster-safe barrier semantics.
- Iteration loop to maintain momentum:
  - Inner loop: case1 only, verify on, Popcorn timing model.
  - Mid loop: periodic Popcorn benchmark checks with archived artifacts.
  - Outer loop: full 4-case local validation before promoting configs.
  - Keep knob settings explicit and logged; avoid hidden default drift.
- Latest update (2026-02-16, evening):
  - Local tuned cta1 case1 loop model remains stable at `~10.451 us/group` with:
    - `UNROLL_N=2`, `WS_UNROLL2_MMA=1`, `EPILOGUE_LD_X32=1`, `TMA_L2_PROMOTION=3`, `FUSE_INPUTS=1`.
  - Live Popcorn re-check against `labs/nvfp4_group_gemm_v2/popcorn_submission_case1_v2.py`:
    - `mode=test`: pass `10/10` (`artifacts/popcorn_checks/20260216T170200Z/popcorn_test.log`).
    - `mode=benchmark`: case1 mean `1771 us total` (`221.375 us/group`) with heavy outliers, but fast-path
      `69.7 us total` (`8.7125 us/group`) (`artifacts/popcorn_checks/20260216T170327Z/benchmark_parsed.json`).
  - cta2/unroll2 experiments in this pass (all still failing verify):
    - Removed unroll2-only SFB rank row shift in mode1 TMA issue paths: no correctness fix.
    - Removed experimental `tmem_c` tile1 idx split: no correctness fix.
    - Switched unroll2 rank partition from TMEM DP to TMEM idx (for C and SF): no correctness fix.
    - Tested high-bit (`sf_id`) rank split: illegal instruction (reverted).
    - Tested `sfb_box_height=64` for cta2 mode1: deadlock/hang (reverted to 128).
  - Current cta2/unroll2 failure signature:
    - Finite but structurally wrong outputs with group-wise MAE around `~75` and max abs around `~455`.
    - Ratio `got/ref` is roughly `~1.9x` on average, indicating a persistent 2-CTA partition/layout mismatch.
  - Latest update (2026-02-16, late evening, CUTLASS layout probe pass):
    - Added a tiny probe at `labs/nvfp4_group_gemm_v2/tmem_sf_frg_probe.cu` that prints encoded TMEM deltas for:
      - `C` fragment `(rank,u)` for 2SM/unroll2,
      - `SFA` fragment `(rank,seg)`,
      - `SFB` fragment `(rank,u,seg)`.
    - Probe confirmed compact CUTLASS SF fragment address deltas in encoded form for VS=16:
      - `SFA`: rank step `+0x4`, seg encoded in top 2 bits.
      - `SFB`: rank step `+0x8`, u step `+0x4`, seg encoded in top 2 bits.
    - Attempted to hard-wire this sf-id-in-address mapping directly into cta2/unroll2 kernel path:
      - Result: `cudaErrorIllegalInstruction` (reverted).
    - Stabilized back to the pre-existing safe cta2/unroll2 mapping (no illegal instruction), but correctness is still failing with the same structural mismatch.
  - Follow-up sweep after probe integration:
    - Kept cta2/unroll2 accumulator rank split in TMEM idx-space (stable/no-illegal path).
    - Switched only cta2/unroll2 SFA/SFB rank split from idx-space to column-space and swept rank stride:
      - tried `8`, `16`, `24`, `32` columns.
      - all still fail case1 verify (max-abs stays roughly in `~435..466` range depending on run).
    - Current best interpretation: scale rank addressing alone is not the sole blocker; cta2/unroll2 still has a deeper 2-CTA partition/layout mismatch.
