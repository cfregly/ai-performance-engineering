# Code instruction router

This router applies recursively to `code/`. Keep it short. The former instruction file and subsequent policy updates live in [the detailed authority](docs/agent-instructions-authority.md); its rules remain authoritative for the scopes routed below. Before changing a routed scope, read the linked section and any directly nested subsections. For an unlisted specialized task, search the authority's headings and load the matching section before editing.

The authority was originally moved without rewriting its text. Resolve relative paths written inside it as if the file were still `code/AGENTS.md`. In particular, its performance-review skill link means [`../.agents/skills/dean-performance-review/SKILL.md`](../.agents/skills/dean-performance-review/SKILL.md), and its audit evidence links start at repository root `docs/`.

## Always-loaded rules

### Work efficiently and follow the repository

- Ask at a material decision point only when the requested outcome and existing evidence do not determine the choice. Inspect neighboring chapters, shared helpers, runtime/profiling harnesses, and established launch flows before introducing a pattern.
- Match existing conventions unless the task requires a deliberate correction. Report any intentional convention change and why the old convention was insufficient.
- Prefer explicit flags and parameters. Do not change a global default without the user's direction or a clearly authorized cross-surface fix; when a default changes, update every affected interface and its documentation together.
- Make the long-term change. Keep benchmark-specific semantics local. Move repeated shared logic with at least two call sites into `core/`, and change the harness only for a cross-cutting infrastructure defect or repeated safe abstraction.

### Safety and repository state

- Do not run destructive Git operations, including `git restore`, `git checkout`, `git reset --hard`, or `git revert`, unless the user explicitly requests the exact operation. Never restore a file to a prior revision on your own.
- Do not delete tracked, untracked, or modified files without explicit user direction. Preserve unexpected local state, report it, and avoid overwriting another owner's work. A file already being edited may be changed only as needed for the task while retaining its existing changes.
- Prefer materialized benchmark and profiler artifacts. Do not introduce symlink-dependent profile pairing; copy a symlinked input to a real, clearly named baseline or candidate artifact before comparison.
- Agents may terminate profilers when necessary within the authorized task without further confirmation. Confirm run ownership, preserve artifacts, log the reason and job/process identifiers, and mark interrupted validation incomplete; follow [Queueing & Monitoring](docs/agent-instructions-authority.md#queueing--monitoring-critical).
- The Amazon book URL in `README.md` is an allowlisted automated-link-check failure caused by bot protection.

### Correctness, verification, and performance claims

- Dogfood every changed reachable path with a real repository invocation when feasible. Each fix needs focused regression coverage, syntax/import validation, and a realistic execution path appropriate to its risk. Record exact commands and outcomes.
- Use the smallest sufficient verification loop while fixing. Run a broad suite only when requested, required by release/CI, or needed to bound a cross-cutting change. Never describe focused checks as a full-suite pass.
- Benchmark-truth tests MUST execute real repo code paths end-to-end. Do not mock successful benchmark execution, profiler capture, correctness, GPU capability, or runtime validity. Narrow control-plane/orchestration tests MAY use `monkeypatch`/`patch` for environment, subprocess, clock, filesystem, process, and external-tool seams when the assertion concerns control flow; retain a real-path test for the surface.
- Treat static performance review as hypothesis generation. Claim a win only after representative equivalent-workload baselines, correctness gates, repeated interleaved control/candidate measurements, noise reporting, and profiler or counter evidence for the mechanism.
- Never use checksums, constants, stale setup outputs, hidden work reduction, or extra timed work to make verification pass. Prefer `VerificationPayloadMixin` and capture the actual timed output. Fail clearly when required verification data is absent.
- Fail fast. Do not add silent fallback, auto-inference, broad exception swallowing, or degraded behavior under the same benchmark name. Fix the root cause or emit an explicit hard diagnostic.

### Hardware and evidence gates

- Required hardware, driver, toolkit, profiler, service, and feature support are hard gates. Use `SKIPPED:` or an equally explicit structured diagnostic for expected unsupported capability. Do not publish or store fallback-path numbers as valid results.
- Canonical GPU benchmark/profiling runs must use the harness clock lock, record application clocks in console telemetry and the run manifest, and complete required `ncu` and `nsys` collection. Do not lock clocks manually with `nvidia-smi`.
- `strict` is the default validity profile. `portable` must be explicitly selected and relaxes only the documented clock, application-clock, telemetry, and virtualization checks. It does not bypass missing CUDA, unsupported platforms, swap, CPU-governor, cgroup, or foreign-process failures.
- Canonical and publish-grade benchmark evidence requires the specified target environment. A user-authorized virtualized current-host run must be labeled non-canonical and does not waive bare-metal or cluster publication gates.
- Preserve structured provenance: run ID, targets, Git commit, hardware key, profile, timestamp, iterations, warmups, clock state, validation issues, and rejection reasons. Separate measured results from unsupported, skipped, advisory, or unaudited checks.

### Public identifier hygiene

This repository is public. Never commit live pod names, Kubernetes namespaces or contexts, node names, organization or tenant IDs, internal hostnames, or experiment labels containing them. Use placeholders such as `<gb300-pod>` and `<namespace>`. Historical identifiers are burned/rotatable context, not permission to reintroduce them. Generic published component names are acceptable.

### API and lifecycle discipline

- Remove deprecated entrypoints, shims, aliases, flags, docs, and tests in one change; update all references to the current API.
- Keep CLI, MCP, dashboard behavior, defaults, help, and docs synchronized. The dashboard API exposes only UI-used behavior unless expansion is explicitly requested. Regenerate MCP docs and update the API reference when tool metadata changes.
- Keep secrets in the repository-root `.env` or `.env.local` only for local integrations. Never print, copy into artifacts, or commit credentials.

## Scoped authority map

| Scope or trigger | Read before changing |
|---|---|
| Performance-sensitive code, benchmark design, hot paths | [Performance Review Workflow](docs/agent-instructions-authority.md#performance-review-workflow-critical), then the Dean skill linked above |
| Toolchain aborts, compatibility shims, architecture transforms, guard removal | [Toolchain Workarounds & Abort Attribution](docs/agent-instructions-authority.md#toolchain-workarounds--abort-attribution-critical) |
| `code/cluster*`, cluster suites, canonical artifacts, or field reports | [Root-Cause First](docs/agent-instructions-authority.md#root-cause-first-critical), [Report Completeness + Sync](docs/agent-instructions-authority.md#report-completeness--sync-critical), [Case Study Contract](docs/agent-instructions-authority.md#case-study-contract-critical), [Discovery + Metadata](docs/agent-instructions-authority.md#discovery--metadata-critical), and [Benchmarks](docs/agent-instructions-authority.md#benchmarks-critical) |
| Field-report edits or cleanup | [Report Update Checklist](docs/agent-instructions-authority.md#report-update-checklist-critical), [Stakeholder Markdown Presentation](docs/agent-instructions-authority.md#stakeholder-markdown-presentation-critical), and [Execution Order](docs/agent-instructions-authority.md#execution-order-critical) |
| Benchmark/profiler runs, expectations, queues, or result analysis | [Benchmark Stability](docs/agent-instructions-authority.md#benchmark-stability-critical), [Provenance Review](docs/agent-instructions-authority.md#provenance-review-critical), [Expectations Files](docs/agent-instructions-authority.md#expectations-files-critical), [Queueing & Monitoring](docs/agent-instructions-authority.md#queueing--monitoring-critical), and [Validity Profiles](docs/agent-instructions-authority.md#validity-profiles-critical) |
| Baseline/optimized pairs, demos, tools, labs, or diagnostics | [Benchmarks vs Tools/Demos](docs/agent-instructions-authority.md#benchmarks-vs-toolsdemos-critical), [Labs](docs/agent-instructions-authority.md#labs-critical), [Hardware Diagnostics](docs/agent-instructions-authority.md#hardware-diagnostics-microbench), and [Chapter Consistency](docs/agent-instructions-authority.md#chapter-consistency) |
| Harness verification, `core/harness`, `core/benchmark`, or verification tests | [Verification Interface](docs/agent-instructions-authority.md#verification-interface-critical), [Benchmark Validity Issues Reference](docs/agent-instructions-authority.md#benchmark-validity-issues-reference), [FAIL FAST](docs/agent-instructions-authority.md#fail-fast---no-fallbacks-no-auto-inference), [Deterministic Seed Pattern](docs/agent-instructions-authority.md#deterministic-seed-pattern-critical), [Tests for New Functionality](docs/agent-instructions-authority.md#tests-for-new-functionality-critical), [Harness Verification Architecture](docs/agent-instructions-authority.md#harness-verification-architecture-important), and [Checksum Verification](docs/agent-instructions-authority.md#checksum-verification-is-not-acceptable-important) |
| Benchmark speedup or memory-goal acceptance | [Achieve MAXIMUM speedup](docs/agent-instructions-authority.md#achieve-maximum-speedup-when-benchmarking-baseline_-versus-optimized_-variants-when-possible), [Multi-GPU Defaults](docs/agent-instructions-authority.md#multi-gpu-defaults-critical), and [Benchmark Example Pairs](docs/agent-instructions-authority.md#benchmark-example-pairs-baseline-vs-optimized) |
| NVFP4 grouped GEMM routing, kernels, Popcorn, ABAB, or TMEM | [NVFP4 Grouped GEMM Perf Playbook](docs/agent-instructions-authority.md#nvfp4-grouped-gemm-perf-playbook-critical), [NVFP4 Group GEMM V2 Learnings](docs/agent-instructions-authority.md#nvfp4-group-gemm-v2-learnings-2026-02-16), and [UTCCP64 + TMEM findings](docs/agent-instructions-authority.md#update-2026-02-18-utccp64--tmem-scale-layout-sm100a-findings) |
| Fresh all-stage `run-e2e` sweep | [Full Sweep Playbook](docs/agent-instructions-authority.md#full-sweep-playbook) and [`code/FULL_SWEEP.md`](FULL_SWEEP.md) |

The detailed authority also contains the full historical validity inventory, real-world incident references, concrete verification patterns, current tuning decisions, negative results, and retirement conditions. Those details are evidence, not automatically current runtime proof; revalidate drift-prone state before making or publishing a claim.
