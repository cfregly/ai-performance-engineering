# Tier-1 Canonical Benchmark Suite

## Purpose
Tier-1 is the repo's smallest high-signal benchmark suite. It exists to answer one practical question quickly:

"Is this repo still producing real, verified performance gains on representative workloads?"

Use tier-1 when you want a canonical health check for releases, regression tracking, or public proof that the baseline/optimized pairs still hold up.

## Run It
```bash
python -m cli.aisp bench run-tier1 --single-gpu --profile minimal
```

That command reads the suite definition from [`configs/benchmark_suites/tier1.yaml`](../configs/benchmark_suites/tier1.yaml) and writes a canonical history package under `artifacts/history/tier1/<run_id>/`.

The recurring CI entrypoint is [`tier1-nightly.yml`](../../.github/workflows/tier1-nightly.yml). It requires the `["self-hosted","linux","x64","gpu","b200","node24-actions"]` runner labels. Apply `node24-actions` only to runners at version 2.327.1 or newer. Apply `b200` only after confirming that every visible GPU reports the exact name `NVIDIA B200` and has MIG disabled. The workflow repeats those checks before it records canonical history. An unverified, partitioned, or different runner cannot accept this job.

## Canonical CI History

The first canonical run must be a manual `main` branch dispatch with all three inputs set:

- `bootstrap_history` is `true`
- `accept_history_anchor` is `true`
- `acceptance_note` contains the public reason for establishing the anchor

Use this path only after the runner labels and GPU preflight are attested. The `tier1-canonical-acceptance` GitHub environment must exist, allow deployments only from `main`, and require a reviewer. The B200 job produces the candidate and uploads its immutable evidence before the protected promotion job starts. A reviewer approves the promotion after that evidence exists.

Restore permits a first anchor when no prior package exists. It can skip a structurally incomplete package only when GitHub proves that package came from a non-success workflow run. It refuses to reset history after a digest failure, unsafe archive path, incomplete discovery, or evidence provenance mismatch.

Every later scheduled or manual run restores cumulative history before dependencies or benchmarks run. Restore considers completed workflow runs even when their job conclusion is failure. It can skip a partial package from a failed run and use an older compatible package. It binds the selected baseline to retained evidence from this workflow on `main`, at the recorded Git commit, with a valid artifact digest.

Normal evidence publication and protected anchor promotion share one short publication lock. Each job restores the live canonical package again after benchmark execution and immediately before it writes history. If a reviewed promotion finishes while an older producer is still running, the older producer merges its evidence into the new anchor and stays ineligible until it is compared under the current contract. Approval waits happen outside the publication lock, so they do not hold that lock. Each concurrency group retains at most 100 pending jobs. GitHub does not guarantee strict dispatch order within the group.

Each run remains in `index.json`. A run passes the release gate only when every configured target succeeded, every required metric is finite and valid, its clean Git commit matches the run manifest, history integrity is clean, and no target regression or deletion remains. Failed, skipped, missing, and malformed runs stay available as evidence but remain ineligible as anchors. Normal publication also keeps a confirmed regression ineligible. A confirmed regression may become an anchor only through a manual `accept_history_anchor` dispatch, a public override reason in `acceptance_note`, immutable evidence upload, and protected post-benchmark promotion. The promotion records the requester, note, workflow run URL, evidence digest, and `accept_history_anchor` acceptance type in `index.json`. The `tier1-canonical-acceptance` environment supplies reviewer approval before promotion. A regression cleared only by a recheck remains rejected even through this path.

The workflow uploads raw main-run and recheck evidence before it uploads cumulative history. If evidence upload fails, history is not published. Candidate, evidence, and history artifacts use 90 day retention. Restore verifies the retained evidence name, digest, commit, workflow, branch, and repository. It does not download old benchmark evidence into the runner workspace. Paths stored in summaries and the index are relative to their artifact roots, so a later run can restore them in a different workspace without exposing a runner path.

The repository does not track a generated canonical history package. GitHub Actions artifacts are the retained source for current canonical results. The Tier-1 dashboard reads a restored or locally supplied history root and shows the latest evidence separately from the latest accepted run.

The selected anchor must be renewed before its evidence reaches 60 days old. A reviewed manual acceptance dispatch creates a new evidence-bound anchor. The promotion utility requires a requester, a nonblank public note, an exact HTTPS Actions run URL, the clean benchmark commit, and the immutable evidence artifact name and digest. Evidence expires after 90 days. If the renewal window is missed and the old evidence expires, normal restore fails closed and an operator must use the audited bootstrap recovery path.

## Current Tier-1 Targets
| Category | Target | Why it is here |
| --- | --- | --- |
| GEMM | `labs/block_scaling:block_scaling` | Blackwell block-scaled GEMM with an explicit software baseline and hardware-accelerated optimized path. |
| Attention | `labs/flashattention4:flashattention4_alibi` | Attention path with a real score modifier and profiler-friendly optimized kernel path. |
| Decode | `labs/persistent_decode:persistent_decode` | Persistent decode serving path where launch-overhead savings materially matter. |
| KV / Memory | `labs/kv_optimization:kv_standard` | Explicit speed versus memory tradeoff for KV-cache optimization. |
| Communication | `ch04:gradient_fusion` | High-signal communication/computation overlap sanity target. |
| End-to-end | `labs/real_world_models:llama_3_1_8b` | Prevents the suite from overfitting to microbenchmarks only. |

## Artifact Contract
Every tier-1 run should produce:

- `artifacts/history/tier1/index.json`
- `artifacts/history/tier1/<run_id>/summary.json`
- `artifacts/history/tier1/<run_id>/regression_summary.md`
- `artifacts/history/tier1/<run_id>/regression_summary.json`
- `artifacts/history/tier1/<run_id>/trend_snapshot.json`

Those files are generated by the suite plumbing in:
- [tier1.py](../core/benchmark/suites/tier1.py)
- [history_index.py](../core/analysis/history_index.py)
- [regressions.py](../core/analysis/regressions.py)
- [trends.py](../core/analysis/trends.py)

## How To Read Tier-1
- `summary.json` is the canonical machine-readable result for the run. It carries per-target baseline time, best speedup, artifact references, and both representative suite-level speedups (`median_speedup`, `geomean_speedup` / `representative_speedup`) plus the raw arithmetic average.
- `regression_summary.md` is the human-facing "what changed since the last canonical run?" document.
- The regression layer follows each target's `optimization_goal`. Speed targets check both relative speedup and absolute optimized latency. Memory targets check both relative savings and absolute optimized memory.
- A subthreshold decline can pass the current run while holding anchor advancement. This prevents several small slowdowns from ratcheting the comparison baseline downward.
- Speedup changes are significant only when they clear both the relative threshold and a small absolute optimized-time floor, so very small kernels do not produce noisy false regressions from sub-`0.05 ms` drift alone.
- `trend_snapshot.json` is the compact time-series summary for dashboards and release notes. Prefer `representative_speedup` or `avg_geomean_speedup` over `avg_speedup` when a single outlier target would skew the arithmetic mean.
- `index.json` is the lookup layer that lets docs and automation find the newest canonical run without path hunting.

## Interpretation Rules
- Treat tier-1 as a relative-performance suite first. The most important numbers are baseline versus optimized deltas under stable harness conditions.
- Do not trust a tier-1 run that is missing verification, manifest data, or profiler-backed paths when those are expected.
- A tier-1 regression is a release-quality signal. Investigate it even if a microbenchmark still looks healthy.
- A tier-1 improvement is only publishable if the benchmark contract stays green and the optimized path is still semantically the same workload.

## Related Docs
- [README.md](../README.md)
- [performance_repo_roadmap.md](./performance_repo_roadmap.md)
- [benchmark_harness_guide.md](./benchmark_harness_guide.md)
