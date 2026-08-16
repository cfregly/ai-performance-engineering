# Optimization Research Campaigns

## Scope

The campaign module adds experiment control around the repository's trusted benchmark harness. It does not run benchmarks, import generated code, edit the evaluator, or promote a candidate. The harness remains responsible for correctness, timing, profiles, and raw artifacts.

Use the three parts together:

1. The opportunity radar proposes and ranks hypotheses.
2. The benchmark harness measures control and candidate revisions.
3. `core.optimization.campaign` records every attempt, applies promotion gates, and writes a compact steering report.

This separation is deliberate. Search policy can change without weakening the evaluator or the evidence contract.

## Research Basis

The implementation combines the parts of several optimization reports that fit this repository:

| Source | Practice adopted here |
| --- | --- |
| [Karpathy's autoresearch controller](https://github.com/karpathy/autoresearch/blob/master/program.md) | Keep the evaluator and objective fixed. Record every attempt. Make one bounded change, measure it, and keep or discard it. |
| [Sankalp's autoresearch report](https://sankalp.bearblog.dev/autoresearch/) | Keep a small beam of different idea families. Use profiles and human review to steer the search when local edits stop teaching you anything. |
| [Michael Goguen's QR report](https://ml-mike.com/writing/qr_v2/) | Preserve failed experiments, inspect every workload case, keep profiles beside experiment records, and assign one concrete direction to each isolated worktree. |
| [GPU Mode QR task](https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/linalg/qr_v2/task.yml) | Treat the evaluator, workload cases, and correctness rules as frozen campaign inputs. |
| [Abseil Performance Hints](https://abseil.io/fast/hints.html) | Profile first, estimate the possible effect, and prefer structural work avoided over small local edits. |

The QR reports also reference blocked compact WY updates, recursive panels, CUDA Graph replay, guarded Cholesky paths, and custom triangular inversion. Those are useful candidates for a dedicated QR lab. They are not safe defaults for unrelated workloads. See the [open QR algorithm notes](https://github.com/fishmingyu/qrv2-gpu-mode/blob/main/ALGORITHM.md), the [LAPACK DGEQRF contract](https://netlib.org/lapack/explore-html/d0/da1/group__geqrf_gade26961283814bb4e62183d9133d8bf5.html), and the [CUDA Graph documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

## Campaign Contract

A campaign freezes these inputs before the first candidate runs:

- objective, metric, direction, and aggregate
- initial control commit and the incumbent lineage rule
- primary workload cases used for the aggregate
- frozen guard cases that may not regress past the declared limit
- workload and environment file hashes
- repeat count, variance limit, minimum effect, paired confidence policy, and evidence requirements
- experiment, wall time, and cost budgets
- beam width

Each completed experiment records:

- one hypothesis and one idea family
- repeated control and candidate samples for every required case
- correctness status and measurement order
- raw benchmark and profile artifact paths
- a code audit and a measured mechanism
- the worktree commit, branch, status, concrete diff, and diff hash
- duration, cost, outcome, and next step

The ledger is append-only. A later status change adds a revision with the same experiment ID. Failed, crashed, parked, and promoted revisions remain available when `PRIORS.md` is regenerated.

## Start a Campaign

Run this from `code/` after creating a frozen workload spec and a baseline run manifest:

```bash
python -m core.optimization.campaign init artifacts/campaigns/launch-overhead \
  --objective "Reduce replay latency without regressing any frozen shape" \
  --metric latency_ms \
  --initial-control-commit "$(git rev-parse HEAD)" \
  --direction lower \
  --primary-case batch-1 \
  --primary-case batch-8 \
  --frozen-case batch-1 \
  --frozen-case batch-8 \
  --frozen-case batch-32 \
  --beam-width 4 \
  --min-trials 5 \
  --require-confidence-bounds \
  --bootstrap-resamples 10000 \
  --min-improvement-pct 2 \
  --max-case-regression-pct 0.5 \
  --workload-spec templates/benchmark_workload_spec.yaml \
  --environment-spec artifacts/baseline/run_manifest.json \
  --max-experiments 40
```

Initialization writes:

```text
artifacts/campaigns/launch-overhead/
  campaign.json
  campaign.sha256
  experiments.jsonl
  experiment-template.json
  REPORT.md
  PRIORS.md
  artifacts/
```

Copy `experiment-template.json` for each candidate. The template starts in `planned` state with unknown correctness and no measurements or evidence. Its `parent_id` is the frozen initial control commit. After a promotion, use the current incumbent commit instead.

Show the incumbent required by the next experiment:

```bash
python -m core.optimization.campaign incumbent \
  artifacts/campaigns/launch-overhead --json
```

Only a manually promoted experiment advances the incumbent. Parked, rejected, crashed, and completed attempts do not advance it. A new measured record is rejected when either `parent_id` or `provenance.control_commit` differs from the current incumbent.

## Artifact-Derived Recording

Use `record-evidence` for unattended or promotion-eligible work. Add `--require-derived-evidence` when the campaign is initialized. That option also makes the paired confidence gate mandatory.

The evidence bundle uses schema `aisp.campaign-benchmark-evidence/v1`. It names every frozen case, declares alternating control and candidate order, and binds each result path to a SHA-256 digest. Every referenced result must contain one completed benchmark target, output and input verification, and baseline and optimized run manifests. The adapter rejects dirty revisions, provenance warnings, runtime limitations, non-strict validity, changed environments, reused files, inconsistent timestamps, and mismatched hashes.

```bash
python -m core.optimization.campaign record-evidence \
  artifacts/campaigns/launch-overhead \
  --experiment /path/to/exp-001.json \
  --evidence /path/to/benchmark-evidence.json \
  --repo /path/to/candidate-worktree \
  --control-revision "$(python -m core.optimization.campaign incumbent \
    artifacts/campaigns/launch-overhead --json | python -c \
    'import json,sys; print(json.load(sys.stdin)["commit"])')"
```

The adapter derives these fields. Draft values for them are ignored:

- repeated control and candidate measurements
- correctness
- measurement protocol
- duration
- raw and profile artifact paths and hashes
- workload and environment hashes
- control and candidate benchmark commits

The Git diff captured by `--repo` must match the commits embedded in the benchmark manifests. The campaign ledger then checks the candidate against the current incumbent under the same file lock used for appends.

For interactive, operator-attested work, use the ordinary `record` command while the candidate worktree still contains the measured patch:

```bash
python -m core.optimization.campaign record \
  artifacts/campaigns/launch-overhead \
  --experiment /path/to/exp-001.json \
  --repo /path/to/candidate-worktree
```

The `--repo` argument captures the candidate's current commit, branch, dirty status, binary diff, and diff hash. It is required for a new measured experiment. A missing or empty candidate diff blocks mechanical promotion by default. Changed or untracked files outside `changed_surface` also block recording. Local raw and profile artifact files receive content hashes when they are recorded. Relative artifact paths resolve from the experiment JSON file and may not escape that directory. Remote artifact URLs are not accepted until a trusted manifest adapter exists.

The ordinary experiment JSON is operator-attested. File hashes prove which evidence files were recorded and detect later changes. Only `record-evidence` derives measurements and correctness from validated benchmark artifacts.

If the candidate is already committed, also pass `--control-revision <incumbent-sha>`. The captured diff will compare that control commit with the candidate worktree's current commit.

Inspect the gate and the active idea beam:

```bash
python -m core.optimization.campaign gate \
  artifacts/campaigns/launch-overhead exp-001

python -m core.optimization.campaign beam \
  artifacts/campaigns/launch-overhead

python -m core.optimization.campaign report \
  artifacts/campaigns/launch-overhead
```

`gate` exits with status 0 only when all mechanical promotion checks pass. It exits with status 2 for a parked, rejected, or inconclusive candidate. Mechanical success is not human approval.

`gate --json` includes the paired bootstrap method, confidence level, resample count, aggregate interval, and per-case intervals. When confidence bounds are required, the aggregate lower bound must clear `min_improvement_pct`. Each frozen-case lower bound must clear the negative `max_case_regression_pct` limit. The gate returns inconclusive when it cannot form the declared number of pairs.

After review, append a new revision of the same experiment with `status` set to `parked`, `rejected`, or `promoted`. Include a concrete `outcome` and `next_step`. This preserves the evidence and updates the generated priors without rewriting history.

## Promotion Rules

A measured aggregate win is not enough. Promotion is blocked when any of these conditions holds:

- correctness did not pass
- a required case or trial is missing
- sample variation exceeds the configured limit
- measurements were not interleaved
- a raw artifact, required profile, code audit, or mechanism is missing
- workload or environment hashes differ from the frozen contract
- git identity or the candidate diff is missing
- any frozen case exceeds the allowed regression
- the primary aggregate improvement is smaller than the declared effect threshold
- a required primary confidence lower bound misses the effect threshold
- a required frozen-case confidence lower bound misses the regression limit
- the record was measured against a stale incumbent

Primary cases determine the aggregate. Guard cases only enforce the regression rule. This prevents a large win on an auxiliary case from hiding a loss on the campaign objective.

## Worktree Operating Rule

Start each candidate worktree from the same incumbent revision. Give it one concrete direction such as launch structure, data layout, precision, or algorithm. Record the candidate even when it crashes or disproves the hypothesis. Do not import generated code into the controller process.

Keep the incumbent unchanged until a candidate passes correctness, repeated measurements, per-case gates, mechanism review, and explicit human approval. A promoted candidate becomes the next parent only after that review.

The JSONL ledger uses POSIX file locks for concurrent readers and writers. It fails closed on platforms without that locking support.

## Opportunity Queue Contract

Generated opportunity queues now use concrete worktree-bound commands. Control commands run from `AISP_CONTROL_CWD`. Candidate and candidate-profile commands require `AISP_CANDIDATE_CWD`. Candidate jobs fail closed unless both worktrees are clean, use different roots, and point to different commits.

Each executable job declares a repeat count and comparison policy. Candidate comparisons use `paired_interleaved` order and alternate control-first with candidate-first runs. The runner writes one result, log set, command, exit code, and content hash per role and repeat. It writes `repeat_run_manifest.json` after all declared repeats. Unsupported or incomplete policies fail before execution.

A failed job receives `FAILED` and an exit code. Independent jobs continue. Dependent jobs receive `BLOCKED`. The queue returns a nonzero status after it has processed every runnable job.

Queue summaries and novelty artifact audits validate marker content, JSON structure, benchmark result content, run manifest provenance, and every declared SHA-256 reference. A file that merely exists does not satisfy the contract.

## Operational Boundary

The module is a durable record and gate, not a scheduler. The former `AutoOptimizer` measurement loop now fails closed. Its retained adapters only move source code and do not produce measurements or promotion decisions.

The dashboard exposes the incumbent, active idea beam, promotion frontier, per-case medians, confidence bounds, frozen-case findings, and hash-checked evidence links at `/campaign`.

Generated candidates never load into the controller process. The worker requires a supported hardened operating-system sandbox before it verifies or times generated code. On an unsupported host it returns a `sandbox_unavailable` result and leaves the candidate as an artifact for review. There is no flag that bypasses this check. Source promotion also requires explicit manual approval or a matching campaign attestation.

No performance result is claimed by this module. A GPU campaign still needs representative hardware, the existing harness, repeated measurements, and profiler evidence.
