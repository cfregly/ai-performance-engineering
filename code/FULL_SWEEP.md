# Full Sweep Playbook

Use this when you need to launch a fresh all-stages `run-e2e`, fix issues as they appear, and carry the run through to a truthful final state.

## Operator Checklist
- Pick a fresh `run_id`.
- Launch a new all-stages run:

```bash
python -m cli.aisp bench run-e2e \
  --run-id <RUN_ID> \
  --run-full-sweep \
  --run-fabric \
  --cluster-preset common-answer-fast \
  --validity-profile strict
```

Slurm is optional for this command: it runs directly on the current host. An external scheduler may instead place the full sweep inside one allocation; it does not need a separate allocation for each target.

- Monitor the run continuously with the repo-native status surface:

```bash
python -m cli.aisp bench run-e2e-status --run-id <RUN_ID> --watch
```

- `run-e2e` auto-arms a detached watcher by default. Re-arm it manually if needed:

```bash
python -m cli.aisp bench watch-e2e --run-id <RUN_ID>
```
- The dashboard now exposes the same normalized source under `/e2e?run_id=<RUN_ID>` instead of reading raw package JSON directly.
- For each failure or suspicious weak result:
  - root-cause it
  - fix it in the most local correct place
  - rerun the affected target directly
  - resume `run-e2e`
- Compare every touched chapter or lab against the matching `book-after/chXX*` content.
- Keep capability-limited outcomes truthful: use `skipped` / `partial`, not fake green.
- Do not disable `nsys` or `ncu`.
- End with:
  - final top-level run status
  - per-stage outcomes
  - rerun ledger
  - exact verification commands
  - explanation of anything still `partial` or unresolved

## Codex Checklist
- Inspect repo state and current modifications before starting.
- Follow existing repo conventions before introducing new harness, launcher, or profiler patterns.
- Keep fixes local to the benchmark, chapter, lab, or cluster path unless the defect is clearly cross-cutting.
- Dogfood every changed runtime path with a real repo invocation.
- Record exact commands, run ids, and artifact paths while working.
- If the e2e run aborts, fix durability or resume behavior before restarting broad reruns.
- Prefer `run-e2e-status` over manual JSON/log joins. It normalizes live child progress, stale-orchestrator detection, watcher state, recent events, ledgers, and the exact resume command.
- `run-e2e` defaults `--full-sweep-suite-timeout 0` so long full-sweep buckets are not killed by the 4-hour aggregate suite watchdog. Per-benchmark and profiler timeouts still apply.
- If a benchmark is unsupported on the current host, emit an explicit `SKIPPED:` result instead of degrading silently.
- If a speed-goal benchmark lands below `1.05x`, ensure status semantics are correct and visible in structured outputs.
- For every touched chapter or lab, check the corresponding `book-after/chXX*` material and fix any meaningful mismatch.
- Preserve provenance packages and historical-failure ledgers; do not hide old failures by deleting artifacts.
- Finish with a truthful summary grouped into:
  - green
  - partial because of host capability limits
  - unresolved failures
  - `book-after` alignment follow-ups

## Objective
1. Launch a brand-new full `run-e2e` across all stages.
2. Fix anything broken, sub-optimal, or misaligned with the corresponding `book-after/chXX*` content.
3. Re-run targeted failures immediately after each fix.
4. Finish with an evidence-backed summary of what is green, what is honestly partial because of host capability limits, and what remains unresolved.

## Start Here
- Review the current repo state and existing local modifications first.
- Do not revert or delete user changes.
- Treat existing modified files as part of the task unless they are clearly unrelated generated artifacts.
- Follow repo conventions before introducing new harness patterns, launcher paths, or workaround flows.

## Fresh Run Contract
- Use a new run id, for example:

```bash
python -m cli.aisp bench run-e2e \
  --run-id 20260327_e2e_full_all_fresh \
  --run-full-sweep \
  --run-fabric \
  --cluster-preset common-answer-fast \
  --validity-profile strict
```

- The run package records watcher metadata and status under:
  - `artifacts/e2e_runs/<RUN_ID>/watcher_status.json`
  - `artifacts/e2e_runs/<RUN_ID>/<RUN_ID>_watcher.launch.log`
- Use `run-e2e-status` for one normalized snapshot instead of manually diffing `summary.json`, `checkpoint.json`, `progress.json`, and child run artifacts.
- The raw run package now includes `preferred_progress_source` and `actions` fields so humans, MCP clients, and dashboard views can discover the authoritative status surface without reconstructing it manually.

- If the host is virtualized, single-GPU, or lacks IB / Spectrum-X management-plane coverage, keep results truthful:
  - do not force `succeeded` when the correct result is `partial`
  - capability-gated multi-GPU work must remain explicit `skipped` / `partial`
  - fabric should only be fully green when the underlying capability contract is truly satisfied
- Keep benchmark and profiler validity checks strict. Use the `portable` profile only for an explicitly partial compatibility run after a strict run proves that the host cannot satisfy a capability gate.
- Never disable `nsys` or `ncu`.

## While The Sweep Is Running
- Monitor the run continuously with `run-e2e-status`.
- For any failed benchmark, failed profiler, broken resume behavior, missing artifact, bad classification, or suspiciously weak result:
  - root-cause it
  - fix it in the most local correct place
  - re-run the affected target with a realistic repo invocation
  - then resume or restart the e2e flow as appropriate
- If a benchmark is not broken but is clearly sub-optimal relative to chapter intent, inspect the matching `book-after/chXX*` material and align code, harness expectations, docs/snippets, or runtime semantics as needed.
- If a chapter’s code and `book-after` are intentionally different, call that out explicitly in the final summary.

## Book Alignment Requirements
- For each touched chapter or lab, compare against the matching `book-after/chXX*` content.
- Fix obvious mismatches in:
  - benchmark name or intent
  - optimization goal
  - capability gating
  - profiler-path behavior
  - code snippet, command, or artifact naming
  - performance story stated in the chapter
- If book and code disagree, fix the real source-of-truth mismatch instead of papering over it in the run summary.

## Historical Context
- Preserve any existing e2e provenance packages and historical-failure ledgers.
- If the new run uncovers issues, produce the same kind of clear structured ledger or equivalent evidence.
- Do not delete old attempts just to make the current run package look clean.

## Required Engineering Behavior
- Dogfood every changed runtime path with a real repo invocation.
- Record exact commands and outcomes.
- Use `apply_patch` for manual edits.
- Prefer local fixes in benchmark/chapter code over weakening the harness.
- Keep structured outputs truthful and auditable.
- If the run aborts mid-flight, fix resume and durability behavior and continue.
- If a benchmark is unsupported on the current host, emit an explicit hard skip rather than a degraded fallback.
- If a speed-goal benchmark lands below `1.05x`, ensure status semantics remain correct.

## Deliverables
At the end of the run, provide:

1. Final top-level run id and terminal state.
2. Exact per-stage outcome for `tier1`, `full_sweep`, `cluster`, and `fabric`.
3. Every code or doc fix made.
4. Every benchmark, lab, or chapter rerun and its outcome.
5. A clear explanation of anything still `partial` and why.
6. Verification commands and artifact paths.
7. Any remaining debt, grouped into:
   - true failures
   - truthful capability-limited partials
   - `book-after` / code alignment follow-ups

Do not stop at analysis. Execute the run, fix issues, and carry it through to a truthful final state.
