# Lab - RecSys Sequence Ranking

## Summary
Benchmarks a modern recommendation-style session-ranking path without pulling in a full legacy RecSys framework stack. The baseline path keeps sparse lookups and candidate scoring intentionally conservative. The optimized path uses vectorized PyTorch, `torch.compile` when available, and a Triton candidate-scoring kernel.

## Problem
Recommendation inference is often dominated by sparse embedding lookups, pooling, and candidate scoring rather than transformer attention. This lab exists to answer one concrete question: how much latency can you save on a session-ranking workload by modernizing the hot path in plain PyTorch and Triton?

## Baseline Path
- per-token sequence pooling with explicit Python-side looping
- one sparse context-table lookup at a time
- one candidate score at a time
- eager PyTorch only

## Optimized Path
- vectorized sequence pooling and context aggregation
- compiled user tower when `torch.compile` is available
- Triton candidate-scoring kernel by default, with a torch fallback for environments where Triton is unavailable
- identical synthetic workload, weights, and verification inputs as the baseline

## Measured Delta
Current validated strict result from `artifacts/runs/20260310_001644__bench__profile_none_targets_labs_recsys_sequence_ranking_sequence_ranking/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `sequence_ranking` | `5.071 ms` | `1.008 ms` | `5.03x` |

Verification stayed clean on the same synthetic batch with `max_abs_diff=2.384185791015625e-07`, and the run now has a repo-native expectation entry in `expectations_b200.json`.

This local strict result was collected under virtualization, so it is useful for repo health and regression tracking, but absolute publish-grade numbers should still be re-collected on bare metal.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/recsys_sequence_ranking:sequence_ranking --profile deep_dive --single-gpu
```

Use the deep-dive path when you want Nsight Systems + Nsight Compute evidence for the sparse ranking hot path instead of a wall-clock number only.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/recsys_sequence_ranking
python -m cli.aisp bench run --targets labs/recsys_sequence_ranking --profile minimal
python -m cli.aisp bench run --targets labs/recsys_sequence_ranking:sequence_ranking --profile minimal --target-extra-arg labs/recsys_sequence_ranking:sequence_ranking="--batch-size 128 --num-candidates 256"
```

## Learning Goals
- See what a recommendation-style sparse workload looks like when expressed as a clean benchmark pair instead of a framework integration demo.
- Compare explicit sparse loops against vectorized PyTorch and Triton on the same synthetic clickstream batch.
- Measure the effect of candidate scoring fusion separately from data-generation or feature-store concerns.
- Keep recommendation-system benchmarking honest by separating inference latency from ETL, offline metrics, and orchestration overhead.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_sequence_ranking.py` | Conservative session-ranking path with per-token, per-table, and per-candidate loops. |
| `optimized_sequence_ranking.py` | Vectorized/compiled path with Triton candidate scoring. |
| `compare_sequence_ranking.py` | Direct local runner for parity and rough speedup checks outside the full harness. |
| `recsys_sequence_ranking_common.py` | Shared synthetic workload generation, deterministic model init, and Triton kernel helpers. |
| `expectations_b200.json` | Strict harness expectation entry generated from the current B200 validation run. |
| `__init__.py` | Lab package marker. |

## Running the Benchmarks
Use the harness path for reproducible artifacts and verification.
```bash
python -m cli.aisp bench list-targets --chapter labs/recsys_sequence_ranking
python -m cli.aisp bench run --targets labs/recsys_sequence_ranking --profile minimal
python -m labs.recsys_sequence_ranking.compare_sequence_ranking --iterations 10 --json
```
- Targets follow the `labs/recsys_sequence_ranking:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/recsys_sequence_ranking:sequence_ranking="--flag value"` to sweep batch size, candidate count, table count, or Zipf skew.
- `--target-extra-arg labs/recsys_sequence_ranking:sequence_ranking="--disable-compile"` keeps the optimized path vectorized but eager.
- `--target-extra-arg labs/recsys_sequence_ranking:sequence_ranking="--score-backend torch"` is useful when you want to isolate `torch.compile` from the Triton kernel contribution.
- `python -m labs.recsys_sequence_ranking.compare_sequence_ranking ...` is a local smoke/perf helper, not a replacement for harness-backed benchmark artifacts.

## Validation Checklist
- `python -m cli.aisp bench list-targets --chapter labs/recsys_sequence_ranking` discovers `sequence_ranking`.
- `python -m cli.aisp bench run --targets labs/recsys_sequence_ranking --profile minimal` keeps baseline and optimized outputs verification-clean on the same synthetic batch.
- `python -m labs.recsys_sequence_ranking.compare_sequence_ranking --iterations 10` reports a small max-abs-diff between the two paths before any speedup claim is trusted.
- `expectations_b200.json` stays aligned with the latest strict B200 run when the workload shape or implementation changes.
- Treat the current virtualized strict run as regression evidence; re-run on bare metal before publishing absolute latency numbers externally.

## Notes
- The workload is intentionally synthetic and weight-light. The point is to benchmark sparse ranking mechanics, not offline recommendation quality.
- This lab keeps ETL, feature stores, and training out of the timed path on purpose. If those become the story, they belong in a separate workflow/playbook lab.
- When Triton is available, the optimized path uses fused Triton kernels for sequence pooling and candidate scoring. The `--score-backend torch` override changes candidate scoring only; sequence pooling may still use its Triton kernel.
