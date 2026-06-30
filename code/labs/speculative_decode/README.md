# Lab - Speculative Decoding

## Summary
Accelerates autoregressive generation by letting a smaller draft model propose multiple tokens that the larger target model verifies in parallel.

## Problem
Speculative decoding only pays off when the draft model is accurate enough that verification overhead is amortized. This lab keeps that tradeoff explicit instead of treating speculation as a guaranteed win.

## Baseline Path
- target-only greedy decode
- simple reference for latency and correctness
- no draft-model parallelism

## Optimized Path
- small draft model proposes multiple tokens per round
- target model verifies the draft batch in parallel
- rejection/correction logic preserves exactness of the target path

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `speculative_decode` | `105.903 ms` | `34.399 ms` | `3.08x` |

That result is why the lab matters: speculation is only interesting when the acceptance rate is high enough to beat the verification cost, and this benchmark pair makes that visible on a deterministic toy-model setup.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/speculative_decode:speculative_decode --profile deep_dive --single-gpu
```

The profiler view is useful here because it shows whether the runtime really shifted work into fewer target-model verification steps instead of just moving cost around.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/speculative_decode
python -m cli.aisp bench run --targets labs/speculative_decode:speculative_decode --profile minimal
```

## Learning Goals
- Measure how draft length and acceptance rate combine into real speedup on a deterministic workload.
- Keep the draft/target comparison exact enough that verification still means something.
- Demonstrate when speculative decoding is helpful and when it is not.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_speculative_decode.py` | Target-only greedy decode baseline. |
| `optimized_speculative_decode.py` | Draft proposals plus batched target verification. |
| `baseline_speculative_decode_trusted.py`, `optimized_speculative_decode_trusted.py` | Trusted-draft serving variant that compares verified speculative decode with direct draft-token placement. |
| `baseline_speculative_decode_transition_table.py`, `optimized_speculative_decode_transition_table.py` | Token-local transition cache variant that removes draft-model calls from repeated trusted decode. |
| `speculative_decode_common.py` | Toy-model helpers and workload setup used by both paths. |
| `expectations_{hardware_key}.json` | Regression thresholds for the benchmark pair. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/speculative_decode
python -m cli.aisp bench run --targets labs/speculative_decode --profile minimal
```
- Targets follow the `labs/speculative_decode:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/speculative_decode:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/speculative_decode:speculative_decode --profile minimal` should show lower end-to-end decode latency for the optimized path.
- The optimized path should remain verification-clean against the target-only reference path.

## Notes
- The lab uses a token-local `TokenMLP` so the benchmark stays deterministic and focused on speculative-decoding mechanics instead of model-download/setup noise.
- This is a good lab for studying acceptance-rate sensitivity before trying the same idea in a full serving stack.
