# Public result evidence

This directory contains public projections of all 24 retained B200 result
records. The projections preserve timing samples, counters, flags, generated
workload code and hashes, correctness evidence, profile summaries, statuses,
and verdict inputs. Operational identifiers and private paths use explicit
placeholders.

## Decision boundaries

- The fixed analyzer returns `PASS` for a qualified steady invocation win.
- The startup route remains `FAIL`. It fails in all four primary seeds, and
  the no-break startup control also fails at seed 1.
- The historical all-gates content condition withheld the planned Whiteboard
  content at experiment closure.
- The current explicit authorization accepts the disclosed startup tradeoff
  and permits this exact research package. It does not change any gate or turn
  the startup failure into a pass.

## Files

- [`analyzer-result.json`](analyzer-result.json) is the full analyzer output
  over the public projections. Only its local result paths are projected to
  repository-relative paths.
- [`analyzer-report.md`](analyzer-report.md) is the analyzer's rendered report.
- [`diagnosis.json`](diagnosis.json) separates call 1 from calls 2 through 10
  for every case and seed.
- [`qualification.json`](qualification.json) records the scientific verdict,
  historical content decision, current publication authorization, and claim
  limits.
- [`publication-manifest.json`](publication-manifest.json) records the original
  and public SHA-256 values plus every changed field path. It does not disclose
  the original private values.

Each `<case>-seed<seed>-flag<arm>/result.json` file is one projected retained
worker record.

From the archive root, rerun the standard-library analyzer with:

```bash
python analyze.py results --out /tmp/nested-graphs-analysis.json \
  --markdown /tmp/nested-graphs-analysis.md
```

These are historical standalone compiler-microbenchmark measurements with
PyTorch Chrome profile evidence. They do not establish a new repository-harness,
NCU, Nsys, serving, model-quality, or multi-GPU qualification.
