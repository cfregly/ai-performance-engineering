# Nested graph-break B200 microbenchmark

Verdict: **PASS** (steady_invocation)

CUDA Graph capture qualification: **PASS**

No-break steady control: **PASS**

No-break cold first-10 control: **FAIL**

This is a controlled compiler microbenchmark. Invocation p95 is not a serving latency percentile.

| Case | Seed | Cold first-10 speedup | Steady median speedup | Candidate/baseline p95 |
|---|---:|---:|---:|---:|
| no_break_depth4 | 0 | 1.0033x | 1.0101x | 0.9962x |
| no_break_depth4 | 1 | 0.9424x | 1.0015x | 1.0056x |
| no_break_depth4 | 2 | 0.9960x | 1.0046x | 1.0152x |
| no_break_depth4 | 3 | 0.9943x | 1.0048x | 0.9516x |
| nested_break_depth4 | 0 | 0.9844x | 2.8223x | 0.3700x |
| nested_break_depth4 | 1 | 0.9605x | 2.9429x | 0.3480x |
| nested_break_depth4 | 2 | 0.9803x | 2.8604x | 0.3595x |
| nested_break_depth4 | 3 | 0.9668x | 2.8505x | 0.3627x |
| nested_break_depth8 | 0 | 0.7511x | 4.5226x | 0.2244x |
| nested_break_depth8 | 1 | 0.7368x | 4.5795x | 0.2212x |
| nested_break_depth8 | 2 | 0.7595x | 4.5302x | 0.2271x |
| nested_break_depth8 | 3 | 0.7558x | 4.4954x | 0.2278x |

The performance decision uses only the predeclared `nested_break_depth8` case. The `no_break_depth4` case is the predeclared drift-control gate; the other depth-4 result is descriptive.

Raw result paths and compiler counters remain in the JSON analysis and source result directories.
