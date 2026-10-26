# B200 Zymtrace capture on PyTorch 2.14.1

Both depth-8 example workers passed on NVIDIA B200 with PyTorch
`2.14.1+cu130`, CUDA `13.0`, and Zymtrace `26.10.1`. The controller ran from
20:21:10 to 20:21:53 UTC on October 26, 2026. This was a 43-second
compatibility and profiler visibility smoke.

| Worker | PID | Matching CUDA records | Events | Flamegraph |
| --- | ---: | ---: | --- | --- |
| Without nested graph-break support | 632510 | 104 | [Download JSON](disabled/zymtrace-export/cuda-events.raw.json) | [Download JSON](disabled/zymtrace-export/cuda-flamegraph.raw.json) |
| With nested graph-break support | 632770 | 188 | [Download JSON](enabled/zymtrace-export/cuda-events.raw.json) | [Download JSON](enabled/zymtrace-export/cuda-flamegraph.raw.json) |

Each worker used seed 0 and a fresh cache. Both passed correctness and runtime
checks. The capture source is pinned to
[`2a4e3fd00`](https://github.com/cfregly/ai-performance-engineering/tree/2a4e3fd00dadf2380a5c413af33bb46c4727fce2/research/2026-10-22_nested-graph-breaks/zymtrace).
See the [runnable example](../../../README.md#run-a-small-example) and
[capture runbook](../../README.md).

The raw backend responses are preserved byte for byte. The second export
contains 292 CUDA records because its time filter overlaps the end of the
first worker. Its audit identifies the 188 records matching PID 632770, the
assigned GPU, and non-null stream IDs. Use each worker's
`zymtrace-export/audit.json` when selecting records. The flamegraphs use the
same project and time filters and may include that overlap.

These sampled exports show that Zymtrace collected CUDA events from both
workers. They do not establish complete per-launch coverage, a new speedup,
or an exact CPU/GPU causal percentage. The published latency measurements
remain the separate unprofiled cohorts. No samples were added to those runs.

The [COMPLETE handback](HANDOFF-COMPLETE.json) records both workers, source
hashes, exports, and cleanup. Both owned worker process groups were empty,
the profiler exited after SIGINT, and the temporary profiling rule was
removed at config version 12. [Immediate cleanup readback](cleanup-readback.json)
showed both GPUs at 0 MiB and no compute processes. The custodian performs its
own fresh resource check before resuming PIQ.

[manifest.json](manifest.json) contains byte sizes, hashes, and the original
archive paths. The earlier rejected PyTorch 2.13 capture remains unchanged in
its separate directory.
