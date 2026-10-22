# Nested graph breaks on B200

This package preserves the experiment behind the B200 `nested_graph_breaks`
comparison. The original worker, analyzer, and capture helpers are unchanged.
It is a standalone research package with a fixed 24-worker matrix. Both arms
execute the same worker, selected with `--enabled 0` or `--enabled 1`.

The retained experiment measured one flag within PyTorch 2.14.1+cu130. It did
not compare PyTorch versions, model throughput, serving latency, or GPU scaling.

## Results and limits

One NVIDIA B200 executed the workload while a second B200 stayed idle. The
runtime used Python 3.12.3, PyTorch 2.14.1+cu130, Triton 3.8.0, and driver
580.173.02. The workload used 1,048,576 FP32 elements and Inductor
`reduce-overhead` with `fullgraph=False`.

| Depth-8 metric | False | True | Comparison |
| --- | ---: | ---: | ---: |
| Synchronized steady median | 407.567 µs | 89.935 µs | 4.531795× |
| First ten compiled-call intervals | 2,411.177 ms | 3,211.677 ms | True takes 33.2% longer |
| Emitted regions, seed 0 | 16 × 4 ATen operations | 2 × 32 ATen operations | Same 64 operations |
| CUDA Graph replay launches per profiled invocation | 16 | 2 | Five untimed invocations per arm |

Latency values are geometric means across four paired seeds. Each worker's
steady value is the median of 100 synchronized calls after 25 warmups. The
four primary speedups range from 4.495352× to 4.579454×. All 48 retained
correctness checkpoints passed with zero recorded error.

The steady-state route and its no-break control passed. The startup route
failed, including one no-break startup-control miss. Those original gates
remain unchanged. The analyzer's predeclared steady-or-startup decision selects
the passing steady route. Acceptance of a disclosed startup cost does not make
the startup route pass.

The first-ten metric sums ten individually synchronized invocation intervals.
It excludes process startup, imports, input creation, and eager-reference work.
Correctness work occurs between calls 1 and 2, so this is not continuous startup
wall time. The entire net increase came from call 1, which takes 786–876 ms
longer with True. Calls 2–10 together save about 19 ms. Larger generated fused
programs are the leading explanation. Exact compiler-phase times were not
recorded, and the no-break control's call-1 excursion remains unexplained.

These historical measurements used the standalone protocol and PyTorch Chrome
profiles. They are not a newly qualified result from the repository harness.
The retained telemetry spans varying SM clocks, and this package does not
assert locked application clocks or successful NCU/Nsys captures for that run.
New harness measurements must satisfy the current harness gates and carry
their own provenance. Do not attach the historical 4.53× figure to a new run.

## Inspect the evidence

- [All 24 result records and qualification](results/README.md)
- [Interactive False/True comparison](graphs/comparison.html)
- [Region overview](graphs/comparison-overview.svg)
- [Extracted operator graphs](graphs/compiled-graphs.json)
- [Frozen protocol with operational identifiers redacted](protocol.json)
- [Source hashes](source-manifest.json)

GitHub displays the source of the interactive HTML. To use its selectors and
zoom controls, serve this directory locally and open `graphs/comparison.html`:

```bash
python -m http.server 8000 --directory research/2026-10-22_nested-graph-breaks
```

The diagrams show emitted Inductor wrapper regions recovered from the original
depth-8, seed-0 caches after the run. They are not complete Dynamo or CUDA Graph
topologies. The replay timelines supply launch evidence, not latency acceptance
measurements. Public evidence copies redact operational identities and paths.
Their manifests record original and public hashes. Measurement arrays and
scientific verdicts remain unchanged.

## Re-analyze the retained measurements

The analyzer needs only the Python standard library. From this directory:

```bash
python analyze.py results --out /tmp/nested-graphs-analysis.json \
  --markdown /tmp/nested-graphs-analysis.md
python -m unittest test_analyze test_counter_capture
```

Run `python -m unittest test_analyze test_counter_capture test_publication` for
the full package check. It validates all 24 public records, source and graph
hashes, original route verdicts, local viewer links, and rejection of corrupted
evidence. It does not initialize CUDA.

The package is separate from the chapter benchmark harness. Naming two wrappers
`baseline_` and `optimized_` would imply that the historical timing came from
that harness. The preserved worker instead records both cold first use and
synchronized steady execution under its fixed protocol. A future harness pair
would need its own hardware validation and measurements.

## Reproduce the original worker

`benchmark.py` is the executed worker, not the site's allocation controller.
It requires a B200, one explicitly selected GPU UUID, fresh cache directories,
and an output directory. It performs real CUDA work. The full protocol uses
three cases, four seeds, and two arms, with a fresh process for each of 24
workers. Seeds 0 and 2 run False then True. Seeds 1 and 3 reverse that order.

For one worker, after acquiring exclusive hardware access and validating the
runtime and current environment requirements:

```bash
export CUDA_VISIBLE_DEVICES="<GPU-UUID>"
export TORCHINDUCTOR_CACHE_DIR="$(mktemp -d)"
export TRITON_CACHE_DIR="$(mktemp -d)"
python benchmark.py --case nested_break_depth8 --enabled 0 --seed 0 \
  --expected-gpu-uuid "$CUDA_VISIBLE_DEVICES" --out /tmp/nested-graphs-false-seed0
```

Repeat with fresh processes, caches, and output paths for every declared cell.
Use `--profile` on both depth-8 seed-0 arms to collect the separate untimed
Chrome profiles required by the historical analyzer. The worker does not
acquire a GPU allocation, lock clocks, collect NCU/Nsys, enforce controller
timeouts, or produce the controller's environment receipts. A direct worker
invocation alone is therefore insufficient for a fully qualified new run.

## Rebuild the visual comparison

Rendering requires Matplotlib and the Graphviz `dot` executable on `PATH`.
From this directory, run:

```bash
python graphs/plot_startup_decomposition.py
python graphs/build_comparison.py
```

The renderer verifies all 18 public wrapper excerpts before generating the
viewer and region diagrams. It reads retained data and does not execute the
workload. Font and renderer versions can change the exported bytes. Any such
change requires a new visual review and updated public asset hashes.

`pytorch-main-dynamo-config.py` is the retained runtime source snapshot used
for provenance. It is not imported as a replacement for installed PyTorch.
PyTorch source is distributed under its [BSD-style license](PYTORCH-LICENSE).
