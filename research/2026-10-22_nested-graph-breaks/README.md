# Nested graph breaks on B200

This package preserves the experiment behind the B200 `nested_graph_breaks`
comparison. The original worker, analyzer, and capture helpers are unchanged.
It is a standalone research package with a fixed 24-worker matrix. Both arms
execute the same worker, selected with `--enabled 0` or `--enabled 1`.

The retained experiment measured one flag within PyTorch 2.14.1+cu130. It did
not compare PyTorch versions, model throughput, serving latency, or GPU scaling.

The separate [bounded B200 follow-up](followup/README.md) adds observed
cold-to-steady crossovers, phase memory and host RSS, a CUDA-Graph-disabled
ablation, and the corrected retained-trace reader. Its diagnostic results do
not replace the historical qualification below. The original source, results,
and figures remain unchanged.

[Download the actual Nsys, NCU, SQLite, CSV, and Chrome captures](profiles/README.md).
Native reports are retained byte for byte and indexed by cohort, arm, and hash.

## Results and limits

One NVIDIA B200 executed the workload while a second B200 stayed idle. The
runtime used Python 3.12.3, PyTorch 2.14.1+cu130, Triton 3.8.0, and driver
580.173.02. The workload used 1,048,576 FP32 elements and Inductor
`reduce-overhead` with `fullgraph=False`.

Here, **support off** means `nested_graph_breaks=False`, and **support on**
means `nested_graph_breaks=True`. The workload keeps the same graph breaks and
the same 64 operations in both arms.

| Depth-8 metric | Support off | Support on | Comparison |
| --- | ---: | ---: | ---: |
| Synchronized steady median | 407.567 µs | 89.935 µs | 4.531795× |
| First ten compiled-call intervals | 2,411.177 ms | 3,211.677 ms | Support on takes 33.2% longer |
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
longer with support on. Calls 2–10 together save about 19 ms. Larger generated fused
programs are the leading explanation. Exact compiler-phase times were not
recorded, and the no-break control's call-1 excursion remains unexplained.

## Independent fresh repeat

Fresh run 022 repeated the same 24-cell compiler microbenchmark on B200. Its
depth-8 steady geometric-mean speedup was **4.541758×**, with individual seed
ratios from 4.438828× to 4.653092×. All 24 correctness records passed. The
steady measurement, control, and capture gates passed. The startup measurement
still failed its fixed rule, although this repeat's startup control passed.

The original **4.531795×** result remains the primary number above. Fresh run
022 is an independent repeat. It does not replace the original records or turn
the failed startup route into a pass. Its retained native Nsys and NCU reports
provide supporting mechanism evidence. Normal unprofiled workers, not the
profiled executions, establish the latency result.

The original measurements used the standalone protocol and PyTorch Chrome
profiles. They are not a newly qualified result from the repository harness.
The retained telemetry spans varying SM clocks, and the original run does not
claim successful NCU or Nsys captures. Fresh run 022 has its own qualification
and native captures under `profiles/fresh022/`. Do not attach the historical
4.53× figure to a new run or merge the two cohorts.

## Inspect the evidence

- [All 24 result records and qualification](results/README.md)
- [Interactive support-off/support-on comparison](graphs/comparison.html)
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
In this execution description, False means nested graph-break support off and
True means support on.

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

## Run a small example

Use the recorded PyTorch 2.14.1+cu130 environment on a B200. The worker compares
the same depth-8 workload with nested graph-break support off and on. Each
process uses fresh compiler caches. The two-arm example below is a quick
reproduction, not the full qualification matrix.

From this directory, choose your B200 UUID and a new output directory:

```bash
python -c "import torch; assert str(torch.__version__) == '2.14.1+cu130', torch.__version__; assert torch.version.cuda == '13.0'"
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-YOUR-B200-UUID
EXAMPLE_ROOT="$(mktemp -d "$PWD/example-run-XXXXXXXX")"
for support in 0 1; do
  env TORCHINDUCTOR_CACHE_DIR="$EXAMPLE_ROOT/$support/inductor" \
      TRITON_CACHE_DIR="$EXAMPLE_ROOT/$support/triton" \
      timeout --kill-after=30s 240s python benchmark.py \
      --case nested_break_depth8 --seed 0 --enabled "$support" \
      --expected-gpu-uuid "$CUDA_VISIBLE_DEVICES" \
      --out "$EXAMPLE_ROOT/$support/result"
done
```

In this command, `0` means without nested graph-break support and `1` means
with support. The worker retains correctness, runtime identity, raw timing
samples, and generated-code counters in each `result.json`. Completed examples
and captures already in the repository remain separate evidence. For Zymtrace,
see the bounded controller and capture requirements in [zymtrace/](zymtrace/).
The [October 26 two-arm B200 capture](zymtrace/captures/torch2141-b200-20261026/)
passed on PyTorch 2.14.1 and includes downloadable Zymtrace event and
flamegraph JSON. It is a separate compatibility smoke, not another speedup
measurement. The current [Zymtrace flamegraphs](zymtrace/views/) weight width
by percentage of captured GPU time after filtering each worker. They replace
the earlier record-count diagrams. Their retained API definition, normalization,
source hashes, renderer, and full weighted stacks are linked beside the images.

## Posts and replies

The [conversation chronology](social/CHRONOLOGY.md) separates the original
October 21 posts, the October 25 self-replies, and the unsent reader responses.
It identifies each recipient, the exact parent, and the Typefully copy bank
or native LinkedIn reply composer. The latest Zymtrace update is a separate
unscheduled draft.
