# Bounded B200 follow-up

This directory contains the code and public evidence summary for the follow-up
questions on the nested-graph-break experiment. The new observations are
separate diagnostics. They do not replace the original 4.531795× qualified
steady-state result, its disclosed startup cost, or failed historical controls.

## Source and images

- [Maintained diagnostic controller and worker](nested_graphs_diagnostic.py)
- [Corrected retained-trace reader](retained_trace_reader.py)
- [Executed diagnostic source](recorded_source/nested_graphs_diagnostic.py)
- [Executed corrected reader](recorded_source/retained_trace_reader.py)
- [Public result summary](results/summary.json)
- [Metadata projection code](publish_summary.py) and [projection manifest](results/projection-manifest.json)
- [Source and evidence hashes](provenance.json)
- [Original structural diagram](../graphs/comparison-overview.png)
- [Original startup diagram](../graphs/startup-decomposition.png)
- [Original region summary](../graphs/region-summary.png)

The images are the original experiment's figures. They do not depict a new
measurement of the CUDA-Graph-disabled ablation.

## Observed execution

Both follow-up phases used NVIDIA B200, PyTorch 2.14.1+cu130, CUDA 13.0,
Python 3.12.3, and driver 580.173.02. The original configuration completed eight
fresh worker processes in 61.296598691958934 seconds. The separate no-CUDA-Graph
configuration completed eight new processes in 51.40245714318007 seconds.
All 16 worker records passed. Both allocations were released and their
cgroups drained and removed. No application-clock setting was changed.

Each configuration had four timing cells, two memory cells, and two untimed
profile cells. Timing used depth 8, two paired seeds with alternating arm
order, fresh processes and caches, and 4,000 synchronized calls. Memory and
profiling ran in separate processes so their observer work was outside timing.
The workload and correctness helper came from the unchanged original package.

These are observations for a synthetic compiler workload. They do not
establish representative training, model, serving, or cross-GPU performance.

## Longer runs and crossover

Continuous wall time begins immediately before the first compiled call. It
includes compilation, capture where enabled, synchronization, and inter-call
overhead. It excludes imports, input setup, and the eager reference.

| Variant | Seed | First and sustained continuous crossover |
| --- | ---: | ---: |
| Original | 0 | Call 1 |
| Original | 1 | Call 3,007 |
| No CUDA Graph replay | 0 | None within 4,000 calls |
| No CUDA Graph replay | 1 | None within 4,000 calls |

Call 1 means the candidate was already ahead at the first observation, not a
measured sub-call crossover. Seed 1's active-call-sum crossover was 3,014.
That metric is distinct from continuous wall time. The two seeds do not
establish a universal break-even threshold. Do not extrapolate a crossover
for the no-replay variant.

## Fixed steady-tail observations

The descriptive rule was the median of the last 1,000 raw synchronized-call
intervals for every timing cell. Both seeds and both configurations are shown.
There is no seed pooling, best-cell selection, clipping, or new pass gate.

| Variant | Seed | False median (ns) | True median (ns) | False / True |
| --- | ---: | ---: | ---: | ---: |
| Original | 0 | 405980.5 | 89129.0 | 4.554976494743574× |
| Original | 1 | 392255.5 | 89523.5 | 4.381592542740174× |
| No CUDA Graph replay | 0 | 177278.0 | 53899.5 | 3.289047208230132× |
| No CUDA Graph replay | 1 | 175962.5 | 53850.0 | 3.2676415970287835× |

CUDA Graph replay was not required for the observed tail advantage. This does
not identify the remaining causal mechanism or an additive CPU/GPU percentage.
Absolute medians were also lower with replay disabled in both arms in this
diagnostic. That does not establish a general best configuration.

## Memory and host RSS

One seed-0 memory pair per configuration was measured independently of timing.
For the original configuration:

| Metric | False | True |
| --- | ---: | ---: |
| Allocated after call 4,000 | 8 MiB | 8 MiB |
| Reserved after call 4,000 | 296 MiB | 168 MiB |
| Peak allocated during first compiled call | 170.5 MiB | 146.5 MiB |
| Host current RSS after call 4,000 | 1410.609375 MiB | 1416.65234375 MiB |

All 20 phase rows, including the no-replay pair, remain in the public summary.
Allocated, reserved, phase peaks, and process RSS are separate measurements.
They do not establish whole-device memory equivalence. No equivalence
tolerance was predeclared.

## Reader correction and causal limits

The executed diagnostic's old profile decomposition selected both CPU
`user_annotation` and same-name GPU `gpu_user_annotation` events. It therefore
reported ten markers for five invocations. That derived geometry is invalid.
The executed source is retained byte for byte for provenance, not recommended
as the current CLI. Use the maintained successor and corrected reader.
The maintained reader also returns failure for malformed or ambiguous event
attribution and unassigned kernels. Its CLI preserves the diagnostic output
while returning a nonzero status. That status correction was checked with
synthetic fixtures and does not change the retained B200 results.

The correction selects exactly five unique, nonoverlapping CPU markers on one
CPU thread. It excludes the five GPU annotation duplicates. Kernels bind first
through an unambiguous CUDA API correlation and otherwise through unique full
interval containment. It retains signed endpoint violations without clipping.

All five invocation records remain visible in every capture. Baseline geometry
is valid for 5/5. Candidate invocations 3, 4, and 5 in each configuration have
signed endpoint violations, leaving 2/5 eligible for geometry aggregates.
Each capture has two trailing CUDA API events outside CPU markers, no
unassigned kernels, and no ambiguous kernel assignments. Do not treat a
selected two-invocation candidate aggregate as whole-capture attribution.

Raw counts across five profile invocations were:

| Variant and arm | CUDA Graph replay events | Kernel events |
| --- | ---: | ---: |
| Original False | 80 | 160 |
| Original True | 10 | 20 |
| No replay False | 0 | 80 |
| No replay True | 0 | 10 |

Even valid trace geometry is descriptive. CPU submission, GPU execution,
waiting, and overlap do not form independent causal shares. Profiling adds
observer overhead. The traces and this ablation do not establish an exact
CPU/GPU causal percentage. See NVIDIA's explanation of
[overhead and latency](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/).

The corrected reader was tested on seven synthetic fixtures and then applied
to the four retained B200 captures. No timing or profiler capture was repeated
for the correction. Raw traces, invalid old outputs, corrected outputs, and
completed original benchmarks were preserved.

## Focused source checks

These checks use the Python standard library and invented parser fixtures.
They do not initialize Torch or CUDA and do not repeat a GPU measurement.

```bash
python -m unittest test_retained_trace_reader test_publish_summary test_diagnostic_profile_reader
python nested_graphs_diagnostic.py plan --horizon 4000
```

The plan must report `torch_imported: false`. The original package's existing
publication tests remain applicable to its unchanged benchmark and figures.

## Run a new diagnostic only with explicit hardware ownership

The controller does not allocate a GPU or acquire ownership. The executor must
first establish exclusive access, a pinned compatible runtime, one physical
GPU UUID, no competing work, and a new absolute output directory. The source
root must match all three original frozen worker hashes.

```bash
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-AUTHORIZED-UUID
python nested_graphs_diagnostic.py run \
  --original-root "$PWD/.." \
  --output-root /absolute/new/primary-output \
  --expected-gpu-uuid "$CUDA_VISIBLE_DEVICES" \
  --cases nested_break_depth8 --seeds 0 1 --diagnostic-seed 0 \
  --horizon 4000 --variants original \
  --child-timeout-seconds 45 --total-budget-seconds 170
```

Run only `--variants no-cudagraph` in another new output directory for the
separate ablation. Do not repeat the original cells to obtain that contrast.
The CLI bounds each child and total runtime, preserves partial results, and
cleans only its own temporary caches. External owner cleanup is still needed
if the controller is forcibly terminated.

The published summary copies B200-calculated values. It does not recompute
statistics or units. Its manifest records original and public hashes and each
metadata redaction. Operational paths, file-system identifiers, and CPU
process/thread identifiers are replaced. Measurement values and all exclusions
are retained. The complete raw archive has SHA-256
`79205f2ce58e3ca9a59e61587acd2bb04a7a7cce1832a40a13bc4039bc5b74fe`.
