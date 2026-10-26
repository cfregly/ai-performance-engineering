# Zymtrace GPU-time flamegraphs

The current public views use **percentage of captured GPU time** for width.
They are derived from the retained October 26, 2026 B200 / PyTorch 2.14.1
exports. They are not screenshots of the Zymtrace interface.

- [Without nested graph-break support](time-weighted/zymtrace-without-nested-graph-breaks.png)
- [With nested graph-break support](time-weighted/zymtrace-with-nested-graph-breaks.png)
- [Full stacks without support](time-weighted/zymtrace-without-nested-graph-breaks.full-stacks.svg)
- [Full stacks with support](time-weighted/zymtrace-with-nested-graph-breaks.full-stacks.svg)
- [Renderer](render_time_profiles.py), [tests](test_render_time_profiles.py),
  [source hashes](provenance.json), and [profile semantics](profile-semantics.json)
- [Verified weight meaning and displayed units](units/authority.json)

The official Zymtrace API describes `JsonEvent.c` as “Time spent in this event.”
The CUDA event filter selects GPU activity. For each worker, the renderer sums
that time weight through its ordered launch stacks and divides by the worker's
sum. The displayed unit is percent of captured GPU time. A common base-time
unit cancels in this ratio. No conversion to seconds or milliseconds is made,
because the public JSON schema does not specify that base unit.

This follows the usual time-weighted flamegraph convention. A long GPU event
contributes proportionally more width than a short event. Each image fills
its own 100% width. Horizontal position groups stack paths and is not chronology.
The raw time weights and normalized fractions are retained in the sidecars.

Worker filtering accepts 104 rows from PID 632510 without support and 188
from PID 632770 with support. The second export also contains 104 rows from
the prior worker. Those rows are excluded by PID before computing any weight.
GPU UUID, CUDA kind, stream presence, and positive finite time weights are
also checked. The native aggregate export is preserved and checked for
arithmetic consistency. It is not used as an unfiltered candidate profile.

These captures include setup, warmup, timing, and correctness. They do not
isolate the steady-state latency cohort. GPU activity can overlap, and GPU
time attributed to Python/native frames is the descendant GPU work those
frames launched. It is not host CPU execution time. The charts do not establish
a CPU/GPU causal percentage or a speedup. No additional sampling extrapolation
is applied. The retained Nsys timeline and separate unprofiled timing records
remain the corresponding timing evidence.

The main image focuses on workload and CUDA mechanism frames while retaining
every accepted event's time weight. The full-stack SVG and JSON retain every
ordered frame. SVG tooltips give inclusive GPU-time percentages and full names.

Regenerate without a GPU or network connection:

```sh
python3 render_profiles.py
python3 -m unittest test_render_time_profiles
```

Python 3 and the existing `rsvg-convert` executable are required. Both current
PNGs are 1800 × 1120, with 400-pixel preview exports. The package gate rejects
count weights, unverified weight meanings, incorrect worker/GPU filters,
source hash mismatches, invalid weights, and mismatched normalized totals.

## Superseded record-count views

The older `zymtrace-*.png`, SVG, and stack JSON files directly in this directory
are retained as historical stack-visibility diagnostics. Their widths count
records. They are superseded for public performance posts by `time-weighted/`.
Their original metadata is retained in [record-count-provenance.json](record-count-provenance.json).
Completed captures and benchmark results have not been changed.
