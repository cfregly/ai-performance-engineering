# Zymtrace CUDA event stack views

These charts render actual retained event stacks from the two October 26,
2026 PyTorch 2.14.1 captures on the NVIDIA B200. They are derived charts,
not screenshots of the Zymtrace interface.

- [Without nested graph-break support](zymtrace-without-nested-graph-breaks.png)
- [With nested graph-break support](zymtrace-with-nested-graph-breaks.png)
- [Input and output hashes, filters, and limits](provenance.json)
- [Renderer](render_profiles.py)
- [Without-support raw event export](../captures/torch2141-b200-20261026/disabled/zymtrace-export/cuda-events.raw.json)
- [With-support raw event export](../captures/torch2141-b200-20261026/enabled/zymtrace-export/cuda-events.raw.json)
- [Without-support raw aggregate flamegraph](../captures/torch2141-b200-20261026/disabled/zymtrace-export/cuda-flamegraph.raw.json)
- [With-support raw aggregate flamegraph](../captures/torch2141-b200-20261026/enabled/zymtrace-export/cuda-flamegraph.raw.json)

Each main image uses a prefix tree of the ordered frame IDs in the retained
`cuda-events.raw.json` export. Each accepted event row contributes one unit
of width. Width does not represent duration, CPU overhead, or a count of all
kernel launches. The export's `c` field is used only for the same positive,
finite validity check as the capture controller. Its units are not inferred.

The main images show every accepted event record with a focused selection of
its frames. They retain CUDA and CUDA-memory frames and function names containing
`ngb_nested_break`, `CUDAGraph`, `CUDAWarmupNode`, `cuLaunchKernel`, or
`cuGraphLaunch`. Other frames are omitted from that projection. Narrow
labels are truncated to fit their bars. Full labels remain in SVG tooltips.
The original ordered frames, including omitted frames, are retained in each
`.stacks.json` sidecar and its zoomable `.full-stacks.svg` view.

The without-support view accepts 104 records from PID 632510. The with-support
view accepts 188 records from PID 632770 and excludes the 104 records from
the prior worker that are present in its raw time-window export. The original
raw flamegraph export cannot be PID-filtered from its aggregate nodes. It is
preserved as a hashed input, while these derived views use the event rows
that retain PID and ordered stack information.

The filter also checks the audited GPU UUID, CUDA kind, stream presence, and
positive finite `c`. Record totals must match the corresponding capture
audit. The source exports cover setup, warmup, timing, and correctness work.
They do not isolate a steady-state timing cohort. The images establish
observed stack visibility, not a speedup, causal overhead split, or timing
comparison. No timeline is reconstructed from the second-resolution `t`
timestamps.

Regenerate with Python 3 and the existing `rsvg-convert` executable:

```sh
python3 render_profiles.py
```

The renderer makes no network or GPU calls and does not modify the captures.
The two main PNG files are 1800 × 1120 pixels.
Each `.preview.png` is a 400-pixel-wide layout preview. Use the main image
or zoomable SVG to read individual function labels.
