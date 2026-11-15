# Retained operator replay

Capture complete tensor values from a caller or an explicit input recipe, then
replay each call through the existing baseline and optimized benchmark pair.
The allowlist contains inclusive int32 prefix scan, FP32 online softmax, and
BF16-to-NVFP4 quantize, add/RMSNorm and SiLU/multiply. The replay does not accept
Python code, module names, kernel source, or commands from a trace.

This adds repeatable input identity and complete correctness checks. It costs
disk space for every input and output, setup time to validate and transfer those
values, and the normal benchmark and profiler work for each event. Use it when
actual values, masks, scales or ragged shapes matter to a diagnosis. A direct
chapter benchmark is enough when its generated workload already answers the
question.

## Capture and validate

Run from the repository root:

```bash
PYTHONPATH=code python -m cli.aisp tools trace-replay -- capture \
  --plan code/labs/trace_replay/examples/capture-plan.json \
  --output-dir artifacts/operator-trace-001 --device cpu

PYTHONPATH=code python -m cli.aisp tools trace-replay -- validate \
  --bundle artifacts/operator-trace-001
```

The example creates illustrative inputs. Its source remains
`generated_by_recipe` after capture. CPU capture executes the existing reference
functions and records `capture_backend: cpu_reference`. It does not execute or
qualify the optimized CUDA kernels. `--device cuda` executes each native baseline
and compares its full result with the reference before retaining the outputs.
NVFP4 native execution requires SM100. CUDA capture is also separate from the
paired performance run.

For observed caller values, use the capture API at the call site:

```python
from labs.trace_replay.runner import capture_event

bundle = capture_event(
    "online_softmax",
    {"rows": x.shape[0], "cols": x.shape[1]},
    {"x": x},
    event_id="softmax-call-001",
    source={
        "kind": "observed_call",
        "description": "Input values observed immediately before this softmax call",
        "caller": "my_application.normalize",
    },
    output_dir="artifacts/observed-softmax-001",
    device="cpu",
)
```

The API retains the supplied values before executing a reference or baseline.
Completed observed calls require contiguous input tensors. The API rejects a
strided view before copying it, so capture cannot silently change its layout.
Describe their actual source. An input from a generated application test remains
test data. A profiler timeline by itself does not contain these values. Capturing
the complete values requires copying them out of the caller, which can perturb
that application. Capture does not measure application latency.

For multiple observed calls, pass an ordered iterable of dictionaries to
`capture_events(calls, output_dir, device="cpu")`. Each dictionary contains
`event_id`, `operation`, `parameters`, `inputs` and `source`, using the same values
as the single-call API. Inputs and metadata are copied before advancing to the
next call, so a generator can reuse its buffers. An optional `status` can be
`completed`, `failed` or `cancelled`. Failed and cancelled source calls must include `error` and can pass
an empty input dictionary when inputs were unavailable. Their places in the
roster remain visible, and the capture does not rerun them or replace their state
with a successful reference call. Available values from failed and cancelled
calls remain diagnostic records, including values supplied through strided views.

Bundles contain `manifest.json`, `manifest.sha256`, and raw contiguous tensor
files. Every descriptor records shape, dtype, byte order, byte count and SHA-256.
BF16 bytes are preserved without converting through a NumPy BF16 dtype. NVFP4
outputs retain separate packed bytes, every scale byte including padding, and
the BF16 residual when present. No checksums or selected output slices substitute
for those tensors. Validation rejects missing, altered, incompatible or unsafe
members. An existing output directory is never replaced.

## Replay on a CUDA host

```bash
PYTHONPATH=code python -m cli.aisp tools trace-replay -- replay \
  --bundle artifacts/operator-trace-001 \
  --output-dir artifacts/operator-replay-001
```

Replay defaults to the standard strict validity profile. On an explicitly
authorized virtualized host, select `--validity-profile portable` and retain that
limitation with the results. This does not remove the CUDA or SM100 requirements.

Each event launches the ordinary paired CLI with both profiler paths enabled:

```bash
PYTHONPATH=code python -m cli.aisp bench run \
  --targets labs/trace_replay:trace_replay \
  --target-extra-arg 'labs/trace_replay:trace_replay=--bundle /absolute/path/operator-trace-001 --event softmax-ragged' \
  --profile minimal
```

The pair requires an explicit bundle and event. It does not silently generate a
default trace. The same selection reaches baseline, optimized, verification and
profiler workers. Each operation uses its existing native benchmark function.
The standard harness supplies timing, clocks, environment checks, complete paired
verification, live input jitter, profiler artifacts and rejection reasons. The
adapter checks every timed output against the reference after timing. The paired
runner also compares the actual baseline and optimized timing-run outputs.
When the live inputs still match the retained bytes, the adapter additionally
compares every output with the captured full result. Jittered inputs use the
current-input reference, so a stale captured output cannot pass that check.
Retained inputs are independent of the harness seed. Live jitter changes the same
input storage read by the operator, including a batch view of the scan vector.

Setup, file validation, input copies, compilation and reference checks are outside
the steady-state timer. Excluded load/transfer, native setup and full correctness
durations appear as custom metrics. Disk usage and capture time remain in the
capture bundle. No new command deadlines or sample limits are imposed by this
tool. Existing harness policies apply.
The CLI suite watchdog is disabled for this ordered replay. Event benchmark and
profiler policies remain those of the standard harness.
Cancellation forwards SIGINT to the owned CLI process group and waits for its
benchmark and profiler cleanup before recording a cancelled event. It does not
kill only the CLI while a detached capture continues running.

## Evidence and limits

`replay.json` joins the immutable bundle digest to separate per-event harness
results, source hashes, commands, logs and receipts. Failed and cancelled capture
events retain their source state and error. They cannot become successful replay
events. Each replay failure gets a distinct retained receipt. One failed, skipped
or rejected event makes the aggregate replay fail. A measured result below the
repository speedup threshold remains the harness's failed result.

The tool dispatches independent operations in recorded order and keeps one event
resident at a time. It does not recreate stream overlap, dependencies between
different calls, allocator history, request arrival times, queueing or serving
latency. The sum of isolated kernel times is not an end-to-end trace latency.
Request arrival replay remains in [serving_comparison](../serving_comparison/README.md).

The input recipe, caller source, capture runtime and capture source hashes remain
separate from the replay runtime and profiling evidence. Replay rejects changed
implementation source instead of transferring an old result to new code. CPU
reference diagnostics and tests cannot establish GPU correctness or a speedup.

One-column softmax and quantize or SiLU/multiply calls with every row masked
remain valid capture inputs. Their outputs are constant under the required input jitter, so
the current replay benchmark reports them as unsupported by that verification
method. It does not bypass the check or call those events successful. Missing
hardware and other native skips remain explicit per-event states, and required
skips prevent aggregate completion.

Run focused checks with:

```bash
PYTHONPATH=code python -m pytest code/tests/test_trace_replay.py -q
```

CUDA-gated tests use real native kernels and complete outputs when the required
hardware exists. CPU tests exercise raw bytes, strict input schemas, actual
reference capture, CLI capture/validation, full typed output contracts, immutable
failure receipts, default verification and a cached-output jitter negative control.
