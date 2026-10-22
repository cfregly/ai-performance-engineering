# B200 arithmetic-intensity verification

This lab separates analytical counts from live GPU observations. Its diagnostic tools
characterize bandwidth and compute ceilings. Its `bf16_gemm` and `native_copy` benchmark
pairs use the repository harness for correctness, profiling, and speedup gates.

The analytical model reads the reviewed single-GPU B200 profile from
`core.benchmark.metrics.BLACKWELL_B200`. It fails if that profile no longer contains
2,250 dense BF16 TFLOP/s and 8,000 decimal GB/s. Those values produce a nominal ridge
point of 281.25 FLOP/byte. They are published specifications, not measurements from the
active GPU.

Run the CPU-only checks from `code/`:

```bash
python -m labs.arithmetic_intensity_b200 formulas --output <artifact-dir>/formulas.json
```

Live commands and harness benchmark pairs use a lab-specific isolated runtime. They
require exact readback of PyTorch 2.13.0+cu130 and the CUDA 13.0 version reported by
PyTorch. This lab's qualification target remains independently pinned. The repository-wide
default is PyTorch 2.14.1+cu130. Retained PyTorch 2.11.0+cu130 results are historical and do
not qualify the current source. The frozen cuBLASLt algorithm must report as supported and
pass full-output correctness on 2.13.0. Reselect it through the bounded search if that check
fails. The lab does not use Triton kernels. An installed Triton version is recorded in
provenance but does not gate a run. Retained harness evidence used Triton 3.6.0, so a prior
3.7.1 pin was not a valid requirement. Use a separate environment for live lab commands.

Every live measurement command requires a named B200 at compute capability 10.0, BF16
support, the strict environment checks, and the repository clock-lock context. The supplied
source commit must equal the live Git `HEAD`, and the complete repository must have no
tracked or untracked changes. There is no portable mode or precision fallback. JSON report
paths are immutable and must not already exist. Each command creates its report with an
exclusive write and rejects a path that appears during execution. Acquire the operator's
existing GPU lease or queue before launching it, and expose only the assigned device. For
example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m labs.arithmetic_intensity_b200 gpu \
  --run-id <run-id> \
  --source-commit <git-commit> \
  --phase all \
  --warmups 5 \
  --repeats 10 \
  --copy-size-mib 4096 \
  --rotation-mib 512 \
  --output <artifact-dir>/measurement.json
```

The copy uses seeded uniform random bytes and a working set much larger than L2. Input
generation occurs outside timing. Effective bandwidth counts one requested read plus one
requested write. The GEMM cases rotate equal-value weight buffers across a requested 512
MiB working set. This reduces warm-L2 reuse while preserving the same mathematical
output. The tool validates the full copy output and each measured GEMM's full BF16 output
against an FP32 reference after timing.

Logical requested bytes remain separate from physical HBM traffic. Every JSON result
leaves `physical_hbm_counter_bytes` null. Collect `dram__bytes_read.sum` and
`dram__bytes_write.sum` with Nsight Compute before placing GEMM points on a measured HBM
roofline. A warm-cache timing divided by the logical byte model is not measured HBM
bandwidth.

Use the repository metric set for NCU and the repository trace modules for Nsight
Systems. Keep artifacts outside the public checkout. The direct `profile.sh` NCU path is
not suitable when unrelated profiler work may exist because that path currently performs
a broad profiler cleanup. A scoped NCU invocation avoids touching foreign work:

```bash
NCU_METRICS=$(PYTHONPATH=. python -c \
  'from core.scripts.harness.metrics_config import BASE_NCU_METRICS; print(",".join(BASE_NCU_METRICS))')

ncu --set full --metrics "$NCU_METRICS" --nvtx \
  --nvtx-include 'ai_b200_gemm_m1_k4096_n4096/' \
  --force-overwrite -o <artifact-dir>/ncu-gemm-m1 \
  python -m labs.arithmetic_intensity_b200 gpu \
  --run-id <run-id>-ncu-m1 --source-commit <git-commit> \
  --phase gemm-m1 --warmups 1 --repeats 1 \
  --output <artifact-dir>/ncu-gemm-m1.json

nsys profile --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --output=<artifact-dir>/nsys-all \
  python -m labs.arithmetic_intensity_b200 gpu \
  --run-id <run-id>-nsys --source-commit <git-commit> \
  --phase all --warmups 1 --repeats 3 \
  --output <artifact-dir>/nsys-all.json
```

Repeat the scoped NCU command with `gemm-m64`, `gemm-large`, and `copy` as needed. The
NVTX range names are `ai_b200_gemm_m64_k4096_n4096`,
`ai_b200_gemm_m8192_k8192_n8192`, and `ai_b200_hbm_copy` for the default shapes.

The `<artifact-dir>` values in this document are portable placeholders. They do not name
published or repository-owned evidence. The tool passes `allow_virtualization=False` and
never selects portable compatibility.
The repository strict validator currently records virtualization as a nonfatal strict
notice, so measurement can continue and the JSON labels the result
`strict_noncanonical_virtualized`. Bare-metal execution, successful NCU and Nsight
Systems artifacts, stable application-clock telemetry, and operator review remain gates
for publication-grade evidence. The tool does not parse profiler artifacts or record the
operator review, so it always emits `publication_eligible=false`.

## Sustained ceiling characterization

`ceiling_tool` runs a fixed, bounded search for the highest sustained ceilings observed
on the assigned B200 under this diagnostic protocol. It is separate from the earlier
representative-shape check above. The default search covers 1, 2, 4, and 8 GiB HBM
payloads, copy/read/write kernel variants, and square BF16 GEMMs at 4096, 8192, 12288,
and 16384 in four physical operand layouts.

```bash
CUDA_VISIBLE_DEVICES=0 python -m labs.arithmetic_intensity_b200.ceiling_tool \
  characterize \
  --run-id <run-id> \
  --source-commit <git-commit> \
  --output <artifact-dir>/characterization.json
```

The screen uses three warmups and seven samples. It retains two copy finalists, one read
finalist, one write finalist, and two BF16 GEMM finalists. Each finalist then runs in five
independent windows. Every window uses fresh seeded data, repository clock preconditioning,
and enough batched invocations to record at least one second of GPU-active CUDA-event time.
The output preserves aggregate timings, normalized per-operation timings, raw samples,
window medians, a deterministic 95 percent bootstrap interval for the median, telemetry,
and every rejection reason.

The HBM inputs are seeded uniform random bytes created outside timing. Copy validation
compares the full output tensor. Read validation compares every fixed-chunk reduction
output against an independent PyTorch reference that covers the full input. Write
validation checks every generated output word. Read and write rates are supporting
diagnostics. Only copy's requested read plus requested write rate supplies the measured
protocol roof. The output labels it `effective requested read plus write bandwidth`.

Before every timed `torch.mm` path, the lab saves the process-wide BF16 reduction setting,
disables reduced-precision BF16 reductions, reads the disabled setting back, and restores
the prior value after the run. Each GEMM uses BF16 Tensor Core dispatch through
`torch.mm(..., out=...)` and checks the full BF16 output against an FP32 matrix
multiplication of the exact BF16 operands. The layout sweep changes physical strides while
preserving logical values. A finalist must pass correctness, the one-second window gate,
and the configured variability gate.

Profile only the selected winners after characterization. Pass the exact candidate fields
from the JSON so the NVTX range and requested-byte accounting remain reproducible:

```bash
python -m labs.arithmetic_intensity_b200.ceiling_tool profile \
  --run-id <run-id>-ncu-copy \
  --source-commit <git-commit> \
  --profile-target bandwidth \
  --bandwidth-operation copy \
  --bandwidth-strategy native_uint4 \
  --profile-payload-mib <payload-mib> \
  --profile-threads <threads> \
  --profile-blocks-per-sm <blocks-per-sm> \
  --output <artifact-dir>/ncu-copy-replay.json

python -m labs.arithmetic_intensity_b200.ceiling_tool profile \
  --run-id <run-id>-nsys-gemm \
  --source-commit <git-commit> \
  --profile-target gemm \
  --profile-gemm-size <size> \
  --profile-gemm-layout <layout> \
  --output <artifact-dir>/nsys-gemm-replay.json
```

Use `ai_b200_peak_<candidate-id>/` as the exact `--nvtx-include` filter for NCU.
The measured roof remains a requested-byte metric until exact-range NCU reports
`dram__bytes_read.sum` and `dram__bytes_write.sum`. Record their sum and its ratio to
requested bytes in the external evidence bundle. Requested-byte arithmetic intensity and
physical-counter arithmetic intensity are separate quantities. Neither the bounded BF16
ceiling nor the requested-byte bandwidth ceiling is a vendor theoretical peak.

## Kernel optimization experiments

`kernel_optimization_tool` performs a bounded implementation screen and a sustained
interleaved confirmation. It is a diagnostic and does not claim a speedup. The discoverable
`baseline_*` and `optimized_*` pairs below are the harness-comparable workloads used for
verification and profiler gates:

- `bf16_gemm`: Torch's default `torch.mm(..., out=...)` path versus a persistent direct
  cuBLASLt plan with an exact frozen supported algorithm.
- `native_copy`: the retained `uint4` grid-stride kernel versus a cache-global,
  four-way-unrolled `uint4` kernel with a frozen launch configuration.

The GEMM pair fixes a 16384-cubed row-major NN BF16 operation, disables reduced-precision
BF16 reductions to preserve FP32 accumulation, fixes beta zero,
and full output semantics. The copy pair fixes an 8 GiB seeded random `uint8` payload and
counts one requested source read plus one requested destination write. Setup allocates and
checks an independent reference result, poisons the private output buffer, and exposes the
public verification output only after the real benchmark path runs. After primary timing,
the capture hook first checks the entire last timed output, then poisons the buffer again,
invokes the implementation once, and checks the entire fresh result before registering the
verification payload. Allocation, reference work, descriptor construction, poisoning, and
this audit invocation remain outside the timed region.

Run the screen once, freeze its exact supported choices in `optimization_benchmarks.py`,
then confirm five fresh-operand AB/BA windows:

```bash
CUDA_VISIBLE_DEVICES=0 python -m labs.arithmetic_intensity_b200.kernel_optimization_tool \
  screen \
  --run-id <run-id>-screen \
  --source-commit <git-commit> \
  --copy-payload-mib 8192 \
  --output <artifact-dir>/screen.json

CUDA_VISIBLE_DEVICES=0 python -m labs.arithmetic_intensity_b200.kernel_optimization_tool \
  confirm \
  --run-id <run-id>-confirm \
  --source-commit <git-commit> \
  --screen <artifact-dir>/screen.json \
  --expected-screen-lab-tree-sha256 <screen-provenance-lab-tree-sha256> \
  --confirmation-windows 5 \
  --confirmation-repeats 15 \
  --minimum-window-gpu-ms 1000 \
  --output <artifact-dir>/confirmation.json
```

Dogfood the actual pairs with strict verification and all three minimal-preset profilers:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cli.aisp bench run \
  --targets labs/arithmetic_intensity_b200:bf16_gemm \
  --targets labs/arithmetic_intensity_b200:native_copy \
  --bench-root . \
  --profile minimal \
  --validity-profile strict \
  --verify-phase gate \
  --single-gpu \
  --run-id <run-id>
```

The 2026-10-14 bounded search measured all eight cuBLASLt heuristics returned for the fixed
GEMM and all nine declared native-copy launch candidates. Final five-window confirmation on
the clean PyTorch 2.11.0+cu130 stack found no clear improvement for either candidate. The
strict harness also rejected both candidates at its 1.05 speedup gate. Those outcomes retain
the existing baseline implementations as the selected paths. They are historical negative
results only. They do not qualify the current source or the current FP32-accumulation policy
because they predate the exact clean-`HEAD` gate and the frozen BF16 reduction setting. A
new run under the current gates is required before making a current-source claim. The 2026
run was virtualized and is labeled
`strict_noncanonical_virtualized`, with `publication_eligible=false`. The 8 TB/s denominator
is the book/DGX up-to nominal reference, while all copy rates here are requested read-plus-
write proxies rather than raw physical HBM bandwidth.

### Dense BF16 nominal-gap timeline

The `diagnose-nominal-gap` command measures sustained clocks, power, temperature, and
throttle state beside the unchanged 16384-cubed row-major NN BF16 workload. It keeps the
scaled-normal operands resident and checks every BF16 output element against an FP32
matmul of those exact operands. A NaN poison before each timed arm makes a stale or no-op
output fail the full-output check.

The command separates a short warm diagnostic from the sustained result. Before every
calibration arm and sustained window, it submits at least three seconds of continuous exact
GEMMs without inserting control work into the CUDA stream. On sampled arms, the observer
starts before settling so observer startup does not create an idle gap before timing. Eight
balanced OFF/ON and ON/OFF pairs then bound the combined overhead of
`query_gpu_telemetry(force_refresh=True)` and `PowerSampler`. The
observer calibration now uses eight pairs and accepts the observed run only when every
paired duration change stays inside the declared limit. Its bootstrap interval is
descriptive and is not an equivalence proof. Missing application clocks, missing power or
throttle fields, fewer than eight complete in-window queries, and midpoint gaps above 250
ms are hard errors. The gap from the window start to the first complete query and the gap
from the last complete query to the window end must also stay within the same bound.
PowerSampler must resolve to the exact CUDA device UUID. A host gap
above 20 ms between completed settling and the timed envelope is also a hard error.
The CLI rejects cadence, gap, and minimum-window combinations that cannot conservatively
fit the required complete queries.

Every sampled sustained window must pass the declared same-window clock and power
no-trend and p10-to-p90 span gate. If any window fails, the report keeps all raw timing and
telemetry for diagnosis, uses `diagnostic_unsettled_no_sustained_kpi`, and suppresses the
sustained throughput summary and nominal fraction.

```bash
CUDA_VISIBLE_DEVICES=0 python -m labs.arithmetic_intensity_b200.kernel_optimization_tool \
  diagnose-nominal-gap \
  --run-id <run-id>-nominal-gap \
  --source-commit <git-commit> \
  --fill-mode scaled_normal \
  --settling-minimum-gpu-ms 3000 \
  --calibration-rounds 8 \
  --sustained-windows 3 \
  --minimum-window-gpu-ms 1000 \
  --telemetry-interval-ms 100 \
  --maximum-telemetry-gap-ms 250 \
  --minimum-in-window-telemetry-samples 8 \
  --output <artifact-dir>/nominal-gap.json
```

The JSON preserves each NVML query interval, actual and application clocks, power,
temperature, throttle bitmask, query cadence, coarse PowerSampler data, and host-phase
energy. CUDA events still have no absolute host timestamp, so the report treats each query
as an interval inside a host envelope and does not claim exact point alignment. It reports
raw measured throughput against the fixed 2,250 TFLOP/s book nominal. It never scales that
nominal by observed clocks.

The exact-work settling range also has a conservative host start and end envelope. Raw
observer samples can therefore be classified as before, inside, or after settling without
claiming an exact CUDA-event timestamp.

`--fill-mode uniform_minus1_plus1`,
`--fill-mode uniform_rms_matched_to_scaled_normal`, and
`--fill-mode zero_power_switching_calibration` are optional fill-sensitivity diagnostics.
The JSON records full-tensor min, max, mean, and RMS summaries for both realized BF16
operands. Uniform[-1,1] has a much larger expected RMS than the scaled-normal primary, so
that case changes amplitude and distribution and cannot isolate distribution alone. The
RMS-matched uniform case changes the distribution while matching the primary's expected
pre-quantization RMS. The zero case tests switching-power sensitivity. None of these fill
cases is a representative primary workload, an accepted optimization, or a replacement for
the scaled-normal headline result.
Resolve the exact CUDA kernel name and Tensor Core counters later with an NVTX-filtered
Nsight Systems and Nsight Compute run. The timeline command leaves those fields unresolved
rather than transferring counters from an older run.

For the later exact-range NCU replay, first query metric support on the installed profiler.
Then collect `gpc__cycles_elapsed.avg.per_second` and the supported SM100 Tensor Core and
issue metrics with both `--clock-control none` and `--cache-control none`. Keep the raw metric
names, values, units, and clock definition in the evidence. Filter the settled range
`ai_b200_nominal_gap_steady_scaled_normal_window_0/` and limit capture to one launch so NCU
does not replay every GEMM in the sustained window. NCU replay timing is mechanism evidence,
not the sustained throughput KPI.

The report also preserves the CUDA `multiProcessorCount` and the independent native SM100
dense BF16 basis of 4,096 FP32-accumulating FMAs, or 8,192 FLOPs, per SM cycle. The exact
architectural model remains unset until the same NVTX range supplies a supported NCU cycle
rate and its clock definition. Sampled NVML clocks stay raw and separate from that exact
counter model.

### Settled cuBLASLt search and peak calibration

`sustained_optimization_tool` extends the frozen v4 timeline without changing its result.
Its search unions the supported heuristics returned with 0, 64 MiB, 512 MiB, and 2 GiB
workspace ceilings, requesting the API maximum of 128 results at each ceiling. This is a
bounded heuristic union, not exhaustive enumeration of every configurable cuBLASLt
algorithm. The report decodes the CUDA 13.0 cluster-shape enum so 1-CTA and 2-CTA entries
remain visible. It must find and remeasure all eight full-output-passing signatures from the
earlier screen. It then adds at most 16 new signatures, preferring 1-CTA and 2-CTA strata.
Search, confirmation, calibration, and replay bind the source tree, Python executable and
version, PyTorch/CUDA build, driver, exact cuBLAS and cuBLASLt runtime versions, full device
identity, and the resolved cuBLASLt algorithm metadata. A mismatch fails closed before a
frozen direct-Lt signature can transfer to the next phase.

Every baseline and candidate arm receives at least three seconds of continuous identical
GEMMs immediately before at least one second of CUDA-event timing. Two ranking pairs use
opposite orders and the exact same scaled-normal operands. This screen only ranks candidates.
It does not claim a speedup or accept a baseline replacement.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  labs.arithmetic_intensity_b200.sustained_optimization_tool search \
  --run-id <run-id>-settled-search \
  --source-commit <git-commit> \
  --workspace-mib 0,64,512,2048 \
  --max-heuristics 128 \
  --maximum-candidates 24 \
  --pair-rounds 2 \
  --settling-minimum-gpu-ms 3000 \
  --minimum-window-gpu-ms 1000 \
  --output <artifact-dir>/settled-search.json
```

Confirm one ranked signature with five fresh-operand AB/BA windows. Both arms use the v4
observer, exact three-second preconditioner, full 268,435,456-element oracle, and same-window
steady-state gate. A candidate must have a fully positive paired bootstrap interval and at
least a 5% median paired improvement before it can advance to the strict repository harness.
Confirmation alone never replaces the baseline. The observer's five-sample arms are sized
separately from the fifteen-sample primary windows so each arm still exceeds one second of
GPU work. If either primary arm has an unsettled window, cross-window summaries and the
paired result are suppressed while the raw windows remain available for diagnosis.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  labs.arithmetic_intensity_b200.sustained_optimization_tool confirm \
  --run-id <run-id>-settled-confirm \
  --source-commit <git-commit> \
  --search <artifact-dir>/settled-search.json \
  --expected-search-lab-tree-sha256 <exact-search-tree-sha256> \
  --candidate-signature '<exact-signature>' \
  --windows 5 \
  --settling-minimum-gpu-ms 3000 \
  --minimum-window-gpu-ms 1000 \
  --output <artifact-dir>/settled-confirm.json
```

The confirmation freezes the implementation used by the separate matrix-value calibration.
If the candidate does not clear the 5% gate, the retained Torch baseline is frozen instead.
Candidate selection uses scaled-normal data only. The calibration rotates two fully
preallocated operand and output sets, uses five balanced interleaved windows per fill, and
checks every timed output against its own full FP32 oracle. It includes scaled normal,
RMS-matched uniform, uniform[-1,1], and all-zero operands. Uniform[-1,1] changes both the
distribution and amplitude, while zero is a low-switching control whose FLOP rate remains
withheld until an exact profiler proves executed dense Tensor Core work.

Two additional low-entropy controls use exact BF16 values of plus or minus 1/16. Every A
row is a real allocated copy of one randomly oriented, balanced sign-pair K vector. B uses
balanced random-sign row pairs. The second control applies one common K permutation to A
columns and B rows, grouping the positive A positions before the negative positions. Their
full value histograms and full FP32 reference outputs must match exactly before timing. The
last fully checked timed outputs must also be bitwise equal and nonzero. This control changes
K ordering while preserving AB. It does not isolate operand switching from accumulator
activity or make an adjacent-word bit-entropy claim.

The five fill orders are a fixed five-row subset of a six-treatment Williams design. Every
fill occupies five distinct ordinal positions, and the two ordering controls occur in both
directions three versus two times. Their exact histograms, reference outputs, strides, and
allocation-alignment classes are checked before any observer or primary timing. Each fill
has its own short sizing diagnostic and eight balanced observer pairs. The tool refuses a
plan above 25 minutes of estimated observer plus primary GPU-active work, leaving five
minutes of the 30-minute phase for allocation, references, checks, and reporting. An
unsettled fill retains raw windows but emits no latency, throughput, power, or energy
aggregate.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  labs.arithmetic_intensity_b200.sustained_optimization_tool calibrate-peak \
  --run-id <run-id>-peak-calibration \
  --source-commit <git-commit> \
  --confirmation <artifact-dir>/settled-confirm.json \
  --expected-confirmation-lab-tree-sha256 <exact-confirm-tree-sha256> \
  --windows 5 \
  --buffer-sets 2 \
  --settling-minimum-gpu-ms 3000 \
  --minimum-window-gpu-ms 1000 \
  --output <artifact-dir>/peak-calibration.json
```

The calibration records clocks, power, temperature, limiter bits, and coarse board joules
per GEMM from the accepted host phase. Board energy is not exact per-kernel energy. Fill and
cache effects are diagnostics and cannot be promoted as an optimization.

Use `profile-replay` to reproduce one post-settling launch from the frozen calibration. Nsys
binds the API call to the actual kernel. NCU must first query metric support on the installed
version. The minimum same-pass set contains `gpc__cycles_elapsed.avg.per_second` and
`gpu__time_duration.sum`, plus a supported native SM100 dense-BF16 instruction metric for
the zero-work proof. Preserve exact names and units. Use `--clock-control none`,
`--cache-control none`, the emitted NVTX include string, and one launch. Profiler timing is
mechanism evidence and never replaces raw CUDA-event timing.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  labs.arithmetic_intensity_b200.sustained_optimization_tool profile-replay \
  --run-id <run-id>-profile-rms \
  --source-commit <git-commit> \
  --peak-calibration <artifact-dir>/peak-calibration.json \
  --expected-peak-lab-tree-sha256 <exact-peak-tree-sha256> \
  --fill-mode uniform_rms_matched_to_scaled_normal \
  --output <artifact-dir>/profile-rms.json

CUDA_VISIBLE_DEVICES=0 python -m \
  labs.arithmetic_intensity_b200.sustained_optimization_tool profile-replay \
  --run-id <run-id>-profile-zero \
  --source-commit <git-commit> \
  --peak-calibration <artifact-dir>/peak-calibration.json \
  --expected-peak-lab-tree-sha256 <exact-peak-tree-sha256> \
  --fill-mode zero_power_switching_calibration \
  --output <artifact-dir>/profile-zero.json
```

Refresh the requested-byte ridge with the same final source and runtime. This command uses
five fresh random 8 GiB sources, full equality, three seconds of exact Torch CUDA tensor-copy
settling, and at least one second of CUDA-event timing per window. Eight balanced observer
pairs use separate five-sample invocation sizing. Every primary window must pass complete
query cadence, memory-clock and power stability, stable application memory clock, P0, and
GPU-identity gates. Graphics clock is retained as a diagnostic rather than a copy acceptance
gate. An unqualified run keeps raw windows and withholds the cross-window requested-byte
summary and 8 TB/s nominal fraction. The read-plus-write rate remains a logical
requested-byte proxy rather than raw physical HBM traffic. Call it a copy-engine path only
after a new matching Nsys trace proves that identity.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  labs.arithmetic_intensity_b200.sustained_optimization_tool calibrate-copy \
  --run-id <run-id>-copy-refresh \
  --source-commit <git-commit> \
  --payload-mib 8192 \
  --windows 5 \
  --settling-minimum-gpu-ms 3000 \
  --minimum-window-gpu-ms 1000 \
  --output <artifact-dir>/copy-refresh.json
```

The finite 90-minute slot reserves 5 minutes for custody and source readback, 7 minutes for
the scaled-normal v4 timeline, 18 minutes for the 24-candidate library screen, 7 minutes for
five-window confirmation, 30 minutes for the interleaved fill calibration, 5 minutes for the
copy refresh, 13 minutes for exact Nsys and NCU captures, and 5 minutes for hashes and
postflight. Stop at each cap and preserve a negative or incomplete result rather than adding
late candidates.

### Explicit PyTorch BLAS preference

Use the GEMM-only lane to compare an explicit PyTorch BLAS preference under the same
screen, output check, clock control, and five-window sustained confirmation protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python -m labs.arithmetic_intensity_b200.ceiling_tool \
  gemm-backend \
  --run-id <run-id> \
  --source-commit <git-commit> \
  --blas-backend cublaslt \
  --output <artifact-dir>/cublaslt-characterization.json
```

The tool records the prior preference, checks the requested preference by readback, and
restores the prior value on exit. PyTorch documents this API as an experimental
preference, and its internal GEMM path may fall back. Preference readback is therefore
not dispatch proof. Capture a cuBLAS API trace or cuBLASLt log for the selected profile
replay, and retain the actual kernel name before attributing a result to cuBLASLt:

```bash
nsys profile --trace=cuda,nvtx,cublas,osrt --force-overwrite=true \
  --output=<artifact-dir>/nsys-cublaslt-winner \
  python -m labs.arithmetic_intensity_b200.ceiling_tool profile \
  --run-id <run-id>-nsys \
  --source-commit <git-commit> \
  --blas-backend cublaslt \
  --profile-target gemm \
  --profile-gemm-size <size> \
  --profile-gemm-layout <layout> \
  --profile-invocations-per-sample <sustained-count> \
  --output <artifact-dir>/nsys-cublaslt-winner.json
```
