# P09 implementation and validation handoff

The 17 files listed in `peak-metrics-source-files.json` implement W1-018, W1-024,
W1-053, W1-056, W1-058, W1-094, W1-108 and the assigned LOCAL-009 architecture
metadata correction. Source and CPU validation passed; GPU acceptance remains HOLD.

The final focused run passed **102 tests, skipped seven CUDA cases, and failed
none in 4.30 seconds**. The new test file accounts for 52 CPU passes and five GPU
skips. Existing metric, peak-import, and microbenchmark tests supply the remaining
50 passes and two skips. The host uses Python 3.12.2 and PyTorch 2.8.0 without CUDA;
it is not the pinned GPU stack. Ruff's selected error rules and scoped diff checks
passed. The real peak CLI exited 1 with an explicit CUDA-unavailable message and
did not produce a peak result.

Original source reproduction demonstrates the HBM read/write factor, decimal
peer-copy conversion, inverted roofline classification caused by wrong B200
peaks, an elapsed-time-dependent purported memory ceiling, acceptance of legacy
HBM artifacts, the SM121-to-SM120 rewrite, and unsupported direct CPU float8 random
generation. The corrected code fixes those mechanisms. Initial, intermediate,
and final validation attempts remain in this directory. The first B200 spec
regression failed against the original constants before they were changed.

HBM targets now require consistent versioned byte/timing metadata. Legacy JSON
files are preserved and rejected for target derivation. The default overall HBM
policy is expressed in corrected read-plus-write units; chapter targets with
unestablished provenance are not blindly doubled. Exact observed SKU and compute
capability select known static profiles. Unknown CUDA SKUs require explicit
specifications, and the CPU fallback is visibly labeled an assumed B200 profile.

The historical GB300 reports retain their original bytes except for added front
notices; hashes verify this. A separate correction document uses primary NVIDIA
SKU tables and sparsity footnotes. The audit's proposed second halving of already
dense H100 rates was rejected. The external `sol-ceilings.yaml` referenced by the
reports is absent from this checkout and was not changed.

Root's independent final source review found no further actionable concern in
the reviewed scope. Earlier review findings—substring B200 matching GB200 and
events recorded on the wrong caller device—were corrected. PyTorch 2.9.1 source
confirms cross-device copies execute on the source current stream; the preliminary
destination-stream hypothesis was corrected before implementation. Cache-sized
copy metadata does not claim verified L2 residency, and peer-copy metadata does
not claim verified NVLink transport.

Remaining gates include the actual Transformer Engine NVFP4 recipe/kernel,
complete FP8 output comparison, two-GPU noncurrent-device timing and peer transfer,
full FP4 numerical calibration, low-precision kernel inspection, cache/link
profiling, and representative repeated measurements under the required runtime
controls. Recipe availability and green CPU arithmetic cannot close these gates.
Architecture metadata and removal of retargeting also do not prove toolchain
support on SM120/121. No GPU lease, compiler, sanitizer, or CUDA execution was used.

No ledger, root plan, Git state, original performance artifact, or other agent's
source was changed by this slice.
