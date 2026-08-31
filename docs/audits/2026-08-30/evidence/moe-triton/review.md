# W1-082 / W1-083: retire incomplete experiments; keep active math honest

The three incomplete experiments are retired. No replacement fused Triton FFN,
grouped GEMM, benchmark measurement, or speedup is claimed. Original finding IDs
are preserved; the root agent owns their final ledger disposition.

## Callers and retirement

The original `triton_fused_moe.py` kernel/wrapper was called only by its own
standalone benchmark. Its first projection covered one intermediate tile, its
store covered one output tile, and the intermediate slicing was unsupported.
The review's verifier also notes that compilation failure made the later
full-FFN throughput print unreachable. We did **not** observe a live inflated
measurement. The local original CLI fails even earlier because Triton is absent;
that dependency failure is preserved, not relabeled as a kernel compilation test.

The two raw experiments in `triton_kernels.py` have no external imports or
launches. Their missing intermediate tiles and group offsets were teaching-code
defects. `level4_triton.py` has a separate definition with the same grouped-kernel
name; it was not confused with this retired one or changed in this slice.

Exact reviewed b57e4c6a9 source copies are retained as `original-triton_fused_moe.py`
and `original-triton_kernels.py`; their hashes and equality with the reviewed
commit are in `original-provenance.json`. The active module no longer defines
the three raw kernels. Legacy `triton_fused_moe(...)` and benchmark entry points
raise `RetiredMoEKernelError`. The CLI exits 3, writes only an explanation to
stderr, and produces no measurement. It points to the separately named complete
sorted PyTorch expert path without silently substituting that backend.

Two existing hotpath test IDs now check these actual failures instead of
asserting allocation/timing spellings in a broken benchmark. All original
hotpath test function IDs remain. The shared timing hygiene test likewise checks
retirement rather than requiring a dead event call.

## Adjacent active-helper correction

The shared model does import the elementwise `fused_silu_mul` helper. Its original
raw-pointer wrapper allocated output with no autograd history and assumed
contiguous storage. Those are separate discoveries, not invented W1-082/083
sub-findings. Executing the original allocation AST on actual CPU tensors
confirms that the output has no gradient requirement; a real CPU transposed-view
address control demonstrates why flat storage order differs. These controls do
not execute or qualify the original Triton kernel.

The helper now validates shape, device, dtype, and strided tensor layout. It
explicitly selects `pytorch_autograd`, `pytorch_cpu`, `pytorch_no_triton`, or
`triton_inference`; the selection is a diagnostic, not proof of kernel execution.
CPU behavior and the existing no-Triton PyTorch fallback are preserved. Either
trainable input routes through differentiable PyTorch SiLU/multiply operations.
Eligible CUDA inference makes contiguous copies inside the call, so copies are
part of its work, and launches on the input device while restoring the caller's
device. Empty tensors launch no kernel. The elementwise kernel computes SiLU in
FP32 before rounding to the input dtype and multiplying, matching the staged
PyTorch operation more closely than rounding sigmoid prematurely.

The shared model imports the real availability flag and documents its CPU and
autograd paths. No other model math changed. A gradient-enabled execution must
not be presented as a Triton-fused measurement.

PyTorch documents the default allocation behavior in
[torch.empty_like](https://docs.pytorch.org/docs/2.9/generated/torch.empty_like.html).
The pointer-offset pattern is illustrated by the official
[Triton vector-add tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py).

## Evidence and limits

The stable ordinary pytest run has **27 passed, 32 skipped**. Eighteen new CPU
checks cover retirement, analytic activation derivatives for either/both inputs,
aliased-view gradients, invalid pair rejection, CPU/empty behavior, complete
shared-model outputs and all input/weight/projection gradients against an
independent per-token reference, and hidden/intermediate tail contributions plus
top-k combination in the explicitly named PyTorch reference. Nine existing
hotpath CPU tests pass. The two tail fixtures use H/I = 65/129 and 129/193, with
only the last intermediate column contributing and one empty expert; the retired
first-tile math cannot satisfy that reference.

Thirty new CUDA cases and two existing CUDA hotpath cases skip honestly. Normal
collection of the single shared hygiene test is blocked by its unrelated
occupancy-lab import of unavailable Triton. Its exact host-only AST function was
executed successfully with real dependencies and no fake GPU/Triton module;
`hygiene-isolated.json` records that narrower result. The complete hygiene module
is **not** reported as passing.

Retained attempts include the original two source-only hotpath passes, the first
retirement run, the unrelated collection error, one test-authoring error (missing
the shared model's required argument), its corrected runs, and an unused-import
error in the first ad-hoc host probe. No attempt was promoted into stronger
evidence than it provides. Python compilation, selected Ruff correctness checks,
and scoped whitespace checks pass.

`code/tests/cuda/run_moe_triton_validation.py` bounds each validation process
group, including compiler descendants. It requires two actual CUDA devices,
Triton, BF16 support, and Compute Sanitizer. It runs all 30 named GPU cases both
normally and under memcheck and rejects missing cases, skips, errors, failures,
timeouts, or missing XML. Those cases check FP32/FP16/BF16 activation outputs and
changed tails for contiguous, transposed, and sliced inputs; CUDA autograd;
noncurrent-device execution; and the separately identified PyTorch full-output
reference. The local preflight returned **HOLD / exit 3**, with no GPU execution.

On an authorized target, supply a new output directory:

```sh
python code/tests/cuda/run_moe_triton_validation.py --output-dir /absolute/new-receipt-dir
```

CUDA/Triton compilation, numeric accuracy, allocator/stream behavior, and memory
sanitizer acceptance remain **HOLD**. There is no fused full-FFN implementation
to qualify after retirement. No hardware lease, remote launch, package install,
historical measurement rewrite, or plan/ledger mutation occurred in this slice.
