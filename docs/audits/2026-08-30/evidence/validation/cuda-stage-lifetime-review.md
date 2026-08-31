# CUDA stage lifetime repair — source-only HOLD

W1-002 and W1-026 are repaired in source, with the inseparable multicast protocol
changes from W1-068. Compilation, target execution, and sanitizer checks have not
run. No CUDA compiler, CUDA device, compute-sanitizer, or candidate CUTLASS headers
are available on this host. The machine-readable receipt records exact hashes.

Seven kernels now wait on each reused stage's prior MMA-completion barrier and
advance its empty parity. First fills do not wait for nonexistent consumers.
Only warp 0 advances pipeline phases; other warps wait at the final CTA
synchronization. The legacy no-MMA-barrier experiment now commits both per-stage
completion and final accumulator completion.

The ordinary cluster kernel still uses independent TMA loads; its header no
longer claims multicast. Logical `grid_m` is kept separate from padded launch
geometry. Padded CTAs execute the same pipeline but never store their output
tile. The multicast epilogue no longer reads the unused beta-zero C tensor, which
also avoids out-of-bounds reads by a padded CTA. This guard relies on the existing
M-multiple-of-128 and N-multiple-of-256 contract; it does not support partial
element tiles.

The experimental multicast kernel publishes barrier initialization before
cluster synchronization. Each empty barrier expects both consumer commits. Each
CTA supplies a distinct B slice to both CTAs, so receiving the full B tile also
proves both producers have crossed their prior stage-empty waits. Expected bytes
come from the complete logical A and B storage tensors, not a multiplied
partition size. Final remote commits are drained and all CTAs synchronize before
shared-memory lifetime ends.

The independent peer review confirmed the local parity/padded-store logic and
identified the need for the cooperative B dependency. The final revised review
status is recorded in `cuda-stage-lifetime-receipt.json`.

PTX permits negative transaction counts and completes a phase only after both
arrival and transaction counts reach zero. It also requires a successful phase
wait before a subsequent-phase arrival. These contracts, plus CuTe's
`arrive.expect_tx` helper, justify the distinction between permitted early
transaction completion and the additional consumer/producer dependency above.
[PTX mbarrier specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier),
[CuTe barrier helpers](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/copy_sm90_desc.hpp),
[CUTLASS multicast commit helper](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/barrier.h).

After obtaining the target's serialized GPU ownership, run from `code/` with the
pinned target environment, using fresh output paths for every attempt:

```sh
timeout 1800 python -m labs.custom_vs_cublas.verify_tcgen05_pipeline --output /tmp/tcgen05-full-output-attempt-001.json
PYTORCH_NO_CUDA_MEMORY_CACHING=1 timeout 1800 compute-sanitizer --tool memcheck --error-exitcode=1 python -m labs.custom_vs_cublas.verify_tcgen05_pipeline --output /tmp/tcgen05-memcheck-attempt-001.json
PYTORCH_NO_CUDA_MEMORY_CACHING=1 timeout 1800 compute-sanitizer --tool synccheck --error-exitcode=1 python -m labs.custom_vs_cublas.verify_tcgen05_pipeline --output /tmp/tcgen05-synccheck-attempt-001.json
```

Capture stdout/stderr and sanitizer exit status beside each receipt. JIT compile
failures remain failures. The runner records source and extension hashes and
rejects source changes during execution. It compares every element using the
repository FP16 tolerance across seven variants, thirty shapes, two seeds, and
four repetitions: 1,680 cases. Short K loops cover unconsumed stages, long K loops
cover repeated phase wraps, and odd M-tile counts exercise padded CTAs.

The CPU probe returned exit 1 with zero executed checks and an explicit CUDA
requirement. This verifies refusal to manufacture results; it is not a kernel
correctness or GPU acceptance result. Original performance claims require fresh
measurement only after actual correctness and memory-safety gates pass.
