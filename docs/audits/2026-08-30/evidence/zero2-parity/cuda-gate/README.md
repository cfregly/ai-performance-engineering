# Common ZeRO workload: independent review and pending CUDA gate

Status: source review and host failure controls passed. Actual two-GPU NCCL/BF16 training is **HOLD**. This host has PyTorch 2.8.0 without CUDA; neither the GPU workers nor any CUDA numerical workload ran. Six host tests passed and the real CUDA test skipped. Both standalone invocations returned exit 3 with a structured HOLD: one without opt-in, one with execution requested but CUDA unavailable. Their complete commands and outputs are retained.

## Source review

The four wrappers delegate model construction, argument parsing and production training to `zero2_common.py`. The baseline single-GPU wrapper initially omitted an explicit `--grad-accum 1` present in its optimized peer; the owner corrected that literal argument discrepancy. No further actionable mismatch was found in the reviewed common path:

- Seven linear layers with six GELUs, FP32 model parameters and BF16 CUDA autocast. The optional synthetic communication parameter is BF16 in both variants.
- Same AdamW options (learning rate, betas 0.9/0.95, weight decay 0.05, fused when supported), rank-specific persistent input buffers, microbatch accumulation and global norm clipping at 1.
- One actual warmup update in both variants; barrier and CUDA synchronization before the same measured loop, final synchronization before stopping the timer, host loss conversion after timing.
- The optimized path changes gradient communication to reduce-scatter/all-gather and partitions optimizer-state ownership. It reconstructs full gradients and performs an explicit optimizer update after backward; it does not retain sharded gradients or overlap the optimizer step.

The independent GPU reference explicitly averages both ranks' losses before clipping, consistent with [PyTorch 2.9 DDP's documented gradient averaging](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html). Optimizer-state ownership is checked against the [ZeRO optimizer contract](https://docs.pytorch.org/docs/2.9/distributed.optim.html), and the reference uses the same [CUDA BF16 autocast](https://docs.pytorch.org/docs/2.9/amp.html) precision context. This is a numerical acceptance design, not runtime evidence.

## Prepared numerical gate

The owned test is `code/tests/test_audit_wave1_zero2_parity_cuda.py`. It requires explicit opt-in, NCCL and exactly two visible allocated CUDA GPUs with native BF16 support. There is no CPU substitute. It executes the actual common `build_training_components` and `training_step`, not a duplicated candidate training loop.

Each rank runs both dense-DDP and RS/AG + sharded AdamW variants, with and without a seven-element BF16 communication payload. The common model has hidden size 17 (2,142 FP32 scalar parameters), batch size 5, accumulation 2, learning rate 0.01 and four consecutive updates (one warmup-equivalent plus three further updates). The optional payload creates an odd-length BF16 bucket; its gradients remain zero as in production, so this is not a nonzero-gradient padding stress test. Rank seeds differ and are replayed identically across variants and by the reference.

An independent dense AdamW reference on each rank uses both ranks' microbatch inputs, averages their losses, and implements global L2 clipping independently. A large final bias forces actual clipping. For all 32 rank/update combinations, the gate saves full candidate/reference parameters, gradients, loss, AdamW state and final microbatch inputs before numerical assertions. It compares every element and both variants, checks unchanged input-buffer addresses, requires actual parameter changes and optimizer step progression, compares full gradients/parameters across rank replicas, and verifies exactly one state owner per parameter for ZeRO versus complete state on both baseline ranks.

Tolerances are explicit in the source: parameters rtol 1e-5/atol 1e-6; gradients 3e-4/3e-6; loss 1e-5/1e-5; optimizer moments 5e-4/3e-7. Replica and input equality are exact. Host controls reject a changed final parameter element, changed final gradient element, NaN loss, and missing parameter. Missing resources produce HOLD with no passed checks. Worker/progress reports and full tensor artifacts are preserved on numerical failure. Distributed operations have a 30-second timeout; the parent has a 180-second polling deadline and terminates/kills remaining workers during cleanup. New attempt directories must not already exist.

Run only after two GPUs have been allocated, from `code/` in the intended environment:

```bash
CUDA_VISIBLE_DEVICES=<two-allocated-device-indices> python tests/test_audit_wave1_zero2_parity_cuda.py --output-dir ../docs/audits/2026-08-30/evidence/zero2-parity/cuda-gate/actual-two-gpu-attempt-1 --execute
```

Alternatively, the opt-in pytest entry point is `AISP_RUN_ZERO2_PARITY_CUDA=1 python -m pytest -q -p no:cacheprovider tests/test_audit_wave1_zero2_parity_cuda.py`; use the standalone command to retain a durable acceptance report. Future receipts record each rank's device model/capability, software version and source hashes separately. No performance claim or cross-hardware qualification is made.

## Separate acceptance gap

`TorchrunScriptBenchmark` builds an unrelated, single linear layer in `setup()` (line 79), evaluates it under inference mode (lines 94-101), and captures that toy output before launching the actual training script (lines 118-127 and 192-193). Its signature dimensions derive from a label hash, not the real training model. Thus a green wrapper signature check cannot validate the real script's loss, gradients, parameters, optimizer updates or workload equivalence. This gate does not change that wrapper, and cannot close its integration gap. The owner must track it separately.

Still pending: actual two-GPU execution, pinned-stack qualification, default hidden-size 10000 / 12 GiB payload scale, torch.compile mode, full CLI/harness training-result integration, GPU sanitizer validation and performance/memory claims. Warmup/timing parity above is source review only; the standalone numerical gate does not measure throughput or exercise the four CLI main functions.
