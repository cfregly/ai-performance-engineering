# moe_cuda router_vectorized: expert-capacity right-sizing (Front M4)

## Verdict

**WIN — 1.3199x median (0.31006 -> 0.23491 ms), 6/6 reps verified per arm,
clean GPU snapshots every rep, loud overflow guard demonstrated.** Capacity
640 -> 320 (60% -> 20% padding) on the lab input; semantics unchanged on any
non-overflowing input, and overflow FAILS the run loudly (never silent drops).
Default flipped ON post-A/B by the banking session; kill switch
AISP_MOE_CUDA_CAPACITY_RIGHTSIZE=0 restores B56.

Final interleaved A/B (6 reps/arm, GPU3 flock, /tmp/frontM4/ab/):
b56 0.31006 ms [0.30830-0.31259] vs rs 0.23491 ms [0.23379-0.23940],
all verified, all exit 0, speedup_median 1.3199. Post-flip validation
(banking session): bare default 396.7x vs baseline (~0.238 ms, right-size
live); kill switch 313.3x (~0.301 ms, B56 restored). Guard demo: forced
capacity=256 -> BENCHMARK FAILED, "capacity overflow guard tripped...822
assignment-pass(es) routed beyond static capacity" (/tmp/frontM4/r04b_*).

## Padding-fraction measurement (the lever's premise, verified)

`/tmp/frontM4/r02_padding_probe.stdout` (lab `setup()` path, seed-42 input,
batch 4096 x top-2 = 8192 assignments over 32 experts):

- Per-expert load: min 214 / mean 256 / max 309.
- Incumbent capacity (B56 and earlier): 2x max load rounded up to 64 = **640**
  -> dense dispatch `[32, 640, 1024]` = 20480 GEMM rows for 8192 live
  assignments = **60% padding**.
- Right-sized capacity: max load rounded up to 64 = **320** -> 10240 rows =
  20% padding; both expert GEMMs and the GELU/dispatch pointwise work halve
  their M dim.

## Approach + correctness argument

(placeholder — filled at completion)

## Evidence

(placeholder — filled at completion)
