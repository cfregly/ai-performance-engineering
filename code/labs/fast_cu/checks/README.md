# Copied-source correctness checks

These tests execute the corrected source in `../upstream`.
Run from `code/labs/fast_cu` with CUDA 13.0+.
The B200 runs below check correctness, not comparative performance.
They cover fixes to the older H100 reduction and scheduler examples. The newer
GB300 NVFP4 r0-r9 kernels are unchanged and are not exercised here.

```bash
mkdir -p /tmp/fast-cu-checks
nvcc -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_100a,code=sm_100a -Iupstream/h100 checks/reduction_reset.cu -o /tmp/fast-cu-checks/reset
nvcc -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_100a,code=sm_100a -Iupstream/h100 checks/reduction_failure.cu -o /tmp/fast-cu-checks/failure
/tmp/fast-cu-checks/reset
```

Each negative case below must return exit code 1:

```bash
/tmp/fast-cu-checks/failure kernel-mismatch
/tmp/fast-cu-checks/failure cub-mismatch
/tmp/fast-cu-checks/failure invalid-length
/tmp/fast-cu-checks/failure invalid-kernel
```

Compile the scheduler test for its actual SM90a target, then run only its host
schedule construction. This launches no H100 GPU kernel and can run on a B200
host. Repeat with `-O3 -DNDEBUG` to check the release build.

```bash
nvcc -std=c++17 -O0 -g --expt-relaxed-constexpr --expt-extended-lambda -gencode=arch=compute_90a,code=sm_90a -Iupstream checks/hilbert_schedule.cu -o /tmp/fast-cu-checks/scheduler -lcuda
CUDA_VISIBLE_DEVICES= /tmp/fast-cu-checks/scheduler
```

The reduction test checks repeated launches with poisoned outputs against an
int64 oracle. The scheduler test checks coverage, duplicate tiles, queue
capacity, and sentinels across six shapes for both headers.

## Full B200 NVFP4 GEMM

Run the port's opt-in correctness tests from `code/`:

```bash
AISP_RUN_FAST_CU_NVFP4_SM100_GPU_TEST=1 python -m pytest tests/test_fast_cu_nvfp4_sm100.py -q
AISP_RUN_FAST_CU_NVFP4_SM100_FULL_GPU_TEST=1 python -m pytest tests/test_fast_cu_nvfp4_sm100.py -q -k full
```

After the profiled harness run, collect five alternating timing rounds on an idle
B200. This uses identical resident inputs, CUDA events, and the harness clock lock.
It records raw round times and restores the entry application clocks and persistence
mode. Reserve the GPU before running it.

```bash
PYTHONPATH=. python labs/fast_cu/checks/measure_nvfp4_sm100.py --output /tmp/nvfp4-sm100-paired.json
```

See [B200 validation](../sm100_validation.md) for the measured results and limits.
