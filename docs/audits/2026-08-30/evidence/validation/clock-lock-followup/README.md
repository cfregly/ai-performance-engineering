# Protection-test clock-lock follow-up

W1-040's remaining GPU clock-lock test previously treated some failed lock
attempts as passes. `before.json` executes the exact old exception handler from
its AST; it does not simulate a CUDA device or a successful lock.

The test now uses the real harness clock-lock context with explicit targets read
from NVML, observes application clocks before and after real CUDA work, and
checks the existing production 50 MHz contract. Recognized permission,
unsupported-operation, missing-tool and missing-NVML-API errors skip locally.
The existing Tier-1 expected-GPU environment contract requires a failure instead.
Generic lock errors, driver errors and observed-clock mismatches propagate.
Production harness/configuration code is unchanged.

The final CPU gate passed **319 tests with 96 explicit skips** (35 CUDA and
61 unsupported/obsolete cases). Of these passes, 42 are new exception-disposition
and diagnostic-comparison controls; none simulates successful GPU execution.
An actual pytest invocation with the Tier-1 expected-GPU contract on this CPU host
fails the clock-lock test as required. Both its failure report and the normal
CUDA skip remain recorded. The final files are `combined-final-v3.*` and
`attested-cpu-negative-final.*`; earlier attempts remain as history.

This repairs the original test-quality defects. It does not implement the
unsupported policies or qualify the skipped CUDA protections. See
`documentation-proposal.md` for exact remaining overclaim corrections in the
root-owned README/generator and factual AGENTS inventory. The prior frozen
protection-coverage receipt remains historical and unchanged.

NVIDIA documents nvidia-smi return codes 3/4/12/13 as unavailable operation,
permission, missing NVML library, or missing NVML function respectively:
[NVIDIA return values](https://docs.nvidia.com/deploy/nvidia-smi/index.html#return-value).
These specific errors never become successful protection evidence.
