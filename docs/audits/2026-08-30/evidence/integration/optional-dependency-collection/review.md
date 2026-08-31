# Optional dependency collection repair

The five assigned test modules now collect 46 cases normally: **41 passed, five CUDA skips**. Before the change all five modules failed collection because their imports required unavailable Triton. No command-level exclusions or blanket module skips were used.

Six production files now defer actual kernel/config imports until requested. CPU schedules, route packing, reference construction, benchmark factories, and minimum-speedup contracts still run through their ordinary imports. The public occupancy names and legacy `FULL_STACK_AUTOTUNE_CONFIGS` access are preserved. Missing Triton raises its actual dependency error when an execution API is requested; there is no alternate backend.

Three test files changed. The two other assigned files pass unchanged. All original test-function IDs remain. New CPU subprocess controls verify that ordinary metadata/packing/factory work leaves kernel modules unloaded; real missing-dependency negative controls verify no fake success. The FlashAttention CUDA tests import lazily, and the optimized test now uses the real attention resolver/kernel instead of a substitute. Their buffer checks and added full-output reference check remain unexecuted CUDA gates here.

The occupancy `--list` and grouped-GEMM `--help` commands pass on the actual CPU environment. Static AST checks preserve every existing grouped-GEMM kernel/helper body and parameter list and all six real autotune constructor expressions. Python syntax compilation and scoped diff checks pass. Parent and peer source reviews found no further actionable concern.

The original files, failed collection, commands, normal pytest/JUnit results, source-equivalence checks and source hashes are retained beside this review. This is a local CPU/source receipt, not pinned-stack, CUDA, sanitizer or performance qualification. The parent owns final full-suite integration; the separately owned hygiene file remains a separate integration epoch.
