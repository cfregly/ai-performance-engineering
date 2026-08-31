# W1-040 / W1-093 documentation disposition proposal

This is a proposed text change for the root-owned README, generator and factual
coverage inventory in `code/AGENTS.md`; it does not edit those files or their
operational instructions. The original 95-row inventory is not 95 independently
implemented or validated protections.

Replace both unconditional “all 95 ... protected” sentences with:

> This threat inventory describes implemented checks, checks with limited scope,
> and unsupported policies. It is not a guarantee that every listed attack is
> detected. The protection-test disposition receipt records what each test
> actually exercises. Unsupported cases and missing CUDA runs are explicit skips,
> never passing coverage.

Replace `Status: OK` with a meaningful status per row. “Scoped check” denotes the
specified mechanism only; it does not claim a generic attack detector or GPU
qualification. The following concrete claims contradict the current production
interfaces and test dispositions and require these cell corrections:

| Inventory row(s) | Mechanism / status replacement |
| --- | --- |
| Output: Invalid Ground Truth | Golden-output caching and output comparison preserve/check the selected reference; they do not validate dataset labels. Scoped check. |
| Output: Uninitialized Memory | No uninitialized-memory provenance detector. Unsupported policy. |
| Workload: Train/Test Overlap; Evaluation: Test Data Leakage; Evaluation: Missing Holdout Sets | No dataset provenance, leakage, or holdout-enforcement interface. Unsupported policy. |
| Location: CPU Spillover | Wall/CUDA timing cross-validation can detect a timing discrepancy; it does not identify per-operation CPU placement. Scoped timing check; placement detector unsupported. |
| Location: Background Thread; Environment: Priority Elevation; Statistical: Background Process Noise | Subprocess execution does not prohibit benchmark-owned threads, lock OS priority, or isolate background CPU processes. Those policies are unsupported. |
| Memory: Fragmentation Effects | Allocator cleanup and memory-growth diagnostics; no fragmentation-equivalence guarantee. Scoped check. |
| Memory: Page Fault Timing; CUDA: Unified Memory Faults | No page-fault/managed-memory event detector. Unsupported policy. |
| Memory: Swap Interference | Environment validation detects enabled swap; it does not disable swap or lock memory. Scoped environment gate. |
| CUDA: Host Callback Escape; Workspace Pre-compute; Persistent Kernel; Undeclared Multi-GPU; Context Switch Overhead; Driver Overhead; Cooperative Launch Abuse; Dynamic Parallelism Hidden | No corresponding host-callback, cuBLAS-workspace, kernel-lifetime, undeclared-device, context-switch, driver-attribution, cooperative-launch, or device-side-launch inspector. Unsupported policy. |
| Compile: Mode Inconsistency; Inductor Asymmetry; Autotuning Variance | No general compiler-mode/backend parity or autotuning-variance guard. Unsupported policy. Existing model-mode tests exercise numerical output comparison only. |
| Compile: Guard Failure Hidden | Observed process-cumulative Dynamo graph counters with availability/source metadata; not resident cache size or a general compile-parity guarantee. Scoped check. |
| Distributed: Topology Mismatch | Comparison of declared topology fields; no ring-versus-tree algorithm field. Scoped signature check. |
| Distributed: Barrier Timing; Gradient Bucketing Mismatch; Async Gradient Timing | No rank-barrier timing, gradient-bucket parity, or asynchronous-gradient completion detector. Unsupported policy. |
| Distributed: Pipeline Bubble Hiding | Declared per-rank workload checks and timing cross-validation; no pipeline-bubble classifier. Scoped checks. |
| Environment: Device Mismatch | Environment inventory does not compare expected and observed GPU identity/capability. Unsupported identity-parity policy. Separate Tier-1 GPU preflight attests its named target. |
| Environment: Frequency Boost | Harness application-clock locking is enabled by actual defaults and strict runs reject lock errors. Real lock/observed-NVML integration awaits CUDA; the GPU-state consistency helper only detects clock drops. Implemented lock, runtime pending. |
| Environment: Memory Overcommit | Memory-growth diagnostics, not an OS/GPU overcommit policy. Scoped diagnostic only. |
| Environment: NUMA Inconsistency | NUMA affinity diagnostics are advisory. They do not pin or reject cross-node affinity. Scoped advisory. |
| Environment: CPU Governor Mismatch | Strict environment validation rejects a non-performance governor; it does not set/lock the governor. Scoped environment gate. |
| Environment: Thermal Throttling | NVML telemetry and temperature/clock-drop/throttling diagnostics; real hardware execution pending. Scoped check. |
| Environment: Power Limit Difference | Power draw is captured; configured power limits are not represented/compared by GPUState. Unsupported power-limit parity policy. |
| Environment: Driver Version Mismatch; Library Version Mismatch | RunManifest captures available provenance; no cross-run driver/cuDNN/cuBLAS version lock. Unsupported version-parity policy. |
| Environment: Virtualization Overhead | Virtualization notice is advisory even in strict mode. No blanket bare-metal rejection. Scoped notice. |
| Statistical: Cherry-picking; Outlier Injection; Variance Gaming; Percentile Selection | Raw supplied samples and derived statistics are preserved/reported. No detector for upstream omissions, injected samples, or malicious selection; no fixed percentile acceptance policy. Scoped reporting. |
| Statistical: Insufficient Samples | Adaptive CUDA iterations target a duration subject to a maximum count. No statistical-power or variance-driven sample-size guarantee. Scoped timing mechanism. |
| Evaluation: Self-Modifying Tests | Config immutability protects configuration values, not test-source files. Test-source immutability policy unsupported. |
| Evaluation: Benchmark Overfitting; Benchmark Memorization | Fresh-input/jitter checks reject specific cached/constant-output behaviors, not general dataset overfitting or contamination. Jitter still has documented pre-perturbation advisory/unsupported exits. Scoped checks. |
| Timing: Timer Granularity | Adaptive measurement duration, not proof of sub-microsecond timer resolution. Real CUDA cases pending. |
| Timing: Warmup Bleed; Location: Warmup Computation | L2 clearing after warmup; this is not a general detector for work moved into warmup. Real CUDA flush-path tests pending; eviction efficacy needs hardware evidence. |
| Timing: Profiler Overhead | The harness timing path does not enable its profiler. No nested-profiler rejection guard. Scoped path check. |

The historical-case summary table must use the same limits: its Invalid Ground
Truth, Data Contamination, Train/Test Overlap, Missing Holdout Sets, Reproducibility,
Cherry-picking, Benchmark Overfitting and Evaluation Integrity rows cannot remain
unqualified `OK`. In particular, replace “RunManifest version locking” with
“Version provenance capture; cross-run version lock unsupported,” and replace
“Dataset isolation + holdout enforcement” / “Held-out evaluation data” with
“No dataset provenance or holdout enforcement.” Evaluation integrity checks do
not enforce source-file immutability.

Do not interpret unmentioned rows as independently reviewed/verified here. Keep
their advertised mechanism narrowly stated and link their applicable test and
runtime gate. No new production detector is proposed by this documentation fix.
