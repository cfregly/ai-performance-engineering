# Retrospective analysis plan

## Status

This document was written after the original experiment and fresh run 022. It
is a retrospective map from the retained evidence to the questions a reader is
likely to ask. It is not a pre-experiment plan and does not change any gate.
The actual fixed plan is `protocol.json`, which predates the retained results.

## Claim under review

The narrow claim is that `torch._dynamo.config.nested_graph_breaks=True`
improved steady synchronized invocation latency for the declared depth-8,
64-operation FP32 compiler microbenchmark on one NVIDIA B200. The baseline is
the same worker with nested graph-break support off. The candidate runs with
support on. The workload itself keeps the same graph breaks. This is not a
serving or model benchmark.

## Questions and retained checks

1. **Correctness:** Require the eager-reference shape, dtype, finite-value, and
   tolerance checks to pass for every worker. Read the original 24 records and
   the independent fresh022 final qualification.
2. **Repeatability:** Report every paired-seed ratio, the observed range, and
   the geometric mean. Keep the original four-seed cohort separate from the
   fresh022 four-seed cohort.
3. **Startup and payback:** Report the fixed first-ten startup verdict and the
   original no-break startup-control failure. Do not infer a universal payback
   point from the steady result.
4. **Memory:** Use only the separately retained follow-up diagnostic for its
   one-pair allocator and RSS observations. Do not generalize it to the primary
   matrix.
5. **Mechanism:** Compare compiled-region and CUDA Graph execution structure.
   Use Nsys and NCU only as supporting evidence because profiling changes the
   execution environment and adds overhead.
6. **Portability:** Limit the claim to the measured B200 and named runtime.
   H100, A100, other PyTorch versions, and other backends were not tested here.
7. **Real-workload relevance:** State that the workload is synthetic and does
   not establish training or serving gains.
8. **Failure boundaries:** Preserve the original failed startup route, failed
   original startup control, and all profiler interpretation limits.

## Evidence mapping

- Original timing and per-seed outcomes: `results/analyzer-result.json`
- Original gate decisions: `results/qualification.json`
- Independent repeat: `profiles/fresh022/qualification/FINAL-SCIENTIFIC-QUALIFICATION.json`
- Accepted profiler evidence: `profiles/fresh022/qualification/STRICT-PROFILER-QUALIFICATION.json`
- Native captures and tracing mode: `profiles/README.md`
- Separate diagnostic answers: `followup/results/summary.json`

No additional GPU run is proposed by this retrospective document.
