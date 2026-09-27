# SGLang failure counter validation on Verda B200

Validated on September 27, 2026 UTC. This follow-up fixes the SGLang P/D telemetry limit recorded in the [initial B200 validation](diagnostic_tools_b200_validation.md). It uses the same two B200 GPUs and Qwen3-8B workload. It does not add an RDMA NIC or qualify cross-node networking.

## Native counter fix

SGLang 0.5.20 declared labeled bootstrap and transfer failure counters without constructing their labeled children at startup. Its multiprocess exporter therefore omitted both families until their first increment. The missing metrics were:

- `sglang:num_bootstrap_failed_reqs_total`
- `sglang:num_transfer_failed_reqs_total`

The new `aisp tools serving-prepare-runtime` command copies the supported package into a separate directory and initializes those two native label sets. The existing failure handlers remain unchanged. The tool accepts the checked version and collector digest only, preserves the installed package, and records the full copied-file manifest and prepared runtime digest.

The source collector SHA256 was `0d85bbc763f44cec6ac7dbab3a70a5b75cd225b7d1a619994b1100fe4a16f4ff`. The prepared collector SHA256 is `6e9975c7a5e7395dfe23cb5233c3b19901d9f46046c9c018c7b217ad48e43378`. The only source change is two native `.labels(**self.labels)` calls at collector initialization.

## Counter and rejection evidence

A real CPU probe instantiated the installed native collector in Prometheus multiprocess mode. The original package exposed neither family before an increment. The prepared package exposed both exact counter families with one zero sample each. Calling SGLang's existing increment methods produced one sample at one for each family in both packages.

A separate GPU run started the prepared SGLang P/D fleet and completed a successful warmup. It then sent a decode request with an invalid bootstrap destination. The decode bootstrap-failure counter increased by one. The prefill failure total stayed unchanged. The serving tool rejected the interval and named the exact decode counter and label set. The run retained both native endpoint scrapes and structured snapshots, then released the GPUs before the accepted comparison began.

The comparison now reads both failure families from both workers. It checks each selected metric and label series before summing. Missing families, disappeared series, counter resets, and positive failure deltas reject the interval. Parsed snapshots are saved before rejection.

## Repeated serving comparison

The full comparison completed with `comparison_valid: true`, eight accepted iterations, and no rejections. It used vLLM 0.26.0, the prepared SGLang 0.5.20 package, and Qwen3-8B on the same two B200s. Each arm had one warmup and two measured repeats. The second repeat ran in reverse order.

- All 16 completed requests produced matching 32-token vectors across the four configurations and both repeats.
- All 24 measured requests echoed the exact prompt token IDs.
- All eight client cancellation checks produced tokens and closed at the configured 2.5 seconds.
- All four P/D iterations recorded three real KV transfers and zero failure deltas. Every selected native SGLang failure series was zero before and after each measured interval.
- All eight lifecycle cleanup records showed empty GPU process tables. Two SGLang monolithic shutdowns exceeded the graceful timeout and required forced termination. The fleet was released in both cases.

This establishes diagnostic correctness for the prepared runtime. It does not establish a performance win or fully graceful SGLang monolithic shutdown.

The [validation receipt](validation/sglang-failure-counters-b200-20260927/serving.json) records native counter snapshots, source and runtime digests, the negative probe, every measured iteration, and GPU release evidence. The same forced SGLang monolithic shutdown sequence was present in the two prior validation runs.

## Regression checks

The final focused suite ran on the Verda host with CUDA hidden: 68 passed, zero failed, zero skipped. These cover runtime preparation, failure-series validation, protocol and lifecycle behavior, launch binding, CLI dispatch, and MCP dispatch. Required Ruff checks and the silent-fallback audit passed. Repository benchmark contracts checked 963 files with zero errors or warnings.

After the GPU run, the preparation tool received a change that limits the length of error messages. The final public CLI reproduced the identical runtime package and manifest used in the GPU tests. Its 10 preparation tests passed again on Verda with zero failures or skips. These repeat a subset of the 68 focused tests. The receipt records both preparation-tool revisions without changing the original run source identity.

From `code/`, the focused command was:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python3 -m pytest -q \
  tests/test_serving_failure_sources.py \
  tests/test_serving_telemetry_integrity.py \
  tests/test_serving_comparison.py \
  tests/test_serving_launch_binding.py \
  tests/test_prepare_sglang_runtime.py \
  tests/test_diagnostics_integration.py
```

The serving run used `--repeats 2 --warmups 1 --ttft-ms 10000 --tpot-ms 2000 --max-itl-ms 5000 --connect-timeout-s 30`. Both GPUs were held at 1,965 MHz SM and 3,996 MHz memory through the repository's `lock_gpu_clocks` contexts. The same request trace and model were used for all four arms. The contexts restored the prior clock and persistence state after the run.

These are diagnostic results. `valid_for_performance_claim` and `publication_ready` remain false. The same-host GPU transfers do not establish network RDMA behavior.

See the [serving guide](../labs/serving_comparison/README.md) for runtime preparation and the comparison command.
