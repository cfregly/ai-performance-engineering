# Serving comparison

This tool replays one request trace with authoritative prompt token ids across four serving arms:

1. vLLM monolithic
2. vLLM prefill and decode disaggregation
3. SGLang monolithic
4. SGLang prefill and decode disaggregation

It starts one arm at a time, records streaming observations, validates output token ids, captures live P/D telemetry, and then stops the complete owned process group. Engine mode also checks `nvidia-smi` before and after every arm. A valid run requires exclusive custody of the same GPU fleet for each arm.

This is a diagnostic tool. `valid_for_performance_claim` and `publication_ready` are always false. Use the repository benchmark and profiler workflow before making a speedup claim.

## Prepare SGLang failure counters

The checked SGLang 0.5.20 collector creates labeled failure counters without initializing their label values. Its multiprocess exporter omits those counters until the first failure. The original B200 runs therefore rejected SGLang P/D before measurement, even though the warmup requests and KV transfers completed.

Prepare an isolated runtime before running this build:

```bash
PYTHONPATH=code python -m cli.aisp tools serving-prepare-runtime -- \
  --package-dir /path/to/site-packages/sglang \
  --output-dir /path/to/sglang-counter-runtime
```

The tool accepts the checked collector source bytes only. It copies the package and initializes the native label values for `sglang:num_bootstrap_failed_reqs_total` and `sglang:num_transfer_failed_reqs_total` at startup. Their existing failure handlers still increment the same counters. It does not change the installed package, replace missing telemetry with zero, or reset a running counter.

Set `lifecycle.environment.PYTHONPATH` in both SGLang profile entries to `/path/to/sglang-counter-runtime:/path/to/ai-performance-engineering/code`. The process group launcher passes this to its children. If a child launch declares its own `PYTHONPATH`, set that value to the same path. Continue to launch `python -m sglang.launch_server`.

Copy `runtime_build_id` from `runtime-receipt.json` into both SGLang profile entries and the P/D provenance file. The receipt records the source and prepared package digests, plus the changed collector file. Regenerate the launch manifest digest after changing its environment.

The comparison still requires native counter samples or exact counter-family declarations. Missing telemetry, counter resets, and positive failure deltas reject the run. Retire this preparation step when a checked upstream build exports these counters at zero and passes the same zero, increment, and serving tests.

The [Verda B200 validation](../../docs/sglang_failure_counter_validation.md) records the native failure probe and the completed two-repeat, four-arm comparison.

## Run it

Start from the repository root:

```bash
PYTHONPATH=code python -m cli.aisp tools serving-compare -- \
  --profile code/labs/serving_comparison/examples/profile.engine.template.json \
  --trace code/labs/serving_comparison/examples/request_trace.jsonl \
  --output-dir artifacts/serving-comparison/run-001 \
  --repeats 4 \
  --warmups 1 \
  --ttft-ms 500 \
  --tpot-ms 50 \
  --max-itl-ms 100
```

Edit the template first. Replace build ids, model paths, GPU ids, clocks, and runtime paths with values from the serving host. The output directory must not exist. The tool never overwrites an earlier result.

The example P/D launch specs use the bundled process group launcher. The SGLang spec follows the `--disaggregation-mode` and `sglang_router --pd-disaggregation` contract. The vLLM spec uses `NixlConnector` and the bundled proxy, which accepts nonempty text or explicit token ids, sends a real prefill request with `do_remote_decode`, passes returned `kv_transfer_params`, and streams decode with `do_remote_prefill`.

The checked template targets vLLM 0.26.0 and SGLang 0.5.20. The validated SGLang 0.5.20 build provides per-event completion `token_ids` and first-event `prompt_token_ids`. Its model gateway rejects an integer-array completion prompt, so the template uses `text_with_token_id_attestation`. The trace text was encoded with the pinned Qwen3 tokenizer and the engine must echo the exact authoritative prompt ids. A cancelled text request must echo those ids before client cancellation or the request is rejected.

## Endpoint and stream contract

Every measured request uses `POST /v1/completions` with deterministic sampling, `stream=true`, `return_token_ids=true`, and usage reporting. The `token_ids` transport sends the trace ids directly. The `text_with_token_id_attestation` transport sends `prompt_text` and requires the first stream event to echo prompt ids that exactly match the trace. A tokenizer round-trip mismatch rejects the request. The client records the monotonic arrival, admission, SSE event, and finish times. It records HTTP failures, deadline expiry, and client stream cancellation.

Token timestamps mean client-observed SSE event times. Text chunks are never split into invented tokens. If one event contains several explicit token ids, those ids retain the same chunk timestamp and the run rejects TPOT and ITL conclusions. TTFT and request completion evidence remain visible.

The request trace is JSONL. Each row uses `serving-comparison.request-trace.v1` and includes:

- `request_id`
- ordered `arrival_ms`
- nonempty `prompt_token_ids`
- `prompt_text` when the profile uses `text_with_token_id_attestation`
- `max_tokens`
- `deadline_ms`
- optional `cancel_after_ms`
- `expected_status`, which is `completed`, `failed`, or `cancelled`
- optional `expected_output_token_ids` for golden correctness

The checked cancellation row allows up to 2,048 output tokens and closes the client stream after 2.5 seconds. This leaves enough time to observe the exact prompt echo while keeping generation active on the validated engines.

The profile selects `exact_token_ids` or `per_request_golden`. It also declares the allowed token mismatch count. Exact comparison uses the first vLLM monolithic repeat as the reference and checks all arms and repeats.

## Fixed fleet and lifecycle

`local_process` is the supported lifecycle backend. `start_command` is an argv list and never passes through a shell. A P/D arm uses `process_group_launcher.py` to keep prefill, decode, and router processes under one owned group.

For engine runs, each arm must set `CUDA_VISIBLE_DEVICES` to the complete fixed fleet. Every `ready_urls` endpoint and native identity endpoint must respond before allocation can pass. The tool retries incomplete owned allocation until the common startup deadline. It rejects a foreign GPU process immediately. It then maps GPU compute PIDs to the launched process tree and to the declared P/D child roles.

The tool reads the observed application graphics clock from `nvidia-smi` and requires an exact match with each declared GPU clock before and after activation. On exit it sends TERM and then a bounded KILL to the owned process group. A failed startup writes `startup-cleanup.json` and proves that the fleet is idle before the next arm.

The profile also fixes `admission_max_concurrency`. The replay client uses that bound. Engine launch commands should set the corresponding native server limit. The template uses vLLM `--max-num-seqs` and SGLang `--max-running-requests`. vLLM 0.26.0 does not expose `--max-num-queued-reqs`.

## Native identity and telemetry

Identity probes query native JSON endpoints and bind live response paths to the pinned model and runtime version. The template uses vLLM `/v1/models` and `/version`, plus SGLang `/server_info`. A label alone cannot claim identity. The expected pinned value must appear in a live assertion.

Prometheus telemetry selectors accept `name` or `names`, exact label filters, `reduce: sum`, `scale`, and `offset`. A selector may set `missing_value: 0` only for an unfiltered counter family. The native response must contain an exact matching `# TYPE ... counter` declaration. The snapshot records that declaration and the selector state. This supports counters that expose no sample before their first increment without hiding a misspelled metric or label. The template includes these native mappings:

| Evidence | vLLM NIXL | SGLang NIXL |
| --- | --- | --- |
| Transfer requests | `vllm:nixl_xfer_time_seconds_count` | `sglang:kv_transfer_latency_ms_count` |
| Transfer time | `vllm:nixl_xfer_time_seconds_sum` | `sglang:kv_transfer_latency_ms_sum`, scaled from ms |
| Transfer failures | `vllm:nixl_num_failed_transfers_total` | Both native failure counters from each SGLang worker endpoint |
| Transfer bytes | `vllm:nixl_bytes_transferred_sum` | `sglang:kv_transfer_total_mb_sum`, scaled by 1,048,576 bytes per MiB |
| Phase requests | `vllm:request_success_total` on the separate P and D endpoints | `sglang:num_requests_total` on the separate P and D endpoints |

SGLang emits its NIXL transfer latency and byte metrics on the prefill endpoint. The decode endpoint supplies decode request activity. Both endpoints can record bootstrap and transfer failures, so both declare `kv_transfer_failures_total`. The tool reports their sum and preserves each selected metric name, label set, and value. It checks every counter series before using that sum. A reset cannot hide a new failure in another series. Other telemetry semantics must have one source.

The required P/D gate is positive transfer request count, transfer time, prefill activity, and decode activity. Each failed-transfer delta must remain zero. The tool retains parsed snapshots and raw scrapes before checking interval validity, including rejected intervals. Transfer bytes, server queue age, and per-pool idle fractions are diagnostics with an explicit `measured` or `unsupported` state. Missing required evidence rejects the run. An unsupported optional signal stays visible and never becomes zero.

The validated same-host setup used NIXL over UCX without an RDMA device. Its evidence covers same-host GPU transfer only. It does not establish cross-node RDMA behavior or performance.

The native names above reflect current upstream documentation and source. Check them against the pinned engine build before running:

- [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM NIXL connector metrics](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- [SGLang P/D disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- [SGLang scheduler metrics source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py)
- [SGLang completion stream protocol](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/openai/protocol.py)

## P/D provenance

Each P/D arm points to a `serving-comparison.pd-provenance.v1` file. It binds the engine, runtime build, connector, transfer backend, proxy, separate P and D GPU pools, endpoints, workload identity, and the exact launch-file digest. `request_path` must be `kv_handoff`. A router that sends a whole request to one engine is rejected as P/D.

The binder accepts the native Python entrypoints shown in the examples and `vllm serve`. vLLM may run through a bounded Singularity or Apptainer `exec --nv --bind` prefix with a pinned image. The outer launcher adds the validated SHA-256 digest and refuses changed bytes. Runtime evidence then matches retained child identities to their observed GPU contexts, including an empty router GPU pool. This proves the declared launch configuration and live process allocation. It does not attest an arbitrary executable or container image by itself.

The profile workload id is also copied into every flat signal. Use the same workload id when collecting fabric, transport, or switch evidence. Cross-layer correlation requires matching resource identity and compatible clocks. Different collectors need a measured clock-skew bound. Trace artifact paths live under `evidence.source_artifact`. They are not network path labels.

## Outputs

`summary.json` uses `serving-comparison.result.v1`. It contains the prompt transport, execution order, per-arm medians, P/D deltas, diagnostic support, correctness rejections, iteration summaries, and a flat `signals` list. Request JSONL files preserve authoritative and echoed prompt ids, explicit output ids, and event timing. Raw and parsed telemetry snapshots are retained under each P/D iteration. P/D iteration summaries retain the validated launch binding and observed role allocation.

Signals use a hashed collector identity in both `clock_domain` and `scope.host`. SLO failures include their wall-clock interval as symptoms so the cross-layer diagnostic tool can compare them with telemetry from the same collector.

## Rejected and unsupported states

The tool rejects public endpoints, missing matrix cells, mismatched engine builds, observed clock mismatches, nonexclusive GPUs, foreign GPU processes, incomplete backend readiness, missing native identity, launch digest or role drift, missing P/D counters, inactive KV transfer, failed KV transfers, counter resets, prompt-token round-trip mismatches, status mismatches, correctness failures, and multi-token event TPOT claims. Control-plane readiness, identity, and telemetry reads use bounded timeouts. Streaming generation reads remain governed by each request deadline.

Only local process lifecycle is supported. Remote SSH launch, Kubernetes lifecycle, chat completions, tool calls, speculative output tolerance, and server-confirmed cancellation are outside this version. Client cancellation means the response stream was closed. It does not prove that the engine stopped work.

The local fixture in `test_serving_comparison.py` validates HTTP, SSE, cancellation, lifecycle, evidence, and artifact behavior. It does not validate vLLM, SGLang, NIXL, GPUs, or performance.
