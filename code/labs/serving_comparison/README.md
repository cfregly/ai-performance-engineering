# Serving comparison

This tool replays one token-id request trace across four serving arms:

1. vLLM monolithic
2. vLLM prefill and decode disaggregation
3. SGLang monolithic
4. SGLang prefill and decode disaggregation

It starts one arm at a time, records streaming observations, validates output token ids, captures live P/D telemetry, and then stops the complete owned process group. Engine mode also checks `nvidia-smi` before and after every arm. A valid run requires exclusive custody of the same GPU fleet for each arm.

This is a diagnostic tool. `valid_for_performance_claim` and `publication_ready` are always false. Use the repository benchmark and profiler workflow before making a speedup claim.

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

Edit the template first. Replace runtime versions, build ids, model paths, GPU ids, clocks, ports, and launch commands with values from the serving host. The output directory must not exist. The tool never overwrites an earlier result.

The example P/D launch specs use the bundled process group launcher. The SGLang spec follows the current `--disaggregation-mode` and `sglang_router --pd-disaggregation` contract. The vLLM spec uses `NixlConnector` and the bundled proxy, which sends a real prefill request with `do_remote_decode`, passes returned `kv_transfer_params`, and streams decode with `do_remote_prefill`.

## Endpoint and stream contract

Every measured request uses `POST /v1/completions` with an explicit list of prompt token ids, deterministic sampling, `stream=true`, `return_token_ids=true`, and usage reporting. The client records the monotonic arrival, admission, SSE event, and finish times. It records HTTP failures, deadline expiry, and client stream cancellation.

Token timestamps mean client-observed SSE event times. Text chunks are never split into invented tokens. If one event contains several explicit token ids, those ids retain the same chunk timestamp and the run rejects TPOT and ITL conclusions. TTFT and request completion evidence remain visible.

The request trace is JSONL. Each row uses `serving-comparison.request-trace.v1` and includes:

- `request_id`
- ordered `arrival_ms`
- nonempty `prompt_token_ids`
- `max_tokens`
- `deadline_ms`
- optional `cancel_after_ms`
- `expected_status`, which is `completed`, `failed`, or `cancelled`
- optional `expected_output_token_ids` for golden correctness

The profile selects `exact_token_ids` or `per_request_golden`. It also declares the allowed token mismatch count. Exact comparison uses the first vLLM monolithic repeat as the reference and checks all arms and repeats.

## Fixed fleet and lifecycle

`local_process` is the supported lifecycle backend. `start_command` is an argv list and never passes through a shell. A P/D arm can use `process_group_launcher.py` to keep prefill, decode, and proxy processes under one owned group.

For engine runs, each arm must set `CUDA_VISIBLE_DEVICES` to the complete fixed fleet. Before launch, the tool requires that fleet to have no compute processes. After readiness, it maps GPU compute PIDs to the launched process tree. It rejects missing GPUs and foreign PIDs. On exit it sends TERM and then a bounded KILL to the owned process group. It verifies that the fleet is idle before the next arm.

The profile also fixes `admission_max_concurrency`. The replay client uses that bound. Engine launch commands must set the corresponding native server limit, such as vLLM `--max-num-queued-reqs` and SGLang `--max-running-requests`.

## Native identity and telemetry

Identity probes query native JSON endpoints and bind live response paths to the pinned model and runtime version. The template uses vLLM `/v1/models` and `/version`, plus SGLang `/server_info`. A label alone cannot claim identity. The expected pinned value must appear in a live assertion.

Prometheus telemetry selectors accept `name` or `names`, exact label filters, `reduce: sum`, `scale`, and `offset`. The template includes these native mappings:

| Evidence | vLLM NIXL | SGLang NIXL |
| --- | --- | --- |
| Transfer requests | `vllm:nixl_xfer_time_seconds_count` | `sglang:kv_transfer_latency_ms_count` |
| Transfer time | `vllm:nixl_xfer_time_seconds_sum` | `sglang:kv_transfer_latency_ms_sum`, scaled from ms |
| Transfer failures | `vllm:nixl_num_failed_transfers_total` | `sglang:num_bootstrap_failed_reqs_total` plus `sglang:num_transfer_failed_reqs_total` |
| Transfer bytes | `vllm:nixl_bytes_transferred_sum` | Marked unsupported in the template |
| Phase requests | Native transfer or request counters on the separate P and D endpoints | Native transfer counters on the separate P and D endpoints |

The required P/D gate is positive transfer request count, transfer time, prefill activity, and decode activity, plus a nondecreasing failure counter. Transfer bytes, server queue age, and per-pool idle fractions are diagnostics with an explicit `measured` or `unsupported` state. Missing required evidence rejects the run. An unsupported optional signal stays visible and never becomes zero.

The native names above reflect current upstream documentation and source. Check them against the pinned engine build before running:

- [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM NIXL connector metrics](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [vLLM production metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- [SGLang P/D disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- [SGLang scheduler metrics source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py)
- [SGLang completion stream protocol](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/openai/protocol.py)

## P/D provenance

Each P/D arm points to a `serving-comparison.pd-provenance.v1` file. It binds the engine, runtime build, connector, transfer backend, proxy, separate P and D GPU pools, endpoints, workload identity, and launch manifest digest. `request_path` must be `kv_handoff`. A router that sends a whole request to one engine is rejected as P/D.

The profile workload id is also copied into every flat signal. Use the same workload id when collecting fabric, transport, or switch evidence. Cross-layer correlation requires matching resource identity and compatible clocks. Different collectors need a measured clock-skew bound. Trace artifact paths live under `evidence.source_artifact`. They are not network path labels.

## Outputs

`summary.json` uses `serving-comparison.result.v1`. It contains the execution order, per-arm medians, P/D deltas, diagnostic support, correctness rejections, iteration summaries, and a flat `signals` list. Request JSONL files preserve explicit token ids and event timing. Raw and parsed telemetry snapshots are retained under each P/D iteration.

Signals use a hashed collector identity in both `clock_domain` and `scope.host`. SLO failures include their wall-clock interval as symptoms so the cross-layer diagnostic tool can compare them with telemetry from the same collector.

## Rejected and unsupported states

The tool rejects public endpoints, missing matrix cells, mismatched engine builds, unlocked engine clocks, nonexclusive GPUs, foreign GPU processes, missing native identity, missing P/D counters, inactive KV transfer, counter resets, status mismatches, correctness failures, and multi-token event TPOT claims.

Only local process lifecycle is supported. Remote SSH launch, Kubernetes lifecycle, chat completions, tool calls, speculative output tolerance, and server-confirmed cancellation are outside this version. Client cancellation means the response stream was closed. It does not prove that the engine stopped work.

The local fixture in `test_serving_comparison.py` validates HTTP, SSE, cancellation, lifecycle, evidence, and artifact behavior. It does not validate vLLM, SGLang, NIXL, GPUs, or performance.
