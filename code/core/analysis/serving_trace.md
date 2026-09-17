# Serving trace analysis

`aisp tools serving-trace` analyzes observed request timestamps using
[serving_trace.py](serving_trace.py). It complements Chapter 16's runtime and
Prometheus metrics with request-level SLO and goodput accounting.

Supply JSONL records with `request_id`, `arrival_s`, `token_timestamps_s`,
`finish_s`, and `status` (`completed`, `failed`, or `cancelled`). Times must share
a clock and unit, be finite, and follow request order. For example, this is an
illustrative schema record, not a retained performance measurement:

```json
{"request_id":"request-1","arrival_s":0.0,"token_timestamps_s":[0.1,0.2,0.3],"finish_s":0.4,"status":"completed"}
```

From `code/`:

```bash
python -m cli.aisp tools serving-trace -- --trace requests.jsonl --ttft-ms 250 --tpot-ms 50 --max-itl-ms 100 --output metrics.json
```

TTFT measures arrival to first token. TPOT is the mean interval between emitted
tokens; single-token requests retain `null` TPOT. An optional maximum inter-token
gap catches stalls hidden by the mean. Goodput counts completed requests meeting
every configured SLO over earliest arrival to latest finish.

Failed and cancelled requests stay in the attainment denominator. Their emitted
tokens stay in throughput. Reports retain original token timestamps and extra
source fields, including failure details; derived metrics overwrite any supplied
fields of the same name. Original files are read without modification. Invalid
records fail explicitly instead of disappearing from the report.

Existing vLLM GPU-event measurements and Prometheus histograms are not silently
reinterpreted as request timestamps. Kernel-generation logs and profiler traces
also have different schemas; retain their original artifacts and source identifiers
when comparing them with request reports.
Source: [serving evaluation](https://arxiv.org/html/2407.07000).
