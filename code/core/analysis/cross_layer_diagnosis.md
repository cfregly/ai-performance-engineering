# Cross-layer diagnosis

Transport rates and network RTT measurements appear in `contextual_measurements`
when their resource and clock intervals match a symptom. They provide context
without being treated as congestion counters or compared with an invented
baseline. Older network captures without collector identity remain analyzable,
but do not export timed signals for this join.

`aisp tools cross-layer-diagnose` joins timed application symptoms and fabric
counter deltas. Each association includes its source artifact hash, resource
scope, overlap duration and clock uncertainty. An association is a hypothesis
to investigate, not a root-cause verdict.

```bash
python -m cli.aisp tools cross-layer-diagnose -- \
  --input /tmp/application-signals.json --input /tmp/fabric-signals.json \
  --run-dir /tmp/cross-layer-analysis
```

Inputs can be diagnostic artifacts containing a `signals` list or a dedicated
`aisp.diagnostic-signals.v1` export. This illustrative counter record is schema
documentation, not a measured result:

```json
{
  "schema": "aisp.diagnostic-signals.v1",
  "signals": [{
    "metric": "pfc_pause_frames", "role": "counter", "semantics": "delta",
    "value": 12, "unit": "frames",
    "start_unix_s": 100, "end_unix_s": 102,
    "clock_domain": "collector-a",
    "scope": {"path": "rail-0", "workload_id": "example-run"}
  }]
}
```

Application exports use `role: symptom` for measured violations such as requests
missing a declared latency objective. Use `role: measurement` for observations
without a declared failure criterion. Preserve actual interval timestamps and
the collector's clock identity. Scope keys must identify the same host, path,
interface or workload. A shared host, path, endpoint identity or workload is
required. An interface name alone can refer to unrelated devices on different
hosts. Shared keys with different values prevent correlation.

For counters, export interval deltas. Cumulative snapshots are rejected. A reset
or unknown unit must be resolved by the collector before a delta can be used.
Zero changes remain observations but do not create positive associations.

The same clock domain means timestamps share a clock. Different domains require
`--max-clock-skew-ms` with a measured bound. The analyzer subtracts that uncertainty
from the interval overlap. Do not supply a guessed bound to force a match.
Synchronized wall clocks do not make CUDA event clocks comparable across GPUs.

The absence of an association does not clear the fabric. Check missing switch
access, the selected rail, collection windows and timestamp alignment. Likewise,
PFC activity during a slow request does not establish that PFC caused the delay.
Repeat an equivalent workload with one controlled change and retain both traces.

These reports are diagnostic artifacts. They do not replace the benchmark
harness's correctness, clock, profiler or repeat requirements.
