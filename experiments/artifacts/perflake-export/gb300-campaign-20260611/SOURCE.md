# PerfLake dry-run export bundle: GB300 kernel-lab campaign (B36..B71)

**Status: PROVISIONAL — DO NOT INGEST.**

## What this is

A staged, on-disk-only sample payload (`ingest.json`) converting the GB300 kernel-lab
campaign's banked runbook entries **B36..B71** into structured rows, for the PerfLake
team to evaluate as a *candidate* ingest shape. This is the perflake-export skill's
"pre-Phase-1 GA dry-run" use case.

- **Schema:** `perflake_gb300_kernel_labs_v0` — **PROVISIONAL**. The locked PerfLake
  ingest shapes today cover inference-serving perf-bench bundles
  (`perflake_inference_perfbench_v0`), MLPerf seeds, and HPC-Perf/NCCL. The GB300
  kernel-lab campaign fits **none** of these; this payload proposes a new shape.
- **Source of truth:** `code/docs/gb300-runbook.md` (sha256 in `provenance.json`),
  banked entries B36..B71, plus one git commit per bank.
- **Nothing was uploaded.** No lake API call, no MCP lake tool was invoked. The
  bundle exists only under
  `experiments/artifacts/perflake-export/gb300-campaign-20260611/`.

## What the operator MUST confirm before any ingest

1. **Schema contract:** `perflake_gb300_kernel_labs_v0` vs the PerfLake Data
   Ingestion API contract (**PB-78 / DATAPLAT-1599**) — the contract is not locked;
   this shape is a proposal, not a conformant payload.
2. **Field conventions:** value fields are number OR verbatim range string (e.g.
   B55 kernel "23.81-23.90" us); the PerfLake team must decide how ranges land.
3. **The B65 bank-id collision:** the runbook contains two distinct B65 entries
   (rows `B65-1`, `B65-2`, both `bank_id_collision: true`). Decide whether to
   renumber upstream in the runbook or carry the collision into the lake.
4. **Redaction stance:** pod/namespace/internal-host strings are redacted
   ("GB300 dev pod"). Confirm this is sufficient if the payload leaves the org.
5. **SoL ceiling registry:** rows reference ceilings by key only
   (`gb300_fp16_nameplate_3.75pf`, `gb300_fp16_real_1.9pf` per the B61 reframing,
   `gb300_hbm3e_peak`, `gb300_nvfp4_peak`). The HBM/NVFP4 absolute peaks are NOT
   inlined (not restated in the parsed entries) — resolve from the evidence docs
   if the lake schema requires absolute values.
6. **Toolchain block:** CUDA 13.2 / NGC 26.05 / CUTLASS 4.2.0 are operator-context
   values, flagged as not independently re-verified in this parse (torch 2.12,
   Triton 3.7, sm_103/152 SMs are runbook-confirmed).

## Bundle contents

| File | Purpose |
|---|---|
| `ingest.json` | The payload: campaign block + 37 rows + field conventions |
| `provenance.json` | Operator, git HEAD, runbook sha256, generation method |
| `SOURCE.md` | This file |
| `VALIDATION.md` | 6 spot-checks: verbatim runbook line vs extracted row fields |

## Row semantics

- `verdict_class` is the normalized leading verdict token of the runbook entry;
  `verdict_raw` preserves the full verdict phrase.
- Honest negatives / ties / audits with no banked before/after pair carry null
  metrics with the measured evidence quoted in `mechanism_one_liner` / `notes`.
- `speedup` is only filled when the runbook states the ratio itself; stated
  before/after figures whose ratio is not stated leave `speedup` null.
- `graphed_dual_figures` (B67) carries the B62 dual-figure discipline where the
  runbook banked both the default-frame and pure-GPU graphed figures.

## Operator-gated next step (DO NOT RUN)

The would-be ingest call, for shape reference only — blocked on PB-78/DATAPLAT-1599:

```
# DO-NOT-RUN — operator-gated; PerfLake ingest contract not locked
perf_tune_report_publish_to_lake \
  --bundle experiments/artifacts/perflake-export/gb300-campaign-20260611/ingest.json \
  --schema perflake_gb300_kernel_labs_v0 \
  --destination s3://perf-lake/gb300-kernel-labs/2026-06-11/ \
  --provenance experiments/artifacts/perflake-export/gb300-campaign-20260611/provenance.json
```
