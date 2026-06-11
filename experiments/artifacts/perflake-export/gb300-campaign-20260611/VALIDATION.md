# Spot-check validation: 6 rows vs verbatim runbook text

Source: `code/docs/gb300-runbook.md` @ git HEAD `c7beb404` (sha256 in `provenance.json`).
Line numbers are from this revision. Quotes are verbatim (wrapped lines joined with their
original leading two-space indent shown as a single space).

---

## 1. Row `B36` (WIN) — runbook lines 1155-1158

> `- WIN + HYPOTHESIS CORRECTION (B36, blackwell_matmul host-overhead deepen): docs/`
> `gb300-blackwell-matmul-hostoverhead.md. labs/blackwell_matmul:blackwell_matmul_tcgen05 (FP16 2048^3)`
> `0.2250 ms -> ~0.155 ms median (~1.45x; 69.61x -> 101-104x vs naive), verify PASS x3, outputs`
> `BIT-IDENTICAL (torch.equal) pre/post.`

| Field | Extracted | Faithful? |
|---|---|---|
| verdict_raw | `WIN + HYPOTHESIS CORRECTION` | yes — verbatim |
| lab / target | `blackwell_matmul` / `blackwell_matmul_tcgen05` | yes |
| baseline/optimized/unit | 0.225 / 0.155 / ms | yes — "0.2250 ms -> ~0.155 ms median" |
| speedup | 1.45 | yes — "(~1.45x ...)" |
| verification | verify PASS x3 | yes |
| numerics_class | bit-identical (torch.equal) | yes |
| sol_pct / key | 11.7 / nameplate | yes — line 1174: "38.9 us = ~440 TFLOPS = 11.7% FP16-SoL" (pre-B61 entry → nameplate key) |
| commit_sha | 8983671c... | yes — `git log`: "...(1.45x, bit-identical); B36" |

## 2. Row `B45` (BREAKTHROUGH) — runbook lines 1347-1350

> `- BREAKTHROUGH (B45, moe_cuda router_vectorized, the B40 lever class generalizes): docs/`
> `gb300-moe-cuda-router-vectorize.md. labs/moe_cuda:router_vectorized optimized arm 8.289 -> 0.500 ms`
> `= 16.5x over the shipped arm (harness 177-270x vs the noisy eager baseline), verify PASS x4,`
> `numerics BIT-EXACT (max_abs_diff 0.0, eager and via replay).`

| Field | Extracted | Faithful? |
|---|---|---|
| verdict_class | BREAKTHROUGH | yes — verbatim |
| lab / target | moe_cuda / router_vectorized | yes |
| baseline/optimized/unit | 8.289 / 0.5 / ms | yes |
| speedup | 16.5 | yes — "= 16.5x over the shipped arm" |
| verification | verify PASS x4 (harness 177-270x ...) | yes |
| numerics_class | bit-exact (max_abs_diff 0.0, eager and via replay) | yes |
| evidence_doc | code/docs/gb300-moe-cuda-router-vectorize.md | yes — file exists in repo |
| commit_sha | 2ad93fe7... | yes — `git log`: "...(8.29->0.50ms, 16.5x, bit-exact); B45" |

## 3. Row `B46` (HONEST_NEGATIVE, null metrics) — runbook lines 1364-1365, 1371-1374

> `- HONEST NEGATIVE (B46, block_scaling tile_k lever, implementation-grade): docs/`
> `gb300-block-scaling-deepen.md.`
> `... (3) The implementable equivalent (peeled short-tail last k-tile, 16->11 MMAs/tile = -31% MMA`
> `work, bit-correct err 0.0) measures EXACT PARITY: 43.62 vs 43.65 us (0.9994x, 6 interleaved rounds)`

| Field | Extracted | Faithful? |
|---|---|---|
| verdict_class | HONEST_NEGATIVE | yes |
| baseline/optimized/speedup | null/null/null | yes by design — the entry banks a PARITY ("43.62 vs 43.65 us (0.9994x)") whose ratio direction is ambiguous in source; numbers preserved verbatim in mechanism_one_liner, not decomposed |
| verification | harness A/B/A all verification PASSED, no B38 regression | yes — line 1374-1375 |
| numerics_class | bit-correct (err 0.0) at parity | yes |
| next_lever | lab CLOSED at structural ceiling | yes — "VERDICT: block_scaling is at its structural ceiling absent a different data layout; closed." (lines 1378-1379) |
| commit_sha | 222a8ac7... | yes — "...; B46 closes the lab" |

## 4. Row `B61` (WIN + SOL REFRAMING, real-ceiling discipline) — runbook lines 1651, 1658-1666

> `- WIN + SOL REFRAMING (B61, capstone early-fill + the persistence pre-check + the REAL dense-FP16 ceiling)`
> `... the 3.75 PFLOPS dense-FP16 "peak" is unreachable on this part -- cuBLAS itself asymptotes at`
> `1.87 PF (8192^3; 11.47us at 2048^3); ... the capstone mainloop runs at ~87% of THAT (14.3 TF/SM, ABOVE cuBLAS's own 12.3)`
> `... RECOMMEND: adopt ~1.9 PF as the GB300 dense-fp16 SoL reference`
> `... SHIPPED WIN: AISP_TCGEN05_EARLY_FILL=1 ... 15.200 -> 14.980us med-of-meds (1.015x, EF1 won all 5 interleaved process-pairs on med AND min)`

| Field | Extracted | Faithful? |
|---|---|---|
| baseline/optimized/unit | 15.2 / 14.98 / us | yes |
| speedup | 1.015 | yes |
| sol_pct / sol_ceiling_key | 87 / `gb300_fp16_real_1.9pf` | yes — "~87% of THAT" where THAT = the real per-SM rate under the ~1.9 PF reframing; the key (not a bare peak) is recorded, registry note carries the 1.87 PF cuBLAS asymptote |
| numerics_class | bit-identical (torch.equal CTA1+CTA2) | yes — line 1668 |
| verification | 5/5 process-pairs; harness 9/9 PASS | yes — lines 1666-1668 |
| commit_sha | 4c8faf40... | yes — "...real GB300 dense-FP16 ceiling is ~1.9PF; B61" |

## 5. Row `B65-1` (WIN, bank-id collision) — runbook lines 1737, 1742-1744; collision at line 1764

> `- WIN + TRAP (B65, dual_2sm persistent clusters + GROUP_M raster, the B63-named lever): docs/`
> `... MEASURED WIN: persistent clusters at CHAMPION occupancy (eo=0, 3 CTAs/SM, warp-split rounds, chunked t2r, 84 regs) + gm=8 raster = 1.0598x paired median, 12/12 wins`
> `(926.0 -> 885.0us; ...)`
>
> and the SECOND B65 at line 1764:
> `- MEASUREMENT BANK + B45-CLASS #4 (B65, dual-figure bank for the frame-bound labs, the B64-named lever): docs/gb300-dual-figure-bank.md.`

| Field | Extracted | Faithful? |
|---|---|---|
| baseline/optimized/unit | 926.0 / 885.0 / us | yes |
| speedup | 1.0598 (12/12) | yes |
| bank_id_collision | true on BOTH rows B65-1 / B65-2 | yes — two distinct B65 entries exist in the runbook; both git commits also carry "; B65" (6ff041ed and b77108e8). Neither dropped, neither renumbered. |
| commit_sha | 6ff041ed... (B65-1) / b77108e8... (B65-2) | yes — matched by entry content (persistent clusters vs dual-figure bank) |
| trap note | cudaOccupancyMaxActiveBlocksPerMultiprocessor unstable on sm_103 | yes — lines 1753-1757 |

## 6. Row `B71` (RE_SWEEP_CLOSE) — runbook lines 1884-1893

> `- RE-SWEEP CLOSE (B71, the B69/B70 unlock measured across the formerly-gated paths): docs/`
> `gb300-sm103a-unlock.md. The Front-T re-sweep under true max-autotune: 1 UNLOCKED WIN + 3 honest`
> `parities + survey clean, ZERO aborts in 15 harness runs.`
> `... WIN: ch14:model_compile_reduced_precision max-autotune 1.130/1.075/1.137x vs default 0.994-1.005x`
> `(median 4.509 -> 4.165 ms = 1.083x over the incumbent arm, verify PASS x3)`

| Field | Extracted | Faithful? |
|---|---|---|
| verdict_class | RE_SWEEP_CLOSE (raw "RE-SWEEP CLOSE") | yes |
| target | ch14:model_compile_reduced_precision (+3 parities listed) | yes |
| baseline/optimized/unit | 4.509 / 4.165 / ms | yes |
| speedup | 1.083 | yes — "median ... = 1.083x over the incumbent arm" |
| verification | verify PASS x3; zero aborts in 15 runs; cold-cache counterfactual control | yes — lines 1886-1890 |
| evidence_doc | code/docs/gb300-sm103a-unlock.md | yes — file exists in repo |
| commit_sha | 8dd29579... | yes — "sm_103a unlock re-sweep: ch14 max-autotune win 1.083x (corrects B23c); B10/B16 closed at parity; B71" |

---

**Result: 6/6 spot-checks faithful.** Known representation choices (range strings,
parity rows with null metrics, the B65 collision) are documented in `SOURCE.md`
"Row semantics" and the payload's `field_conventions` block.
