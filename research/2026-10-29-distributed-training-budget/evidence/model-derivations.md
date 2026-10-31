# Distributed training example: exact derivations

This document defines the public hypothetical workbook. Numbers are arithmetic outputs, not measured GPU results. The code and tests reproduce them with Python's standard library. `system-budget.json` retains every per-rank resource total; `reader-summary.json` supplies compact figure labels.

## 1. Model and physical ranks

There are `DP=2` replicas, `PP=4` stages per replica, and eight physical ranks per stage: `2×4×8=64` GPUs. A stage occupies one node. Each rack has four nodes. A rank has identity `(replica,stage,local)`, with global ID `(replica×4+stage)×8+local` and node ID `replica×4+stage`.

The twelve layers divide evenly into three per stage. Each layer has Q/K/V/O attention projections, two RMSNorm gains, a router, and sixteen two-matrix GELU experts. Hidden width `H=2048`; expert width `F=4096`; query heads `Hq=16`; K/V heads `Hkv=2`; head dimension `D=128`. Causal self-attention uses one sequence of `S=2048` tokens per microbatch. Each replica processes eight microbatches; the global step has `2×8×2048=32768` tokens. Input embedding and output head are untied with vocabulary `V=32768`. They live on stage 0 and stage 3 respectively. A final RMSNorm lives on stage 3. There are no biases, shared experts or dropout. This is a synthetic model contract, not a named deployed model.

The attention rank layout is `local=cp×4+tp`. TP groups are `[0,1,2,3]` and `[4,5,6,7]`; CP groups are `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`. CP owns contiguous sequence halves; TP owns four query heads and one quarter of O's input features. The chosen head-sharded GQA policy gives K/V head 0 to TP 0/1 and head 1 to TP 2/3. K/V weight shards therefore have two TP copies and two CP copies. Q/O shards have two CP copies. The MoE reuses exactly these eight ranks as EP8: rank `e` owns experts `2e` and `2e+1`. EP, TP and CP do not multiply the physical device count.

Norm and router weights are replicated across all eight ranks. Their gradients use explicit unique-token contribution ownership, then one eight-rank sum. Redundant TP representations of a logical tensor do not create duplicate norm or router gradient terms.

## 2. Unique and allocated parameters

For one layer:

| Parameter family | Unique parameters |
| --- | ---: |
| Q and O | `2H² = 8,388,608` |
| K and V | `2H(Hkv D) = 1,048,576` |
| Sixteen two-matrix experts | `E×2HF = 268,435,456` |
| Router | `HE = 32,768` |
| Two RMSNorm gains | `2H = 4,096` |
| Total | `277,909,504` |

The model has `12×277,909,504+2VH+H = 3,469,133,824` unique parameters. Muon covers `12×277,872,640` parameters. AdamW covers the router/norm parameters, two `VH` matrices and the final norm.

One physical rank holds, per layer, `2H²/TP + 2HD + (E/8)×2HF = 36,175,872` Muon parameters and `HE+2H = 36,864` AdamW parameters. A stage has three such layers. Stages 0 and 3 additionally hold `VH/4 = 16,777,216` embedding or head parameters per rank; stage 3 has `H` final norm gains. Across all physical copies, the allocation is `7,489,748,992` parameter elements. That physical-copy number is a storage count; the DP gradient ledger and global norm use the unique count.

BF16 model weights plus FP32 master, gradient and one Muon momentum use `2+4+4+4=14` bytes per allocated Muon element. AdamW uses two FP32 moments, hence `18` bytes per allocated element. This unsharded optimizer-state choice deliberately leaves CP/TP redundant states allocated; a different state-sharding policy must change both capacity and communication.

## 3. Token ownership, routing and four exchange phases

After attention, every CP half has four TP representations. Select one 256-token contiguous chunk on each local rank. Token `i` selects attention owner `floor(i/256)`; its EP token owner is `(floor(i/256)+1) mod 8`. This is a bijection over the eight equal token chunks and is deliberately nonidentity, so a real reshard is visible.

The synthetic balanced route selects experts `i mod 16` and `(i+7) mod 16`, with gate weights 0.5 each. Every token has two distinct destinations and unit total gate weight. There are `top_k×S=4096` assignments, 256 per expert and 512 per physical rank. Exactly 3584 assignments cross an EP rank boundary; 512 stay local. Expert work is not repeated for TP copies.

A routed vector is `H×2=4096` bytes. Add 16 bytes of token ID, expert ID, gate weight and metadata. Dispatch, return/combine, backward dispatch and backward return each exchange that payload. The total remote routed bytes per layer/microbatch/replica are `3584×4112×4 = 58,949,632`. Across twelve layers, eight microbatches and two replicas, this is `11,318,329,344` bytes. An independent conservative count exchange sends four bytes to every other EP rank in each phase: `8×7×4×4×12×8×2 = 172,032` bytes.

Attention→EP shuffle sends one 1 MiB chunk per rank. EP→attention sends one inverse-shuffle chunk and allgathers its four chunks inside each TP group, for another `1+3=4 MiB` per rank. Backward mirrors these transitions, selecting a unique owner only after the TP derivative sum has completed. Total reshard payload is therefore `2×(1+4)=10 MiB` per rank/layer/microbatch. The step total is `10 MiB×64×3×8 = 16,106,127,360` bytes. Do not differentiate a tensor replication and also sum an already allreduced derivative a second time.

The routing precision and gradient summation choices are part of the contract. Activation and activation-gradient messages use BF16, while parameter gradients accumulate and communicate in FP32. The arithmetic checks establish logical ownership and conservation, not equivalence to a particular finite-precision training implementation.

## 4. Attention and boundary collectives

Each CP rank owns `S/2=1024` query tokens. Per rank/layer/microbatch, the local K+V segment has `1024×128×2×2 = 524,288` bytes. A forward CP2 allgather and a backward CP2 dK/dV reduce-scatter each send that amount per rank. Their step total is `1,610,612,736` bytes. This symmetric exchange is a declared policy: with contiguous causal halves, the forward high→low K/V send and backward low→high dK/dV send are unnecessary to the causal FLOP model. Those sends are `805,306,368` bytes of the total; no optimized causal exchange is claimed. Full gathered K/V are retained for backward, so no K/V matrix recomputation is inserted into the dense-GEMM count.

Each TP rank's replicated CP activation has `1024×2048×2 = 4 MiB`. A ring allreduce over `p=4` sends `2(p−1)/p=1.5` times that payload per participant. O's forward output reduction and QKV's backward input-gradient reduction require two such collectives, `12 MiB` per rank/layer/microbatch, totaling `19,327,352,832` bytes. The input embedding's forward output reduction and the final norm/head backward input-gradient reduction add `1,610,612,736` bytes. Vocabulary-parallel stable cross-entropy uses a maximum and a sum over 1024 FP32 scalars in each TP group, totaling `1,572,864` bytes.

A pipeline boundary sends only one unique 256-token chunk per rank, then the destination allgathers within TP4 to restore its CP token view. Forward and backward boundary payload is `2×S×H×2 = 16 MiB` per boundary/microbatch/replica. Across three boundaries, eight microbatches and two replicas this is `768 MiB = 805,306,368` off-node bytes. Destination reconstruction adds `2,415,919,104` intra-node bytes. Sending all four TP duplicates across nodes instead would be a different, more expensive policy.

## 5. Gradients, clipping and whole-matrix optimization

Microbatch losses are normalized by `1/32768` per token. Parameter gradients accumulate over eight local microbatches. Local sums complete before the DP sum: Q/O use CP2; K/V use CP2 plus the duplicate TP pair; router/norm use unique contributions across eight ranks; embedding/head gradients use CP2. Complete expert matrices have one EP owner. Only canonical unique shards enter DP2.

For ring allreduce, per-rank sent bytes are `2(p−1)/p×payload`. DP2 therefore sends one FP32 unique-gradient tensor's worth per replica: `2×3,469,133,824×4 = 27,753,070,592` bytes across the step. It is not `physical_parameter_copies×4`, and it is not doubled again by adding received bytes.

The primary plan declares full DP allreduce completion before its global L2 norm. This is a schedule choice, not forced by clipping alone: a norm only needs each unique reduced gradient at its term owner. Every unique parameter contributes exactly once; only replica 0 contributes norm terms, while replica 1 contributes zero to the final scalar sum. Stage-local scalars reach eight stage leaders, which allreduce one FP64 scalar; the result is distributed locally. The scalar path adds 896 intra-node bytes and 112 off-node bytes, with placement-dependent rack classification. Muon and AdamW use the same resulting clipping scalar.

Muon applies to Q, K, V, O and each complete expert MLP matrix. AdamW applies to embeddings, output head, router and norm gains. Attention matrices gather FP32 gradient, master weight and momentum (`12 B/parameter`) onto local rank 0. Q/O gather three of four shards; K/V gather one of two unique head shards. The owner updates the full matrix, scatters updated BF16 model/FP32 master/FP32 momentum (`10 B/parameter`), and publishes all redundant copies. Experts already hold whole matrices, so their Muon updates require no intra-stage matrix gather. AdamW's elementwise policy updates canonical embedding/head shards and norm/router copies, then publishes BF16 weight/FP32 master/two FP32 moments (`14 B/parameter`).

Gather traffic for the twelve layers and two replicas is `(¾×2H²+½×2H(HkvD))×12 B×12×2 = 1,962,934,272` bytes. Muon scatter/publication is `4,404,019,200` bytes. AdamW publication is `3,845,201,920` bytes. These are explicit costs of this simple owner policy. They do not establish that canonical gathering is the best implementation. Every updated matrix/state is published before the next-step global barrier releases a forward pass.

The declared five-iteration finite Newton–Schulz recurrence uses `n=min(rows,cols)`, `m=max(rows,cols)`, `A=XXᵀ`, `A2=A@A`, `B=bA+cA2`, `X=aX+B@X`. The dense work is `5×(4n²m+2n³)` FLOPs per matrix. Coefficients can be `a=3.4445,b=−4.7750,c=2.0315`; the operation count is unchanged by coefficient values. Momentum update, Frobenius normalization and weight update add a declared `9nm` scalar FLOPs; each iteration adds `3n²+2nm` scalar FLOPs. Square roots, divisions, clipping reductions, casts/copies and AdamW scalar work are separately excluded from dense-GEMM floors. A finite iteration is approximate, not exact SVD/polar decomposition. NS products couple shards; the small numerical test rejects treating independent shard transforms as this whole-matrix policy.

## 6. Training work and stage balance

For CP rank `c`, let its query indices be `[cS/2,(c+1)S/2)`. Its causal pair count is `q_end(q_end+1)/2−q_start(q_start+1)/2`. The two CP counts sum to `S(S+1)/2`; high-CP attention has more causal query work.

Forward per-rank projection work is `2(S/2)×(2H²/TP+2HD)` FLOPs. Forward causal QK/AV work is `4×(Hq/TP)×D×causal_pairs`. Forward expert work is `4×(top_k S/8)×HF`; forward router GEMM work is `2(S/8)HE`. Each dense forward has two dense backward products, so the training total is three times forward for these terms. Stage 3 adds the full head's forward plus two backward GEMMs: `6(S/2)H(V/TP)` per rank/microbatch. Embedding lookups are not dense matrix multiplications.

Across all physical ranks, training dense work is `127,075,734,257,664` FLOPs. Each of stages 0–2 accounts for `28,470,398,681,088` across both data replicas; stage 3 accounts for `41,664,538,214,400`. The full head adds `13,194,139,533,312` and makes stage 3's work `46.343361%` larger than the others. Intentional TP K/V projection duplication is included. This is dense work, not achieved utilization; softmax, RMSNorm, GELU, routing decisions, residuals, loss, communication kernels and indexing take additional work.

Muon dense work is `342,359,896,227,840` physical FLOPs for the separate optimizer phase, including both DP copies. It must not be added to training FLOPs to report a larger MFU. Canonical rank 0 owns extra attention optimizer work, so its work is used for the optimizer compute floor. Elementwise AdamW still consumes time even though it is not assigned a fabricated service estimate here.

## 7. Pipeline lifetime and memory

The fixed schedule is noninterleaved 1F1B: stage `s` warms up with `min(PP−s−1,M)` forwards, then alternates the next forward with the oldest backward, then drains backwards. Per-stage order, predecessor forward, successor backward and its own forward are all explicit DAG dependencies. With every F and B assigned one slot and transfers assigned zero latency, makespan is 22 slots; each stage is busy 16 slots. Maximum live forward activations are `[4,3,2,1]` microbatches. The six idle slots are `6/22=27.27%` of elapsed slots or `6/16=(PP−1)/M=37.5%` of busy slots. Both belong only to that equal-service illustration. The head imbalance and actual service rates are not calibrated into it.

Capacity is the sum of all declared state, retained tensors and workspace maxima; reusable maxima are deliberately summed for a conservative ledger. It retains CP/TP residual views, a separate unique EP residual copy, norm outputs/statistics, Q, full K/V, attention output, router logits/metadata, expert inputs/preactivations/outputs and combined output for every live microbatch. GELU output is recomputed pointwise from retained preactivation for backward. Model matrix work is not rematerialized. Stage 3 retains final norm input/output and two local FP32 logits-sized arrays for logits and dlogits, plus loss statistics. All eight microbatches' token IDs and targets have an explicit reserve.

Active backward gradient buffers, two expert stream slots per local expert, global route metadata, TP/CP collective buffers, PP buffers, 128 MiB gradient send/receive buckets, attention matrix gather reserve, NS workspace, 64 MiB publication buffer, FP32 dK/dV accumulators and norm scalar buffers are included. NS scratch reserves `4×(2nm+3n²)` bytes for the largest applicable whole matrix. No quadratic score tensor or deliberate dK/dV tile-partial array is part of this main-model allocation contract.

Balanced routing retains 512 expert assignments per rank/layer/microbatch. The skew stress case retains all 4096 assignments on one rank's two experts. Streaming 128 assignments per expert per chunk never drops work. Balanced experts need two chunks; skewed experts need sixteen. Chunking bounds transient staging, but does not erase retained backward tensors or the extra serial work on the hot owner. The performance resource screens use the balanced route only; the skew ledger is a capacity stress case.

All listed allocations are summed before any conditional capacity statement. The remaining device bytes are not available-memory measurements: framework/context/allocator reserve and fragmentation, additional implementation/GEMM/collective/transport buffers, graph pools, compilation and checkpoint allocations are excluded. There is no runnable implementation fit claim.

## 8. Placement and resource decision

Placement A maps replica 0's four stages to rack 0 and replica 1's stages to rack 1. DP2 crosses racks; all PP stays within a rack. Placement B maps stages 0/1 of both replicas to rack 0 and stages 2/3 to rack 1. All DP2 stays inside racks, while PP stage 1→2 crosses racks. The same directed logical message list is reclassified; total bytes and computation do not change.

Assumed capabilities, chosen for this thought experiment, are 500 TFLOP/s BF16 training dense work per GPU; 100 TFLOP/s FP32 Muon dense work per GPU; 3 TB/s HBM per GPU; 160 GB/s per GPU per direction inside a node; 40 GB/s per node NIC per direction; 160 GB/s aggregate intra-rack injected payload per rack; and 80 GB/s per rack uplink per direction. Endpoint links are full duplex. Intra-node fabric has no extra shared bottleneck in this model. These are generic hypothetical inputs, not a hardware specification or a measurement.

Each directed payload is counted once. Aggregate cross-rack payload is both directions added; the rack-uplink floor instead divides the busiest one direction by one direction's capacity. A's DP payload is `13,876,535,296` bytes in each direction, plus 14 norm-scalar bytes per direction. B's crossing PP payload is `134,217,728` bytes per direction plus 28 norm-scalar bytes per direction. Intra-node and intra-rack injection counts are not physical switch-hop expansion; protocol bytes are excluded.

The category-to-phase map partitions every directed transfer exactly once. `attention_cp_kv`, `attention_tp_activation`, `boundary_activation_tp`, `data_parallel_unique_gradients`, `expert_count_exchange`, `expert_dispatch_combine`, `gradient_boundary`, `gradient_local_adam`, `gradient_local_attention`, `head_cross_entropy`, `pipeline_boundary`, `pipeline_reconstruct` and `token_owner_reshard` are pre-clipping. `global_clip_norm` is at the clipping barrier. `muon_whole_matrix_gather`, `muon_scatter_publish` and `adam_state_publish` follow clipping. Thus every off-node byte except the norm scalars is pre-clipping in this declared full-allreduce-before-norm plan. The plan chooses full DP completion before replica 0 supplies all norm terms; clipping alone does not require allgather first. Gather/state publication traffic is intra-node and post-clipping.

Compute floors divide the busiest rank's work by its assumed per-GPU capability. For this declared phase assignment, pre-clipping service is at least the maximum of training compute and that phase's GPU/node/rack payload screens. Post-clipping service is at least the maximum of FP32 Muon compute, partial optimizer HBM and that phase's payload screens. Add these two phase maxima: the chosen schedule places the norm barrier between them even though the busiest training and optimizer ranks differ. Muon gathering after clipping is a declared placement of work; gathering alone does not require the scalar. No pair of simultaneous node/rack/GPU resources is added within a phase. Norm-scalar service between phases is conservatively omitted from the ordered sum; full-step resource screens retain every scalar byte. This is a necessary floor, not a modeled DAG makespan or assumed attainable overlap.

A's binding pre-clipping rack direction is `13,876,535,296/80e9 = 0.1734566912 s`. B's binding pre-clipping node is `3,670,466,560/40e9 = 0.091761664 s`, sent and received separately. Its pre-clipping largest within-rack injection also requires `14,144,978,944/160e9 = 0.0884061184 s`; these resource screens overlap and are not added. The busiest rank's post-clipping FP32 Muon count is `6,717,261,742,080/100e12 = 0.0671726174208 s`. The ordered necessary floors are therefore A `0.2406293086208 s` and B `0.1589342814208 s`.

The partial optimizer HBM floor uses 22 bytes per canonically updated Muon parameter and 30 per AdamW parameter for one persistent-state read/write, then grants the assumed 128 MiB phase cache against both reads and writes. It omits all other HBM activity and is intentionally permissive. The selected FP32 Newton–Schulz recurrence departs from the pinned Muon reference's BF16 input/iteration; its 67.17 ms bound is a cost of this declared policy. This teaching-scale update has only 32,768 tokens; larger token budgets can change training's share of the floor.

For a proposed step budget `T`, reject the design if any necessary floor exceeds `T`. At the fixed illustrative `T=150 ms`, both A and B are excluded for the primary full-DP-allreduce-before-norm schedule together with the FP32 single-owner attention Muon policy. Placement reduces the floor but these choices together leave B at 158.93 ms. If all floors for a different budget are below `T`, the design is merely not ruled out. Stronger rack service or a different collective, optimizer precision or ownership policy requires a separately stated model. The comparison does not prove B runs faster, and the ratio between floors must not be reported as speedup.

### Alternative schedule without changing the primary comparison

An analytical alternative splits each DP2 directed allreduce payload into equal reduce-scatter and allgather halves. Reduce-scatter precedes the global norm. Replica 0 and replica 1 each contribute disjoint norm terms for `1,734,566,912` unique parameter elements, totaling `3,469,133,824` exactly once. The global scalar route and bytes remain unchanged. After the same norm/common clipping, allgather restores the full reduced gradients while the phase floor grants optimistic overlap with the same FP32 single-owner attention Muon work. Whole-matrix updates still depend on their consumed gradient shards; no full DAG feasibility is demonstrated. This preserves mathematical gradient/norm/update semantics and every total directed payload.

A sends `6,938,267,648` DP bytes per rack direction in each half: pre `6,938,267,648/80e9=0.0867283456 s`; post `max(0.0867283456,0.0671726174208)=0.0867283456 s`; sum `0.1734566912 s`.

For B, stage 3 has `(3×277,909,504+32768×2048+2048)×4=3,603,357,696` DP bytes. Half plus its `8×2048×2048×2=67,108,864` sent PP bytes gives `1,868,787,712` pre-clipping node bytes. Its pre floor is `1,868,787,712/40e9=0.0467196928 s`. The post-clipping DP node floor is `1,801,678,848/40e9=0.0450419712 s`, below FP32 Muon's `0.0671726174208 s`, so the ordered alternative sum is `0.1138923102208 s`.

Norm-scalar bytes are retained in alternative traffic ledgers but their service is omitted from every alternative screen, including full-step screens. This explains why A's stated permissive alternative floor excludes the tiny scalar cost. At 150 ms, alternative A is excluded and B is merely not ruled out. These are alternative necessary floors, not a replacement primary baseline, executable schedule, feasibility proof, runtime or speedup.

## 9. Independent attention companion mapping

The companion uses causal MQA `N=4096,Hq=16,Hkv=1,D=128`, with 128-token Q/K tiles and unique full FP32 dK/dV partials per head-tilepair. It is not this main model. There are `32×33/2×16=8448` head-tilepairs and `8448×2×128×128×4=1056 MiB` of scratch.

If four query heads go to each TP rank and the 32 query tiles divide into two CP halves, low-CP rank tilepair count is `4×sum(1..16)=544`, scratch 68 MiB. High CP gets `4×sum(17..32)=1568`, scratch 196 MiB. Across eight ranks, `4×(68+196)=1056 MiB`, conserving unique partial work. At an ideal 128 MiB cache per rank, the one-write/one-read scratch floor is zero for each low rank and `2×(196−128)=136 MiB` for each high rank. Its per-high-rank division by a separate assumed 8 TB/s is `0.017825792 ms`; this is not an aggregate divided by one GPU and not a full attention or step time. CP/TP gradient reductions and the changed system dataflow need their own accounting before connecting it to any distributed training budget.

The exact generated ledgers follow. All bytes below are integers; GB uses `10^9` bytes, GiB uses `2^30` bytes. GB/s uses `10^9` bytes/s, TB/s uses `10^12` bytes/s and TFLOP/s uses `10^12` FLOPs/s. The companion's 8 TB/s is separate from the main workbook's 3 TB/s HBM assumption, and no main floor consumes companion quantities.

## Generated exact traffic ledger

| Category | A intra-node B | A intra-rack B | A inter-rack B | B intra-rack B | B inter-rack B |
| --- | ---: | ---: | ---: | ---: | ---: |
| adam state publish | 3,845,201,920 | 0 | 0 | 0 | 0 |
| attention cp kv | 1,610,612,736 | 0 | 0 | 0 | 0 |
| attention tp activation | 19,327,352,832 | 0 | 0 | 0 | 0 |
| boundary activation tp | 1,610,612,736 | 0 | 0 | 0 | 0 |
| data parallel unique gradients | 0 | 0 | 27,753,070,592 | 27,753,070,592 | 0 |
| expert count exchange | 172,032 | 0 | 0 | 0 | 0 |
| expert dispatch combine | 11,318,329,344 | 0 | 0 | 0 | 0 |
| global clip norm | 896 | 84 | 28 | 56 | 56 |
| gradient boundary | 2,147,483,648 | 0 | 0 | 0 | 0 |
| gradient local adam | 49,774,592 | 0 | 0 | 0 | 0 |
| gradient local attention | 2,214,592,512 | 0 | 0 | 0 | 0 |
| head cross entropy | 1,572,864 | 0 | 0 | 0 | 0 |
| muon scatter publish | 4,404,019,200 | 0 | 0 | 0 | 0 |
| muon whole matrix gather | 1,962,934,272 | 0 | 0 | 0 | 0 |
| pipeline boundary | 0 | 805,306,368 | 0 | 536,870,912 | 268,435,456 |
| pipeline reconstruct | 2,415,919,104 | 0 | 0 | 0 | 0 |
| token owner reshard | 16,106,127,360 | 0 | 0 | 0 | 0 |

## Generated resource screens

| Necessary screen | A milliseconds | B milliseconds |
| --- | ---: | ---: |
| barrier ordered phase floor | 240.629308621 | 158.934281421 |
| gpu intranode payload | 9.339089150 | 9.339089150 |
| necessary step floor | 240.629308621 | 158.934281421 |
| node offnode payload | 91.761664350 | 91.761664350 |
| optimizer Muon dense matmul | 67.172617421 | 67.172617421 |
| optimizer persistent state HBM partial | 1.025235627 | 1.025235627 |
| ordered training plus Muon compute | 72.535303520 | 72.535303520 |
| post clipping gpu intranode payload | 3.503078400 | 3.503078400 |
| post clipping node offnode payload | 0.000000000 | 0.000000000 |
| post clipping phase | 67.172617421 | 67.172617421 |
| post clipping rack interrack payload | 0.000000000 | 0.000000000 |
| post clipping rack intrarack payload | 0.000000000 | 0.000000000 |
| pre clipping gpu intranode payload | 5.836010400 | 5.836010400 |
| pre clipping node offnode payload | 91.761664000 | 91.761664000 |
| pre clipping phase | 173.456691200 | 91.761664000 |
| pre clipping rack interrack payload | 173.456691200 | 1.677721600 |
| pre clipping rack intrarack payload | 2.516582400 | 88.406118400 |
| rack interrack payload | 173.456691375 | 1.677721950 |
| rack intrarack payload | 2.516582662 | 88.406118575 |
| training dense matmul | 5.362686099 | 5.362686099 |

### Analytical reduce-scatter-first alternative

Same total payload/math; norm service omitted. Post allgather and optimizer receive optimistic overlap, not a demonstrated schedule.

| Necessary screen | A milliseconds | B milliseconds |
| --- | ---: | ---: |
| pre clipping phase | 86.728345600 | 46.719692800 |
| post clipping phase | 86.728345600 | 67.172617421 |
| necessary step floor | 173.456691200 | 113.892310221 |

## Generated declared balanced allocation ledger

All columns are bytes per physical rank. Repeated state sizes refer to each stage's per-rank parameter allocation. These are conditional declared sums, not observed allocations.

| Allocation | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
| --- | ---: | ---: | ---: | ---: |
| BF16 model weights | 250,830,848 | 217,276,416 | 217,276,416 | 250,834,944 |
| FP32 Adam first and second moments | 135,102,464 | 884,736 | 884,736 | 135,118,848 |
| FP32 Muon momentum | 434,110,464 | 434,110,464 | 434,110,464 | 434,110,464 |
| FP32 accumulated gradients | 501,661,696 | 434,552,832 | 434,552,832 | 501,669,888 |
| FP32 gradient reduction bucket send receive | 134,217,728 | 134,217,728 | 134,217,728 | 134,217,728 |
| FP32 master weights | 501,661,696 | 434,552,832 | 434,552,832 | 501,669,888 |
| Muon NS matrix workspace | 117,440,512 | 117,440,512 | 117,440,512 | 117,440,512 |
| TP CP collective buffers | 9,437,184 | 9,437,184 | 9,437,184 | 9,437,184 |
| active backward tensor gradients conservative | 54,618,112 | 54,618,112 | 54,618,112 | 54,618,112 |
| all step token ids and targets int64 | 262,144 | 262,144 | 262,144 | 262,144 |
| attention backward dK dV FP32 | 2,097,152 | 2,097,152 | 2,097,152 | 2,097,152 |
| expert stream double buffer | 2,105,344 | 2,105,344 | 2,105,344 | 2,105,344 |
| global norm scalar buffers | 64 | 64 | 64 | 64 |
| optimizer publish buffer | 67,108,864 | 67,108,864 | 67,108,864 | 67,108,864 |
| pipeline send receive buffers | 2,097,152 | 2,097,152 | 2,097,152 | 2,097,152 |
| retained layer activations | 327,708,672 | 245,781,504 | 163,854,336 | 81,927,168 |
| route count and sort metadata full sequence | 65,792 | 65,792 | 65,792 | 65,792 |
| whole attention matrix gather reserve | 50,331,648 | 50,331,648 | 50,331,648 | 50,331,648 |
| cross entropy statistics FP32 | 0 | 0 | 0 | 8,192 |
| head logits and dlogits FP32 | 0 | 0 | 0 | 67,108,864 |
| retained final norm input output stats | 0 | 0 | 0 | 8,392,704 |
| **Total declared live** | **2,590,857,536** | **2,206,940,480** | **2,125,013,312** | **2,420,622,656** |
| **Worst-skew declared live** | **3,412,941,120** | **2,852,863,296** | **2,594,775,360** | **2,714,223,936** |

## Generated equal-unit 1F1B rows

Each column is one assumed unit, and `..` is idle. F/B service is one unit for this illustration only.

```text
stage 0: F0 F1 F2 F3 .. .. .. B0 F4 B1 F5 B2 F6 B3 F7 B4 .. B5 .. B6 .. B7
stage 1: .. F0 F1 F2 .. .. B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 .. B6 .. B7 ..
stage 2: .. .. F0 F1 .. B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 .. B7 .. ..
stage 3: .. .. .. F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 .. .. ..
```
