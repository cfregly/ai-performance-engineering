#!/usr/bin/env python3
"""A public hypothetical training-system workbook, Python standard library only.

This counts declared logical payloads, mathematical dense-matmul work and a
declared allocation contract. It neither benchmarks a stack nor predicts a step
time. Byte counts exclude transport headers/retries; resource floors are
necessary conditions under user-supplied assumed capabilities.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path


# Dependency phases of the declared message plan. Unknown categories must fail
# rather than silently granting optimizer overlap to newly added traffic.
TRAFFIC_PHASE = {
    **dict.fromkeys(("attention_cp_kv", "attention_tp_activation",
                    "boundary_activation_tp", "data_parallel_unique_gradients",
                    "expert_count_exchange", "expert_dispatch_combine",
                    "gradient_boundary", "gradient_local_adam",
                    "gradient_local_attention", "head_cross_entropy",
                    "pipeline_boundary", "pipeline_reconstruct",
                    "token_owner_reshard"), "pre_clipping"),
    "global_clip_norm": "clipping_barrier",
    **dict.fromkeys(("muon_whole_matrix_gather", "muon_scatter_publish",
                    "adam_state_publish"), "post_clipping"),
}
REDUCE_SCATTER_FIRST_PHASE = {
    **{k: v for k, v in TRAFFIC_PHASE.items() if k != "data_parallel_unique_gradients"},
    "data_parallel_reduce_scatter": "pre_clipping",
    "data_parallel_allgather": "post_clipping",
}


@dataclass(frozen=True)
class Inputs:
    replicas: int = 2
    stages: int = 4
    ranks_per_stage: int = 8
    racks: int = 2
    layers: int = 12
    hidden: int = 2048
    expert_width: int = 4096
    experts: int = 16
    top_k: int = 2
    sequence: int = 2048
    microbatches: int = 8
    vocab: int = 32768
    query_heads: int = 16
    kv_heads: int = 2
    head_dim: int = 128
    tp: int = 4
    cp: int = 2
    bf16_bytes: int = 2
    fp32_bytes: int = 4
    routing_metadata_bytes: int = 16
    expert_chunk_tokens: int = 128
    newton_schulz_iterations: int = 5
    device_capacity_bytes: int = 80_000_000_000
    assumed_training_matmul_flops_s: int = 500_000_000_000_000
    assumed_optimizer_fp32_flops_s: int = 100_000_000_000_000
    assumed_hbm_bytes_s: int = 3_000_000_000_000
    assumed_optimizer_phase_cache_bytes: int = 128 * 1024 ** 2
    assumed_gpu_intranode_send_bytes_s: int = 160_000_000_000
    assumed_node_offnode_send_bytes_s: int = 40_000_000_000
    assumed_rack_intrarack_send_bytes_s: int = 160_000_000_000
    assumed_rack_interrack_send_bytes_s: int = 80_000_000_000

    def validate(self):
        assert self.replicas == 2 and self.stages == 4 and self.racks == 2
        assert self.ranks_per_stage == self.tp * self.cp == 8
        assert self.experts % self.ranks_per_stage == 0
        assert self.layers % self.stages == 0
        assert self.query_heads * self.head_dim == self.hidden
        assert self.query_heads % self.tp == 0
        assert self.tp % self.kv_heads == 0  # K/V duplicate within TP pairs.
        assert self.sequence % self.ranks_per_stage == 0
        assert self.sequence % self.cp == 0 and self.vocab % self.tp == 0
        assert self.top_k == 2 and self.experts == 16


def gid(replica, stage, local, x=Inputs()):
    return (replica * x.stages + stage) * x.ranks_per_stage + local


def identity(rank, x=Inputs()):
    node, local = divmod(rank, x.ranks_per_stage)
    replica, stage = divmod(node, x.stages)
    return replica, stage, local


def rack_of(rank, placement, x=Inputs()):
    replica, stage, _ = identity(rank, x)
    if placement == "A":
        return replica
    if placement == "B":
        return stage // 2
    raise ValueError(placement)


class Traffic:
    def __init__(self, x, phase_by_category=None):
        self.x = x
        self.transfers = []
        self.phase_by_category = TRAFFIC_PHASE if phase_by_category is None else phase_by_category

    def send(self, category, source, target, nbytes):
        assert nbytes >= 0 and int(nbytes) == nbytes
        if source != target and nbytes:
            self.transfers.append((category, source, target, int(nbytes)))

    def ring_allreduce(self, category, ranks, payload):
        """Sum reduce-scatter + allgather, direct ring links, sent bytes."""
        p = len(ranks)
        amount = Fraction(2 * (p - 1), p) * payload
        assert amount.denominator == 1
        for i, rank in enumerate(ranks):
            self.send(category, rank, ranks[(i + 1) % p], amount.numerator)

    def ring_allgather(self, category, ranks, full_payload):
        p = len(ranks)
        amount = Fraction(p - 1, p) * full_payload
        assert amount.denominator == 1
        for i, rank in enumerate(ranks):
            self.send(category, rank, ranks[(i + 1) % p], amount.numerator)

    def broadcast(self, category, source, targets, payload):
        # Explicit direct fanout, not an uncounted ideal multicast.
        for target in targets:
            self.send(category, source, target, payload)

    def summary(self, placement, phase=None, exclude_norm=False):
        by_category = defaultdict(lambda: defaultdict(int))
        by_gpu = defaultdict(lambda: defaultdict(int))
        by_gpu_received = defaultdict(lambda: defaultdict(int))
        by_node = defaultdict(int)
        by_node_received = defaultdict(int)
        intra_rack = defaultdict(int)
        inter_rack = defaultdict(int)
        for category, source, target, amount in self.transfers:
            if exclude_norm and category == "global_clip_norm":
                continue
            category_phase = self.phase_by_category[category] if phase is not None else None
            if phase is not None and category_phase != phase:
                continue
            sn, tn = source // self.x.ranks_per_stage, target // self.x.ranks_per_stage
            sr, tr = rack_of(source, placement, self.x), rack_of(target, placement, self.x)
            tier = "intra_node" if sn == tn else ("intra_rack" if sr == tr else "inter_rack")
            by_category[category][tier] += amount
            by_gpu[source][tier] += amount
            by_gpu_received[target][tier] += amount
            if sn != tn:
                by_node[sn] += amount
                by_node_received[tn] += amount
            if tier == "intra_rack":
                intra_rack[sr] += amount
            if tier == "inter_rack":
                inter_rack[f"{sr}->{tr}"] += amount
        totals = {tier: sum(v.get(tier, 0) for v in by_category.values())
                  for tier in ("intra_node", "intra_rack", "inter_rack")}
        return {
            "basis": "aggregate logical bytes SENT, each directed transfer counted once",
            "totals_bytes": totals,
            "by_category_bytes": dict(sorted(by_category.items())),
            "per_gpu_sent_bytes": {str(k): dict(v) for k, v in sorted(by_gpu.items())},
            "per_gpu_received_bytes": {str(k): dict(v) for k, v in sorted(by_gpu_received.items())},
            "per_node_offnode_sent_bytes": dict(sorted(by_node.items())),
            "per_node_offnode_received_bytes": dict(sorted(by_node_received.items())),
            "per_rack_intrarack_sent_bytes": dict(sorted(intra_rack.items())),
            "interrack_directed_bytes": dict(sorted(inter_rack.items())),
            "max_gpu_intranode_sent_bytes": max((v["intra_node"] for v in by_gpu.values()), default=0),
            "max_gpu_intranode_received_bytes": max((v["intra_node"] for v in by_gpu_received.values()), default=0),
            "max_node_offnode_sent_bytes": max(by_node.values(), default=0),
            "max_node_offnode_received_bytes": max(by_node_received.values(), default=0),
        }


def parameters(x):
    h, f, v = x.hidden, x.expert_width, x.vocab
    matrices = [("Q", h, h), ("K", h, x.kv_heads * x.head_dim),
                ("V", h, x.kv_heads * x.head_dim), ("O", h, h)]
    unique_muon_layer = sum(a * b for _, a, b in matrices) + x.experts * 2 * h * f
    unique_adam_layer = h * x.experts + 2 * h
    unique_total = x.layers * (unique_muon_layer + unique_adam_layer) + 2 * v * h + h
    local_muon_layer = 2 * h * h // x.tp + 2 * h * x.head_dim + (x.experts // 8) * 2 * h * f
    local_adam_layer = unique_adam_layer
    return {
        "attention_muon_matrices": [{"name": n, "rows": a, "cols": b} for n, a, b in matrices],
        "unique_muon_parameters_per_layer": unique_muon_layer,
        "unique_adam_parameters_per_layer": unique_adam_layer,
        "unique_model_parameters": unique_total,
        "local_muon_parameters_per_layer": local_muon_layer,
        "local_adam_parameters_per_layer": local_adam_layer,
        "physical_parameter_copies": sum(
            (local_muon_layer + local_adam_layer) * (x.layers // 4)
            + (v * h // x.tp if stage in (0, 3) else 0)
            + (h if stage == 3 else 0)
            for _replica in range(2) for stage in range(4) for _local in range(8)),
    }


def token_mapping(x):
    """A nonidentity reshard: selected TP chunk -> next EP rank, no copies."""
    chunk = x.sequence // 8
    return [{"token": token, "cp": token // (x.sequence // 2),
             "selected_tp": (token // chunk) % 4,
             "attention_owner_rank": token // chunk,
             "expert_token_owner_rank": (token // chunk + 1) % 8}
            for token in range(x.sequence)]


def routes(x, skew=False):
    mapping = token_mapping(x)
    result = []
    for row in mapping:
        token, owner = row["token"], row["expert_token_owner_rank"]
        expert_ids = (0, 1) if skew else (token % x.experts, (token + 7) % x.experts)
        for expert in expert_ids:
            result.append({"token": token, "owner": owner, "expert": expert,
                           "destination": expert // (x.experts // 8), "weight": 0.5})
    return result


def chunk_routes(x, skew=False):
    by_expert = defaultdict(list)
    for route in routes(x, skew):
        by_expert[route["expert"]].append(route)
    return [(expert, items[start:start + x.expert_chunk_tokens])
            for expert, items in sorted(by_expert.items())
            for start in range(0, len(items), x.expert_chunk_tokens)]


def build_traffic(x):
    t = Traffic(x)
    h, n, b = x.hidden, x.sequence, x.bf16_bytes
    layers = x.layers // 4
    for replica in range(2):
        for stage in range(4):
            ranks = [gid(replica, stage, local, x) for local in range(8)]
            tp_groups = [ranks[cp * 4:cp * 4 + 4] for cp in range(2)]
            per_cp_activation = n // 2 * h * b
            cp_kv_payload = n // 2 * x.head_dim * 2 * b
            for _layer in range(layers):
                # Declared symmetric forward KV allgather and backward dKV
                # reduce-scatter policy, including causally unnecessary sends.
                # It retains TP K/V duplication and distinct CP query work.
                for tp in range(4):
                    a, z = ranks[tp], ranks[4 + tp]
                    for source, target in ((a, z), (z, a)):
                        t.send("attention_cp_kv", source, target, 2 * cp_kv_payload * x.microbatches)
                # O output reduction in F, QKV input-gradient reduction in B.
                for group in tp_groups:
                    t.ring_allreduce("attention_tp_activation", group,
                                     2 * per_cp_activation * x.microbatches)
                # CP2TP4 -> unique EP8 chunks (one cyclic permutation). Reverse
                # path needs inverse permutation and TP token allgather. The
                # backward paths use already reduced logical gradients and
                # select one owner; no second sum of duplicate TP derivatives.
                chunk_payload = n // 8 * h * b
                for local, source in enumerate(ranks):
                    target = ranks[(local + 1) % 8]
                    t.send("token_owner_reshard", source, target, 2 * chunk_payload * x.microbatches)
                    t.send("token_owner_reshard", target, source, 2 * chunk_payload * x.microbatches)
                for group in tp_groups:
                    t.ring_allgather("token_owner_reshard", group,
                                     2 * per_cp_activation * x.microbatches)
                # Four activation-vector exchanges: dispatch/combine and their
                # backward adjoints. Balanced exact routing is enumerated.
                route_bytes = h * b + x.routing_metadata_bytes
                for route in routes(x):
                    src, dst = ranks[route["owner"]], ranks[route["destination"]]
                    t.send("expert_dispatch_combine", src, dst, 2 * route_bytes * x.microbatches)
                    t.send("expert_dispatch_combine", dst, src, 2 * route_bytes * x.microbatches)
                # Conservative independent count exchange in each of 4 phases.
                for source in ranks:
                    for target in ranks:
                        t.send("expert_count_exchange", source, target, 4 * 4 * x.microbatches)
                # Accumulate all microbatches first, then sum replicated weight
                # gradients. Q/O across CP2; K/V across CP2 x duplicate TP pair.
                for tp in range(4):
                    t.ring_allreduce("gradient_local_attention", [ranks[tp], ranks[4 + tp]],
                                     2 * h * h // 4 * 4)
                for pair in (0, 2):
                    t.ring_allreduce("gradient_local_attention",
                                     [ranks[pair], ranks[pair + 1], ranks[4 + pair], ranks[5 + pair]],
                                     2 * h * x.head_dim * 4)
                t.ring_allreduce("gradient_local_adam", ranks, (h * x.experts + 2 * h) * 4)

                # Whole-matrix Muon: gather gradient/master/momentum (12 B/p)
                # onto local rank 0, update there, scatter state (10 B/p), then
                # publish state to every redundant CP/TP copy (10 B/p).
                # Experts already own complete matrices and need no gather.
                for name in ("Q", "O"):
                    shard = h * h // 4
                    for tp in range(1, 4):
                        t.send("muon_whole_matrix_gather", ranks[tp], ranks[0], shard * 12)
                        t.send("muon_scatter_publish", ranks[0], ranks[tp], shard * 10)
                    for tp in range(4):
                        t.send("muon_scatter_publish", ranks[tp], ranks[4 + tp], shard * 10)
                for name in ("K", "V"):
                    shard = h * x.head_dim
                    t.send("muon_whole_matrix_gather", ranks[2], ranks[0], shard * 12)
                    t.send("muon_scatter_publish", ranks[0], ranks[2], shard * 10)
                    for canonical in (0, 2):
                        t.broadcast("muon_scatter_publish", ranks[canonical],
                                    [ranks[canonical + 1], ranks[canonical + 4], ranks[canonical + 5]],
                                    shard * 10)
                t.broadcast("adam_state_publish", ranks[0], ranks[1:],
                            (h * x.experts + 2 * h) * 14)

            if stage in (0, 3):
                boundary_shard = x.vocab * h // 4
                for tp in range(4):
                    t.ring_allreduce("gradient_boundary", [ranks[tp], ranks[4 + tp]], boundary_shard * 4)
                    t.send("adam_state_publish", ranks[tp], ranks[4 + tp], boundary_shard * 14)
                for group in tp_groups:
                    # Input embedding F or final norm input-gradient B.
                    t.ring_allreduce("boundary_activation_tp", group, per_cp_activation * x.microbatches)
            if stage == 3:
                t.ring_allreduce("gradient_local_adam", ranks, h * 4)
                t.broadcast("adam_state_publish", ranks[0], ranks[1:], h * 14)
                for group in tp_groups:
                    # Vocabulary-parallel stable cross entropy: maximum + sum.
                    t.ring_allreduce("head_cross_entropy", group, 2 * (n // 2) * 4 * x.microbatches)

            # Norm: sum local unique parameter-gradient squares to stage leader.
            # Only replica 0 contributes stage sums to the following global
            # allreduce, so duplicate DP gradients are not counted twice.
            for local in range(1, 8):
                t.send("global_clip_norm", ranks[local], ranks[0], 8)
            t.broadcast("global_clip_norm", ranks[0], ranks[1:], 8)

        # Pipeline sends only a unique sequence chunk per TP rank; destination
        # reconstructs its CP2TP4 logical view inside the node. Same for dX.
        for stage in range(3):
            for local in range(8):
                source, target = gid(replica, stage, local, x), gid(replica, stage + 1, local, x)
                chunk = n // 8 * h * b * x.microbatches
                t.send("pipeline_boundary", source, target, chunk)
                t.send("pipeline_boundary", target, source, chunk)
            for endpoint in (stage, stage + 1):
                for cp in range(2):
                    group = [gid(replica, endpoint, cp * 4 + tp, x) for tp in range(4)]
                    t.ring_allgather("pipeline_reconstruct", group, n // 2 * h * b * x.microbatches)

    # DP allreduce only unique parameter shards, NOT all physical replicas.
    p = parameters(x)
    for stage in range(4):
        counts = [0] * 8
        for local in range(8):
            counts[local] += layers * (x.experts // 8) * 2 * h * x.expert_width
        for tp in range(4):
            counts[tp] += layers * 2 * h * h // 4
        for canonical in (0, 2):
            counts[canonical] += layers * 2 * h * x.head_dim
        counts[0] += layers * (h * x.experts + 2 * h)
        if stage in (0, 3):
            for tp in range(4):
                counts[tp] += x.vocab * h // 4
        if stage == 3:
            counts[0] += h
        for local, count in enumerate(counts):
            t.ring_allreduce("data_parallel_unique_gradients",
                             [gid(0, stage, local, x), gid(1, stage, local, x)], count * 4)
    leaders = [gid(replica, stage, 0, x) for replica in range(2) for stage in range(4)]
    t.ring_allreduce("global_clip_norm", leaders, 8)
    return t


def reduce_scatter_first_traffic(original):
    """Same DP2 ring payload, with RS before norm and allgather after clip.

    These are two explicit directed transfers per original DP transfer. The
    norm scalar path is unchanged, but both replicas now contribute disjoint
    reduced-gradient norm terms. This is an alternative analytical schedule.
    """
    alternative = Traffic(original.x, REDUCE_SCATTER_FIRST_PHASE)
    for category, source, target, amount in original.transfers:
        if category == "data_parallel_unique_gradients":
            assert amount % 2 == 0
            alternative.send("data_parallel_reduce_scatter", source, target, amount // 2)
            alternative.send("data_parallel_allgather", source, target, amount // 2)
        else:
            alternative.send(category, source, target, amount)
    return alternative


def norm_ownership(original, reduce_scatter_first=False):
    """Intervals index unique canonical parameters, never physical copies."""
    intervals = []
    cursor = 0
    totals = {"0": 0, "1": 0}
    for category, source, target, amount in original.transfers:
        replica, stage, local = identity(source, original.x)
        if category != "data_parallel_unique_gradients" or replica != 0:
            continue
        count = amount // original.x.fp32_bytes
        assert count * original.x.fp32_bytes == amount
        owners = [(source, cursor, cursor + count)]
        if reduce_scatter_first:
            assert count % 2 == 0
            owners = [(source, cursor, cursor + count // 2),
                      (target, cursor + count // 2, cursor + count)]
        for owner, begin, end in owners:
            dp, _, _ = identity(owner, original.x)
            intervals.append({"owner_rank": owner, "replica": dp,
                              "unique_parameter_begin": begin,
                              "unique_parameter_end": end,
                              "elements": end - begin, "stage": stage, "local": local})
            totals[str(dp)] += end - begin
        cursor += count
    assert cursor == parameters(original.x)["unique_model_parameters"]
    return {"policy": "both DP replicas contribute disjoint reduced halves" if reduce_scatter_first else "replica 0 contributes all unique norm terms; replica 1 contributes zero",
            "unique_parameter_elements": cursor, "elements_by_replica": totals,
            "owner_intervals": intervals}


def ns_work(rows, cols, iterations):
    """Declared finite NS recurrence; counts dense matmuls and scalar work.

    n=min(rows,cols), m=max. Each iteration forms A=XX^T, A2=A@A,
    B=bA+cA2, X=aX+B@X. This is an approximate iterative transform, not
    an exact polar factor. GEMM FLOPs convention is 2*m*n*k.
    """
    n, m = sorted((rows, cols))
    p = n * m
    return {"dense_matmul_flops": iterations * (4 * n * n * m + 2 * n ** 3),
            "elementwise_flops": 9 * p + iterations * (3 * n * n + 2 * p),
            "matrix_workspace_bytes": (2 * p + 3 * n * n) * 4}


def compute(x):
    h, n, f = x.hidden, x.sequence, x.expert_width
    by_rank = []
    per_stage = defaultdict(int)
    muon_shapes = [(m["rows"], m["cols"]) for m in parameters(x)["attention_muon_matrices"]]
    for rank in range(64):
        replica, stage, local = identity(rank, x)
        cp = local // 4
        query_end, query_start = (cp + 1) * n // 2, cp * n // 2
        causal_pairs = query_end * (query_end + 1) // 2 - query_start * (query_start + 1) // 2
        projection_forward = 2 * (n // 2) * (2 * h * h // 4 + 2 * h * x.head_dim)
        attention_forward = 4 * (x.query_heads // 4) * x.head_dim * causal_pairs
        expert_forward = 4 * (x.top_k * n // 8) * h * f
        router_forward = 2 * (n // 8) * h * x.experts
        layer_training = 3 * (projection_forward + attention_forward + expert_forward + router_forward)
        training = (x.layers // 4) * x.microbatches * layer_training
        if stage == 3:
            training += x.microbatches * 6 * (n // 2) * h * (x.vocab // 4)
        # Muon update executes once per parameter per replica. Attention whole
        # matrices update serially on stage-local rank 0. Four complete expert
        # matrices per rank (two experts, two matrices each) update locally.
        expert_ns = ns_work(h, f, x.newton_schulz_iterations)
        optimizer = (x.layers // 4) * 4 * expert_ns["dense_matmul_flops"]
        optimizer_elementwise = (x.layers // 4) * 4 * expert_ns["elementwise_flops"]
        muon_updated = (x.layers // 4) * 4 * h * f
        adam_updated = 0
        if local == 0:
            for a, b in muon_shapes:
                work = ns_work(a, b, x.newton_schulz_iterations)
                optimizer += (x.layers // 4) * work["dense_matmul_flops"]
                optimizer_elementwise += (x.layers // 4) * work["elementwise_flops"]
                muon_updated += (x.layers // 4) * a * b
            adam_updated += (x.layers // 4) * (h * x.experts + 2 * h)
            if stage == 3:
                adam_updated += h
        if stage in (0, 3) and local < 4:
            adam_updated += x.vocab * h // 4
        # One read/write of persistent optimizer state, irrespective of cache
        # traffic: Muon 4(g)+4(m read)+4(m write)+4(w read)+4(w write)+2(model).
        # Adam adds second moment read/write: 30 bytes per updated parameter.
        hbm = muon_updated * 22 + adam_updated * 30
        row = {"rank": rank, "replica": replica, "stage": stage, "cp": cp,
               "training_dense_matmul_flops": training,
               "muon_dense_matmul_flops": optimizer,
               "muon_elementwise_flops": optimizer_elementwise,
               "adam_parameters_updated": adam_updated,
               "optimizer_persistent_state_min_bytes": hbm}
        by_rank.append(row)
        per_stage[stage] += training
    # Physical work includes intentional K/V projection duplication and DP2.
    return {"training_flop_scope": "forward+backward dense GEMMs only; includes full output head, excludes optimizer from training FLOPs",
            "by_rank": by_rank,
            "training_dense_matmul_flops_physical": sum(r["training_dense_matmul_flops"] for r in by_rank),
            "muon_dense_matmul_flops_physical": sum(r["muon_dense_matmul_flops"] for r in by_rank),
            "stage_training_dense_matmul_flops_both_replicas": dict(per_stage),
            "max_rank_training_dense_matmul_flops": max(r["training_dense_matmul_flops"] for r in by_rank),
            "max_rank_muon_dense_matmul_flops": max(r["muon_dense_matmul_flops"] for r in by_rank),
            "max_rank_optimizer_persistent_state_min_bytes": max(r["optimizer_persistent_state_min_bytes"] for r in by_rank),
            "excluded_training_operations": ["RMSNorm, router softmax/top-k, activation GELU, residual adds, attention softmax, loss reductions, embedding indexing, communication kernels"],
            "excluded_optimizer_operations": ["AdamW square roots/divides/elementwise arithmetic, clipping scalar reductions, Muon norm square root, copies/casts and communication packing"]}


def schedule(stages=4, microbatches=8, forward_service=1, backward_service=1):
    """Noninterleaved 1F1B stage order, earliest start from dependency DAG.

    Unit service and zero transfer latency are an illustration, never a
    calibration of the resource floors. No overlap engine is assumed.
    """
    orders = {}
    for stage in range(stages):
        warmup = min(stages - stage - 1, microbatches)
        order = [("F", mb) for mb in range(warmup)]
        for i in range(microbatches - warmup):
            order += [("F", i + warmup), ("B", i)]
        order += [("B", mb) for mb in range(microbatches - warmup, microbatches)]
        orders[stage] = order
    events, pending = {}, {(stage, kind, mb) for stage, ops in orders.items() for kind, mb in ops}
    while pending:
        progressed = False
        for key in sorted(pending):
            stage, kind, mb = key
            deps = []
            ix = orders[stage].index((kind, mb))
            if ix:
                pk, pm = orders[stage][ix - 1]
                deps.append((stage, pk, pm))
            if kind == "F" and stage:
                deps.append((stage - 1, "F", mb))
            if kind == "B":
                deps.append((stage, "F", mb))
                if stage + 1 < stages:
                    deps.append((stage + 1, "B", mb))
            if not all(dep in events for dep in deps):
                continue
            start = max((events[dep]["end"] for dep in deps), default=0)
            service = forward_service if kind == "F" else backward_service
            events[key] = {"stage": stage, "operation": kind, "microbatch": mb,
                           "start": start, "end": start + service,
                           "dependencies": [list(dep) for dep in deps]}
            pending.remove(key)
            progressed = True
        if not progressed:
            raise ValueError("Schedule dependency cycle")
    live = {}
    for stage in range(stages):
        count, maximum = 0, 0
        for kind, _ in orders[stage]:
            count += 1 if kind == "F" else -1
            maximum = max(maximum, count)
        assert count == 0
        live[stage] = maximum
    makespan = max(e["end"] for e in events.values())
    busy = microbatches * (forward_service + backward_service)
    return {"units": "assumed service slots, no hardware time units",
            "forward_service": forward_service, "backward_service": backward_service,
            "transfer_latency": 0, "makespan_slots": makespan,
            "busy_slots_per_stage": busy,
            "idle_fraction_per_stage": (makespan - busy) / makespan,
            "maximum_live_microbatches_by_stage": live,
            "events": sorted(events.values(), key=lambda e: (e["start"], e["stage"]))}


def capacity(x, pipeline, worst_skew=False):
    h, n, b, f = x.hidden, x.sequence, x.bf16_bytes, x.expert_width
    p = parameters(x)
    assignments = x.top_k * n if worst_skew else x.top_k * n // 8
    # Conservative retain-every-listed-tensor, no rematerialization contract.
    per_layer = {
        "input_residual_cp_tp_replicated": n // 2 * h * b,
        "norm1_output_cp_tp_replicated": n // 2 * h * b,
        "Q_local_heads": n // 2 * (h // 4) * b,
        "gathered_K_and_V": 2 * n * x.head_dim * b,
        "attention_output_local_heads": n // 2 * (h // 4) * b,
        "post_attention_residual_cp_tp_replicated": n // 2 * h * b,
        "post_attention_residual_unique_EP_copy": n // 8 * h * b,
        "norm2_unique_token_output": n // 8 * h * b,
        "router_FP32_logits": n // 8 * x.experts * 4,
        "route_metadata": n // 8 * x.top_k * x.routing_metadata_bytes,
        "expert_inputs_retained": assignments * h * b,
        "expert_hidden_retained": assignments * f * b,
        "expert_outputs_retained": assignments * h * b,
        "combined_MoE_output_unique_tokens": n // 8 * h * b,
        "RMSNorm_inverse_rms_statistics": (n // 2 + n // 8) * 4,
        "attention_logsumexp_FP32": (n // 2) * (x.query_heads // 4) * 4,
    }
    rows = []
    for stage in range(4):
        muon = p["local_muon_parameters_per_layer"] * (x.layers // 4)
        adam = p["local_adam_parameters_per_layer"] * (x.layers // 4)
        if stage in (0, 3):
            adam += x.vocab * h // 4
        if stage == 3:
            adam += h
        live = pipeline["maximum_live_microbatches_by_stage"][stage]
        ledger = {
            "BF16_model_weights": (muon + adam) * 2,
            "FP32_master_weights": (muon + adam) * 4,
            "FP32_accumulated_gradients": (muon + adam) * 4,
            "FP32_Muon_momentum": muon * 4,
            "FP32_Adam_first_and_second_moments": adam * 8,
            "retained_layer_activations": sum(per_layer.values()) * (x.layers // 4) * live,
            "active_backward_tensor_gradients_conservative": 2 * sum(per_layer.values()),
            "expert_stream_double_buffer": 2 * (x.experts // 8) * x.expert_chunk_tokens * (h * b + 16),
            "route_count_and_sort_metadata_full_sequence": x.top_k * n * 16 + 8 * 8 * 4,
            "TP_CP_collective_buffers": 2 * (n // 2 * h * b) + 2 * (n // 2 * x.head_dim * 2 * b),
            "pipeline_send_receive_buffers": 2 * (n // 8 * h * b),
            "FP32_gradient_reduction_bucket_send_receive": 2 * 64 * 1024 ** 2,
            "whole_attention_matrix_gather_reserve": 3 * h * h * 4,
            "Muon_NS_matrix_workspace": max(ns_work(h, f, 5)["matrix_workspace_bytes"], ns_work(h, h, 5)["matrix_workspace_bytes"]),
            "optimizer_publish_buffer": 64 * 1024 ** 2,
            "attention_backward_dK_dV_FP32": 2 * n * x.head_dim * 4,
            "global_norm_scalar_buffers": 8 * 8,
            "all_step_token_ids_and_targets_int64": x.microbatches * n * 2 * 8,
        }
        if stage == 3:
            ledger["retained_final_norm_input_output_stats"] = live * (2 * (n // 2) * h * b + (n // 2) * 4)
            # Keep logits for B instead of recomputing the vocabulary head.
            ledger["head_logits_and_dlogits_FP32"] = live * 2 * (n // 2) * (x.vocab // 4) * 4
            ledger["cross_entropy_statistics_FP32"] = live * 2 * (n // 2) * 4
        total = sum(ledger.values())
        rows.append({"stage": stage, "live_microbatches": live, "ledger_bytes": ledger,
                     "declared_live_bytes": total,
                     "assumed_device_bytes": x.device_capacity_bytes,
                     "unallocated_bytes_before_excluded_runtime_items": x.device_capacity_bytes - total})
    return {"routing_case": "all routes on one rank's two experts" if worst_skew else "exact balanced route pattern",
            "basis": "sum of declared live allocations; workspace maxima deliberately summed even when reusable",
            "per_layer_per_live_microbatch_bytes": per_layer,
            "by_stage_per_rank": rows,
            "excluded_allocations": ["CUDA/framework contexts and allocator fragmentation/reserve", "implementation GEMM/collective library workspaces beyond declared buffers", "transport registration and library persistent buffers", "graph capture pools, compilation/autotuning caches, debug/profiler allocations", "checkpoint save/load buffers and dataloader host/pinned memory"],
            "capacity_conclusion": "Conditional ledger only. No actual implementation capacity or runnable fit claim; runtime items require measurement."}


def payload_floors(x, ledger):
    """Concurrent resource screens for one set of directed transfers."""
    return {
        "gpu_intranode_payload_seconds": max(ledger["max_gpu_intranode_sent_bytes"], ledger["max_gpu_intranode_received_bytes"]) / x.assumed_gpu_intranode_send_bytes_s,
        "node_offnode_payload_seconds": max(ledger["max_node_offnode_sent_bytes"], ledger["max_node_offnode_received_bytes"]) / x.assumed_node_offnode_send_bytes_s,
        "rack_intrarack_payload_seconds": max(ledger["per_rack_intrarack_sent_bytes"].values(), default=0) / x.assumed_rack_intrarack_send_bytes_s,
        "rack_interrack_payload_seconds": max(ledger["interrack_directed_bytes"].values(), default=0) / x.assumed_rack_interrack_send_bytes_s,
    }


def floors(x, traffic, work, traffic_by_phase, schedule_label="declared_full_allreduce_before_norm", traffic_for_service=None):
    result = {}
    for placement, ledger in traffic.items():
        training = work["max_rank_training_dense_matmul_flops"] / x.assumed_training_matmul_flops_s
        optimizer = work["max_rank_muon_dense_matmul_flops"] / x.assumed_optimizer_fp32_flops_s
        # Grant all assumed cache to avoiding compulsory persistent-state
        # traffic across the optimizer phase boundary, once on read and once
        # on write. This deliberately favors the candidate. This is a partial
        # HBM floor, not the entire training/NS HBM traffic.
        opt_hbm = max(0, work["max_rank_optimizer_persistent_state_min_bytes"]
                      - 2 * x.assumed_optimizer_phase_cache_bytes) / x.assumed_hbm_bytes_s
        values = {
            "training_dense_matmul_seconds": training,
            "optimizer_Muon_dense_matmul_seconds": optimizer,
            "optimizer_persistent_state_HBM_partial_seconds": opt_hbm,
            **payload_floors(x, ledger if traffic_for_service is None else traffic_for_service[placement]),
        }
        pre = payload_floors(x, traffic_by_phase[placement]["pre_clipping"])
        post = payload_floors(x, traffic_by_phase[placement]["post_clipping"])
        values.update({"pre_clipping_" + k: v for k, v in pre.items()})
        values.update({"post_clipping_" + k: v for k, v in post.items()})
        # The supplied schedule declares which transfers precede clipping.
        # Full DP allreduce-before-norm is a choice, not forced by clipping:
        # reduce-scatter-first instead places DP allgather in the post phase.
        # In both policies optimizer work follows the norm, even when the
        # busiest training and optimizer ranks are different GPUs.
        # Max within a phase grants arbitrary overlap, making a necessary floor.
        # Sum the disjoint ordered phases. Norm-scalar service is omitted from
        # this sum (never assigned to pre traffic); full-step screens retain it.
        values["pre_clipping_phase_seconds"] = max(training, *pre.values())
        values["post_clipping_phase_seconds"] = max(optimizer, opt_hbm, *post.values())
        values["ordered_training_plus_Muon_compute_seconds"] = training + optimizer
        values["barrier_ordered_phase_floor_seconds"] = values["pre_clipping_phase_seconds"] + values["post_clipping_phase_seconds"]
        values["necessary_step_floor_seconds"] = max(values.values())
        result[placement] = values
    return {"interpretation": "necessary floors for the named declared schedule and capabilities; not universal clipping requirements, attainable runtime, speedup or MFU",
            "schedule": schedule_label,
            "full_step_payload_screens_omit_norm_service": traffic_for_service is not None,
            "assumptions": "BF16 training and the declared FP32 single-owner attention Muon policy use distinct assumed per-GPU capabilities. Service numbers are illustrative, not measured. Max grants optimistic arbitrary overlap within each phase, including post-norm DP allgather with optimizer when present; actual matrix/gradient dependencies can raise the bound. Add phase maxima only across the declared norm barrier. Norm-scalar service is omitted from the ordered sum; full-step service screens omit it only where separately flagged. Compute/traffic screens use balanced routing; worst-skew capacity is separate.",
            "by_placement": result}


def companion_mapping():
    # Independent MQA workload; deliberately not the main model's GQA workload.
    rows = []
    for cp in range(2):
        start, end = cp * 16, (cp + 1) * 16
        pairs_per_head = sum(query_tile + 1 for query_tile in range(start, end))
        pairs = 4 * pairs_per_head
        scratch = pairs * 2 * 128 * 128 * 4
        traffic = max(0, 2 * (scratch - 128 * 1024 ** 2))
        rows.append({"cp": cp, "ranks_in_cp": 4, "head_tile_pairs_per_rank": pairs,
                     "scratch_bytes_per_rank": scratch,
                     "scratch_HBM_floor_bytes_per_rank": traffic,
                     "floor_seconds_at_separate_assumed_8TB_s": traffic / 8_000_000_000_000})
    return {"scope": "separate causal MQA N4096,Hq16,Hkv1,D128,tile128; unique FP32 dK/dV partial arrays; not a main-model step calibration",
            "cache_bytes_per_rank_assumed": 128 * 1024 ** 2, "by_cp": rows}


def workbook(x=Inputs()):
    x.validate()
    t = build_traffic(x)
    traffic = {placement: t.summary(placement) for placement in ("A", "B")}
    phases = {placement: {phase: t.summary(placement, phase) for phase in
                         ("pre_clipping", "clipping_barrier", "post_clipping")}
              for placement in ("A", "B")}
    work = compute(x)
    pipe = schedule(x.stages, x.microbatches)
    resource_floors = floors(x, traffic, work, phases)
    alt = reduce_scatter_first_traffic(t)
    alt_traffic = {placement: alt.summary(placement) for placement in ("A", "B")}
    alt_phases = {placement: {phase: alt.summary(placement, phase) for phase in
                             ("pre_clipping", "clipping_barrier", "post_clipping")}
                  for placement in ("A", "B")}
    alt_service = {placement: alt.summary(placement, exclude_norm=True) for placement in ("A", "B")}
    alt_floors = floors(x, alt_traffic, work, alt_phases,
                        "analytical_alternative_reduce_scatter_norm_allgather", alt_service)
    route_counts = defaultdict(int)
    for route in routes(x):
        route_counts[route["expert"]] += 1
    return {"state": "public hypothetical analytical workbook; no measured performance result",
            "inputs": asdict(x), "parameters": parameters(x),
            "token_layout": {"attention": "CP2 x TP4 on ranks cp*4+tp", "experts": "EP8 on the SAME eight ranks; two whole experts/rank",
                             "reshard": "choose one 256-token TP chunk per rank, cyclic shift to next EP owner; inverse shift plus TP allgather restores attention view"},
            "step_order": ["accumulate microbatch gradients, normalizing loss by global 32768 tokens",
                           "sum distinct CP/TP token/head gradient contributions into canonical unique shards",
                           "declared schedule: complete full DP2 allreduce of each unique gradient shard",
                           "declared norm ownership: replica 0 contributes ALL unique Muon/AdamW terms; replica 1 contributes zero",
                           "apply the same clipping scalar to both optimizer families",
                           "whole-matrix Muon gather/update/scatter and canonical AdamW updates",
                           "publish updated parameter/state copies; complete global update barrier before next step"],
            "routing": {"assignments_per_microbatch": len(routes(x)), "assignments_by_expert": dict(route_counts),
                        "balanced_chunks": len(chunk_routes(x)), "worst_skew_chunks": len(chunk_routes(x, True)), "drop_policy": "none; stream all chunks"},
            "traffic": traffic, "traffic_by_phase": phases,
            "traffic_category_phase": TRAFFIC_PHASE,
            "norm_ownership": norm_ownership(t),
            "compute": work, "pipeline": pipe,
            "capacity_balanced": capacity(x, pipe), "capacity_worst_skew": capacity(x, pipe, True),
            "resource_floors": resource_floors,
            "hypothetical_budget_screen": budget_screen(resource_floors),
            "alternative_reduce_scatter_first": {
                "scope": "analytical alternative, not a replacement primary baseline or an executed/feasible schedule",
                "step_order": ["same local gradient sums", "DP2 reduce-scatter",
                               "both replicas contribute disjoint reduced-gradient norm terms", "global norm and common clip",
                               "DP allgather and optimizer granted optimistic overlap, respecting consumed-gradient dependencies in any implementation",
                               "same updated-weight/optimizer-state publication and final barrier"],
                "traffic_category_phase": REDUCE_SCATTER_FIRST_PHASE,
                "traffic": alt_traffic, "traffic_by_phase": alt_phases,
                "norm_ownership": norm_ownership(t, True),
                "resource_floors": alt_floors,
                "hypothetical_budget_screen": budget_screen(alt_floors),
                "limits": "Same total payload and mathematical gradient/norm/update semantics. Norm-scalar bytes retained but their service is omitted from all alternative screens. Max allows optimistic allgather/optimizer overlap; no full DAG feasibility, capacity validation, runtime or speedup claim."},
            "companion_mapping": companion_mapping()}


def budget_screen(resource_floors, budget_seconds=0.150):
    """A necessary-condition rejection only, never a feasibility certificate."""
    return {"budget_seconds": budget_seconds,
            "by_placement": {placement: {
                "necessary_floor_seconds": f["necessary_step_floor_seconds"],
                "floor_minus_budget_seconds": f["necessary_step_floor_seconds"] - budget_seconds,
                "status": "ruled_out" if f["necessary_step_floor_seconds"] > budget_seconds else "not_ruled_out"}
                for placement, f in resource_floors["by_placement"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = workbook()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
