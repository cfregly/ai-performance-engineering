#!/usr/bin/env python3
"""Conservation, gradient ownership, whole-matrix and dependency checks."""
import math
import json
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import unittest
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

from reader_summary import summary
from verify_analytical_results import expected_values, validate_document, verify
from system_budget import (Inputs, Traffic, TRAFFIC_PHASE, build_traffic, budget_screen, capacity,
                           chunk_routes, companion_mapping, compute, gid,
                           identity, ns_work, parameters, rack_of, routes,
                           reduce_scatter_first_traffic, schedule, token_mapping, workbook)


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def finite_ns(a, iterations=5):
    """Tiny numerical demonstration only; not a performance implementation."""
    transposed = len(a) > len(a[0])
    x = transpose(a) if transposed else [row[:] for row in a]
    norm = math.sqrt(sum(v * v for row in x for v in row)) + 1e-7
    x = [[v / norm for v in row] for row in x]
    for _ in range(iterations):
        aa = matmul(x, transpose(x))
        a2 = matmul(aa, aa)
        b = [[-4.7750 * v + 2.0315 * w for v, w in zip(vr, wr)] for vr, wr in zip(aa, a2)]
        bx = matmul(b, x)
        x = [[3.4445 * v + w for v, w in zip(vr, wr)] for vr, wr in zip(x, bx)]
    return transpose(x) if transposed else x


class SystemBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = Inputs()
        cls.t = build_traffic(cls.x)
        cls.w = workbook(cls.x)

    def test_retained_reader_summary_matches_current_calculator_bytes(self):
        retained = Path(__file__).resolve().parents[1] / 'evidence/reader-summary.json'
        calculated = json.dumps(summary(self.w), separators=(',', ':'), sort_keys=True) + '\n'
        self.assertEqual(retained.read_bytes(), calculated.encode('utf-8'))

    def test_declared_parallelism_uses_64_not_512_gpus(self):
        self.assertEqual(2 * 4 * 8, 64)
        self.assertEqual(self.x.tp * self.x.cp, self.x.ranks_per_stage)
        self.assertEqual(len({gid(r, s, l) for r in range(2) for s in range(4) for l in range(8)}), 64)

    def test_rank_identity_round_trip(self):
        for rank in range(64):
            self.assertEqual(gid(*identity(rank)), rank)

    def test_attention_groups_partition_each_stage(self):
        for replica in range(2):
            for stage in range(4):
                tp = [{gid(replica, stage, cp * 4 + t) for t in range(4)} for cp in range(2)]
                cp = [{gid(replica, stage, c * 4 + t) for c in range(2)} for t in range(4)]
                self.assertEqual(set.union(*tp), set.union(*cp))
                self.assertEqual(sum(map(len, tp)), 8)
                self.assertEqual(sum(map(len, cp)), 8)

    def test_placements_put_four_nodes_in_each_rack(self):
        for placement in ("A", "B"):
            self.assertEqual(Counter(rack_of(n * 8, placement) for n in range(8)), {0: 4, 1: 4})

    def test_A_all_DP_interrack_B_all_DP_intrarack(self):
        for stage in range(4):
            self.assertNotEqual(rack_of(gid(0, stage, 0), "A"), rack_of(gid(1, stage, 0), "A"))
            self.assertEqual(rack_of(gid(0, stage, 0), "B"), rack_of(gid(1, stage, 0), "B"))

    def test_B_only_middle_pipeline_boundary_crosses_rack(self):
        for replica in range(2):
            crossed = [s for s in range(3) if rack_of(gid(replica, s, 0), "B") != rack_of(gid(replica, s + 1, 0), "B")]
            self.assertEqual(crossed, [1])

    def test_unique_parameter_count_independent_formula(self):
        # Per block: Q/O, grouped K/V, 16 two-matrix experts, router, two norms.
        layer = 2 * 2048 ** 2 + 2 * 2048 * 256 + 16 * 2 * 2048 * 4096 + 2048 * 16 + 2 * 2048
        expected = 12 * layer + 2 * 32768 * 2048 + 2048
        self.assertEqual(expected, 3_469_133_824)
        self.assertEqual(parameters(self.x)["unique_model_parameters"], expected)

    def test_token_reshard_has_one_owner_per_token(self):
        m = token_mapping(self.x)
        self.assertEqual(len(m), 2048)
        self.assertEqual(len({r["token"] for r in m}), 2048)
        self.assertEqual(Counter(r["attention_owner_rank"] for r in m), dict.fromkeys(range(8), 256))
        self.assertEqual(Counter(r["expert_token_owner_rank"] for r in m), dict.fromkeys(range(8), 256))
        self.assertTrue(all((r["expert_token_owner_rank"] - 1) % 8 == r["attention_owner_rank"] for r in m))

    def test_routing_has_two_distinct_experts_and_unit_gate_sum(self):
        by_token = defaultdict(list)
        for r in routes(self.x):
            by_token[r["token"]].append(r)
        for rr in by_token.values():
            self.assertEqual(len({r["expert"] for r in rr}), 2)
            self.assertEqual(sum(r["weight"] for r in rr), 1)

    def test_balanced_routing_conserves_all_work(self):
        rr = routes(self.x)
        self.assertEqual(len(rr), 4096)
        self.assertEqual(Counter(r["expert"] for r in rr), dict.fromkeys(range(16), 256))
        self.assertEqual(Counter(r["destination"] for r in rr), dict.fromkeys(range(8), 512))
        self.assertEqual(sum(r["owner"] != r["destination"] for r in rr), 3584)

    def test_chunking_no_drop_even_under_worst_skew(self):
        for skew in (False, True):
            original = {(r["token"], r["expert"]) for r in routes(self.x, skew)}
            chunks = chunk_routes(self.x, skew)
            flattened = [(r["token"], r["expert"]) for _, chunk in chunks for r in chunk]
            self.assertEqual(len(flattened), 4096)
            self.assertEqual(set(flattened), original)
            self.assertTrue(all(0 < len(chunk) <= 128 for _, chunk in chunks))
        skew = Counter(r["expert"] for r in routes(self.x, True))
        self.assertEqual(skew, {0: 2048, 1: 2048})

    def test_collective_sent_bytes_count_once_not_sent_plus_received(self):
        t = Traffic(self.x)
        t.ring_allreduce("test", [0, 1, 2, 3], 1024)
        self.assertEqual(sum(amount for _, _, _, amount in t.transfers), 6144)
        self.assertTrue(all(amount == 1536 for _, _, _, amount in t.transfers))

    def test_DP_bytes_use_unique_parameters_not_physical_copies(self):
        expected = 2 * 3_469_133_824 * 4
        actual = sum(n for cat, _, _, n in self.t.transfers if cat == "data_parallel_unique_gradients")
        self.assertEqual(actual, expected)
        self.assertLess(actual, 4 * parameters(self.x)["physical_parameter_copies"])

    def test_pipeline_boundary_bytes_independent_equation(self):
        # 2 replicas * 3 boundaries * 8 microbatches * F/B * token x hidden x BF16.
        expected = 2 * 3 * 8 * 2 * 2048 * 2048 * 2
        actual = sum(n for cat, _, _, n in self.t.transfers if cat == "pipeline_boundary")
        self.assertEqual(actual, expected)
        self.assertEqual(actual, 768 * 1024 ** 2)

    def test_reshard_is_explicit_and_avoids_ep_tp_cp_product(self):
        # 10 MiB/rank/block/microbatch across 64 ranks, 3 blocks/node, 8 MBs.
        actual = sum(n for cat, _, _, n in self.t.transfers if cat == "token_owner_reshard")
        self.assertEqual(actual, 10 * 1024 ** 2 * 64 * 3 * 8)

    def test_expert_payload_all_four_phases(self):
        expected = 3584 * (2048 * 2 + 16) * 4 * 12 * 8 * 2
        actual = sum(n for cat, _, _, n in self.t.transfers if cat == "expert_dispatch_combine")
        self.assertEqual(actual, expected)

    def test_all_payload_conserved_when_placement_changes(self):
        total = sum(n for _, _, _, n in self.t.transfers)
        for placement in ("A", "B"):
            ledger = self.w["traffic"][placement]
            self.assertEqual(sum(ledger["totals_bytes"].values()), total)
            self.assertEqual(sum(sum(v.values()) for v in ledger["per_gpu_sent_bytes"].values()), total)

    def test_placement_crossrack_counts(self):
        a, b = self.w["traffic"]["A"], self.w["traffic"]["B"]
        self.assertEqual(a["totals_bytes"]["inter_rack"], 2 * 3_469_133_824 * 4 + 28)
        self.assertEqual(b["totals_bytes"]["inter_rack"], 256 * 1024 ** 2 + 56)
        self.assertEqual(a["totals_bytes"]["intra_node"], b["totals_bytes"]["intra_node"])

    def test_headline_rack_totals_conserve_exact_offnode_payload(self):
        expected = {
            "A": {"inter_rack": 27_753_070_620, "intra_rack": 805_306_452},
            "B": {"inter_rack": 268_435_512, "intra_rack": 28_289_941_560},
        }
        for placement, counts in expected.items():
            with self.subTest(placement=placement):
                totals = self.w["traffic"][placement]["totals_bytes"]
                self.assertEqual(totals["inter_rack"], counts["inter_rack"])
                self.assertEqual(totals["intra_rack"], counts["intra_rack"])
                self.assertEqual(totals["inter_rack"] + totals["intra_rack"], 28_558_377_072)

    def test_headline_busiest_node_sends_and_receives_exact_full_update_bytes(self):
        for placement in ("A", "B"):
            with self.subTest(placement=placement):
                ledger = self.w["traffic"][placement]
                sent = ledger["per_node_offnode_sent_bytes"]
                received = ledger["per_node_offnode_received_bytes"]
                busiest = max(sent, key=sent.get)
                self.assertEqual(sent[busiest], 3_670_466_574)
                self.assertEqual(received[busiest], sent[busiest])
                self.assertEqual(ledger["max_node_offnode_sent_bytes"], 3_670_466_574)
                self.assertEqual(ledger["max_node_offnode_received_bytes"], 3_670_466_574)

    def test_Muon_full_matrix_gather_has_all_three_FP32_states(self):
        # Q/O: 3/4 nonlocal; K/V: 1/2 nonlocal. Repeated 12 blocks, DP2.
        expected = int((2 * 2048 ** 2 * 3 / 4 + 2 * 2048 * 256 * 1 / 2) * 12 * 12 * 2)
        actual = sum(n for cat, _, _, n in self.t.transfers if cat == "muon_whole_matrix_gather")
        self.assertEqual(actual, expected)

    def test_finite_whole_matrix_NS_differs_from_per_shard_NS(self):
        a = [[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 3.0, 1.0]]
        whole = finite_ns(a)
        left = finite_ns([row[:2] for row in a])
        right = finite_ns([row[2:] for row in a])
        shards = [l + r for l, r in zip(left, right)]
        self.assertGreater(max(abs(v - w) for vr, wr in zip(whole, shards) for v, w in zip(vr, wr)), 0.01)

    def test_global_norm_uses_unique_completed_gradients(self):
        # Logical gradients [3,4,12], replicated in distinct multiplicities.
        values = [3, 4, 12]
        physical = [v for v, copies in zip(values, (2, 4, 8)) for _ in range(copies)]
        correct = math.sqrt(sum(v * v for v in values))
        self.assertEqual(correct, 13)
        self.assertGreater(math.sqrt(sum(v * v for v in physical)), correct)
        # DP contributions form one unique global gradient before norm.
        replica0, replica1 = [1, 1, 4], [2, 3, 8]
        self.assertEqual([a + b for a, b in zip(replica0, replica1)], values)
        self.assertNotEqual(sum(v * v for v in replica0 + replica1), correct ** 2)

    def test_NS_work_has_cubic_Gram_and_rectangular_products(self):
        w = ns_work(2, 4, 5)
        self.assertEqual(w["dense_matmul_flops"], 5 * (4 * 2 * 2 * 4 + 2 * 2 ** 3))
        self.assertEqual(w, ns_work(4, 2, 5))

    def test_full_head_cost_is_not_ignored(self):
        stages = self.w["compute"]["stage_training_dense_matmul_flops_both_replicas"]
        # stages keys are ints before JSON serialization.
        expected_head = 2 * 8 * 6 * 2048 * 2048 * 32768
        self.assertEqual(stages[3] - stages[0], expected_head)
        self.assertGreater(stages[3] / stages[0], 1.46)

    def test_causal_CP_work_sums_to_full_attention(self):
        n = self.x.sequence
        low = (n // 2) * (n // 2 + 1) // 2
        high = n * (n + 1) // 2 - low
        self.assertEqual(low + high, n * (n + 1) // 2)
        self.assertGreater(high, 2 * low)

    def test_headline_later_CP_dense_work_ratio_uses_complete_rank_work(self):
        ranks = {row["rank"]: row for row in self.w["compute"]["by_rank"]}
        for stage in range(3):
            with self.subTest(stage=stage):
                earlier = ranks[gid(0, stage, 0)]["training_dense_matmul_flops"]
                later = ranks[gid(0, stage, 4)]["training_dense_matmul_flops"]
                self.assertEqual(earlier, 1_702_090_506_240)
                self.assertEqual(later, 1_856_709_328_896)
                self.assertEqual(Fraction(later, earlier), Fraction(24593, 22545))
                self.assertEqual(round((later / earlier - 1) * 100, 2), 9.08)

    def test_reader_summary_emits_later_CP_work_percent_for_each_early_stage(self):
        percent = summary(self.w)["compute"]["later_CP_extra_percent_by_stage"]
        self.assertEqual(percent, {0: 9.084054, 1: 9.084054, 2: 9.084054})
        for stage, value in percent.items():
            with self.subTest(stage=stage):
                self.assertEqual(round(value, 2), 9.08)

    def test_1F1B_event_count_serialization_and_dependencies(self):
        p = self.w["pipeline"]
        events = {(e["stage"], e["operation"], e["microbatch"]): e for e in p["events"]}
        self.assertEqual(len(events), 64)
        for event in events.values():
            for dependency in event["dependencies"]:
                self.assertLessEqual(events[tuple(dependency)]["end"], event["start"])
        for stage in range(4):
            row = sorted((e for e in events.values() if e["stage"] == stage), key=lambda e: e["start"])
            self.assertEqual(len(row), 16)
            self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(row, row[1:])))

    def test_1F1B_live_buffers_and_bubble_are_derived(self):
        p = self.w["pipeline"]
        self.assertEqual(p["makespan_slots"], 22)
        self.assertEqual(p["maximum_live_microbatches_by_stage"], {0: 4, 1: 3, 2: 2, 3: 1})
        self.assertAlmostEqual(p["idle_fraction_per_stage"], 6 / 22)
        # Changing microbatch count changes the modeled bubble.
        self.assertEqual(schedule(4, 1)["makespan_slots"], 8)
        self.assertEqual(schedule(4, 16)["makespan_slots"], 38)

    def test_capacity_is_sum_of_declared_allocations(self):
        for key in ("capacity_balanced", "capacity_worst_skew"):
            c = self.w[key]
            self.assertTrue(c["excluded_allocations"])
            for row in c["by_stage_per_rank"]:
                self.assertEqual(row["declared_live_bytes"], sum(row["ledger_bytes"].values()))
                self.assertEqual(row["assumed_device_bytes"] - row["declared_live_bytes"], row["unallocated_bytes_before_excluded_runtime_items"])
        for a, b in zip(self.w["capacity_balanced"]["by_stage_per_rank"], self.w["capacity_worst_skew"]["by_stage_per_rank"]):
            self.assertGreater(b["declared_live_bytes"], a["declared_live_bytes"])

    def test_capacity_uses_14_byte_Muon_and_18_byte_Adam_states(self):
        row = self.w["capacity_balanced"]["by_stage_per_rank"][1]
        p = parameters(self.x)
        expected = 3 * (p["local_muon_parameters_per_layer"] * 14 + p["local_adam_parameters_per_layer"] * 18)
        keys = ("BF16_model_weights", "FP32_master_weights", "FP32_accumulated_gradients", "FP32_Muon_momentum", "FP32_Adam_first_and_second_moments")
        self.assertEqual(sum(row["ledger_bytes"][k] for k in keys), expected)

    def test_capacity_retains_EP_raw_residual_for_norm_backward(self):
        c = self.w["capacity_balanced"]
        self.assertEqual(c["per_layer_per_live_microbatch_bytes"]["post_attention_residual_unique_EP_copy"], 1024 ** 2)

    def test_capacity_reserves_all_step_ids_and_targets(self):
        for row in self.w["capacity_balanced"]["by_stage_per_rank"]:
            self.assertEqual(row["ledger_bytes"]["all_step_token_ids_and_targets_int64"], 8 * 2048 * 2 * 8)

    def test_binary_bytes_and_decimal_service_units(self):
        self.assertEqual(1024 ** 2, 1_048_576)
        self.assertEqual(134_217_728 / 80_000_000_000, 0.0016777216)

    def test_resource_screen_does_not_add_parallel_network_resources(self):
        for placement in ("A", "B"):
            f = self.w["resource_floors"]["by_placement"][placement]
            self.assertEqual(f["ordered_training_plus_Muon_compute_seconds"], f["training_dense_matmul_seconds"] + f["optimizer_Muon_dense_matmul_seconds"])
            self.assertEqual(f["necessary_step_floor_seconds"], max(v for k, v in f.items() if k != "necessary_step_floor_seconds"))

    def test_clipping_phases_partition_every_directed_transfer(self):
        self.assertEqual(set(TRAFFIC_PHASE), {cat for cat, _, _, _ in self.t.transfers})
        for placement in ("A", "B"):
            complete = self.w["traffic"][placement]
            phases = self.w["traffic_by_phase"][placement]
            for tier, total in complete["totals_bytes"].items():
                self.assertEqual(sum(p["totals_bytes"][tier] for p in phases.values()), total)
            self.assertEqual(set(phases["post_clipping"]["by_category_bytes"]),
                             {"muon_whole_matrix_gather", "muon_scatter_publish", "adam_state_publish"})
            self.assertEqual(set(phases["clipping_barrier"]["by_category_bytes"]), {"global_clip_norm"})
            self.assertNotIn("global_clip_norm", phases["pre_clipping"]["by_category_bytes"])
            self.assertEqual(phases["post_clipping"]["totals_bytes"]["inter_rack"], 0)
            self.assertEqual(phases["post_clipping"]["totals_bytes"]["intra_rack"], 0)

    def test_pre_clipping_counts_exclude_norm_scalars(self):
        a = self.w["traffic_by_phase"]["A"]["pre_clipping"]
        b = self.w["traffic_by_phase"]["B"]["pre_clipping"]
        self.assertEqual(a["interrack_directed_bytes"], {"0->1": 13_876_535_296, "1->0": 13_876_535_296})
        self.assertEqual(b["interrack_directed_bytes"], {"0->1": 134_217_728, "1->0": 134_217_728})
        self.assertEqual(a["max_node_offnode_sent_bytes"], 3_670_466_560)
        self.assertEqual(b["max_node_offnode_sent_bytes"], 3_670_466_560)
        self.assertEqual(b["per_rack_intrarack_sent_bytes"], {0: 14_144_962_560, 1: 14_144_978_944})

    def test_global_barrier_adds_pre_network_to_post_Muon_floor(self):
        post = 6_717_261_742_080 / 100_000_000_000_000
        expected_pre = {"A": 13_876_535_296 / 80_000_000_000,
                        "B": 3_670_466_560 / 40_000_000_000}
        for placement, pre in expected_pre.items():
            f = self.w["resource_floors"]["by_placement"][placement]
            self.assertEqual(f["pre_clipping_phase_seconds"], pre)
            self.assertEqual(f["post_clipping_phase_seconds"], post)
            self.assertEqual(f["necessary_step_floor_seconds"], pre + post)
            # This declared full-allreduce/replica-0 norm schedule prevents
            # overlap of its DP allgather with post-norm optimizer work.
            self.assertGreater(f["necessary_step_floor_seconds"], max(pre, post))
            self.assertGreater(f["necessary_step_floor_seconds"], 0.150)

    def test_phase_maxima_grant_overlap_only_within_each_phase(self):
        for placement in ("A", "B"):
            f = self.w["resource_floors"]["by_placement"][placement]
            pre = [f["training_dense_matmul_seconds"]] + [v for k, v in f.items() if k.startswith("pre_clipping_") and "payload" in k]
            post = [f["optimizer_Muon_dense_matmul_seconds"], f["optimizer_persistent_state_HBM_partial_seconds"]] + [v for k, v in f.items() if k.startswith("post_clipping_") and "payload" in k]
            self.assertEqual(f["barrier_ordered_phase_floor_seconds"], max(pre) + max(post))
            self.assertLess(f["barrier_ordered_phase_floor_seconds"], sum(pre) + sum(post))

    def test_unchanged_150ms_target_excludes_both_placements(self):
        screen = self.w["hypothetical_budget_screen"]
        self.assertEqual(screen["budget_seconds"], 0.150)
        self.assertEqual({p: row["status"] for p, row in screen["by_placement"].items()},
                         {"A": "ruled_out", "B": "ruled_out"})
        self.assertAlmostEqual(screen["by_placement"]["A"]["floor_minus_budget_seconds"], 0.0906293086208)
        self.assertAlmostEqual(screen["by_placement"]["B"]["floor_minus_budget_seconds"], 0.0089342814208)
        # B's node-only bound clears 150 ms; its ordered update bound does not.
        self.assertLess(self.w["resource_floors"]["by_placement"]["B"]["node_offnode_payload_seconds"], screen["budget_seconds"])

    def test_unknown_transfer_category_cannot_silently_overlap_optimizer(self):
        t = Traffic(self.x)
        t.send("new_unclassified_dependency", 0, 8, 1024)
        with self.assertRaises(KeyError):
            t.summary("A", "pre_clipping")

    def test_alternative_preserves_every_directed_payload_and_norm_scalar(self):
        alternative = reduce_scatter_first_traffic(self.t)
        before = Counter()
        after = Counter()
        for cat, src, dst, amount in self.t.transfers:
            before[cat, src, dst] += amount
        for cat, src, dst, amount in alternative.transfers:
            if cat in ("data_parallel_reduce_scatter", "data_parallel_allgather"):
                cat = "data_parallel_unique_gradients"
            after[cat, src, dst] += amount
        self.assertEqual(before, after)
        dp_half = 3_469_133_824 * 4  # half of the two-replica allreduce payload
        for cat in ("data_parallel_reduce_scatter", "data_parallel_allgather"):
            self.assertEqual(sum(n for c, _, _, n in alternative.transfers if c == cat), dp_half)
        for placement in ("A", "B"):
            alt = self.w["alternative_reduce_scatter_first"]
            for key in ("totals_bytes", "per_gpu_sent_bytes", "per_gpu_received_bytes",
                        "per_node_offnode_sent_bytes", "per_node_offnode_received_bytes"):
                self.assertEqual(alt["traffic"][placement][key], self.w["traffic"][placement][key])
            self.assertEqual(alt["traffic_by_phase"][placement]["clipping_barrier"],
                             self.w["traffic_by_phase"][placement]["clipping_barrier"])

    def test_alternative_norm_owners_cover_unique_parameters_once(self):
        primary = self.w["norm_ownership"]
        alternative = self.w["alternative_reduce_scatter_first"]["norm_ownership"]
        self.assertEqual(primary["elements_by_replica"], {"0": 3_469_133_824, "1": 0})
        self.assertEqual(alternative["elements_by_replica"], {"0": 1_734_566_912, "1": 1_734_566_912})
        for policy in (primary, alternative):
            intervals = sorted(policy["owner_intervals"], key=lambda r: r["unique_parameter_begin"])
            self.assertEqual(intervals[0]["unique_parameter_begin"], 0)
            self.assertEqual(intervals[-1]["unique_parameter_end"], 3_469_133_824)
            self.assertTrue(all(a["unique_parameter_end"] == b["unique_parameter_begin"] for a, b in zip(intervals, intervals[1:])))
            self.assertEqual(sum(row["elements"] for row in intervals), parameters(self.x)["unique_model_parameters"])
            for row in intervals:
                self.assertEqual(row["replica"], identity(row["owner_rank"], self.x)[0])

    def test_reduced_shard_norm_and_clip_match_full_gradient_math(self):
        # Two replicas' distinct data contributions; reduced vector is [3,4,12,0].
        contributions = ([1, 1, 4, -2], [2, 3, 8, 2])
        full = [a + b for a, b in zip(*contributions)]
        reduced_halves = [[contributions[0][i] + contributions[1][i] for i in indices]
                          for indices in (range(2), range(2, 4))]
        full_norm = math.sqrt(sum(g * g for g in full))
        distributed_norm = math.sqrt(sum(g * g for shard in reduced_halves for g in shard))
        self.assertEqual(full_norm, 13)
        self.assertEqual(full_norm, distributed_norm)
        scalar = min(1.0, 6.5 / distributed_norm)
        gathered = [g * scalar for shard in reduced_halves for g in shard]
        expected = [g * min(1.0, 6.5 / full_norm) for g in full]
        self.assertEqual(gathered, expected)
        # Both whole-matrix optimizer owners consume the same reconstructed input.
        self.assertEqual(finite_ns([gathered[:2], gathered[2:]]), finite_ns([expected[:2], expected[2:]]))

    def test_alternative_phase_floors_follow_independent_byte_equations(self):
        alt = self.w["alternative_reduce_scatter_first"]
        a = alt["resource_floors"]["by_placement"]["A"]
        b = alt["resource_floors"]["by_placement"]["B"]
        half_DP_direction = 3_469_133_824 * 4 / 2
        self.assertEqual(a["pre_clipping_phase_seconds"], half_DP_direction / 80_000_000_000)
        self.assertEqual(a["post_clipping_phase_seconds"], half_DP_direction / 80_000_000_000)
        # Stage 3 has three layers, a full head and final norm. Its one adjacent
        # PP boundary sends 8 microbatches of 2048*2048 BF16 activation/gradient.
        stage3_DP = (3 * 277_909_504 + 32768 * 2048 + 2048) * 4
        stage3_PP_sent = 8 * 2048 * 2048 * 2
        self.assertEqual(b["pre_clipping_phase_seconds"], (stage3_DP / 2 + stage3_PP_sent) / 40_000_000_000)
        self.assertEqual(b["post_clipping_phase_seconds"], 6_717_261_742_080 / 100_000_000_000_000)
        self.assertAlmostEqual(a["necessary_step_floor_seconds"], 0.1734566912)
        self.assertAlmostEqual(b["necessary_step_floor_seconds"], 0.1138923102208)
        self.assertTrue(alt["resource_floors"]["full_step_payload_screens_omit_norm_service"])
        self.assertEqual({p: v["status"] for p, v in alt["hypothetical_budget_screen"]["by_placement"].items()},
                         {"A": "ruled_out", "B": "not_ruled_out"})

    def test_reader_summary_millisecond_floor_keys_match_units(self):
        labels = summary(self.w)
        alternative = labels["alternative_DP_schedule"]["phase_and_floor_ms"]
        expected_keys = {"pre_clipping_phase", "post_clipping_phase", "necessary_step_floor"}
        for placement in ("A", "B"):
            self.assertEqual(set(alternative[placement]), expected_keys)
            self.assertTrue(all(not key.endswith("_seconds") for key in
                                labels["traffic_and_screens"][placement]["floors_ms"]))
            source = self.w["alternative_reduce_scatter_first"]["resource_floors"]["by_placement"][placement]
            for key in expected_keys:
                self.assertEqual(alternative[placement][key], round(source[key + "_seconds"] * 1000, 9))
        self.assertEqual(alternative["B"]["necessary_step_floor"], 113.892310221)

    def test_reader_summary_decision_and_binding_labels_match_primary_floors(self):
        labels = summary(self.w)
        expected_bindings = {
            "A": "pre-clip rack interconnect + post-clip FP32 Muon",
            "B": "pre-clip node NIC + post-clip FP32 Muon",
        }
        for placement, binding in expected_bindings.items():
            with self.subTest(placement=placement):
                self.assertEqual(labels["traffic_and_screens"][placement]["binding_screen"], binding)
                floor = labels["traffic_and_screens"][placement]["floors_ms"]["necessary_step_floor"]
                self.assertIn(f"{placement} {floor:.9f}", labels["decision"])
        self.assertIn("excludes BOTH at 150 ms", labels["decision"])

    def test_reader_summary_tracks_changed_capabilities_and_microbatches(self):
        changed = workbook(replace(
            self.x, microbatches=2,
            assumed_training_matmul_flops_s=250_000_000_000_000,
            assumed_optimizer_fp32_flops_s=500_000_000_000_000,
            assumed_hbm_bytes_s=1_500_000_000_000,
            assumed_optimizer_phase_cache_bytes=64 * 1024 ** 2,
            device_capacity_bytes=64_000_000_000,
            assumed_gpu_intranode_send_bytes_s=80_000_000_000,
            assumed_node_offnode_send_bytes_s=41_500_000_000,
            assumed_rack_intrarack_send_bytes_s=80_000_000_000,
            assumed_rack_interrack_send_bytes_s=800_000_000_000))
        labels = summary(changed)
        self.assertEqual(labels["assumed_capabilities"], {
            "training_BF16_TF_s_GPU": 250, "Muon_FP32_TF_s_GPU": 500,
            "HBM_TB_s_GPU": 1.5, "HBM_phase_cache_MiB": 64, "device_GB": 64,
            "GPU_intranode_GB_s_each_direction": 80,
            "node_NIC_GB_s_each_direction": 41.5,
            "rack_intrafabric_GB_s_aggregate_injection": 80,
            "rack_uplink_GB_s_each_direction": 800})
        pipeline = labels["unit_1F1B"]
        self.assertEqual(pipeline["slots"], 10)
        self.assertEqual(pipeline["busy_each_stage"], 4)
        self.assertEqual(pipeline["idle_fraction"], 6 / 10)
        self.assertEqual(pipeline["idle_over_busy"], 6 / 4)
        self.assertEqual(pipeline["max_live_microbatches"], [2, 2, 2, 1])
        self.assertEqual(labels["model"]["microbatches_per_DP"], 2)
        self.assertEqual(labels["model"]["tokens_per_step"], 8192)
        # Moving DP payload into the slower rack fabric makes B rack-bound.
        self.assertEqual(labels["traffic_and_screens"]["A"]["binding_screen"],
                         "pre-clip node NIC + post-clip FP32 Muon")
        self.assertEqual(labels["traffic_and_screens"]["B"]["binding_screen"],
                         "pre-clip rack fabric + post-clip FP32 Muon")
        self.assertIn("excludes B at 150 ms", labels["decision"])

    def test_reader_summary_budget_verdicts_follow_the_selected_budget(self):
        changed = deepcopy(self.w)
        changed["hypothetical_budget_screen"] = budget_screen(changed["resource_floors"], 0.300)
        alternative = changed["alternative_reduce_scatter_first"]
        alternative["hypothetical_budget_screen"] = budget_screen(alternative["resource_floors"], 0.300)
        labels = summary(changed)
        self.assertIn("excludes neither placement at 300 ms", labels["decision"])
        limit = labels["alternative_DP_schedule"]["limit"]
        self.assertIn("At 300 ms", limit)
        self.assertIn("A is not certified feasible", limit)
        self.assertIn("B is not certified feasible", limit)

    def test_reader_summary_rejects_static_labels_for_a_different_contract(self):
        for field in ("geometry", "service", "schedule"):
            with self.subTest(field=field):
                changed = deepcopy(self.w)
                if field == "geometry":
                    changed["inputs"]["sequence"] = 4096
                elif field == "service":
                    changed["pipeline"] = schedule(4, 8, forward_service=2, backward_service=1)
                else:
                    changed["resource_floors"]["schedule"] = "different_collective_order"
                with self.assertRaises(ValueError):
                    summary(changed)

    def test_alternative_phase_partition_moves_only_DP_allgather(self):
        alt = self.w["alternative_reduce_scatter_first"]
        for placement in ("A", "B"):
            phases = alt["traffic_by_phase"][placement]
            self.assertIn("data_parallel_reduce_scatter", phases["pre_clipping"]["by_category_bytes"])
            self.assertNotIn("data_parallel_allgather", phases["pre_clipping"]["by_category_bytes"])
            self.assertIn("data_parallel_allgather", phases["post_clipping"]["by_category_bytes"])
            for tier, total in alt["traffic"][placement]["totals_bytes"].items():
                self.assertEqual(sum(phase["totals_bytes"][tier] for phase in phases.values()), total)

    def test_companion_mapping_is_separate_and_conserves_partial_bytes(self):
        c = companion_mapping()
        low, high = c["by_cp"]
        self.assertEqual((low["head_tile_pairs_per_rank"], high["head_tile_pairs_per_rank"]), (544, 1568))
        self.assertEqual(4 * (low["scratch_bytes_per_rank"] + high["scratch_bytes_per_rank"]), 1056 * 1024 ** 2)
        self.assertEqual(high["scratch_HBM_floor_bytes_per_rank"], 136 * 1024 ** 2)
        self.assertEqual(low["scratch_HBM_floor_bytes_per_rank"], 0)


class AnalyticalResultsContractTests(unittest.TestCase):
    """Export regressions: the oracle is current calculator output, not the JSON."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.w = workbook()
        cls.s = summary(cls.w)
        cls.expected = expected_values(cls.w, cls.s)
        cls.document = json.loads((cls.root / "evidence/analytical-results.json").read_text())
        cls.pins = {key: {"path": path, "sha256": hashlib.sha256((cls.root / path).read_bytes()).hexdigest()}
                    for key, path in (("code", "code/system_budget.py"),
                                      ("source_results", "evidence/reader-summary.json"),
                                      ("full_workbook_results", "evidence/system-budget.json"))}

    def test_retained_analytical_export_matches_all_current_values_units_and_pins(self):
        self.assertEqual(verify(self.root)["named_values"], 38)

    def test_every_exported_value_rejects_numeric_drift(self):
        for i, row in enumerate(self.document["values"]):
            with self.subTest(name=row["name"]):
                changed = deepcopy(self.document)
                changed["values"][i]["value"] += 1
                with self.assertRaisesRegex(ValueError, "wrong current value"):
                    validate_document(changed, self.expected, self.pins)

    def test_every_exported_value_rejects_wrong_units(self):
        for i, row in enumerate(self.document["values"]):
            with self.subTest(name=row["name"]):
                changed = deepcopy(self.document)
                changed["values"][i]["unit"] = "wrong unit"
                with self.assertRaisesRegex(ValueError, "wrong unit"):
                    validate_document(changed, self.expected, self.pins)

    def test_export_rejects_missing_duplicate_extra_or_renamed_rows(self):
        for change in ("missing", "duplicate", "extra", "renamed"):
            with self.subTest(change=change):
                changed = deepcopy(self.document)
                if change == "missing":
                    changed["values"].pop()
                elif change == "duplicate":
                    changed["values"].append(deepcopy(changed["values"][0]))
                elif change == "extra":
                    changed["values"].append({"name": "unknown", "value": 0, "unit": "GB"})
                else:
                    changed["values"][0]["name"] = "unknown"
                with self.assertRaisesRegex(ValueError, "duplicate value name|value names differ"):
                    validate_document(changed, self.expected, self.pins)

    def test_export_rejects_each_stale_hash_or_wrong_source_path(self):
        for key in self.pins:
            for field in ("sha256", "path"):
                with self.subTest(pin=key, field=field):
                    changed = deepcopy(self.document)
                    changed[key][field] = "stale"
                    with self.assertRaisesRegex(ValueError, "path/SHA256 pin"):
                        validate_document(changed, self.expected, self.pins)

    def test_projection_is_checked_against_workbook_not_only_reader_summary(self):
        changed = deepcopy(self.s)
        changed["compute"]["physical_training_dense_TF"] += 1
        with self.assertRaisesRegex(ValueError, "summary/workbook projection differs"):
            expected_values(self.w, changed)

    def test_comparison_budget_tracks_declared_assumption_without_copying_150(self):
        changed_w, changed_s = deepcopy(self.w), deepcopy(self.s)
        changed_w["hypothetical_budget_screen"]["budget_seconds"] *= 2
        budget_ms = changed_w["hypothetical_budget_screen"]["budget_seconds"] * 1000
        changed_s["decision"] = "Declared comparison at " + f"{budget_ms:g} ms: synthetic only"
        projected = expected_values(changed_w, changed_s)
        self.assertEqual(projected["comparison_budget"]["value"], budget_ms)
        self.assertNotEqual(projected["comparison_budget"]["value"], self.expected["comparison_budget"]["value"])

    def test_optimized_verifier_rejects_wrong_value_and_all_three_stale_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "package"
            for relative in ("code/system_budget.py", "code/reader_summary.py", "code/verify_analytical_results.py",
                             "evidence/analytical-results.json", "evidence/reader-summary.json", "evidence/system-budget.json"):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.root / relative, destination)
            target = root / "evidence/analytical-results.json"
            command = [sys.executable, "-O", str(root / "code/verify_analytical_results.py")]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            good = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(good.returncode, 0, good.stderr)
            for change in ("value", *self.pins, "retained_source_results", "retained_full_workbook_results"):
                with self.subTest(change=change):
                    mutated = deepcopy(self.document)
                    if change == "value":
                        mutated["values"][0]["value"] += 1
                    elif change.startswith("retained_"):
                        pin = change.removeprefix("retained_")
                        retained = root / self.pins[pin]["path"]
                        retained.write_bytes(retained.read_bytes() + b" ")
                        mutated[pin]["sha256"] = hashlib.sha256(retained.read_bytes()).hexdigest()
                    else:
                        mutated[change]["sha256"] = "0" * 64
                    target.write_text(json.dumps(mutated))
                    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    expected_error = ("wrong current value" if change == "value" else
                                      "retained output differs" if change.startswith("retained_") else
                                      "path/SHA256 pin")
                    self.assertIn(expected_error, result.stderr)
                    if change.startswith("retained_"):
                        shutil.copyfile(self.root / self.pins[pin]["path"], retained)


if __name__ == "__main__":
    unittest.main(verbosity=2)
