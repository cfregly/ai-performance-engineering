#!/usr/bin/env python3
"""Emit a compact source of truth for text/diagram labels, not performance data."""
import argparse
import json
from pathlib import Path
from system_budget import workbook


# Static group, routing and ownership descriptions belong to this geometry.
FROZEN_GEOMETRY = {
    "replicas": 2, "stages": 4, "ranks_per_stage": 8, "racks": 2,
    "layers": 12, "hidden": 2048, "expert_width": 4096, "experts": 16,
    "top_k": 2, "sequence": 2048, "vocab": 32768, "query_heads": 16,
    "kv_heads": 2, "head_dim": 128, "tp": 4, "cp": 2,
    "bf16_bytes": 2, "fp32_bytes": 4, "routing_metadata_bytes": 16,
    "expert_chunk_tokens": 128, "newton_schulz_iterations": 5,
}


def scaled(value, unit):
    """Keep exact integer labels while retaining fractional capabilities."""
    return value // unit if value % unit == 0 else value / unit


def binding_screen(floors):
    """Name the largest resource screen in each ordered phase."""
    payload_labels = {
        "gpu_intranode_payload_seconds": "GPU interconnect",
        "node_offnode_payload_seconds": "node NIC",
        "rack_intrarack_payload_seconds": "rack fabric",
        "rack_interrack_payload_seconds": "rack interconnect",
    }
    pre = {"training compute": floors["training_dense_matmul_seconds"]}
    post = {
        "FP32 Muon": floors["optimizer_Muon_dense_matmul_seconds"],
        "optimizer HBM": floors["optimizer_persistent_state_HBM_partial_seconds"],
    }
    for key, label in payload_labels.items():
        pre[label] = floors["pre_clipping_" + key]
        post[label] = floors["post_clipping_" + key]
    return f"pre-clip {max(pre, key=pre.get)} + post-clip {max(post, key=post.get)}"


def summary(r):
    x = r["inputs"]
    for key, expected in FROZEN_GEOMETRY.items():
        if x[key] != expected:
            raise ValueError(f"Static diagram labels require {key}={expected}, got {x[key]}")
    if r["resource_floors"]["schedule"] != "declared_full_allreduce_before_norm":
        raise ValueError("Primary decision labels require the declared full-allreduce-before-norm schedule")
    p = r["pipeline"]
    if p["forward_service"] != p["backward_service"]:
        raise ValueError("The unit 1F1B label requires equal forward and backward service")
    slots = []
    stage_ids = sorted(p["maximum_live_microbatches_by_stage"])
    for stage in stage_ids:
        row = [".."] * p["makespan_slots"]
        for e in p["events"]:
            if e["stage"] == stage:
                row[e["start"]] = e["operation"] + str(e["microbatch"])
        slots.append(" ".join(row))
    traffic = {}
    for placement, ledger in r["traffic"].items():
        fs = r["resource_floors"]["by_placement"][placement]
        traffic[placement] = {
            "aggregate_sent_bytes": ledger["totals_bytes"],
            "interrack_bytes_each_direction": max(ledger["interrack_directed_bytes"].values()),
            "max_GPU_intranode_bytes_one_direction": max(ledger["max_gpu_intranode_sent_bytes"], ledger["max_gpu_intranode_received_bytes"]),
            "max_node_offnode_bytes_one_direction": max(ledger["max_node_offnode_sent_bytes"], ledger["max_node_offnode_received_bytes"]),
            "pre_clip_interrack_bytes_each_direction": max(r["traffic_by_phase"][placement]["pre_clipping"]["interrack_directed_bytes"].values()),
            "pre_clip_max_node_sent_bytes": r["traffic_by_phase"][placement]["pre_clipping"]["max_node_offnode_sent_bytes"],
            "floors_ms": {k.removesuffix("_seconds"): round(v * 1000, 9) for k, v in fs.items()
                          if not ("payload" in k and k.startswith(("pre_clipping_", "post_clipping_")))},
            "binding_screen": binding_screen(fs),
        }
    stages = r["compute"]["stage_training_dense_matmul_flops_both_replicas"]
    cp_training = {stage: {0: 0, 1: 0} for stage in range(x["stages"] - 1)}
    for row in r["compute"]["by_rank"]:
        if row["stage"] in cp_training:
            cp_training[row["stage"]][row["cp"]] += row["training_dense_matmul_flops"]
    cp_extra_percent = {stage: round((work[1] / work[0] - 1) * 100, 6)
                        for stage, work in cp_training.items()}
    screen = r["hypothetical_budget_screen"]
    excluded = [placement for placement, row in screen["by_placement"].items()
                if row["status"] == "ruled_out"]
    exclusion = "BOTH" if len(excluded) == len(traffic) else ", ".join(excluded) or "neither placement"
    budget_ms = screen["budget_seconds"] * 1000
    floor_labels = ", ".join(f"{placement} {row['floors_ms']['necessary_step_floor']:.9f}"
                             for placement, row in traffic.items())
    alternative_screen = r["alternative_reduce_scatter_first"]["hypothetical_budget_screen"]
    alternative_budget_ms = alternative_screen["budget_seconds"] * 1000
    alternative_verdicts = ". ".join(
        f"{placement} is ruled out" if row["status"] == "ruled_out"
        else f"{placement} is not certified feasible"
        for placement, row in alternative_screen["by_placement"].items())
    return {
        "state": "FROZEN analytical hypothetical inputs and derived labels; no measured runtime/speedup/MFU",
        "model": {"GPUs":x["replicas"] * x["stages"] * x["ranks_per_stage"],
                  "nodes":x["replicas"] * x["stages"], "GPUs_per_node":x["ranks_per_stage"],
                  "racks":x["racks"], "DP":x["replicas"], "PP":x["stages"],
                  "layers":x["layers"], "hidden":x["hidden"], "expert_width":x["expert_width"],
                  "experts":x["experts"], "top_k":x["top_k"], "S":x["sequence"],
                  "microbatches_per_DP":x["microbatches"],
                  "tokens_per_step":x["replicas"] * x["microbatches"] * x["sequence"],
                  "vocab_untied_input_output":x["vocab"], "Hq":x["query_heads"],
                  "Hkv":x["kv_heads"], "D":x["head_dim"],
                  "unique_parameters":r["parameters"]["unique_model_parameters"]},
        "groups_each_node": {"attention_TP4_groups":[[0,1,2,3],[4,5,6,7]],
                             "attention_CP2_groups":[[0,4],[1,5],[2,6],[3,7]],
                             "EP8_same_ranks":[0,1,2,3,4,5,6,7],
                             "KV_duplicate_TP_pairs":[[0,1],[2,3]],
                             "reshard":"unique 256-token chunk/rank -> next EP rank; inverse + TP allgather; 10MiB/rank/layer/MB F+B"},
        "placement_nodes_by_rack": {"A":[[0,1,2,3],[4,5,6,7]],"B":[[0,1,4,5],[2,3,6,7]],
                                   "node_identity":"node=replica*4+stage; each node owns one stage"},
        "assumed_capabilities": {
            "training_BF16_TF_s_GPU":scaled(x["assumed_training_matmul_flops_s"], 10 ** 12),
            "Muon_FP32_TF_s_GPU":scaled(x["assumed_optimizer_fp32_flops_s"], 10 ** 12),
            "HBM_TB_s_GPU":scaled(x["assumed_hbm_bytes_s"], 10 ** 12),
            "HBM_phase_cache_MiB":scaled(x["assumed_optimizer_phase_cache_bytes"], 1024 ** 2),
            "device_GB":scaled(x["device_capacity_bytes"], 10 ** 9),
            "GPU_intranode_GB_s_each_direction":scaled(x["assumed_gpu_intranode_send_bytes_s"], 10 ** 9),
            "node_NIC_GB_s_each_direction":scaled(x["assumed_node_offnode_send_bytes_s"], 10 ** 9),
            "rack_intrafabric_GB_s_aggregate_injection":scaled(x["assumed_rack_intrarack_send_bytes_s"], 10 ** 9),
            "rack_uplink_GB_s_each_direction":scaled(x["assumed_rack_interrack_send_bytes_s"], 10 ** 9)},
        "traffic_and_screens":traffic,
        "compute": {"physical_training_dense_TF":r["compute"]["training_dense_matmul_flops_physical"]/1e12,
                    "physical_Muon_dense_TF_separate":r["compute"]["muon_dense_matmul_flops_physical"]/1e12,
                    "stage_training_TF_both_DP":[stages[s]/1e12 for s in range(4)],
                    "stage3_extra_percent_full_head":round((stages[3]/stages[0]-1)*100,6),
                    "later_CP_extra_percent_by_stage":cp_extra_percent,
                    "scalar_ops":"RMSNorm/softmax/GELU/AdamW/copies not included in dense GEMM floors"},
        "unit_1F1B": {"assumed_F_and_B_service":p["forward_service"],
                      "zero_transfer_latency":p["transfer_latency"] == 0,
                      "slots":p["makespan_slots"], "busy_each_stage":p["busy_slots_per_stage"],
                      "idle_fraction":p["idle_fraction_per_stage"],
                      "idle_over_busy":(p["makespan_slots"] - p["busy_slots_per_stage"]) / p["busy_slots_per_stage"],
                      "max_live_microbatches":[p["maximum_live_microbatches_by_stage"][stage] for stage in stage_ids],"rows":slots,
                      "limit":"slot illustration assumes equal stage service; actual head imbalance not calibrated"},
        "declared_capacity": {"balanced_live_GB_per_stage":[v["declared_live_bytes"]/1e9 for v in r["capacity_balanced"]["by_stage_per_rank"]],
                              "worst_skew_live_GB_per_stage":[v["declared_live_bytes"]/1e9 for v in r["capacity_worst_skew"]["by_stage_per_rank"]],
                              "states":"BF16 weights + FP32 master/grad; Muon one momentum =14B/p, AdamW two moments =18B/p",
                              "includes":"retained activations, route buffers/metadata, backward grads, full-head logits+dlogits, collectives, pipeline, gather, NS, publication buffers",
                              "excludes":"framework/context/allocator, extra library/transport workspaces, graph pools, compilation, checkpoint buffers; conditional ledger only"},
        "optimizer_and_routing": {"Muon":"Q/K/V/O and complete two-matrix experts; 5 finite NS iterations; not exact polar/SVD",
                                  "AdamW":"untied input embedding, output head, router and RMSNorm",
                                  "owner":"whole attention matrices gather gradient/master/momentum to stage rank0, update/scatter/publish; experts whole local",
                                  "barrier":"declared choice: full DP allreduce -> replica0 all unique norm terms -> clip -> FP32 single-owner attention Muon/AdamW -> publish; not forced by clipping alone",
                                  "routing":"4096 assignments/MB; balanced256/expert; two128-token chunks/expert. Worst two-expert skew2048/expert:16chunks each, no drops; longer serial work"},
        "decision":f"Primary full-DP-allreduce-before-norm schedule plus FP32 single-owner attention Muon excludes {exclusion} at {budget_ms:g} ms: {floor_labels} ms. These are declared choices, not clipping requirements or runtimes.",
        "alternative_DP_schedule": {
            "order":"RS -> disjoint norm terms on both replicas -> clip -> allgather granted optimistic overlap with same optimizer",
            "norm_elements_each_replica":r["alternative_reduce_scatter_first"]["norm_ownership"]["elements_by_replica"],
            "phase_and_floor_ms": {placement:{key.removesuffix("_seconds"):round(f[key]*1000,9) for key in
                                               ("pre_clipping_phase_seconds","post_clipping_phase_seconds","necessary_step_floor_seconds")}
                                  for placement,f in r["alternative_reduce_scatter_first"]["resource_floors"]["by_placement"].items()},
            "limit":f"Same total bytes/math. Norm service omitted. At {alternative_budget_ms:g} ms: {alternative_verdicts}. Actual DAG can raise floors. Alternative does not replace primary."},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = json.dumps(summary(workbook()), separators=(",", ":"), sort_keys=True) + "\n"
    assert len(text.encode()) <= 7000, len(text.encode())
    args.output.write_text(text)
