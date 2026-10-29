#!/usr/bin/env python3
"""Emit a compact source of truth for text/diagram labels, not performance data."""
import argparse
import json
from pathlib import Path
from system_budget import workbook


def summary(r):
    p = r["pipeline"]
    slots = []
    for stage in range(4):
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
            "binding_screen": "pre-clip rack interconnect + post-clip FP32 Muon" if placement == "A" else "pre-clip node NIC + post-clip FP32 Muon",
        }
    stages = r["compute"]["stage_training_dense_matmul_flops_both_replicas"]
    return {
        "state": "FROZEN analytical hypothetical inputs and derived labels; no measured runtime/speedup/MFU",
        "model": {"GPUs":64, "nodes":8, "GPUs_per_node":8, "racks":2, "DP":2, "PP":4,
                  "layers":12, "hidden":2048, "expert_width":4096, "experts":16, "top_k":2,
                  "S":2048, "microbatches_per_DP":8, "tokens_per_step":32768,
                  "vocab_untied_input_output":32768, "Hq":16, "Hkv":2, "D":128,
                  "unique_parameters":r["parameters"]["unique_model_parameters"]},
        "groups_each_node": {"attention_TP4_groups":[[0,1,2,3],[4,5,6,7]],
                             "attention_CP2_groups":[[0,4],[1,5],[2,6],[3,7]],
                             "EP8_same_ranks":[0,1,2,3,4,5,6,7],
                             "KV_duplicate_TP_pairs":[[0,1],[2,3]],
                             "reshard":"unique 256-token chunk/rank -> next EP rank; inverse + TP allgather; 10MiB/rank/layer/MB F+B"},
        "placement_nodes_by_rack": {"A":[[0,1,2,3],[4,5,6,7]],"B":[[0,1,4,5],[2,3,6,7]],
                                   "node_identity":"node=replica*4+stage; each node owns one stage"},
        "assumed_capabilities": {"training_BF16_TF_s_GPU":500,"Muon_FP32_TF_s_GPU":100,
                                 "HBM_TB_s_GPU":3,"HBM_phase_cache_MiB":128,"device_GB":80,
                                 "GPU_intranode_GB_s_each_direction":160,"node_NIC_GB_s_each_direction":40,
                                 "rack_intrafabric_GB_s_aggregate_injection":160,"rack_uplink_GB_s_each_direction":80},
        "traffic_and_screens":traffic,
        "compute": {"physical_training_dense_TF":r["compute"]["training_dense_matmul_flops_physical"]/1e12,
                    "physical_Muon_dense_TF_separate":r["compute"]["muon_dense_matmul_flops_physical"]/1e12,
                    "stage_training_TF_both_DP":[stages[s]/1e12 for s in range(4)],
                    "stage3_extra_percent_full_head":round((stages[3]/stages[0]-1)*100,6),
                    "scalar_ops":"RMSNorm/softmax/GELU/AdamW/copies not included in dense GEMM floors"},
        "unit_1F1B": {"assumed_F_and_B_service":1,"zero_transfer_latency":True,"slots":22,
                      "busy_each_stage":16,"idle_fraction":6/22,"idle_over_busy":6/16,"max_live_microbatches":[4,3,2,1],"rows":slots,
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
        "decision":"Primary full-DP-allreduce-before-norm schedule plus FP32 single-owner attention Muon excludes BOTH at 150 ms: A 240.629308621, B 158.934281421 ms. These are declared choices, not clipping requirements or runtimes.",
        "alternative_DP_schedule": {
            "order":"RS -> disjoint norm terms on both replicas -> clip -> allgather granted optimistic overlap with same optimizer",
            "norm_elements_each_replica":r["alternative_reduce_scatter_first"]["norm_ownership"]["elements_by_replica"],
            "phase_and_floor_ms": {placement:{key:round(f[key]*1000,9) for key in
                                               ("pre_clipping_phase_seconds","post_clipping_phase_seconds","necessary_step_floor_seconds")}
                                  for placement,f in r["alternative_reduce_scatter_first"]["resource_floors"]["by_placement"].items()},
            "limit":"Same total bytes/math. Norm service omitted. At 150 ms only A excluded; B not certified feasible. Actual DAG can raise floors. Alternative does not replace primary."},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = json.dumps(summary(workbook()), separators=(",", ":"), sort_keys=True) + "\n"
    assert len(text.encode()) <= 7000, len(text.encode())
    args.output.write_text(text)
