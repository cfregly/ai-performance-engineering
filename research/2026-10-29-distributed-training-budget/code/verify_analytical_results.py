#!/usr/bin/env python3
"""Check the retained 38-row analytical export against the current calculator.

This is an export/data-contract check, not independent scientific qualification.
The comparison budget is a declared assumption from the workbook, not a result.
All production checks use explicit exceptions and remain active under python -O.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import reader_summary
import system_budget


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_values(w, s):
    """Bind export names/units to fresh summary and workbook fields, not numbers.

    The two projections must agree. Unit conversions and published rounding are
    part of this export contract. This does not re-prove the workbook's physics.
    """
    rows = {}

    def add(name, unit, summary_value, workbook_value):
        require(name not in rows, f"duplicate expected name: {name}")
        require(summary_value == workbook_value,
                f"summary/workbook projection differs: {name}")
        rows[name] = {"name": name, "unit": unit, "value": summary_value}

    x, model = w["inputs"], s["model"]
    add("GPUs", "GPUs", model["GPUs"], x["replicas"] * x["stages"] * x["ranks_per_stage"])
    add("nodes", "nodes", model["nodes"], x["replicas"] * x["stages"])
    add("racks", "racks", model["racks"], x["racks"])
    add("unique_parameters", "parameters", model["unique_parameters"], w["parameters"]["unique_model_parameters"])
    add("tokens_per_step", "tokens/update", model["tokens_per_step"], x["replicas"] * x["microbatches"] * x["sequence"])
    stage_work = w["compute"]["stage_training_dense_matmul_flops_both_replicas"]
    add("last_stage_extra_matrix_work", "percent", s["compute"]["stage3_extra_percent_full_head"],
        round((stage_work[x["stages"] - 1] / stage_work[0] - 1) * 100, 6))
    for name, key, source_key in (
        ("physical_training_dense_work", "physical_training_dense_TF", "training_dense_matmul_flops_physical"),
        ("physical_muon_dense_work", "physical_Muon_dense_TF_separate", "muon_dense_matmul_flops_physical"),
    ):
        add(name, "TFLOP/update", s["compute"][key], w["compute"][source_key] / 10 ** 12)
    for placement in ("A", "B"):
        ledger = w["traffic"][placement]
        view = s["traffic_and_screens"][placement]
        # This fixed two-rack export has equal directed rack payloads. Check the
        # condition before converting its per-direction summary into both ways.
        directed = list(ledger["interrack_directed_bytes"].values())
        require(len(directed) == 2 and len(set(directed)) == 1,
                f"two equal rack directions required: {placement}")
        add(f"{placement}_interrack_payload_both_directions", "B/update",
            view["interrack_bytes_each_direction"] * len(directed), sum(directed))
        add(f"{placement}_maximum_node_offnode_payload_one_direction", "B/update",
            view["max_node_offnode_bytes_one_direction"],
            max(ledger["max_node_offnode_sent_bytes"], ledger["max_node_offnode_received_bytes"]))
        for exported, key in (("pre_clipping_phase", "pre_clipping_phase"),
                              ("post_clipping_phase", "post_clipping_phase"),
                              ("barrier_ordered_phase_floor", "necessary_step_floor")):
            add(f"{placement}_{exported}", "ms/update", view["floors_ms"][key],
                round(w["resource_floors"]["by_placement"][placement][key + "_seconds"] * 1000, 9))
    for label, source in (("balanced", "capacity_balanced"), ("worst_skew", "capacity_worst_skew")):
        by_stage = w[source]["by_stage_per_rank"]
        require(len(by_stage) == x["stages"], f"wrong capacity stage count: {label}")
        for stage, capacity in enumerate(by_stage):
            add(f"stage_{stage}_{label}_declared_live_allocation", "GB",
                s["declared_capacity"][label + "_live_GB_per_stage"][stage],
                capacity["declared_live_bytes"] / 10 ** 9)
    # No measured budget is inferred: this row is the workbook's declared input.
    budget_ms = w["hypothetical_budget_screen"]["budget_seconds"] * 1000
    require(f" at {budget_ms:g} ms:" in s["decision"], "summary decision uses a different declared budget")
    add("comparison_budget", "ms/update", budget_ms, budget_ms)
    p, pv = w["pipeline"], s["unit_1F1B"]
    add("ideal_pipeline_service_slots", "unit-service slots", pv["slots"], p["makespan_slots"])
    add("ideal_pipeline_idle_fraction", "fraction of elapsed slots", pv["idle_fraction"], p["idle_fraction_per_stage"])
    add("ideal_pipeline_idle_over_busy", "fraction of busy slots", pv["idle_over_busy"],
        (p["makespan_slots"] - p["busy_slots_per_stage"]) / p["busy_slots_per_stage"])
    alt, av = w["alternative_reduce_scatter_first"], s["alternative_DP_schedule"]
    for placement in ("A", "B"):
        for key in ("pre_clipping_phase", "post_clipping_phase", "necessary_step_floor"):
            add(f"RS_first_{placement}_{key}", "ms/update necessary lower bound",
                av["phase_and_floor_ms"][placement][key],
                round(alt["resource_floors"]["by_placement"][placement][key + "_seconds"] * 1000, 9))
    for replica in range(x["replicas"]):
        add(f"RS_first_unique_norm_elements_replica_{replica}", "parameters",
            av["norm_elements_each_replica"][str(replica)], alt["norm_ownership"]["elements_by_replica"][str(replica)])
    require(len(rows) == 38, "this version of the analytical export requires exactly 38 named rows")
    return rows


def validate_document(document, expected, pins):
    """Validate caller-supplied artifact data; used by the file verifier and tests."""
    require(document.get("schema_version") == 1, "wrong analytical schema version")
    require(document.get("result_kind") == "synthetic_arithmetic", "wrong analytical result kind")
    values = document.get("values")
    require(isinstance(values, list), "values must be a list")
    actual = {}
    for row in values:
        require(isinstance(row, dict) and set(row) == {"name", "value", "unit"}, "invalid value row fields")
        name = row["name"]
        require(isinstance(name, str), "value name must be a string")
        require(name not in actual, f"duplicate value name: {name}")
        actual[name] = row
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    require(not missing and not extra, f"value names differ: missing={sorted(missing)}, extra={sorted(extra)}")
    for name, expected_row in expected.items():
        row = actual[name]
        require(row["unit"] == expected_row["unit"], f"wrong unit: {name}")
        value = row["value"]
        require(type(value) in (int, float) and math.isfinite(value), f"invalid numeric value: {name}")
        require(value == expected_row["value"], f"wrong current value: {name}")
    for key, pin in pins.items():
        require(document.get(key) == pin, f"stale or wrong {key} path/SHA256 pin")


def verify(package_root=None):
    root = (package_root or Path(__file__).resolve().parents[1]).resolve()
    for module, filename in ((system_budget, "system_budget.py"), (reader_summary, "reader_summary.py")):
        require(Path(module.__file__).resolve() == root / "code" / filename,
                f"wrong imported source module: {filename}")
    w = system_budget.workbook()
    s = reader_summary.summary(w)
    fresh = {
        "evidence/system-budget.json": (json.dumps(w, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "evidence/reader-summary.json": (json.dumps(s, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8"),
    }
    for path, data in fresh.items():
        require((root / path).read_bytes() == data, f"retained output differs from current calculator: {path}")
    pins = {}
    for key, path in (("code", "code/system_budget.py"),
                      ("source_results", "evidence/reader-summary.json"),
                      ("full_workbook_results", "evidence/system-budget.json")):
        pins[key] = {"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()}
    expected = expected_values(w, s)
    document = json.loads((root / "evidence/analytical-results.json").read_text())
    validate_document(document, expected, pins)
    return {"status": "PASS", "named_values": len(expected), "pins": pins,
            "reader_summary_code_sha256": hashlib.sha256((root / "code/reader_summary.py").read_bytes()).hexdigest(),
            "scope": "current 38-row export, units, values, source pins and fresh retained serializations; no scientific qualification"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.package_root), indent=2, sort_keys=True))
