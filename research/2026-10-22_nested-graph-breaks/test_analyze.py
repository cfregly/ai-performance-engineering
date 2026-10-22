#!/usr/bin/env python3
"""CPU-only semantic tests for the benchmark evidence analyzer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import struct
import unittest
from typing import Any, Dict, List

import analyze


def _sample(seed: int) -> Dict[str, Any]:
    indices = [
        0,
        1,
        17,
        1024,
        analyze.INPUT_ELEMENTS // 3,
        analyze.INPUT_ELEMENTS // 2,
        analyze.INPUT_ELEMENTS - 1,
    ]
    values = [seed + offset / 100.0 for offset in range(len(indices))]
    packed = b"".join(struct.pack("!d", float(value)) for value in values)
    return {
        "indices": indices,
        "values": values,
        "sha256": hashlib.sha256(packed).hexdigest(),
        "hash_encoding": "network-order float64 values at selected indices",
    }


def _p95(values: List[float]) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * 0.95
    low = math.floor(rank)
    high = math.ceil(rank)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _record(case: str, enabled: int, seed: int, speedup: float) -> Dict[str, Any]:
    spec = analyze.CASES[case]
    baseline_latency = 10.0 + seed * 0.1
    latency = baseline_latency / speedup if enabled else baseline_latency
    cold_call = (100.0 + seed) / speedup if enabled else (100.0 + seed)
    throughput_call = (9.0 + seed * 0.05) / speedup if enabled else (9.0 + seed * 0.05)
    cold = [cold_call] * analyze.COLD_CALLS
    steady = [latency] * analyze.LATENCY_TRIALS
    blocks = [
        {
            "total_ms": throughput_call * analyze.THROUGHPUT_CALLS_PER_BLOCK,
            "per_call_ms": throughput_call,
        }
        for _ in range(analyze.THROUGHPUT_BLOCKS)
    ]
    math_signature = {
        "depth": spec["depth"],
        "pre": ["sin", "multiply:0.5", "residual_add", "cos"],
        "post": ["sin", "multiply:0.25", "residual_add", "cos"],
    }
    math_hash = hashlib.sha256(
        json.dumps(math_signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    generated_source = "synthetic workload for " + case
    if spec["explicit_graph_breaks"]:
        generated_source += "\ntorch._dynamo.graph_break()"
    source_hash = hashlib.sha256(generated_source.encode()).hexdigest()
    gpu_uuid = f"GPU-TEST-{seed % 2}"
    cache_base = f"/tmp/synthetic/{case}-{enabled}-{seed}"
    fixed_environment = {
        "CUDA_CACHE_DISABLE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
        "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
        "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "0",
        "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
        "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "0",
    }
    inductor_options = {
        "compile_threads": 1,
        "fx_graph_cache": False,
        "fx_graph_remote_cache": False,
        "autotune_remote_cache": False,
        "bundled_autotune_remote_cache": False,
        "triton.cudagraphs": False,
        "triton.cudagraph_skip_dynamic_graphs": False,
        "max_autotune": False,
    }
    functorch_options = {
        "enable_autograd_cache": False,
        "enable_remote_autograd_cache": False,
        "bundled_autograd_cache": False,
    }
    profile_requested = case == analyze.PRIMARY_CASE and seed == 0
    profile = (
        {
            "requested": True,
            "status": "captured",
            "evidence_complete": True,
            "trace_sha256": "d" * 64,
            "cuda_graph_launch_event_count": 5,
            "cuda_kernel_event_count": 10,
            "cudagraph_step_marker_used": False,
            "output_lifetime": "released before each subsequent invocation",
        }
        if profile_requested
        else {"requested": False, "status": "not_requested"}
    )
    return {
        "schema": analyze.INPUT_SCHEMA,
        "benchmark_id": analyze.BENCHMARK_ID,
        "scope": analyze.SCOPE,
        "status": "pass",
        "case": case,
        "enabled": enabled,
        "seed": seed,
        "profile_requested": profile_requested,
        "treatment": {
            "name": "torch._dynamo.config.nested_graph_breaks",
            "requested": bool(enabled),
            "actual": bool(enabled),
            "only_intended_arm_difference": True,
        },
        "workload": {
            "depth": spec["depth"],
            "explicit_graph_breaks": spec["explicit_graph_breaks"],
            "pre_pointwise_ops_per_frame": 4,
            "post_pointwise_ops_per_frame": 4,
            "pointwise_math": ["sin", "multiply", "residual_add", "cos"],
            "input_elements": analyze.INPUT_ELEMENTS,
            "input_bytes": analyze.INPUT_BYTES,
            "input_dtype": "torch.float32",
            "input_distribution": "torch.randn with per-process CUDA generator",
            "generated_source": generated_source,
            "generated_source_sha256": source_hash,
            "math_signature": math_signature,
            "math_signature_sha256": math_hash,
        },
        "compile": {
            "api": "torch.compile",
            "backend": "inductor",
            "mode": "reduce-overhead",
            "fullgraph": False,
            "mode_options": {"triton.cudagraphs": True},
        },
        "device": {
            "type": "cuda",
            "logical_index": 0,
            "torch_name": "NVIDIA B200",
            "expected_gpu_uuid": gpu_uuid,
            "cuda_visible_devices": gpu_uuid,
            "observed_uuid": gpu_uuid,
        },
        "correctness": {
            "pass": True,
            "first_compiled_call": {
                "pass": True,
                "assert_close_pass": True,
                "finite": True,
                "shape_match": True,
                "dtype_match": True,
                "rtol": 1e-4,
                "atol": 1e-5,
                "reference_shape": [analyze.INPUT_ELEMENTS],
                "compiled_shape": [analyze.INPUT_ELEMENTS],
                "reference_dtype": "torch.float32",
                "compiled_dtype": "torch.float32",
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "sample_output": _sample(seed),
            },
            "post_steady_compiled_call": {
                "pass": True,
                "assert_close_pass": True,
                "finite": True,
                "shape_match": True,
                "dtype_match": True,
                "rtol": 1e-4,
                "atol": 1e-5,
                "reference_shape": [analyze.INPUT_ELEMENTS],
                "compiled_shape": [analyze.INPUT_ELEMENTS],
                "reference_dtype": "torch.float32",
                "compiled_dtype": "torch.float32",
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "sample_output": _sample(seed),
            },
        },
        "timing": {
            "first_compiled_call_host_ms": cold[0],
            "cold_first_10": {
                "raw_host_ms": cold,
                "total_host_ms": sum(cold),
                "mean_host_ms": statistics.fmean(cold),
                "median_host_ms": statistics.median(cold),
            },
            "warmup_calls": analyze.WARMUP_CALLS,
            "steady_invocation_latency": {
                "raw_host_ms": steady,
                "median_host_ms": statistics.median(steady),
                "mean_host_ms": statistics.fmean(steady),
                "p95_host_ms": _p95(steady),
            },
            "batched_throughput": {
                "calls_per_block": analyze.THROUGHPUT_CALLS_PER_BLOCK,
                "blocks": blocks,
                "median_per_call_ms": throughput_call,
            },
            "compiler_counter_stability": {
                "before_steady_latency": {"stats.unique_graphs": 1},
                "after_steady_latency": {"stats.unique_graphs": 1},
                "after_batched_throughput": {"stats.unique_graphs": 1},
                "steady_latency_delta": {},
                "batched_throughput_delta": {},
                "stable": True,
            },
        },
        "dynamo": {
            "all": {"stats": {"unique_graphs": 1}},
            "stats": {"unique_graphs": 1},
            "graph_break_reason_counts": (
                {"Call to torch._dynamo.graph_break()": 1}
                if spec["explicit_graph_breaks"]
                else {}
            ),
            "graph_break_event_sum": spec["explicit_graph_breaks"],
            "graph_break_distinct_reason_count": spec["explicit_graph_breaks"],
            "unique_graphs": 1,
        },
        "inductor_metrics": {"available": True, "generated_kernel_count": 1},
        "profile": profile,
        "runtime_warnings": [],
        "runtime": {
            "python": "3.12.0 synthetic",
            "platform": "linux-synthetic",
            "hostname": "b200-host",
            "torch_version": "2.14.0.dev",
            "torch_git_version": "a" * 40,
            "torch_cuda_version": "13.0",
            "cudnn_version": 99999,
            "num_threads": 1,
            "num_interop_threads": 1,
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "inductor_options": inductor_options,
            "functorch_options": functorch_options,
            "fixed_environment": fixed_environment,
        },
        "controller_contract": {
            "expected_gpu_uuid": gpu_uuid,
            "cuda_visible_devices": gpu_uuid,
            "cache_dirs": {
                "TORCHINDUCTOR_CACHE_DIR": {
                    "path": cache_base + "/inductor",
                    "initially_empty": True,
                    "initial_entries": [],
                },
                "TRITON_CACHE_DIR": {
                    "path": cache_base + "/triton",
                    "initially_empty": True,
                    "initial_entries": [],
                },
            },
        },
        "provenance": {
            "worker_source_sha256": "b" * 64,
            "flag_source_snapshot_sha256": "c" * 64,
            "flag_source_snapshot_line": 587,
            "stderr_capture_owner": "controller",
        },
        "_source_path": f"synthetic/{case}/{enabled}/{seed}/result.json",
    }


def _valid_matrix(primary_speedup: float = 1.20) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for case in analyze.CASES:
        for enabled in analyze.ARMS:
            for seed in analyze.SEEDS:
                speedup = primary_speedup if case == analyze.PRIMARY_CASE else 1.0
                records.append(_record(case, enabled, seed, speedup))
    return records


def _drift_no_break_cold_control(records: List[Dict[str, Any]]) -> None:
    target = next(
        record
        for record in records
        if record["case"] == "no_break_depth4"
        and record["enabled"] == 1
        and record["seed"] == 3
    )
    baseline_cold_call = 100.0 + target["seed"]
    drifted_cold_call = baseline_cold_call / 1.06
    cold = [drifted_cold_call] * analyze.COLD_CALLS
    target["timing"]["first_compiled_call_host_ms"] = cold[0]
    target["timing"]["cold_first_10"].update(
        {
            "raw_host_ms": cold,
            "total_host_ms": sum(cold),
            "mean_host_ms": statistics.fmean(cold),
            "median_host_ms": statistics.median(cold),
        }
    )


class AnalyzerSemanticTests(unittest.TestCase):
    def test_known_fixed_primary_ratios_pass_steady_gate(self) -> None:
        analysis = analyze.analyze_records(_valid_matrix(primary_speedup=1.20))
        self.assertEqual(analysis["decision"]["verdict"], "PASS")
        self.assertEqual(analysis["decision"]["win_kind"], "steady_invocation")
        primary = analysis["cases"][analyze.PRIMARY_CASE]
        self.assertAlmostEqual(
            primary["steady_median_invocation"]["geometric_mean_ratio"], 1.20
        )
        self.assertGreaterEqual(
            primary["steady_median_invocation"]["observed_min_seed_ratio"], 1.10
        )
        self.assertIsNone(primary["steady_median_invocation"]["coverage_claim"])

    def test_missing_arm_is_rejected(self) -> None:
        records = _valid_matrix()
        records = [
            record
            for record in records
            if not (
                record["case"] == analyze.PRIMARY_CASE
                and record["enabled"] == 1
                and record["seed"] == 2
            )
        ]
        with self.assertRaisesRegex(
            analyze.ValidationError, "missing result identities"
        ):
            analyze.analyze_records(records)

    def test_seed_identity_must_change_sampled_output(self) -> None:
        records = _valid_matrix()
        selected = [
            record
            for record in records
            if record["case"] == "nested_break_depth4" and record["enabled"] == 0
        ]
        repeated_first = copy.deepcopy(
            selected[0]["correctness"]["first_compiled_call"]["sample_output"]
        )
        repeated_final = copy.deepcopy(
            selected[0]["correctness"]["post_steady_compiled_call"]["sample_output"]
        )
        for record in selected[1:]:
            record["correctness"]["first_compiled_call"]["sample_output"] = (
                copy.deepcopy(repeated_first)
            )
            record["correctness"]["post_steady_compiled_call"]["sample_output"] = (
                copy.deepcopy(repeated_final)
            )
        with self.assertRaisesRegex(
            analyze.ValidationError, "seeded inputs/outputs are not distinct"
        ):
            analyze.analyze_records(records)

    def test_wrong_flag_is_rejected(self) -> None:
        records = _valid_matrix()
        target = next(record for record in records if record["enabled"] == 1)
        target["treatment"]["actual"] = False
        with self.assertRaisesRegex(analyze.ValidationError, "treatment.actual"):
            analyze.analyze_records(records)

    def test_wrong_runtime_is_rejected(self) -> None:
        records = _valid_matrix()
        records[0]["runtime"]["inductor_options"]["compile_threads"] = 2
        with self.assertRaisesRegex(analyze.ValidationError, "compile_threads"):
            analyze.analyze_records(records)

    def test_profile_scope_requires_both_seed_zero_primary_arms(self) -> None:
        records = _valid_matrix()
        target = next(
            record
            for record in records
            if record["case"] == analyze.PRIMARY_CASE
            and record["enabled"] == 1
            and record["seed"] == 0
        )
        target["profile_requested"] = False
        target["profile"] = {"requested": False, "status": "not_requested"}
        with self.assertRaisesRegex(analyze.ValidationError, "profile_requested"):
            analyze.analyze_records(records)

    def test_correctness_failure_is_rejected(self) -> None:
        records = _valid_matrix()
        records[0]["correctness"]["first_compiled_call"]["assert_close_pass"] = False
        records[0]["correctness"]["first_compiled_call"]["pass"] = False
        records[0]["correctness"]["pass"] = False
        with self.assertRaisesRegex(analyze.ValidationError, "assert_close_pass"):
            analyze.analyze_records(records)

    def test_timed_recompile_counter_delta_is_rejected(self) -> None:
        records = _valid_matrix()
        records[0]["timing"]["compiler_counter_stability"]["stable"] = False
        records[0]["timing"]["compiler_counter_stability"]["steady_latency_delta"] = {
            "stats.unique_graphs": 1
        }
        with self.assertRaisesRegex(
            analyze.ValidationError, "compiler_counter_stability"
        ):
            analyze.analyze_records(records)

    def test_missing_capture_evidence_blocks_qualified_win_but_keeps_measurement(
        self,
    ) -> None:
        records = _valid_matrix(primary_speedup=1.20)
        target = next(
            record
            for record in records
            if record["case"] == analyze.PRIMARY_CASE
            and record["enabled"] == 0
            and record["seed"] == 0
        )
        target["status"] = "benchmark_pass_profile_failed"
        target["profile"] = {
            "requested": True,
            "status": "failed_missing_required_markers",
            "evidence_complete": False,
            "error": "synthetic missing cudaGraphLaunch marker",
        }
        analysis = analyze.analyze_records(records)
        self.assertTrue(analysis["decision"]["measurement_gate_pass"])
        self.assertFalse(analysis["decision"]["capture_qualification_pass"])
        self.assertEqual(analysis["decision"]["verdict"], "INCONCLUSIVE")
        self.assertFalse(
            analysis["capture_qualification"]["cuda_graph_claim_permitted"]
        )

    def test_no_break_control_drift_blocks_qualified_win(self) -> None:
        records = _valid_matrix(primary_speedup=1.20)
        for record in records:
            if (
                record["case"] == "no_break_depth4"
                and record["enabled"] == 1
                and record["seed"] == 3
            ):
                baseline_latency = 10.0 + record["seed"] * 0.1
                drifted_latency = baseline_latency / 1.06
                steady = [drifted_latency] * analyze.LATENCY_TRIALS
                record["timing"]["steady_invocation_latency"].update(
                    {
                        "raw_host_ms": steady,
                        "median_host_ms": statistics.median(steady),
                        "mean_host_ms": statistics.fmean(steady),
                        "p95_host_ms": _p95(steady),
                    }
                )
                break
        analysis = analyze.analyze_records(records)
        self.assertTrue(analysis["decision"]["measurement_gate_pass"])
        self.assertTrue(analysis["decision"]["capture_qualification_pass"])
        self.assertFalse(analysis["decision"]["control_validity_pass"])
        self.assertEqual(analysis["control_validity"]["steady"]["status"], "FAIL")
        self.assertEqual(analysis["decision"]["verdict"], "INCONCLUSIVE")

    def test_cold_control_drift_blocks_only_startup_route(self) -> None:
        steady_records = _valid_matrix(primary_speedup=1.20)
        _drift_no_break_cold_control(steady_records)
        steady_analysis = analyze.analyze_records(steady_records)
        self.assertEqual(steady_analysis["decision"]["verdict"], "PASS")
        self.assertEqual(steady_analysis["decision"]["win_kind"], "steady_invocation")
        self.assertTrue(steady_analysis["decision"]["steady_control_validity_pass"])
        self.assertFalse(steady_analysis["decision"]["startup_control_validity_pass"])

        startup_records = _valid_matrix(primary_speedup=1.20)
        for record in startup_records:
            if record["case"] == analyze.PRIMARY_CASE and record["enabled"] == 1:
                baseline_latency = 10.0 + record["seed"] * 0.1
                steady = [baseline_latency] * analyze.LATENCY_TRIALS
                record["timing"]["steady_invocation_latency"].update(
                    {
                        "raw_host_ms": steady,
                        "median_host_ms": statistics.median(steady),
                        "mean_host_ms": statistics.fmean(steady),
                        "p95_host_ms": _p95(steady),
                    }
                )
        startup_analysis = analyze.analyze_records(startup_records)
        self.assertEqual(startup_analysis["decision"]["verdict"], "PASS")
        self.assertEqual(startup_analysis["decision"]["win_kind"], "startup_only")

        drifted_records = copy.deepcopy(startup_records)
        _drift_no_break_cold_control(drifted_records)
        analysis = analyze.analyze_records(drifted_records)
        self.assertTrue(analysis["decision"]["measurement_gate_pass"])
        self.assertEqual(analysis["decision"]["measurement_win_kind"], "startup_only")
        self.assertTrue(analysis["decision"]["steady_control_validity_pass"])
        self.assertFalse(analysis["decision"]["startup_control_validity_pass"])
        self.assertEqual(
            analysis["control_validity"]["cold_first_10"]["status"], "FAIL"
        )
        self.assertEqual(analysis["decision"]["verdict"], "INCONCLUSIVE")

    def test_graph_break_mechanism_evidence_is_required(self) -> None:
        missing_nested_break = _valid_matrix()
        nested_target = next(
            record
            for record in missing_nested_break
            if record["case"] == "nested_break_depth8"
            and record["enabled"] == 1
            and record["seed"] == 2
        )
        nested_target["dynamo"]["graph_break_reason_counts"] = {}
        nested_target["dynamo"]["graph_break_event_sum"] = 0
        nested_target["dynamo"]["graph_break_distinct_reason_count"] = 0
        with self.subTest("nested case must retain a positive break histogram"):
            with self.assertRaisesRegex(
                analyze.ValidationError, "lacks positive graph-break events"
            ):
                analyze.analyze_records(missing_nested_break)

        false_control_break = _valid_matrix()
        control_target = next(
            record
            for record in false_control_break
            if record["case"] == "no_break_depth4"
            and record["enabled"] == 0
            and record["seed"] == 1
        )
        control_target["dynamo"]["graph_break_reason_counts"] = {
            "Call to torch._dynamo.graph_break()": 1
        }
        control_target["dynamo"]["graph_break_event_sum"] = 1
        control_target["dynamo"]["graph_break_distinct_reason_count"] = 1
        with self.subTest("no-break control must stay break-free"):
            with self.assertRaisesRegex(
                analyze.ValidationError, "no-break control recorded graph-break events"
            ):
                analyze.analyze_records(false_control_break)

    def test_unfamiliar_positive_graph_break_reason_is_accepted(self) -> None:
        records = _valid_matrix(primary_speedup=1.20)
        target = next(
            record
            for record in records
            if record["case"] == "nested_break_depth4"
            and record["enabled"] == 1
            and record["seed"] == 3
        )
        target["dynamo"]["graph_break_reason_counts"] = {
            "future PyTorch wording with no stable API token": 1
        }
        target["dynamo"]["graph_break_event_sum"] = 1
        target["dynamo"]["graph_break_distinct_reason_count"] = 1
        analysis = analyze.analyze_records(records)
        self.assertEqual(analysis["validation"]["status"], "PASS")

    def test_zero_unique_graphs_is_rejected(self) -> None:
        records = _valid_matrix()
        records[0]["dynamo"]["stats"]["unique_graphs"] = 0
        with self.assertRaisesRegex(analyze.ValidationError, "positive integer"):
            analyze.analyze_records(records)


if __name__ == "__main__":
    unittest.main()
