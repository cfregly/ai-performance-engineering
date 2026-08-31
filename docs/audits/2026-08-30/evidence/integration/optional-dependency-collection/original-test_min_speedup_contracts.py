import json
from pathlib import Path

from ch03.baseline_gemm import BaselineGemmBenchmark
from ch08.baseline_nvfp4_mlp import BaselineChapter8NVFP4MLPBenchmark
from ch08.baseline_thresholdtma import BaselineThresholdTMABenchmark
from ch08.baseline_tiling import BaselineTilingBenchmark
from ch08.baseline_tiling_tcgen05 import BaselineTilingBenchmarkTCGen05
from core.benchmark.expectations import ExpectationEntry, RunProvenance
from core.analysis.refresh_expectations_from_run import _expectation_example_key as refresh_expectation_example_key
from core.benchmark.bench_commands import _expectation_example_key as bench_command_expectation_example_key
from core.harness.run_benchmarks import (
    _format_failed_no_speedup,
    _resolve_local_contract_from_expectation,
    _should_fail_no_speedup,
    build_expectation_metadata,
    expectation_example_key,
)
from labs.fullstack_cluster.optimized_cluster_gemm_tcgen05_cta2 import (
    get_benchmark as get_fullstack_cluster_gemm_tcgen05_cta2_benchmark,
)
from labs.occupancy_tuning.optimized_proton_matmul_bm64_bn64_bk32_nw2 import (
    get_benchmark as get_low_warp_proton_benchmark,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _provenance() -> RunProvenance:
    return RunProvenance(
        git_commit="deadbee",
        hardware_key="b200",
        profile_name="minimal",
        timestamp="2026-03-28T00:00:00+00:00",
        iterations=10,
        warmup_iterations=5,
        execution_environment="virtualized",
        validity_profile="portable",
    )


def test_should_fail_no_speedup_respects_local_minimum_speedup() -> None:
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.01,
            "minimum_required_speedup": 1.02,
        }
    ) is True
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.03,
            "minimum_required_speedup": 1.02,
        }
    ) is False


def test_should_fail_no_speedup_defaults_to_generic_gate_without_local_contract() -> None:
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.04,
        }
    ) is True
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.05,
        }
    ) is False


def test_format_failed_no_speedup_uses_local_minimum_speedup() -> None:
    rendered = _format_failed_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.01,
            "minimum_required_speedup": 1.02,
        }
    )
    assert "1.01x" in rendered
    assert "1.02x threshold" in rendered


def test_build_expectation_metadata_includes_minimum_required_speedup() -> None:
    metadata = build_expectation_metadata(
        {
            "example": "cache_aware_disagg",
            "type": "python",
            "optimization_goal": "speed",
            "baseline_time_ms": 10.0,
            "minimum_required_speedup": 1.02,
        },
        {"file": "optimized_cache_aware_disagg.py", "technique": "optimized", "time_ms": 9.2},
        git_commit="deadbee",
    )
    assert metadata["minimum_required_speedup"] == 1.02


def test_expectation_entry_round_trip_preserves_minimum_required_speedup() -> None:
    entry = ExpectationEntry(
        example="cache_aware_disagg",
        type="python",
        optimization_goal="speed",
        baseline_time_ms=10.0,
        best_optimized_time_ms=9.2,
        provenance=_provenance(),
        best_optimization_name="optimized",
        best_optimization_file="optimized_cache_aware_disagg.py",
        best_optimization_technique="optimized",
        minimum_required_speedup=1.02,
    )

    serialized = entry.to_dict()
    assert serialized["metadata"]["minimum_required_speedup"] == 1.02

    restored = ExpectationEntry.from_dict(serialized)
    assert restored.minimum_required_speedup == 1.02


def test_resolve_local_contract_from_expectation_requires_explicit_speed_floor() -> None:
    entry = ExpectationEntry(
        example="launch_bounds",
        type="python",
        optimization_goal="speed",
        baseline_time_ms=10.072,
        best_optimized_time_ms=10.0,
        provenance=_provenance(),
        best_optimization_name="optimized_launch_bounds",
    )

    goal, minimum_required_speedup = _resolve_local_contract_from_expectation(entry)

    assert goal == "speed"
    assert minimum_required_speedup is None


def test_resolve_local_contract_from_expectation_keeps_non_speed_goal_without_threshold() -> None:
    entry = ExpectationEntry(
        example="padding_aware_transformer",
        type="python",
        optimization_goal="memory",
        baseline_time_ms=10.0,
        best_optimized_time_ms=12.0,
        baseline_memory_mb=1000.0,
        best_optimized_memory_mb=250.0,
        provenance=_provenance(),
        best_optimization_name="optimized_padding_aware_transformer",
    )

    goal, minimum_required_speedup = _resolve_local_contract_from_expectation(entry)

    assert goal == "memory"
    assert minimum_required_speedup is None


def test_block_scaling_expectation_uses_explicit_local_floor_below_historical_best() -> None:
    payload = json.loads((REPO_ROOT / "labs" / "block_scaling" / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = ExpectationEntry.from_dict(payload["examples"]["block_scaling"])

    assert entry.minimum_required_speedup == 1.75
    assert entry.best_speedup > entry.minimum_required_speedup


def test_small_effect_b200_examples_keep_explicit_local_speed_floors() -> None:
    ch06_payload = json.loads((REPO_ROOT / "ch06" / "expectations_b200.json").read_text(encoding="utf-8"))
    launch_bounds = ExpectationEntry.from_dict(ch06_payload["examples"]["launch_bounds"])
    launch_bounds_cuda = ExpectationEntry.from_dict(ch06_payload["examples"]["launch_bounds_cuda"])

    assert launch_bounds.minimum_required_speedup == 1.005
    assert launch_bounds.best_speedup > launch_bounds.minimum_required_speedup
    assert launch_bounds_cuda.minimum_required_speedup == 1.005
    assert launch_bounds_cuda.best_speedup > launch_bounds_cuda.minimum_required_speedup


def test_model_compile_reduced_precision_b200_uses_explicit_local_floor() -> None:
    payload = json.loads((REPO_ROOT / "ch14" / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = ExpectationEntry.from_dict(payload["examples"]["model_compile_reduced_precision"])

    assert entry.minimum_required_speedup == 1.01
    assert entry.best_speedup > entry.minimum_required_speedup


def test_expectation_example_key_keeps_cuda_examples_with_cuda_suffix_stable() -> None:
    for helper in (
        expectation_example_key,
        bench_command_expectation_example_key,
        refresh_expectation_example_key,
    ):
        assert helper("launch_bounds_cuda", "cuda") == "launch_bounds_cuda"
        assert helper("double_buffered_pipeline", "cuda") == "double_buffered_pipeline_cuda"


def test_persistent_decode_nvlink_offload_is_comparison_contract_while_paged_offload_stays_speed() -> None:
    payload = json.loads((REPO_ROOT / "labs" / "persistent_decode" / "expectations_b200.json").read_text(encoding="utf-8"))
    nvlink_entry = ExpectationEntry.from_dict(payload["examples"]["nvlink_offload"])
    paged_entry = ExpectationEntry.from_dict(payload["examples"]["paged_kv_offload"])

    assert nvlink_entry.optimization_goal == "comparison"
    assert nvlink_entry.minimum_required_speedup is None
    assert paged_entry.optimization_goal == "speed"
    assert paged_entry.minimum_required_speedup is None


def test_fullstack_cluster_tcgen05_cta2_stays_comparison_contract() -> None:
    benchmark = get_fullstack_cluster_gemm_tcgen05_cta2_benchmark()

    assert benchmark.get_optimization_goal() == "comparison"
    assert benchmark.baseline_alias == "cluster_gemm_tcgen05"


def test_occupancy_tuning_low_warp_schedule_stays_comparison_contract() -> None:
    payload = json.loads((REPO_ROOT / "labs" / "occupancy_tuning" / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = ExpectationEntry.from_dict(payload["examples"]["proton_matmul_bm64_bn64_bk32_nw2"])
    benchmark = get_low_warp_proton_benchmark()

    assert entry.optimization_goal == "comparison"
    assert entry.minimum_required_speedup is None
    assert benchmark.get_optimization_goal() == "comparison"


def test_nvfp4_group_gemm_shape_surface_uses_canonical_frontdoor_and_comparison_shapes() -> None:
    payload = json.loads((REPO_ROOT / "labs" / "nvfp4_group_gemm" / "expectations_b200.json").read_text(encoding="utf-8"))

    assert "nvfp4_group_gemm_case0" not in payload["examples"]
    assert "nvfp4_group_gemm_case1" not in payload["examples"]
    assert "nvfp4_group_gemm_case2" not in payload["examples"]
    assert "nvfp4_group_gemm_case3" not in payload["examples"]

    canonical = ExpectationEntry.from_dict(payload["examples"]["nvfp4_group_gemm"])
    winner = ExpectationEntry.from_dict(payload["examples"]["nvfp4_group_gemm_g2_n3072_k4096"])
    loser_one = ExpectationEntry.from_dict(payload["examples"]["nvfp4_group_gemm_g8_n4096_k7168"])
    loser_two = ExpectationEntry.from_dict(payload["examples"]["nvfp4_group_gemm_g8_n7168_k2048"])
    loser_three = ExpectationEntry.from_dict(payload["examples"]["nvfp4_group_gemm_g2_n4096_k1536"])

    assert canonical.optimization_goal == "speed"
    assert canonical.minimum_required_speedup == 1.005
    assert canonical.best_speedup > canonical.minimum_required_speedup

    assert winner.optimization_goal == "speed"
    assert winner.minimum_required_speedup == 1.005
    assert winner.best_speedup > winner.minimum_required_speedup

    assert loser_one.optimization_goal == "comparison"
    assert loser_one.minimum_required_speedup is None
    assert loser_two.optimization_goal == "comparison"
    assert loser_two.minimum_required_speedup is None
    assert loser_three.optimization_goal == "comparison"
    assert loser_three.minimum_required_speedup is None


def test_ch08_bridge_comparisons_keep_comparison_contracts() -> None:
    expectation_paths = (
        REPO_ROOT / "ch08" / "expectations_b200.json",
        REPO_ROOT / "ch08" / "expectations_2x_b200.json",
        REPO_ROOT / "ch08" / "expectations_4x_gb200.json",
    )
    expected_keys = {
        "nvfp4_mlp",
        "thresholdtma",
        "thresholdtma_cuda",
        "tiling",
        "tiling_tcgen05",
    }
    for path in expectation_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))["examples"]
        present = sorted(expected_keys.intersection(payload.keys()))
        assert present, path.name
        for key in present:
            entry = ExpectationEntry.from_dict(payload[key])
            assert entry.optimization_goal == "comparison", (path.name, key)
            assert entry.minimum_required_speedup is None, (path.name, key)

    assert object.__new__(BaselineChapter8NVFP4MLPBenchmark).get_optimization_goal() == "comparison"
    assert object.__new__(BaselineThresholdTMABenchmark).get_optimization_goal() == "comparison"
    assert object.__new__(BaselineTilingBenchmark).get_optimization_goal() == "comparison"
    assert object.__new__(BaselineTilingBenchmarkTCGen05).get_optimization_goal() == "comparison"


def test_ch03_gemm_stays_comparison_contract() -> None:
    expectation_paths = (
        REPO_ROOT / "ch03" / "expectations_b200.json",
        REPO_ROOT / "ch03" / "expectations_4x_gb200.json",
    )
    for path in expectation_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))["examples"]
        entry = ExpectationEntry.from_dict(payload["gemm"])
        assert entry.optimization_goal == "comparison", path.name
        assert entry.minimum_required_speedup is None, path.name

    assert object.__new__(BaselineGemmBenchmark).get_optimization_goal() == "comparison"
