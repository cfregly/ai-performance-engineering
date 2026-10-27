from __future__ import annotations

import argparse
import copy
import hashlib
import math
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import labs.arithmetic_intensity_b200.ceiling_tool as ceiling_tool
import labs.arithmetic_intensity_b200.kernel_optimization_tool as kernel_tool
import labs.arithmetic_intensity_b200.optimization_benchmarks as optimization_benchmarks
import labs.arithmetic_intensity_b200.sustained_optimization_tool as sustained_tool
import labs.arithmetic_intensity_b200.verification_tool as verification_tool
from labs.arithmetic_intensity_b200.ceiling_protocol import (
    bandwidth_candidates,
    bootstrap_median_ci,
    confirmation_invocations_per_sample,
    gemm_candidates,
    parse_layout_csv,
    parse_positive_csv,
)
from labs.arithmetic_intensity_b200.ceiling_tool import _preferred_blas_backend
from labs.arithmetic_intensity_b200.formulas import (
    formula_report,
    gemm_logical_cost,
    memory_bound_ms,
)
from labs.arithmetic_intensity_b200.kernel_optimization_tool import (
    _nominal_gap_steady_state_qualification,
    _validate_nominal_gap_args,
    _validate_screen_for_confirmation,
)
from labs.arithmetic_intensity_b200.kernel_optimization_tool import (
    _parser as kernel_optimization_parser,
)
from labs.arithmetic_intensity_b200.optimization_benchmarks import (
    COPY_PAYLOAD_BYTES,
    GEMM_SIZE,
    SELECTED_COPY_BLOCKS_PER_SM,
    SELECTED_COPY_THREADS,
    SELECTED_GEMM_ALGORITHM_SIGNATURE,
    BaselineBf16GemmBenchmark,
    BaselineNativeCopyBenchmark,
)
from labs.arithmetic_intensity_b200.optimization_protocol import (
    alternating_pair_order,
    copy_launch_candidates,
    integrate_power_samples_over_window,
    observer_overhead_summary,
    paired_window_summary,
    select_algorithm_index,
    summarize_nominal_gap_telemetry,
)
from labs.arithmetic_intensity_b200.sustained_optimization_tool import (
    PEAK_PHASE_GPU_ACTIVE_BUDGET_MS,
    PRIOR_FULL_OUTPUT_PASSING_SIGNATURES,
    WORKSPACE_CEILINGS_BYTES,
    _balanced_fill_order,
    _bounded_candidate_signatures,
    _cluster_shape_metadata,
    _copy_window_steady_state_qualification,
    _invocation_counts,
    _k_order_control_operands,
    _parse_fill_modes,
    _planned_peak_gpu_active_ms,
    _validate_observer_args,
)
from labs.arithmetic_intensity_b200.sustained_optimization_tool import (
    _parser as sustained_optimization_parser,
)
from labs.arithmetic_intensity_b200.verification_tool import (
    _PACKAGE_ROOT,
    _bf16_fp32_accumulation_policy,
    _package_sha256,
    _preflight_output,
    _publication_status,
    _require_bf16_fp32_accumulation_policy,
    _rotation_count,
    _validate_lab_runtime_versions,
    _validate_live_source_commit,
    _write_report,
)


@pytest.mark.parametrize("triton", [None, "3.6.0", "3.7.1"])
def test_lab_runtime_contract_accepts_exact_torch_cuda_and_nonbinding_triton(
    triton: str | None,
) -> None:
    observed = _validate_lab_runtime_versions(
        pytorch="2.14.1+cu130",
        pytorch_cuda="13.0",
        triton=triton,
    )

    assert observed == {
        "qualification_role": "current_source_target",
        "pytorch": "2.14.1+cu130",
        "pytorch_cuda": "13.0",
        "triton": triton,
        "triton_role": "recorded_only_nonbinding_unused_by_lab",
    }


@pytest.mark.parametrize(
    ("pytorch", "pytorch_cuda", "mismatch"),
    [
        ("2.9.1+cu130", "13.0", "PyTorch expected 2.14.1+cu130"),
        ("2.11.0+cu130", "13.0", "PyTorch expected 2.14.1+cu130"),
        ("2.13.0+cu130", "13.0", "PyTorch expected 2.14.1+cu130"),
        ("2.14.1+cu130", "12.8", "PyTorch CUDA expected 13.0"),
        ("2.14.1", None, "PyTorch CUDA expected 13.0"),
    ],
)
def test_lab_runtime_contract_rejects_torch_or_cuda_drift(
    pytorch: str,
    pytorch_cuda: str | None,
    mismatch: str,
) -> None:
    with pytest.raises(verification_tool.DiagnosticGateError) as error:
        _validate_lab_runtime_versions(
            pytorch=pytorch,
            pytorch_cuda=pytorch_cuda,
            triton="3.6.0",
        )
    assert mismatch in str(error.value)


def test_all_live_entrypoint_seams_gate_runtime_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectCudaAccess:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"CUDA was accessed through {name} before the runtime gate")

    device = torch.device("cpu")
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch, "cuda", RejectCudaAccess())
    monkeypatch.setattr(
        verification_tool,
        "_validate_live_source_commit",
        lambda _source_commit: None,
    )
    monkeypatch.setattr(
        ceiling_tool,
        "_validate_live_source_commit",
        lambda _source_commit: None,
    )

    entrypoints = (
        lambda: verification_tool.run_gpu_diagnostic(
            argparse.Namespace(
                warmups=1,
                repeats=1,
                large_size=1,
                rotation_mib=1,
                source_commit="a" * 40,
            )
        ),
        lambda: ceiling_tool._strict_runtime(
            torch,
            device_index=0,
            source_commit="a" * 40,
            run_id="runtime-gate-test",
        ),
        lambda: optimization_benchmarks._require_b200(device),
    )

    for entrypoint in entrypoints:
        with pytest.raises(
            verification_tool.DiagnosticGateError,
            match=r"PyTorch expected 2\.14\.1\+cu130, observed 2\.11\.0\+cu130",
        ):
            entrypoint()


def test_reviewed_b200_profile_and_ridge_are_exact() -> None:
    report = formula_report()
    profile = report["hardware_profile"]

    assert profile["dense_bf16_tflops"] == 2250.0
    assert profile["hbm_bandwidth_decimal_gb_per_s"] == 8000.0
    assert profile["ridge_flop_per_byte"] == 281.25
    assert profile["verified_against_reviewed_values"] is True


@pytest.mark.parametrize(
    ("m", "expected_flops", "expected_bytes", "expected_intensity"),
    [
        (1, 33_554_432, 33_570_816, 0.9995119570522206),
        (64, 2_147_483_648, 34_603_008, 62.06060606060606),
    ],
)
def test_beta_zero_gemm_counts(
    m: int,
    expected_flops: int,
    expected_bytes: int,
    expected_intensity: float,
) -> None:
    result = gemm_logical_cost(m=m, n=4096, k=4096, beta_zero=True)

    assert result["flops"] == expected_flops
    assert result["logical_requested_bytes"] == expected_bytes
    assert result["logical_byte_breakdown"]["c_read_bytes"] == 0
    assert math.isclose(
        result["logical_arithmetic_intensity_flop_per_byte"],
        expected_intensity,
        rel_tol=1e-15,
    )
    assert result["physical_hbm_counter_bytes"] is None


def test_kv_crossover_and_transfer_bounds() -> None:
    report = formula_report()
    kv = report["kv_cache_model"]
    bounds = report["ideal_hbm_transfer_bounds"]

    assert kv["kv_bytes_per_context_token_per_sequence"] == 147_456
    assert kv["kv_bytes_per_context_position_all_sequences"] == 9_437_184
    assert math.isclose(
        kv["weight_kv_equality_context_tokens"],
        1604.2921278211807,
        rel_tol=1e-15,
    )
    assert kv["example_combined_weight_and_kv_bytes"] == 23_369_224_448
    assert kv["example_combined_weight_and_kv_decimal_gb"] == 23.369224448
    assert bounds["weights_only_ms"] == 1.8925
    assert bounds["example_weight_and_kv_ms"] == 2.921153056


def test_nonzero_beta_adds_a_full_c_read() -> None:
    beta_zero = gemm_logical_cost(m=64, n=4096, k=4096, beta_zero=True)
    beta_nonzero = gemm_logical_cost(m=64, n=4096, k=4096, beta_zero=False)

    assert (
        beta_nonzero["logical_requested_bytes"] - beta_zero["logical_requested_bytes"]
        == 64 * 4096 * 2
    )


def test_weight_rotation_exceeds_target_without_silent_truncation() -> None:
    weight_bytes = 4096 * 4096 * 2
    count = _rotation_count(weight_bytes=weight_bytes, target_rotation_mib=512)

    assert count == 16
    assert count * weight_bytes >= 512 * 1024 * 1024


@pytest.mark.parametrize("value", [0, -1])
def test_transfer_bound_rejects_invalid_byte_counts(value: int) -> None:
    with pytest.raises(ValueError, match="byte_count must be a positive integer"):
        memory_bound_ms(value, 8000.0)


def test_ceiling_search_matrix_is_bounded_and_unique() -> None:
    bandwidth = bandwidth_candidates([1024, 2048, 4096, 8192])
    gemm = gemm_candidates([4096, 8192, 12288, 16384], ["nn", "nt", "tn", "tt"])

    assert len(bandwidth) == 116
    assert len({candidate.candidate_id for candidate in bandwidth}) == len(bandwidth)
    assert sum(candidate.operation == "copy" for candidate in bandwidth) == 44
    assert sum(candidate.operation == "read" for candidate in bandwidth) == 36
    assert sum(candidate.operation == "write" for candidate in bandwidth) == 36
    assert len(gemm) == 16
    assert len({candidate.candidate_id for candidate in gemm}) == len(gemm)


def test_sustained_window_invocations_reach_requested_duration() -> None:
    invocations = confirmation_invocations_per_sample(
        5.0,
        repeats=15,
        minimum_window_gpu_ms=1000.0,
    )

    assert invocations == 14
    assert invocations * 15 * 5.0 >= 1000.0


def test_bootstrap_median_interval_is_deterministic() -> None:
    first = bootstrap_median_ci([10.0, 11.0, 12.0, 13.0, 14.0], seed=7, resamples=1000)
    second = bootstrap_median_ci([10.0, 11.0, 12.0, 13.0, 14.0], seed=7, resamples=1000)

    assert first == second
    assert first["low"] <= 12.0 <= first["high"]


def test_ceiling_csv_parsers_fail_closed() -> None:
    assert parse_positive_csv("1024,2048", name="payload") == [1024, 2048]
    assert parse_layout_csv("nn,NT,tt") == ["nn", "nt", "tt"]
    with pytest.raises(ValueError, match="unique"):
        parse_positive_csv("1024,1024", name="payload")
    with pytest.raises(ValueError, match="unsupported"):
        parse_layout_csv("nn,nx")


def test_lab_tree_hash_covers_native_sources() -> None:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in Path(_PACKAGE_ROOT).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".cpp", ".cu", ".cuh", ".md", ".py", ".yaml"}
    )
    for path in paths:
        digest.update(path.relative_to(_PACKAGE_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    assert (_PACKAGE_ROOT / "ceiling_kernels.cu") in paths
    assert _package_sha256() == digest.hexdigest()


def test_live_source_commit_requires_exact_clean_head(monkeypatch: pytest.MonkeyPatch) -> None:
    head = "a" * 40
    outputs = {
        ("rev-parse", "--verify", "HEAD"): head,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }
    monkeypatch.setattr(
        verification_tool,
        "_git_output",
        lambda *arguments: outputs[tuple(arguments)],
    )

    assert _validate_live_source_commit(head) == head
    with pytest.raises(RuntimeError, match="exactly match"):
        _validate_live_source_commit("b" * 40)

    outputs[("status", "--porcelain=v1", "--untracked-files=all")] = " M code/example.py"
    with pytest.raises(RuntimeError, match="clean Git source tree"):
        _validate_live_source_commit(head)


def test_reports_are_exclusive_and_preserve_existing_bytes(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    report = {"status": "complete"}

    _preflight_output(output)
    _write_report(report, output)
    original = output.read_bytes()

    with pytest.raises(FileExistsError, match="immutable report"):
        _preflight_output(output)
    with pytest.raises(FileExistsError):
        _write_report({"status": "replacement"}, output)
    assert output.read_bytes() == original

    kernel_output = tmp_path / "kernel-report.json"
    kernel_tool._write_report(kernel_output, report)
    kernel_original = kernel_output.read_bytes()
    with pytest.raises(FileExistsError):
        kernel_tool._write_report(kernel_output, {"status": "replacement"})
    assert kernel_output.read_bytes() == kernel_original


def test_formulas_cli_stays_cpu_only_and_refuses_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "formulas.json"

    assert verification_tool.main(["formulas", "--output", str(output)]) == 0
    original = output.read_bytes()
    assert verification_tool.main(["formulas", "--output", str(output)]) == 2

    assert output.read_bytes() == original
    assert '"status": "rejected"' in capsys.readouterr().out


@pytest.mark.parametrize("previous", [False, True])
def test_bf16_reduction_policy_is_frozen_and_restored(previous: bool) -> None:
    matmul = SimpleNamespace(allow_bf16_reduced_precision_reduction=previous)
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)),
    )

    with _bf16_fp32_accumulation_policy(fake_torch) as policy:
        assert policy == {"allow_bf16_reduced_precision_reduction": False}
        assert matmul.allow_bf16_reduced_precision_reduction is False
        _require_bf16_fp32_accumulation_policy(fake_torch)
    assert matmul.allow_bf16_reduced_precision_reduction is previous
    if previous:
        with pytest.raises(RuntimeError, match="must be disabled"):
            _require_bf16_fp32_accumulation_policy(fake_torch)
    else:
        _require_bf16_fp32_accumulation_policy(fake_torch)


def test_publication_status_stays_false_for_declared_profiler_files() -> None:
    status = _publication_status(
        {"ncu": ["capture.ncu-rep"], "nsys": ["capture.nsys-rep"]},
        virtualized=False,
    )

    assert status["publication_eligible"] is False
    assert "operator_review_is_not_recorded_by_this_tool" in status["publication_blockers"]


def test_explicit_blas_preference_is_read_back_and_restored() -> None:
    class Backend(Enum):
        Cublas = 1
        Cublaslt = 2

    current = Backend.Cublas

    def preference(requested=None):
        nonlocal current
        previous = current
        if requested is not None:
            current = Backend.Cublaslt if requested == "cublaslt" else requested
        return previous if requested is not None else current

    torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(preferred_blas_library=preference),
        )
    )

    with _preferred_blas_backend(torch, "cublaslt") as metadata:
        assert current is Backend.Cublaslt
        assert metadata["observed_during_measurement"] == "cublaslt"

    assert current is Backend.Cublas
    assert metadata["restored"] == "cublas"


def test_optimization_candidate_matrices_are_bounded_and_deterministic() -> None:
    launches = copy_launch_candidates()

    assert len(launches) == 9
    assert len({item["candidate_id"] for item in launches}) == len(launches)
    assert alternating_pair_order(0) == ("baseline", "candidate")
    assert alternating_pair_order(1) == ("candidate", "baseline")


def test_exact_cublaslt_signature_selection_fails_closed() -> None:
    algorithms = [
        {"heuristic_index": 0, "signature": "id=66,tile=10"},
        {"heuristic_index": 1, "signature": "id=66,tile=313"},
    ]

    assert select_algorithm_index(algorithms, "id=66,tile=313") == 1
    with pytest.raises(RuntimeError, match="not frozen"):
        select_algorithm_index(algorithms, "UNSET_PENDING_BOUNDED_SCREEN")
    with pytest.raises(RuntimeError, match="found 0"):
        select_algorithm_index(algorithms, "id=66,tile=999")


def test_paired_window_summary_uses_every_window_and_classifies_interval() -> None:
    baseline = [100.0, 101.0, 99.0, 100.5, 100.25]
    candidate = [110.0, 111.1, 108.9, 110.55, 110.275]

    summary = paired_window_summary(baseline, candidate, seed=7)

    assert len(summary["window_candidate_vs_baseline_pct"]) == 5
    assert math.isclose(summary["candidate_vs_baseline_pct"], 10.0, rel_tol=1e-12)
    assert summary["bootstrap_95_ci_pct"]["low"] > 0.0
    assert summary["verdict"] == "improved"


@pytest.mark.parametrize(
    "benchmark_class,tensor_names",
    [
        (BaselineBf16GemmBenchmark, ("a", "b", "output")),
        (BaselineNativeCopyBenchmark, ("source", "output")),
    ],
)
def test_optimization_verification_capture_requires_timed_path(
    benchmark_class: type, tensor_names: tuple[str, ...]
) -> None:
    benchmark = object.__new__(benchmark_class)
    benchmark._benchmark_has_run = False
    for name in tensor_names:
        setattr(benchmark, name, object())

    with pytest.raises(RuntimeError, match=r"benchmark_fn\(\) must run"):
        benchmark.capture_verification_payload()


def test_confirmation_screen_lineage_and_workload_fail_closed() -> None:
    screen_hash = "a" * 64
    source_commit = "b" * 40
    device = {
        "uuid": "GPU-test-b200",
        "name": "NVIDIA B200",
        "compute_capability": [10, 0],
    }
    provenance = {
        "lab_tree_sha256": screen_hash,
        "source_commit": source_commit,
        "device": device,
        "pytorch": "2.14.1+cu130",
        "pytorch_cuda": "13.0",
    }
    screen = {
        "schema_version": 1,
        "kind": "b200_kernel_optimization_bounded_screen",
        "status": "screen_complete",
        "provenance": provenance,
        "gemm": {
            "shape": {"m": GEMM_SIZE, "n": GEMM_SIZE, "k": GEMM_SIZE},
            "layout": "row_major_nn",
            "dtype": "bf16",
            "compute_type": "fp32_accumulation",
            "beta_zero": True,
            "selected_signature": SELECTED_GEMM_ALGORITHM_SIGNATURE,
            "candidates": [
                {
                    "signature": SELECTED_GEMM_ALGORITHM_SIGNATURE,
                    "ranking_eligible": True,
                    "correctness": {"passed": True},
                }
            ],
        },
        "copy": {
            "payload_bytes": COPY_PAYLOAD_BYTES,
            "selected_candidate_id": "selected-copy",
            "selected_threads": SELECTED_COPY_THREADS,
            "selected_blocks_per_sm": SELECTED_COPY_BLOCKS_PER_SM,
            "candidates": [
                {
                    "candidate_id": "selected-copy",
                    "threads": SELECTED_COPY_THREADS,
                    "blocks_per_sm": SELECTED_COPY_BLOCKS_PER_SM,
                    "ranking_eligible": True,
                    "correctness": {"passed": True},
                }
            ],
            "byte_accounting": {
                "scope": "requested source read plus requested destination write",
                "is_raw_physical_hbm_bandwidth": False,
            },
        },
    }
    current = copy.deepcopy(provenance)
    current["lab_tree_sha256"] = "c" * 64

    lineage = _validate_screen_for_confirmation(
        screen,
        expected_screen_lab_tree_sha256=screen_hash,
        source_commit=source_commit,
        current_provenance=current,
    )
    assert lineage["screen_lab_tree_sha256"] == screen_hash
    assert lineage["confirmation_lab_tree_sha256"] == "c" * 64

    bad_payload = copy.deepcopy(screen)
    bad_payload["copy"]["payload_bytes"] -= 1
    with pytest.raises(ValueError, match="8 GiB"):
        _validate_screen_for_confirmation(
            bad_payload,
            expected_screen_lab_tree_sha256=screen_hash,
            source_commit=source_commit,
            current_provenance=current,
        )

    with pytest.raises(ValueError, match="frozen digest"):
        _validate_screen_for_confirmation(
            screen,
            expected_screen_lab_tree_sha256="d" * 64,
            source_commit=source_commit,
            current_provenance=current,
        )


def _nominal_gap_telemetry_sample(sequence: int, start_s: float) -> dict[str, object]:
    return {
        "sequence": sequence,
        "query_start_monotonic_s": start_s,
        "query_end_monotonic_s": start_s + 0.01,
        "query_start_timestamp_utc": f"2026-10-14T16:00:{sequence:02d}+00:00",
        "query_end_timestamp_utc": f"2026-10-14T16:00:{sequence:02d}.01+00:00",
        "telemetry_timestamp_utc": f"2026-10-14T16:00:{sequence:02d}+00:00",
        "gpu_index": 0.0,
        "graphics_clock_mhz": 1500.0 + sequence,
        "memory_clock_mhz": 4000.0,
        "applications_clock_sm_mhz": 1965.0,
        "applications_clock_memory_mhz": 4000.0,
        "power_draw_w": 980.0 + sequence,
        "temperature_gpu_c": 55.0,
        "utilization_gpu_pct": 100.0,
        "performance_state": 0.0,
        "throttle_reasons": 4.0,
    }


def test_nominal_gap_telemetry_requires_complete_in_window_queries() -> None:
    samples = [_nominal_gap_telemetry_sample(index, 0.10 + index * 0.12) for index in range(6)]

    summary = summarize_nominal_gap_telemetry(
        samples,
        window_start_monotonic_s=0.0,
        window_end_monotonic_s=0.85,
        minimum_samples=5,
        expected_gpu_index=0,
        requested_interval_s=0.1,
        maximum_gap_s=0.2,
    )

    assert summary["sample_count"] == 6
    assert summary["cadence"]["estimated_missed_samples"] == 0
    assert math.isclose(
        summary["cadence"]["boundary_gap_s"]["last_query_end_to_window_end"],
        0.14,
    )
    assert summary["metrics"]["graphics_clock_mhz"]["p50"] == 1502.5
    assert summary["throttle_reasons"]["software_power_cap_active_fraction"] == 1.0
    assert summary["steady_state_diagnostics"]["heuristic_is_a_measurement_acceptance_gate"] is True

    delayed_samples = [
        _nominal_gap_telemetry_sample(index, start)
        for index, start in enumerate((0.10, 0.20, 0.39, 0.49, 0.59))
    ]
    delayed_summary = summarize_nominal_gap_telemetry(
        delayed_samples,
        window_start_monotonic_s=0.0,
        window_end_monotonic_s=0.75,
        minimum_samples=5,
        expected_gpu_index=0,
        requested_interval_s=0.1,
        maximum_gap_s=0.2,
    )
    assert delayed_summary["cadence"]["estimated_missed_samples"] == 1

    samples[2]["applications_clock_sm_mhz"] = None
    with pytest.raises(RuntimeError, match="applications_clock_sm_mhz"):
        summarize_nominal_gap_telemetry(
            samples,
            window_start_monotonic_s=0.0,
            window_end_monotonic_s=0.85,
            minimum_samples=5,
            expected_gpu_index=0,
            requested_interval_s=0.1,
            maximum_gap_s=0.2,
        )


def test_nominal_gap_telemetry_rejects_insufficient_samples_and_cadence_gap() -> None:
    too_few = [_nominal_gap_telemetry_sample(index, 0.1 + index * 0.1) for index in range(4)]
    with pytest.raises(RuntimeError, match="insufficient in-window"):
        summarize_nominal_gap_telemetry(
            too_few,
            window_start_monotonic_s=0.0,
            window_end_monotonic_s=1.0,
            minimum_samples=5,
            expected_gpu_index=0,
            requested_interval_s=0.1,
            maximum_gap_s=0.2,
        )

    trailing_stall = [_nominal_gap_telemetry_sample(index, 0.1 + index * 0.1) for index in range(8)]
    with pytest.raises(RuntimeError, match="maximum boundary gap"):
        summarize_nominal_gap_telemetry(
            trailing_stall,
            window_start_monotonic_s=0.0,
            window_end_monotonic_s=1.2,
            minimum_samples=8,
            expected_gpu_index=0,
            requested_interval_s=0.1,
            maximum_gap_s=0.25,
        )

    gap_samples = [
        _nominal_gap_telemetry_sample(index, start)
        for index, start in enumerate((0.1, 0.2, 0.3, 0.7, 0.8))
    ]
    with pytest.raises(RuntimeError, match="maximum gap"):
        summarize_nominal_gap_telemetry(
            gap_samples,
            window_start_monotonic_s=0.0,
            window_end_monotonic_s=1.0,
            minimum_samples=5,
            expected_gpu_index=0,
            requested_interval_s=0.1,
            maximum_gap_s=0.2,
        )


def test_nominal_gap_power_integration_uses_only_the_host_phase() -> None:
    summary = integrate_power_samples_over_window(
        [
            {"timestamp_epoch_s": 0.0, "total_power_w": 100.0},
            {"timestamp_epoch_s": 1.0, "total_power_w": 200.0},
            {"timestamp_epoch_s": 2.0, "total_power_w": 100.0},
        ],
        window_start_epoch_s=0.5,
        window_end_epoch_s=1.5,
    )

    assert summary["energy_joules"] == 175.0
    assert summary["time_weighted_power_w"] == 175.0
    assert summary["phase_duration_s"] == 1.0


def test_nominal_gap_observer_overhead_requires_every_observed_pair_within_limit() -> None:
    summary = observer_overhead_summary(
        [10.0] * 8,
        [10.01] * 8,
        limit_pct=0.5,
        seed=7,
    )

    assert math.isclose(summary["median_sampler_overhead_pct"], 0.1, rel_tol=1e-12)
    assert summary["every_observed_pair_within_limit"] is True
    assert summary["within_observer_overhead_limit"] is True
    assert summary["bootstrap_interval_role"] == "descriptive_only_not_an_equivalence_proof"

    outlier = observer_overhead_summary(
        [10.0] * 8,
        [10.1, *([10.01] * 7)],
        limit_pct=0.5,
        seed=7,
    )
    assert math.isclose(outlier["median_sampler_overhead_pct"], 0.1, rel_tol=1e-12)
    assert outlier["every_observed_pair_within_limit"] is False
    assert outlier["within_observer_overhead_limit"] is False


def test_nominal_gap_sustained_kpi_requires_every_window_to_be_settled() -> None:
    def window(index: int, passed: bool) -> dict[str, object]:
        return {
            "window": index,
            "measurement": {
                "observation": {
                    "in_window_telemetry": {
                        "steady_state_diagnostics": {
                            "passes_declared_steady_state_heuristic": passed
                        }
                    }
                }
            },
        }

    qualified = _nominal_gap_steady_state_qualification(
        [window(0, True), window(1, True), window(2, True)],
        expected_window_count=3,
    )
    assert qualified["sustained_kpi_qualified"] is True

    unsettled = _nominal_gap_steady_state_qualification(
        [window(0, True), window(1, False), window(2, True)],
        expected_window_count=3,
    )
    assert unsettled["sustained_kpi_qualified"] is False
    assert unsettled["unsettled_windows_retained_for_diagnosis"] is True

    with pytest.raises(RuntimeError, match="lacks the required steady-state gate"):
        _nominal_gap_steady_state_qualification(
            [{"window": 0, "measurement": {}}],
            expected_window_count=1,
        )


def test_nominal_gap_cli_defaults_are_bounded_and_fill_modes_are_explicit(
    tmp_path: Path,
) -> None:
    parser = kernel_optimization_parser()
    args = parser.parse_args(
        [
            "diagnose-nominal-gap",
            "--run-id",
            "local-control-plane-test",
            "--source-commit",
            "a" * 40,
            "--output",
            str(tmp_path / "nominal-gap.json"),
        ]
    )

    _validate_nominal_gap_args(args)
    assert args.fill_mode == "scaled_normal"
    assert args.settling_minimum_gpu_ms == 3000.0
    assert args.calibration_rounds == 8
    assert args.sustained_windows == 3
    assert args.telemetry_interval_ms == 100.0
    assert args.maximum_telemetry_gap_ms == 250.0
    assert args.minimum_in_window_telemetry_samples == 8

    args.fill_mode = "zero_power_switching_calibration"
    _validate_nominal_gap_args(args)
    args.fill_mode = "uniform_rms_matched_to_scaled_normal"
    _validate_nominal_gap_args(args)
    args.telemetry_interval_ms = 50.0
    args.maximum_telemetry_gap_ms = 250.0
    with pytest.raises(ValueError, match="at most 2.5 times"):
        _validate_nominal_gap_args(args)
    args.telemetry_interval_ms = 125.0
    with pytest.raises(ValueError, match="cannot conservatively fit"):
        _validate_nominal_gap_args(args)
    args.telemetry_interval_ms = 100.0
    args.calibration_rounds = 4
    with pytest.raises(ValueError, match="calibration_rounds"):
        _validate_nominal_gap_args(args)
    args.calibration_rounds = 8
    args.settling_minimum_gpu_ms = 2999.0
    with pytest.raises(ValueError, match="settling_minimum_gpu_ms"):
        _validate_nominal_gap_args(args)


def test_sustained_search_cluster_inventory_decodes_one_and_two_cta_shapes() -> None:
    one_cta = _cluster_shape_metadata(2)
    two_cta_x = _cluster_shape_metadata(3)
    two_cta_y = _cluster_shape_metadata(5)
    unknown = _cluster_shape_metadata(99)

    assert one_cta == {
        "cluster_shape": "1x1x1",
        "cluster_cta_count": 1,
        "cluster_shape_decode_status": "decoded_cuda13_cublaslt_enum",
    }
    assert two_cta_x["cluster_shape"] == "2x1x1"
    assert two_cta_x["cluster_cta_count"] == 2
    assert two_cta_y["cluster_shape"] == "1x2x1"
    assert two_cta_y["cluster_cta_count"] == 2
    assert unknown["cluster_shape"] is None
    assert unknown["cluster_cta_count"] is None
    assert unknown["cluster_shape_decode_status"] == "unknown_cuda13_enum_value"


def test_sustained_search_cap_retains_prior_signatures_and_prefers_one_cta() -> None:
    inventory = []
    for index, signature in enumerate(PRIOR_FULL_OUTPUT_PASSING_SIGNATURES):
        inventory.append(
            {
                "signature": signature,
                "cluster_cta_count": 2,
                "selected_workspace_ceiling_bytes": 0,
                "workspace_bytes": 0,
                "stages_id": 35,
                "tile_id": 500 + index,
            }
        )
    inventory.extend(
        [
            {
                "signature": "extra_four_cta",
                "cluster_cta_count": 4,
                "selected_workspace_ceiling_bytes": 0,
                "workspace_bytes": 0,
                "stages_id": 35,
                "tile_id": 1,
            },
            {
                "signature": "extra_one_cta",
                "cluster_cta_count": 1,
                "selected_workspace_ceiling_bytes": 64 << 20,
                "workspace_bytes": 0,
                "stages_id": 35,
                "tile_id": 2,
            },
        ]
    )

    selected = _bounded_candidate_signatures(
        inventory,
        required_signatures=PRIOR_FULL_OUTPUT_PASSING_SIGNATURES,
        maximum_candidates=9,
    )

    assert selected[:8] == list(PRIOR_FULL_OUTPUT_PASSING_SIGNATURES)
    assert selected[8] == "extra_one_cta"
    with pytest.raises(RuntimeError, match="Required prior"):
        _bounded_candidate_signatures(
            inventory[:-3],
            required_signatures=PRIOR_FULL_OUTPUT_PASSING_SIGNATURES,
            maximum_candidates=8,
        )


def test_sustained_search_and_peak_cli_defaults_are_finite(tmp_path: Path) -> None:
    parser = sustained_optimization_parser()
    search = parser.parse_args(
        [
            "search",
            "--run-id",
            "local-control-plane-test",
            "--source-commit",
            "a" * 40,
            "--output",
            str(tmp_path / "search.json"),
        ]
    )
    assert search.workspace_bytes == WORKSPACE_CEILINGS_BYTES
    assert search.maximum_candidates == 24
    assert search.max_heuristics == 128
    assert search.pair_rounds == 2
    assert search.settling_minimum_gpu_ms == 3000.0
    assert search.minimum_window_gpu_ms == 1000.0

    peak = parser.parse_args(
        [
            "calibrate-peak",
            "--run-id",
            "local-control-plane-test",
            "--source-commit",
            "a" * 40,
            "--output",
            str(tmp_path / "peak.json"),
            "--confirmation",
            str(tmp_path / "confirmation.json"),
            "--expected-confirmation-lab-tree-sha256",
            "b" * 64,
        ]
    )
    assert peak.windows == 5
    assert peak.buffer_sets == 2
    assert peak.calibration_rounds == 8
    assert peak.fill_modes == (
        "scaled_normal",
        "uniform_rms_matched_to_scaled_normal",
        "uniform_minus1_plus1",
        "k_order_random_balanced",
        "k_order_positive_then_negative",
        "zero_power_switching_calibration",
    )

    copy_args = parser.parse_args(
        [
            "calibrate-copy",
            "--run-id",
            "local-control-plane-test",
            "--source-commit",
            "a" * 40,
            "--output",
            str(tmp_path / "copy.json"),
        ]
    )
    assert copy_args.payload_bytes == 8 << 30
    assert copy_args.windows == 5
    assert copy_args.calibration_rounds == 8
    assert copy_args.calibration_repeats == 5
    assert copy_args.minimum_telemetry_samples == 8


def test_peak_fill_parser_and_balanced_orders_fail_closed() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="fill modes and order are frozen"):
        _parse_fill_modes(
            "scaled_normal,uniform_minus1_plus1,k_order_random_balanced,"
            "k_order_positive_then_negative,zero_power_switching_calibration"
        )
    with pytest.raises(argparse.ArgumentTypeError, match="fill modes and order are frozen"):
        _parse_fill_modes(
            "scaled_normal,uniform_minus1_plus1,k_order_random_balanced,"
            "k_order_positive_then_negative,zero_power_switching_calibration,"
            "uniform_rms_matched_to_scaled_normal"
        )

    fills = ("S", "R", "U", "A", "P", "Z")
    orders = [_balanced_fill_order(fills, index) for index in range(5)]
    assert all(set(order) == set(fills) for order in orders)
    assert len({tuple(order) for order in orders}) == 5
    for fill in fills:
        positions = [order.index(fill) for order in orders]
        assert len(set(positions)) == 5
    assert sum(order.index("A") < order.index("P") for order in orders) == 3
    assert sum(order.index("P") < order.index("A") for order in orders) == 2
    with pytest.raises(ValueError, match="exactly six"):
        _balanced_fill_order(fills[:-1], 0)
    with pytest.raises(ValueError, match="exactly five"):
        _balanced_fill_order(fills, 5)


def test_observer_arguments_require_eight_pairs_and_dense_window_coverage() -> None:
    valid = SimpleNamespace(
        calibration_rounds=8,
        calibration_repeats=5,
        minimum_telemetry_samples=8,
        telemetry_interval_ms=100.0,
        maximum_telemetry_gap_ms=250.0,
        minimum_window_gpu_ms=1000.0,
        observer_overhead_limit_pct=0.5,
    )
    _validate_observer_args(valid)

    for field, value, message in (
        ("calibration_rounds", 4, "exactly eight"),
        ("minimum_telemetry_samples", 7, "minimum_telemetry_samples"),
        ("maximum_telemetry_gap_ms", 300.0, "maximum_telemetry_gap_ms"),
    ):
        invalid = copy.copy(valid)
        setattr(invalid, field, value)
        with pytest.raises(ValueError, match=message):
            _validate_observer_args(invalid)


def test_k_order_control_preserves_histograms_and_exact_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sustained_tool, "GEMM_SIZE", 64)
    original_a, original_b, original_contract = _k_order_control_operands(
        torch,
        device=torch.device("cpu"),
        seed=314159,
        permuted=False,
    )
    clustered_a, clustered_b, clustered_contract = _k_order_control_operands(
        torch,
        device=torch.device("cpu"),
        seed=314159,
        permuted=True,
    )

    assert original_a.is_contiguous() and original_b.is_contiguous()
    assert clustered_a.is_contiguous() and clustered_b.is_contiguous()
    assert torch.equal(original_a[0], original_a[-1])
    assert torch.equal(original_a[:, 0::2], -original_a[:, 1::2])
    assert bool(torch.all(clustered_a[:, :32] > 0))
    assert bool(torch.all(clustered_a[:, 32:] < 0))
    assert torch.equal(
        torch.sort(original_a.flatten()).values, torch.sort(clustered_a.flatten()).values
    )
    assert torch.equal(
        torch.sort(original_b.flatten()).values, torch.sort(clustered_b.flatten()).values
    )
    assert torch.equal(
        original_a.float() @ original_b.float(), clustered_a.float() @ clustered_b.float()
    )
    assert "randomly oriented" in original_contract["arrangement"]
    assert "positive A" in clustered_contract["arrangement"]


def test_sustained_invocation_sizing_has_three_second_and_one_second_margins() -> None:
    settling, timed = _invocation_counts(
        6.0,
        repeats=5,
        settling_minimum_gpu_ms=3000.0,
        minimum_window_gpu_ms=1000.0,
    )

    assert settling == 750
    assert timed == 40
    assert settling * 6.0 >= 4500.0
    assert timed * 5 * 6.0 >= 1200.0


def test_observer_and_primary_peak_plan_use_separate_counts_and_fit_budget() -> None:
    records = []
    for fill_index in range(6):
        short_p50_ms = 5.5 + 0.1 * fill_index
        settling, primary = _invocation_counts(
            short_p50_ms,
            repeats=15,
            settling_minimum_gpu_ms=3000.0,
            minimum_window_gpu_ms=1000.0,
        )
        observer_settling, observer = _invocation_counts(
            short_p50_ms,
            repeats=5,
            settling_minimum_gpu_ms=3000.0,
            minimum_window_gpu_ms=1000.0,
        )
        assert observer * 5 * short_p50_ms >= 1200.0
        assert primary * 15 * short_p50_ms >= 1200.0
        assert observer > primary
        assert observer_settling == settling
        records.append(
            {
                "short_p50_ms": short_p50_ms,
                "settling_invocations": settling,
                "observer_invocations_per_sample": observer,
                "primary_invocations_per_sample": primary,
            }
        )

    planned_ms = _planned_peak_gpu_active_ms(
        records,
        observer_pairs=8,
        observer_repeats=5,
        primary_windows=5,
        primary_repeats=15,
    )
    assert planned_ms > 6 * (16 + 5) * (3000.0 + 1000.0)
    assert planned_ms < PEAK_PHASE_GPU_ACTIVE_BUDGET_MS


def test_copy_window_qualification_gates_memory_power_cadence_and_runtime_state() -> None:
    samples = [_nominal_gap_telemetry_sample(index, 0.05 + index * 0.12) for index in range(8)]
    timeline = summarize_nominal_gap_telemetry(
        samples,
        window_start_monotonic_s=0.0,
        window_end_monotonic_s=1.0,
        minimum_samples=8,
        expected_gpu_index=0,
        requested_interval_s=0.1,
        maximum_gap_s=0.25,
    )
    measurement = {"observation": {"in_window_telemetry": timeline}}

    qualified = _copy_window_steady_state_qualification(
        measurement,
        expected_gpu_index=0,
    )
    assert qualified["passed"] is True
    assert qualified["memory_clock"]["stable"] is True
    assert qualified["power"]["stable"] is True
    assert qualified["runtime_state"]["p0_passed"] is True
    assert qualified["graphics_clock_role"] == "diagnostic_only_not_a_copy_acceptance_gate"

    drifting_samples = copy.deepcopy(samples)
    for sample in drifting_samples[4:]:
        sample["memory_clock_mhz"] = 3000.0
    drifting_timeline = summarize_nominal_gap_telemetry(
        drifting_samples,
        window_start_monotonic_s=0.0,
        window_end_monotonic_s=1.0,
        minimum_samples=8,
        expected_gpu_index=0,
        requested_interval_s=0.1,
        maximum_gap_s=0.25,
    )
    rejected = _copy_window_steady_state_qualification(
        {"observation": {"in_window_telemetry": drifting_timeline}},
        expected_gpu_index=0,
    )
    assert rejected["passed"] is False
    assert "memory_clock_stability" in rejected["failed_gates"]


def test_native_extension_exports_exact_blas_runtime_identity() -> None:
    source = (
        Path(__file__).parents[1]
        / "labs"
        / "arithmetic_intensity_b200"
        / "optimization_bindings.cpp"
    ).read_text()

    assert "cublasCreate(&cublas_handle)" in source
    assert "cublasGetVersion(cublas_handle, &cublas_version)" in source
    assert "cublasDestroy(cublas_handle)" in source
    assert "cublasGetVersion(nullptr" not in source
    assert "cublasLtGetVersion()" in source
    assert 'module.def(\n      "blas_runtime_versions"' in source


@pytest.mark.parametrize(
    "source_name",
    ["ceiling_kernels.cu", "optimization_kernels.cu"],
)
def test_native_uint4_copy_sources_enforce_alignment_and_nonoverlap(
    source_name: str,
) -> None:
    source = (
        Path(__file__).parents[1]
        / "labs"
        / "arithmetic_intensity_b200"
        / source_name
    ).read_text()

    assert source.count("% alignof(uint4) == 0") >= 2
    assert "source storage offset must preserve 16-byte uint4 alignment" in source
    assert "destination storage offset must preserve 16-byte uint4 alignment" in source
    assert "source_address + byte_count <= destination_address" in source
    assert "destination_address + byte_count <= source_address" in source
    assert "source and destination must not overlap" in source
