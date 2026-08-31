"""Negative controls for artifact-owned hardware identity metadata."""

from __future__ import annotations

import json
import math

import pytest

from core.analysis import deep_profiling_report as report


@pytest.mark.parametrize("raw", ["1e999", "-1e999", "nan", "N/A"])
def test_parse_float_never_returns_nonfinite_values(raw) -> None:
    assert report.parse_float(raw) is None


@pytest.mark.parametrize("raw", ["1,0", "12,34", "1(foo)0", "1 (unit) 0"])
def test_parse_float_rejects_malformed_numeric_evidence(raw) -> None:
    assert report.parse_float(raw) is None


def test_parse_float_accepts_only_well_formed_nsight_grouping_and_unit_suffix() -> None:
    assert report.parse_float("1,234.5") == pytest.approx(1234.5)
    assert report.parse_float("123.4 (bytes)") == pytest.approx(123.4)


@pytest.mark.parametrize(
    ("major", "minor"),
    [
        (10.9, 3.0),
        (10.0, 3.8),
        (math.inf, 3.0),
        (10.0, math.inf),
        (-1.0, 0.0),
    ],
)
def test_fractional_or_nonfinite_capability_metadata_is_invalid(major, minor) -> None:
    metrics = report.KernelMetrics("fixture")
    metrics.metrics[report.COMPUTE_CAPABILITY_MAJOR_KEYS[0]] = report.RawMetric(
        report.COMPUTE_CAPABILITY_MAJOR_KEYS[0], major
    )
    metrics.metrics[report.COMPUTE_CAPABILITY_MINOR_KEYS[0]] = report.RawMetric(
        report.COMPUTE_CAPABILITY_MINOR_KEYS[0], minor
    )

    selection = report.resolve_hardware_selection([metrics], None)

    assert selection.provenance == "source_ncu_compute_capability_invalid"
    assert selection.profile == "unknown"
    assert selection.specs is None
    assert selection.compute_capability == "invalid"


def test_nonfinite_capability_csv_is_preserved_as_invalid_identity(tmp_path) -> None:
    csv_path = tmp_path / "invalid-capability.csv"
    csv_path.write_text(
        '"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        '"0","kernel","device__attribute_compute_capability_major","1e999",""\n'
        '"1","kernel","device__attribute_compute_capability_minor","3",""\n',
        encoding="utf-8",
    )

    parsed = report.parse_ncu_csv(csv_path)
    selection = report.resolve_hardware_selection(parsed.values(), None)

    assert selection.provenance == "source_ncu_compute_capability_invalid"
    assert selection.specs is None


@pytest.mark.parametrize(
    ("header", "row", "message"),
    [
        (
            '"ID","Metric Name","Metric Value","Metric Unit"',
            '"1","gpu__time_duration.sum","1","ms"',
            "kernel name",
        ),
        (
            '"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"',
            '\"\",\"kernel\",\"gpu__time_duration.sum\",\"1\",\"ms\"',
            "launch ID",
        ),
    ],
)
def test_metric_rows_require_kernel_name_and_launch_id(
    tmp_path, header, row, message
) -> None:
    csv_path = tmp_path / "missing-launch-identity.csv"
    csv_path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        report.parse_ncu_csv(csv_path)


def test_missing_launch_identity_is_a_structured_cli_failure(tmp_path) -> None:
    csv_path = tmp_path / "missing-kernel-name.csv"
    output_path = tmp_path / "report.json"
    csv_path.write_text(
        '"ID","Metric Name","Metric Value","Metric Unit"\n'
        '"1","gpu__time_duration.sum","1","ms"\n',
        encoding="utf-8",
    )

    result = report.main(
        ["--ncu-csv", str(csv_path), "--output-json", str(output_path)]
    )
    payload = json.loads(output_path.read_text())

    assert result == 2
    assert payload["success"] is False
    assert payload["inputs"][0]["status"] == "error"
    assert "kernel name" in payload["inputs"][0]["error"]


def test_repeated_kernel_launches_keep_separate_counter_sets(tmp_path) -> None:
    csv_path = tmp_path / "repeated.csv"
    rows = ['"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"']
    for launch_id, duration, flops, dram_bytes in (
        ("1", "1", "100", "1000"),
        ("2", "2", "900", "3000"),
    ):
        rows.extend(
            [
                f'"{launch_id}","same_kernel","gpu__time_duration.sum","{duration}","ms"',
                f'"{launch_id}","same_kernel","flop_count_sp","{flops}","flop"',
                f'"{launch_id}","same_kernel","dram__bytes.sum","{dram_bytes}","byte"',
            ]
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    parsed = report.parse_ncu_csv(csv_path, capture_id="fixture-capture")

    assert set(parsed) == {
        "same_kernel [capture=fixture-capture,launch_id=1]",
        "same_kernel [capture=fixture-capture,launch_id=2]",
    }
    first = parsed["same_kernel [capture=fixture-capture,launch_id=1]"]
    second = parsed["same_kernel [capture=fixture-capture,launch_id=2]"]
    assert report.metric_in_unit(
        first.get("gpu__time_duration.sum")
    ) == pytest.approx(1.0)
    assert report.compute_flops(first) == 100.0
    assert report.compute_hbm_bytes(first) == 1000.0
    assert report.metric_in_unit(
        second.get("gpu__time_duration.sum")
    ) == pytest.approx(2.0)
    assert report.compute_flops(second) == 900.0
    assert report.compute_hbm_bytes(second) == 3000.0


def test_conflicting_duplicate_metric_across_inputs_is_rejected() -> None:
    left = report.KernelMetrics("kernel [capture=same,launch_id=1]")
    right = report.KernelMetrics("kernel [capture=same,launch_id=1]")
    left.metrics["dram__bytes.sum"] = report.RawMetric(
        "dram__bytes.sum", 100.0, "byte"
    )
    right.metrics["dram__bytes.sum"] = report.RawMetric(
        "dram__bytes.sum", 200.0, "byte"
    )

    with pytest.raises(report.ConflictingMetricError, match="Conflicting duplicate"):
        report.merge_kernel_metrics(
            [{left.name: left}, {right.name: right}]
        )


def test_complementary_metrics_for_same_launch_can_merge() -> None:
    left = report.KernelMetrics("kernel [capture=same,launch_id=1]")
    right = report.KernelMetrics("kernel [capture=same,launch_id=1]")
    left.metrics["flop_count_sp"] = report.RawMetric(
        "flop_count_sp", 100.0, "flop"
    )
    right.metrics["dram__bytes.sum"] = report.RawMetric(
        "dram__bytes.sum", 200.0, "byte"
    )

    merged = report.merge_kernel_metrics(
        [{left.name: left}, {right.name: right}]
    )

    assert set(merged[left.name].metrics) == {"flop_count_sp", "dram__bytes.sum"}


def test_content_capture_identity_is_path_independent_and_changes_with_bytes(
    tmp_path,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"same artifact bytes")
    second.write_bytes(b"same artifact bytes")

    initial = report._capture_namespace(first)

    assert initial == report._capture_namespace(second)
    assert initial.startswith("sha256:")
    assert str(tmp_path) not in initial

    first.write_bytes(b"replacement artifact bytes")
    assert report._capture_namespace(first) != initial


def test_unrelated_csv_files_cannot_complete_each_others_roofline(tmp_path) -> None:
    compute_csv = tmp_path / "compute.csv"
    traffic_csv = tmp_path / "traffic.csv"
    header = '"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
    compute_csv.write_text(
        header
        + '"1","same_kernel","gpu__time_duration.sum","1","ms"\n'
        + '"1","same_kernel","flop_count_sp","100","flop"\n'
    )
    traffic_csv.write_text(
        header + '"1","same_kernel","dram__bytes.sum","1000","byte"\n'
    )

    compute = report.parse_ncu_csv(compute_csv)
    traffic = report.parse_ncu_csv(traffic_csv)
    merged = report.merge_kernel_metrics([compute, traffic])

    assert len(merged) == 2
    assert {metrics.capture_id for metrics in merged.values()} == {
        report._capture_namespace(compute_csv),
        report._capture_namespace(traffic_csv),
    }
    assert all(
        report.derive_roofline(
            metrics, report.get_architecture_specs_for_profile("b200")
        )[0]
        is None
        for metrics in merged.values()
    )


def test_standard_device_and_cc_columns_validate_exact_artifact_profile(tmp_path) -> None:
    csv_path = tmp_path / "b200.csv"
    csv_path.write_text(
        '"ID","Process ID","Device ID","Device","CC","Kernel Name",'
        '"Metric Name","Metric Value","Metric Unit"\n'
        '"1","42","0","NVIDIA B200","10.0","kernel",'
        '"gpu__time_duration.sum","1","ms"\n'
        '"1","42","0","NVIDIA B200","10.0","kernel",'
        '"flop_count_sp","100","flop"\n'
        '"1","42","0","NVIDIA B200","10.0","kernel",'
        '"dram__bytes.sum","1000","byte"\n'
    )

    parsed = report.parse_ncu_csv(csv_path)
    inferred = report.resolve_hardware_selection(parsed.values(), None)
    compatible = report.resolve_hardware_selection(parsed.values(), "b200")
    mismatched = report.resolve_hardware_selection(parsed.values(), "h100-sxm")

    assert inferred.profile == "b200"
    assert inferred.provenance == "source_ncu_exact_device_identity"
    assert inferred.specs is not None
    assert compatible.specs is not None
    assert compatible.provenance == "explicit_cli_artifact_compute_capability_compatible"
    assert mismatched.specs is None
    assert mismatched.provenance == "explicit_cli_artifact_compute_capability_mismatch"


def test_process_and_device_fields_prevent_same_launch_id_collision(tmp_path) -> None:
    csv_path = tmp_path / "multi-device.csv"
    csv_path.write_text(
        '"ID","Host Name","Process ID","Device ID","Device","CC",'
        '"Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        '"1","node","10","0","NVIDIA B200","10.0","kernel",'
        '"flop_count_sp","100","flop"\n'
        '"1","node","11","1","NVIDIA H100 SXM","9.0","kernel",'
        '"flop_count_sp","200","flop"\n'
    )

    parsed = report.parse_ncu_csv(csv_path)
    selection = report.resolve_hardware_selection(parsed.values(), "b200")

    assert len(parsed) == 2
    assert any("process_id=10" in key and "device_id=0" in key for key in parsed)
    assert any("process_id=11" in key and "device_id=1" in key for key in parsed)
    assert selection.specs is None
    assert selection.provenance == "explicit_cli_artifact_compute_capability_ambiguous"


def test_device_name_and_capability_from_different_captures_do_not_form_identity() -> None:
    name_only = report.KernelMetrics(
        "kernel [capture=name]",
        capture_id="name",
        device_name="NVIDIA B200",
    )
    capability_only = report.KernelMetrics(
        "kernel [capture=capability]",
        capture_id="capability",
        compute_capability=(10, 0),
    )

    selection = report.resolve_hardware_selection(
        [name_only, capability_only], declared_profile=None
    )

    assert selection.profile == "unknown"
    assert selection.specs is None
    assert selection.provenance == "source_ncu_device_identity_unvalidated"


def test_exact_identity_cannot_supply_ceilings_to_identity_free_capture() -> None:
    identified = report.KernelMetrics(
        "kernel [capture=identified]",
        capture_id="identified",
        device_name="NVIDIA B200",
        compute_capability=(10, 0),
    )
    unidentified = report.KernelMetrics(
        "kernel [capture=unidentified]",
        capture_id="unidentified",
    )

    selection = report.resolve_hardware_selection([identified, unidentified], None)

    assert selection.profile == "unknown"
    assert selection.specs is None
    assert selection.provenance == "source_ncu_device_identity_unvalidated"


def test_malformed_metric_identity_cannot_authorize_device_profile(tmp_path) -> None:
    csv_path = tmp_path / "malformed-identity.csv"
    csv_path.write_text(
        '"ID","Device","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        '"1","NVIDIA B200","kernel",'
        '"device__attribute_compute_capability_major","10oops",""\n'
        '"1","NVIDIA B200","kernel",'
        '"device__attribute_compute_capability_minor","0",""\n'
    )

    parsed = report.parse_ncu_csv(csv_path)
    selection = report.resolve_hardware_selection(parsed.values(), None)

    assert report.parse_float("10oops") is None
    assert selection.profile == "unknown"
    assert selection.specs is None
    assert selection.provenance == "source_ncu_compute_capability_invalid"


def test_no_utilization_metrics_do_not_default_binding_to_compute() -> None:
    metrics = report.KernelMetrics("low_ai")
    for name, value, unit in (
        ("gpu__time_duration.sum", 1.0, "ms"),
        ("flop_count_sp", 1_000_000_000.0, "flop"),
        ("dram__bytes.sum", 1_000_000_000.0, "byte"),
    ):
        metrics.metrics[name] = report.RawMetric(name, value, unit)

    roofline, *_ = report.derive_roofline(
        metrics,
        report.get_architecture_specs_for_profile("b200"),
    )

    assert roofline is not None
    assert roofline.binding == "unknown"
    assert roofline.is_memory_bound is True
    assert roofline.is_compute_bound is False
    assert roofline.is_tmem_bound is False
