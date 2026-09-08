from __future__ import annotations

import json

import pytest

from labs.ozaki_scheme.lab_utils import (
    format_result_row,
    parse_float_csv,
    parse_int_csv,
    parse_metrics,
    summarize_reproducibility,
)


def test_ozaki_reference_exposes_declared_secondary_pair_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from labs.ozaki_scheme.accuracy_policy import (
        DEFAULT_POLICY_PATH,
        configured_accuracy,
    )
    from labs.ozaki_scheme.baseline_ozaki_scheme import BaselineOzakiSchemeBenchmark
    from labs.ozaki_scheme.optimized_ozaki_scheme_dynamic import (
        OptimizedOzakiSchemeDynamicBenchmark,
    )
    from labs.ozaki_scheme.optimized_ozaki_scheme_fixed import (
        OptimizedOzakiSchemeFixedBenchmark,
    )

    monkeypatch.setenv("AISP_OZAKI_ACCURACY_POLICY", str(DEFAULT_POLICY_PATH))
    policy = json.loads(DEFAULT_POLICY_PATH.read_text())
    reference = BaselineOzakiSchemeBenchmark()
    expected = (0.0, max(item["checksum_atol"] for item in policy["variants"].values()))
    assert reference.get_output_tolerance() == expected
    assert expected[1] > 0.0
    for variant, benchmark_type in (
        ("dynamic", OptimizedOzakiSchemeDynamicBenchmark),
        ("fixed", OptimizedOzakiSchemeFixedBenchmark),
    ):
        candidate = benchmark_type()
        native_gate_args, tolerance = configured_accuracy(variant)
        assert candidate.get_output_tolerance() == tolerance
        assert tolerance[1] <= reference.get_output_tolerance()[1]
        for index in range(0, len(native_gate_args), 2):
            flag, value = native_gate_args[index:index + 2]
            assert candidate._run_args[candidate._run_args.index(flag) + 1] == value


def test_ozaki_reference_stays_exact_without_an_accuracy_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from labs.ozaki_scheme.baseline_ozaki_scheme import BaselineOzakiSchemeBenchmark

    monkeypatch.delenv("AISP_OZAKI_ACCURACY_POLICY", raising=False)
    assert BaselineOzakiSchemeBenchmark().get_output_tolerance() == (0.0, 0.0)


def test_ozaki_lab_parse_metrics_captures_strategy_and_checksum() -> None:
    stdout = """
VARIANT: ozaki_dynamic
EMULATION_STRATEGY: eager
EMULATION_USED: 1
RETAINED_BITS: 4
TFLOPS: 11.178
MAX_ABS_ERROR: 3.0e-06
MEAN_ABS_ERROR: 0.0
RESULT_CHECKSUM: 1.2345000000e+02
TIME_MS: 12.296
"""

    metrics = parse_metrics(stdout)

    assert metrics["variant"] == "ozaki_dynamic"
    assert metrics["emulation_strategy"] == "eager"
    assert metrics["emulation_used"] == 1
    assert metrics["retained_bits"] == 4
    assert metrics["tflops"] == 11.178
    assert metrics["checksum"] == 123.45
    assert metrics["time_ms"] == 12.296


def test_ozaki_lab_csv_parsers_preserve_order() -> None:
    assert parse_int_csv("6,8,10,12") == [6, 8, 10, 12]
    assert parse_float_csv("1e-1,1e-2,1e-3") == [1e-1, 1e-2, 1e-3]


def test_ozaki_lab_reproducibility_summary_flags_stable_records() -> None:
    records = [
        {"checksum": 10.0, "retained_bits": 4, "emulation_used": 1},
        {"checksum": 10.0, "retained_bits": 4, "emulation_used": 1},
        {"checksum": 10.0, "retained_bits": 4, "emulation_used": 1},
    ]

    summary = summarize_reproducibility(records)

    assert summary == {
        "run_count": 3,
        "checksum_stable": True,
        "retained_bits_stable": True,
        "emulation_used_stable": True,
    }


def test_ozaki_lab_reproducibility_summary_flags_drift() -> None:
    records = [
        {"checksum": 10.0, "retained_bits": 4, "emulation_used": 1},
        {"checksum": 10.1, "retained_bits": 4, "emulation_used": 1},
    ]

    summary = summarize_reproducibility(records)

    assert summary["checksum_stable"] is False
    assert summary["retained_bits_stable"] is True
    assert summary["emulation_used_stable"] is True


def test_ozaki_lab_result_row_reports_speedup() -> None:
    row = format_result_row(
        "Ozaki dynamic",
        {
            "time_ms": 2.0,
            "tflops": 100.0,
            "retained_bits": 4,
            "emulation_used": 1,
            "max_abs_error": 1e-6,
            "mean_abs_error": 0.0,
        },
        baseline_ms=10.0,
    )

    assert "| Ozaki dynamic | 2.000 | 100.000 | 5.00x | 4 | 1 | 1.000e-06 | 0.000e+00 |" == row
