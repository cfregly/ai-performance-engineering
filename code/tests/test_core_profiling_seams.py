"""Focused compatibility tests for core-owned offline profiling helpers."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_architecture_specs_seventh_positional_argument_remains_cpu_gpu_bandwidth() -> None:
    core_roofline = importlib.import_module("core.analysis.kernel_roofline")

    specs = core_roofline.ArchitectureSpecs("fixture", 1.0, 2.0, 4.0, 2.0, 8.0, 900.0)

    assert specs.cpu_gpu_bandwidth_gbs == pytest.approx(900.0)
    assert specs.profile_source == "explicit_caller"
    assert specs.peak_tensor_fp16_tflops is None


def test_ch08_roofline_entrypoint_reexports_core_analyzer(monkeypatch) -> None:
    core_roofline = importlib.import_module("core.analysis.kernel_roofline")
    chapter_roofline = importlib.import_module("ch08.roofline")

    assert chapter_roofline.ArchitectureSpecs is core_roofline.ArchitectureSpecs
    assert chapter_roofline.RooflineAnalyzer is core_roofline.RooflineAnalyzer
    assert chapter_roofline.get_architecture_specs is core_roofline.get_architecture_specs

    specs = core_roofline.ArchitectureSpecs(
        name="fixture",
        peak_fp32_tflops=100.0,
        peak_fp16_tflops=200.0,
        peak_fp8_tflops=400.0,
        peak_tf32_tflops=200.0,
        memory_bandwidth_gbs=1_000.0,
    )
    monkeypatch.setattr(core_roofline, "get_architecture_specs", lambda: specs)
    result = chapter_roofline.RooflineAnalyzer().analyze_kernel(
        kernel_time_ms=2.0,
        flops=1e12,
        bytes_transferred=2e9,
        precision="fp32",
    )

    assert result["achieved_tflops"] == pytest.approx(500.0)
    assert result["achieved_bandwidth_gbs"] == pytest.approx(1_000.0)
    assert result["arithmetic_intensity"] == pytest.approx(500.0)
    assert result["ridge_point"] == pytest.approx(100.0)
    assert result["is_compute_bound"] is True
    assert result["is_memory_bound"] is False


def test_core_roofline_uses_peak_bandwidth_to_classify_binding_roof() -> None:
    core_roofline = importlib.import_module("core.analysis.kernel_roofline")
    specs = core_roofline.ArchitectureSpecs(
        name="fixture",
        peak_fp32_tflops=100.0,
        peak_fp16_tflops=200.0,
        peak_fp8_tflops=400.0,
        peak_tf32_tflops=200.0,
        memory_bandwidth_gbs=1_000.0,
    )
    analyzer = core_roofline.RooflineAnalyzer(specs)

    memory_bound = analyzer.analyze_kernel(
        kernel_time_ms=1.0,
        flops=1e9,
        bytes_transferred=1e9,
    )
    compute_bound = analyzer.analyze_kernel(
        kernel_time_ms=2_000.0,
        flops=200e9,
        bytes_transferred=1e9,
    )

    assert memory_bound["arithmetic_intensity"] == pytest.approx(1.0)
    assert memory_bound["ridge_point"] == pytest.approx(100.0)
    assert memory_bound["memory_bound_tflops"] == pytest.approx(1.0)
    assert memory_bound["is_memory_bound"] is True
    assert memory_bound["is_compute_bound"] is False
    assert compute_bound["arithmetic_intensity"] == pytest.approx(200.0)
    assert compute_bound["memory_bound_tflops"] == pytest.approx(200.0)
    assert compute_bound["is_memory_bound"] is False
    assert compute_bound["is_compute_bound"] is True


def test_core_roofline_detects_the_current_cuda_device(monkeypatch) -> None:
    core_roofline = importlib.import_module("core.analysis.kernel_roofline")
    torch = pytest.importorskip("torch")
    observed_indices = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    def properties(index):
        observed_indices.append(index)
        return SimpleNamespace(name="NVIDIA B200", major=10, minor=0)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)

    specs = core_roofline.get_architecture_specs()

    assert observed_indices == [1]
    assert specs.name == "NVIDIA B200"


def test_ch08_ieee_fp32_matmul_policy_is_restored() -> None:
    chapter = importlib.import_module("ch08.roofline")
    torch = pytest.importorskip("torch")
    previous = torch.backends.cuda.matmul.allow_tf32

    with chapter._ieee_fp32_matmul():
        assert torch.backends.cuda.matmul.allow_tf32 is False

    assert torch.backends.cuda.matmul.allow_tf32 is previous


@pytest.mark.parametrize(
    ("kernel_time_ms", "flops", "bytes_transferred", "message"),
    [
        (0.0, 1.0, 1.0, "kernel_time_ms must be positive"),
        (1.0, -1.0, 1.0, "flops must be non-negative"),
        (1.0, 1.0, 0.0, "bytes_transferred must be positive"),
    ],
)
def test_core_roofline_rejects_invalid_measurements(
    kernel_time_ms,
    flops,
    bytes_transferred,
    message,
) -> None:
    core_roofline = importlib.import_module("core.analysis.kernel_roofline")
    specs = core_roofline.get_architecture_specs_for_profile("b200")
    analyzer = core_roofline.RooflineAnalyzer(specs)

    with pytest.raises(ValueError, match=message):
        analyzer.analyze_kernel(
            kernel_time_ms=kernel_time_ms,
            flops=flops,
            bytes_transferred=bytes_transferred,
        )


def test_core_consumers_import_stdlib_only_nsys_parser() -> None:
    core_roofline = importlib.import_module("core.analysis.kernel_roofline")
    core_nsys = importlib.import_module("core.profiling.nsight_systems")
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")

    assert deep_report.RooflineAnalyzer is core_roofline.RooflineAnalyzer
    assert deep_report.NsightSystemsReportParser is core_nsys.NsightSystemsReportParser
    assert nsys_summary.NsightSystemsReportParser is core_nsys.NsightSystemsReportParser


def test_ch17_capture_wrapper_delegates_to_streaming_core_parser(
    monkeypatch,
    tmp_path,
) -> None:
    core_nsys = importlib.import_module("core.profiling.nsight_systems")
    chapter_guide = importlib.import_module("ch17.blackwell_profiling_guide")

    assert issubclass(
        chapter_guide.NsightSystemsProfiler,
        core_nsys.NsightSystemsReportParser,
    )

    rows = [
        {"Name": "attention_a", "Time (%)": "12.5"},
        {"Name": "copy", "Time (%)": "90.0"},
        {"Name": "attention_b", "Time (%) [sum]": '"40.0"'},
        {"Name": "attention_invalid", "Time (%)": "N/A"},
    ]
    report_path = tmp_path / "capture.nsys-rep"
    report_path.touch()

    def fake_stats(_report, section):
        return rows if section == "cuda_gpu_kern_sum" else []

    monkeypatch.setattr(
        chapter_guide.NsightSystemsProfiler,
        "_run_nsys_stats",
        staticmethod(fake_stats),
    )
    summary = chapter_guide.NsightSystemsProfiler.summarize_report(
        str(report_path),
        kernel_regex="^attention",
        top_k=2,
        print_summary=False,
    )

    assert [row["Name"] for row in summary["kernels"]] == [
        "attention_b",
        "attention_a",
    ]


def test_nsys_ranking_excludes_rows_without_identity_or_valid_percentage() -> None:
    core_nsys = importlib.import_module("core.profiling.nsight_systems")
    rows = [
        {"Name": "valid", "Time (%)": "75"},
        {"Name": "missing_time"},
        {"Time (%)": "50"},
        {"Name": "over_100", "Time (%)": "900"},
        {"Name": "ambiguous", "Time (%)": "10", "Time (%) [sum]": "11"},
    ]

    ranked = core_nsys.NsightSystemsReportParser._filter_and_rank_kernels(
        rows, None, 10
    )

    assert ranked == [{"Name": "valid", "Time (%)": "75"}]


def test_nsys_stats_rejects_unrecognized_header(monkeypatch, tmp_path) -> None:
    core_nsys = importlib.import_module("core.profiling.nsight_systems")
    report_path = tmp_path / "capture.nsys-rep"
    report_path.touch()
    monkeypatch.setattr(
        core_nsys.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="Unexpected,Columns\nvalue,other\n"
        ),
    )

    with pytest.raises(RuntimeError, match="recognized Name/Time"):
        core_nsys.NsightSystemsReportParser._run_nsys_stats(
            report_path, "cuda_gpu_kern_sum"
        )


def test_nsys_stats_rejects_malformed_row_width(monkeypatch, tmp_path) -> None:
    core_nsys = importlib.import_module("core.profiling.nsight_systems")
    report_path = tmp_path / "capture.nsys-rep"
    report_path.touch()
    monkeypatch.setattr(
        core_nsys.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="Time (%),Name\n75,kernel,unexpected\n"
        ),
    )

    with pytest.raises(RuntimeError, match="malformed row"):
        core_nsys.NsightSystemsReportParser._run_nsys_stats(
            report_path, "cuda_gpu_kern_sum"
        )


def test_nsys_parser_uses_valid_reports_on_temporary_evidence_copy(
    monkeypatch,
    tmp_path,
) -> None:
    core_nsys = importlib.import_module("core.profiling.nsight_systems")
    report_path = tmp_path / "capture.nsys-rep"
    original = b"immutable profile evidence"
    report_path.write_bytes(original)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        section = command[command.index("--report") + 1]
        if section == "cuda_gpu_kern_sum":
            output = "Time (%),Name\n75.0,kernel\n"
        else:
            output = "Time (%),Name\n100.0,cudaLaunchKernel\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(core_nsys.subprocess, "run", fake_run)

    summary = core_nsys.NsightSystemsReportParser.summarize_report(
        str(report_path), print_summary=False
    )

    assert summary["report"] == str(report_path.resolve())
    assert report_path.read_bytes() == original
    assert [command[command.index("--report") + 1] for command in commands] == [
        "cuda_api_sum",
        "cuda_gpu_kern_sum",
    ]
    assert all("--force-export" not in command for command in commands)
    assert all(Path(command[-1]).parent != report_path.parent for command in commands)
    assert all(not Path(command[-1]).exists() for command in commands)


def test_nsys_source_receipt_digest_changes_when_same_path_bytes_change(
    monkeypatch,
    tmp_path,
) -> None:
    core_nsys = importlib.import_module("core.profiling.nsight_systems")
    report_path = tmp_path / "capture.nsys-rep"
    monkeypatch.setattr(
        core_nsys.NsightSystemsReportParser,
        "_run_nsys_stats",
        staticmethod(lambda _report, _section: []),
    )
    report_path.write_bytes(b"first capture")
    first = core_nsys.NsightSystemsReportParser.summarize_report(
        str(report_path), print_summary=False
    )
    report_path.write_bytes(b"replacement capture")
    second = core_nsys.NsightSystemsReportParser.summarize_report(
        str(report_path), print_summary=False
    )

    assert first["source"]["status"] == "parsed"
    assert first["source"]["sha256"] != second["source"]["sha256"]
    assert first["source"]["size_bytes"] == len(b"first capture")
    assert second["source"]["size_bytes"] == len(b"replacement capture")


def test_nsight_profile_command_is_preserved_through_chapter_alias() -> None:
    chapter_guide = importlib.import_module("ch17.blackwell_profiling_guide")

    command = chapter_guide.NsightSystemsProfiler.profile_command(
        "workload.py",
        "capture",
        duration=7,
        ib_switch_guids=["a", "b"],
        use_hardware_trace=False,
        include_sqlite_export=False,
    )

    assert command.startswith("nsys profile -o capture --trace=cuda,osrt,nvtx,ucx,gds")
    assert "--duration=7" in command
    assert "--ib-switch-metrics-device=a,b" in command
    assert "--export=sqlite" not in command
    assert command.endswith("python workload.py")


def test_ch17_nsys_context_exposes_only_effective_constructor_options() -> None:
    chapter_guide = importlib.import_module("ch17.blackwell_profiling_guide")

    signature = inspect.signature(chapter_guide.NsightSystemsProfiler)

    assert list(signature.parameters) == ["output_name", "trace_nvtx"]
    assert signature.parameters["trace_nvtx"].kind is inspect.Parameter.KEYWORD_ONLY


def test_ch17_pytorch_profiler_completes_schedule_and_writes_trace(
    tmp_path,
    capsys,
) -> None:
    torch = pytest.importorskip("torch")
    chapter_guide = importlib.import_module("ch17.blackwell_profiling_guide")
    calls = 0

    def workload() -> None:
        nonlocal calls
        calls += 1
        torch.add(torch.ones(8), 1)

    chapter_guide.profile_with_pytorch_profiler(
        workload,
        output_dir=str(tmp_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )

    trace_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert calls == 9
    assert trace_files
    assert all(path.stat().st_size > 0 for path in trace_files)
    assert "TensorBoard trace saved" in capsys.readouterr().out


def test_ch17_nsys_analysis_returns_parsed_status_and_propagates_failure(
    monkeypatch,
    capsys,
) -> None:
    chapter_guide = importlib.import_module("ch17.blackwell_profiling_guide")
    expected_summary = {
        "report": "/tmp/capture.nsys-rep",
        "summary": [{"Name": "cudaLaunchKernel"}],
        "kernels": [{"Name": "fixture_kernel"}],
    }
    monkeypatch.setattr(
        chapter_guide.NsightSystemsProfiler,
        "summarize_report",
        classmethod(lambda cls, report_path, **kwargs: expected_summary),
    )

    status = chapter_guide.NsightSystemsProfiler.analyze_blackwell_metrics("/tmp/capture.nsys-rep")

    assert status == {
        "status": "parsed",
        "tool": "nsight_systems",
        "report_path": "/tmp/capture.nsys-rep",
        "summary": expected_summary,
    }
    assert "Open the trace" in capsys.readouterr().out

    def fail_parse(cls, report_path, **kwargs):
        raise RuntimeError("nsys stats failed")

    monkeypatch.setattr(
        chapter_guide.NsightSystemsProfiler,
        "summarize_report",
        classmethod(fail_parse),
    )
    with pytest.raises(RuntimeError, match="nsys stats failed"):
        chapter_guide.NsightSystemsProfiler.analyze_blackwell_metrics("/tmp/broken.nsys-rep")
    assert "Open the trace" not in capsys.readouterr().out


def test_ch17_ncu_command_emits_raw_csv_with_complete_scalar_counter_families() -> None:
    chapter_guide = importlib.import_module("ch17.blackwell_profiling_guide")

    command = chapter_guide.NsightComputeProfiler.profile_kernel_command(
        "workload.py",
        "artifacts/capture",
        kernel_filter="fixture_kernel",
    )
    argv = shlex.split(command)
    metrics = set(argv[argv.index("--metrics") + 1].split(","))

    for operation_letter in ("f", "h", "d"):
        assert {
            f"smsp__sass_thread_inst_executed_op_{operation_letter}fma_pred_on.sum",
            f"smsp__sass_thread_inst_executed_op_{operation_letter}add_pred_on.sum",
            f"smsp__sass_thread_inst_executed_op_{operation_letter}mul_pred_on.sum",
        } <= metrics
    assert {
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
    } <= metrics
    assert not any("tensor" in metric or "fp8" in metric for metric in metrics)
    assert "smsp__sass_thread_inst_executed_op_dmma_pred_on.sum" not in metrics
    assert argv[argv.index("--page") + 1] == "raw"
    assert argv[argv.index("--log-file") + 1] == "artifacts/capture.csv"
    assert argv[argv.index("--export") + 1] == "artifacts/capture.ncu-rep"
    assert argv[argv.index("--kernel-name") + 1] == "fixture_kernel"


def test_ch17_ncu_analysis_parses_supported_raw_csv(tmp_path, capsys) -> None:
    chapter_guide = importlib.import_module("ch17.blackwell_profiling_guide")
    csv_path = tmp_path / "capture.csv"
    header = '"ID","Kernel Name","Device","CC","Metric Name","Metric Value","Metric Unit"\n'
    rows = [
        ("gpu__time_duration.sum", "2", "ms"),
        ("dram__bytes_read.sum", "1000", "byte"),
        ("dram__bytes_write.sum", "2000", "byte"),
        ("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "5", "inst"),
        ("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", "7", "inst"),
        ("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", "11", "inst"),
    ]
    csv_path.write_text(
        header
        + "".join(
            f'"1","fixture_kernel","NVIDIA B200","10.0","{metric}","{value}","{unit}"\n'
            for metric, value, unit in rows
        ),
        encoding="utf-8",
    )

    status = chapter_guide.NsightComputeProfiler.analyze_blackwell_kernel(str(csv_path))

    assert status["status"] == "parsed"
    assert status["tool"] == "nsight_compute"
    assert status["artifact_type"] == "ncu_raw_csv"
    assert status["report_path"] == str(csv_path.resolve())
    assert status["kernel_count"] == 1
    assert status["kernels"][0]["metric_count"] == len(rows)
    parsed_metrics = status["kernels"][0]["metrics"]
    assert {metric for metric, _value, _unit in rows} == set(parsed_metrics)
    assert parsed_metrics["gpu__time_duration.sum"] == {
        "value": 2.0,
        "unit": "ms",
        "section": None,
    }
    assert "Parsed 1 kernel launch" in capsys.readouterr().out


def test_ch17_ncu_analysis_rejects_missing_empty_binary_and_unrecognized_artifacts(
    tmp_path,
) -> None:
    chapter_guide = importlib.import_module("ch17.blackwell_profiling_guide")
    analyze = chapter_guide.NsightComputeProfiler.analyze_blackwell_kernel

    with pytest.raises(FileNotFoundError, match="artifact not found"):
        analyze(str(tmp_path / "missing.csv"))

    empty_csv = tmp_path / "empty.csv"
    empty_csv.touch()
    with pytest.raises(ValueError, match="CSV is empty"):
        analyze(str(empty_csv))

    empty_report = tmp_path / "empty.ncu-rep"
    empty_report.touch()
    with pytest.raises(ValueError, match=r"\.ncu-rep artifact is empty"):
        analyze(str(empty_report))

    binary_report = tmp_path / "capture.ncu-rep"
    binary_report.write_bytes(b"\x00NVREP\xff")
    with pytest.raises(ValueError, match=r"Binary \.ncu-rep.*export"):
        analyze(str(binary_report))

    binary_csv = tmp_path / "binary.csv"
    binary_csv.write_bytes(b"\x00\xff")
    with pytest.raises(ValueError, match="text, not binary"):
        analyze(str(binary_csv))

    malformed_csv = tmp_path / "malformed.csv"
    malformed_csv.write_text("not,ncu,csv\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or unrecognized"):
        analyze(str(malformed_csv))


def test_complete_blackwell_analysis_returns_success_only_after_both_parse(
    monkeypatch,
) -> None:
    chapter_guide = importlib.import_module("ch17.blackwell_profiling_guide")
    nsys_status = {"status": "parsed", "tool": "nsight_systems"}
    ncu_status = {"status": "parsed", "tool": "nsight_compute"}
    monkeypatch.setattr(
        chapter_guide.NsightSystemsProfiler,
        "analyze_blackwell_metrics",
        staticmethod(lambda report_path: nsys_status),
    )
    monkeypatch.setattr(
        chapter_guide.NsightComputeProfiler,
        "analyze_blackwell_kernel",
        staticmethod(lambda report_path: ncu_status),
    )
    monkeypatch.setattr(
        chapter_guide.BlackwellMetricsGuide,
        "print_metric_guide",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr(
        chapter_guide.HBMMemoryAnalyzer,
        "print_hbm3e_best_practices",
        staticmethod(lambda: None),
    )

    result = chapter_guide.run_complete_blackwell_analysis("capture.nsys-rep", "capture.csv")

    assert result == {
        "status": "success",
        "hardware_profile": "b200",
        "nsys": nsys_status,
        "ncu": ncu_status,
    }

    def fail_ncu(report_path):
        raise ValueError("invalid NCU CSV")

    monkeypatch.setattr(
        chapter_guide.NsightComputeProfiler,
        "analyze_blackwell_kernel",
        staticmethod(fail_ncu),
    )
    with pytest.raises(ValueError, match="invalid NCU CSV"):
        chapter_guide.run_complete_blackwell_analysis(
            "capture.nsys-rep",
            "broken.csv",
        )

    monkeypatch.setattr(
        chapter_guide.NsightComputeProfiler,
        "analyze_blackwell_kernel",
        staticmethod(lambda report_path: {"status": "failed"}),
    )
    with pytest.raises(RuntimeError, match="did not return parsed status"):
        chapter_guide.run_complete_blackwell_analysis(
            "capture.nsys-rep",
            "failed.csv",
        )


def test_offline_core_imports_block_torch_and_chapter_access() -> None:
    script = """
import builtins
import sys

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch.") or name.startswith("ch17"):
        raise AssertionError(f"offline import attempted {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
import core.analysis.deep_profiling_report
import core.profiling.nsight_systems
import core.profiling.nsys_summary
assert "torch" not in sys.modules
assert not any(name.startswith("ch17") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _offline_kernel_metrics(deep_report):
    metrics = deep_report.KernelMetrics("fixture_kernel")
    for name, value, unit in (
        ("gpu__time_duration.sum", 2.0, "ms"),
        ("flop_count_sp", 1e12, "flop"),
        ("dram__bytes.sum", 2e9, "byte"),
    ):
        metrics.metrics[name] = deep_report.RawMetric(name, value, unit)
    return metrics


def test_offline_roofline_uses_explicit_specs_without_local_detection(monkeypatch) -> None:
    core_roofline = importlib.import_module("core.analysis.kernel_roofline")
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = _offline_kernel_metrics(deep_report)

    def reject_local_detection():
        raise AssertionError("offline analysis consulted local CUDA")

    monkeypatch.setattr(core_roofline, "get_architecture_specs", reject_local_detection)
    selection = deep_report.resolve_hardware_selection([metrics], "b200")
    roofline, *_ = deep_report.derive_roofline(metrics, selection.specs)
    unknown, *_ = deep_report.derive_roofline(metrics, None)

    assert selection.provenance == "explicit_cli"
    assert selection.as_dict()["specs"]["name"] == "NVIDIA B200"
    assert "core.benchmark.metrics:b200" in selection.as_dict()["specs"]["profile_source"]
    assert roofline is not None
    assert roofline.peak_tflops == pytest.approx(75.0)
    assert roofline.peak_bandwidth_gbs == pytest.approx(8_000.0)
    assert unknown is None


@pytest.mark.parametrize(("major", "minor"), [(10, 3), (12, 0), (12, 1)])
def test_offline_roofline_does_not_infer_sku_from_compute_capability(
    major,
    minor,
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = _offline_kernel_metrics(deep_report)
    metrics.metrics["device__attribute_compute_capability_major"] = deep_report.RawMetric(
        "device__attribute_compute_capability_major",
        float(major),
    )
    metrics.metrics["device__attribute_compute_capability_minor"] = deep_report.RawMetric(
        "device__attribute_compute_capability_minor",
        float(minor),
    )

    selection = deep_report.resolve_hardware_selection([metrics], None)

    assert selection.profile == "unknown"
    assert selection.provenance == "source_ncu_compute_capability_only"
    assert selection.compute_capability == f"{major}.{minor}"
    assert selection.specs is None

    del metrics.metrics["device__attribute_compute_capability_minor"]
    incomplete = deep_report.resolve_hardware_selection([metrics], None)
    assert incomplete.profile == "unknown"
    assert incomplete.provenance == "source_ncu_compute_capability_incomplete"
    assert incomplete.specs is None


@pytest.mark.parametrize("profile", ["gb10", "b300", "h100", "cpu"])
def test_unvalidated_or_ambiguous_profiles_are_rejected(profile) -> None:
    core_roofline = importlib.import_module("core.analysis.kernel_roofline")

    with pytest.raises(ValueError, match="Unknown or unvalidated hardware profile"):
        core_roofline.get_architecture_specs_for_profile(profile)


@pytest.mark.parametrize(
    ("hardware_args", "expected_profile", "expected_provenance", "has_roofline"),
    [
        (["--hardware-profile", "b200"], "b200", "explicit_cli", True),
        ([], "unknown", "default_unknown", False),
    ],
)
def test_deep_report_records_hardware_provenance(
    tmp_path,
    hardware_args,
    expected_profile,
    expected_provenance,
    has_roofline,
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    csv_path = tmp_path / "metrics.csv"
    output_path = tmp_path / "report.json"
    csv_path.write_text(
        '"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        '"1","fixture_kernel","gpu__time_duration.sum","2","ms"\n'
        '"1","fixture_kernel","flop_count_sp","1000000000000","flop"\n'
        '"1","fixture_kernel","dram__bytes.sum","2000000000","byte"\n'
    )

    result = deep_report.main(
        [
            "--ncu-csv",
            str(csv_path),
            "--output-json",
            str(output_path),
            *hardware_args,
        ]
    )
    payload = json.loads(output_path.read_text())

    assert result == 0
    assert payload["schema_version"] == "deep_profiling_report.v1"
    assert payload["success"] is True
    assert payload["inputs"][0]["status"] == "parsed"
    assert payload["inputs"][0]["kernel_count"] == 1
    assert payload["inputs"][0]["capture_id"].startswith("sha256:")
    assert payload["advisories"][0]["source"]["capture_id"] == payload["inputs"][0]["capture_id"]
    assert payload["advisories"][0]["source"]["artifacts"] == [str(csv_path.resolve())]
    assert payload["hardware"]["profile"] == expected_profile
    assert payload["hardware"]["provenance"] == expected_provenance
    assert (payload["advisories"][0]["roofline"] is not None) is has_roofline
    if not has_roofline:
        assert (
            "no ceiling-based bottleneck classification"
            in payload["advisories"][0]["recommendations"][0]
        )


def test_utilization_percentages_do_not_fabricate_roofline_work() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("utilization_only")
    for name, value, unit in (
        ("gpu__time_duration.sum", 2.0, "ms"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed", 50.0, "%"),
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed", 25.0, "%"),
    ):
        metrics.metrics[name] = deep_report.RawMetric(name, value, unit)
    specs = deep_report.get_architecture_specs_for_profile("b200")

    roofline, duration_ms, flops, bytes_transferred, _ = deep_report.derive_roofline(
        metrics,
        specs,
    )
    advisory = deep_report.build_advisory(metrics, specs)

    assert roofline is None
    assert duration_ms == pytest.approx(2.0)
    assert flops is None
    assert bytes_transferred is None
    assert advisory.roofline is None
    assert advisory.flops is None
    assert advisory.bytes_transferred is None
    assert advisory.sm_util_pct == pytest.approx(50.0)
    assert advisory.dram_util_pct == pytest.approx(25.0)
    assert "utilization percentages are not substitutes" in advisory.recommendations[0]


def test_compute_bytes_uses_strict_non_overlapping_family_precedence() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("byte_families")

    def add(name, value, unit="byte"):
        metrics.metrics[name] = deep_report.RawMetric(name, value, unit)

    add("dram__bytes.sum", 100.0)
    add("dram__bytes_read.sum", 60.0)
    add("dram__bytes_write.sum", 40.0)
    add("gpu__dram_sectors_read.sum", 2.0, "sector")
    add("gpu__dram_sectors_write.sum", 3.0, "sector")
    add("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", 1_000.0)
    add("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum", 2_000.0)

    assert deep_report.compute_bytes(metrics) == pytest.approx(100.0)

    del metrics.metrics["dram__bytes.sum"]
    assert deep_report.compute_bytes(metrics) == pytest.approx(100.0)

    del metrics.metrics["dram__bytes_read.sum"]
    del metrics.metrics["dram__bytes_write.sum"]
    assert deep_report.compute_bytes(metrics) == pytest.approx(160.0)

    del metrics.metrics["gpu__dram_sectors_write.sum"]
    assert deep_report.compute_bytes(metrics) == pytest.approx(3_000.0)

    metrics.metrics.clear()
    add("dram__bytes_read.sum", 60.0)
    assert deep_report.compute_bytes(metrics) is None


def test_compute_flops_uses_strict_counter_family_precedence() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("flop_families")

    def add(name, value):
        metrics.metrics[name] = deep_report.RawMetric(name, value, "flop")

    add("flop_count_sp", 100.0)
    add("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 10.0)

    assert deep_report.compute_flops(metrics) == pytest.approx(100.0)

    del metrics.metrics["flop_count_sp"]
    assert deep_report.pick_precision(metrics) == "fp32"
    assert deep_report.compute_flops(metrics) is None
    add("sm__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0)
    add("sm__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0)
    assert deep_report.compute_flops(metrics) == pytest.approx(20.0)


def test_complete_sass_counter_family_is_required_and_scopes_do_not_double_count() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("complete_family")

    def add(scope, operation, value):
        name = f"{scope}__sass_thread_inst_executed_op_f{operation}_pred_on.sum"
        metrics.metrics[name] = deep_report.RawMetric(name, value, "instruction")

    add("smsp", "fma", 10.0)
    assert deep_report.compute_flops(metrics) is None

    add("smsp", "add", 3.0)
    add("smsp", "mul", 4.0)
    add("sm", "fma", 1_000.0)
    add("sm", "add", 1_000.0)
    add("sm", "mul", 1_000.0)

    assert deep_report.compute_flops(metrics) == pytest.approx(27.0)


@pytest.mark.parametrize(
    ("sp_value", "expected_precision"),
    [(0.0, "fp64"), (100.0, "mixed")],
)
def test_zero_aggregate_counter_does_not_claim_another_precision(
    sp_value, expected_precision
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("mixed_aggregate_and_sass")
    metrics.metrics["flop_count_sp"] = deep_report.RawMetric("flop_count_sp", sp_value, "flop")
    for operation, value in (("dfma", 10.0), ("dadd", 2.0), ("dmul", 3.0)):
        name = f"smsp__sass_thread_inst_executed_op_{operation}_pred_on.sum"
        metrics.metrics[name] = deep_report.RawMetric(name, value, "instruction")

    assert deep_report.pick_precision(metrics) == expected_precision
    if expected_precision == "mixed":
        assert deep_report.compute_flops(metrics) is None
    else:
        assert deep_report.compute_flops(metrics) == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("sp_value", "hp_value", "expected_precision", "expected_flops"),
    [
        (100.0, 0.0, "fp32", 100.0),
        (0.0, 100.0, "fp16", 100.0),
        (0.0, 0.0, "operation_free", 0.0),
    ],
)
def test_producer_shaped_zero_counters_do_not_create_mixed_precision(
    sp_value, hp_value, expected_precision, expected_flops
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("producer_shaped")
    metrics.metrics["flop_count_sp"] = deep_report.RawMetric(
        "flop_count_sp", sp_value, "flop"
    )
    metrics.metrics["flop_count_hp"] = deep_report.RawMetric(
        "flop_count_hp", hp_value, "flop"
    )

    assert deep_report.pick_precision(metrics) == expected_precision
    assert deep_report.compute_flops(metrics) == pytest.approx(expected_flops)


def test_precision_specific_scalar_rooflines_do_not_use_tensor_or_other_family_peaks() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    specs = deep_report.get_architecture_specs_for_profile("b200")

    fp16 = deep_report.KernelMetrics("scalar_fp16")
    fp64 = deep_report.KernelMetrics("scalar_fp64")
    mixed = deep_report.KernelMetrics("mixed")
    for metrics in (fp16, fp64, mixed):
        metrics.metrics["gpu__time_duration.sum"] = deep_report.RawMetric(
            "gpu__time_duration.sum", 1.0, "ms"
        )
        metrics.metrics["dram__bytes.sum"] = deep_report.RawMetric(
            "dram__bytes.sum", 1_000.0, "byte"
        )
    fp16.metrics["smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"] = deep_report.RawMetric(
        "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", 10.0, "instruction"
    )
    fp16.metrics["smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"] = deep_report.RawMetric(
        "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum", 2.0, "instruction"
    )
    fp16.metrics["smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"] = deep_report.RawMetric(
        "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum", 3.0, "instruction"
    )
    fp64.metrics["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"] = deep_report.RawMetric(
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", 10.0, "instruction"
    )
    fp64.metrics["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"] = deep_report.RawMetric(
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", 2.0, "instruction"
    )
    fp64.metrics["smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"] = deep_report.RawMetric(
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum", 3.0, "instruction"
    )
    mixed.metrics["flop_count_sp"] = deep_report.RawMetric("flop_count_sp", 10.0, "flop")
    mixed.metrics["flop_count_hp"] = deep_report.RawMetric("flop_count_hp", 10.0, "flop")

    fp16_roof, *_ = deep_report.derive_roofline(fp16, specs)
    fp64_roof, *_ = deep_report.derive_roofline(fp64, specs)
    mixed_roof, *_ = deep_report.derive_roofline(mixed, specs)

    assert deep_report.pick_precision(fp16) == "fp16"
    assert deep_report.compute_flops(fp16) == pytest.approx(25.0)
    assert fp16_roof is not None
    assert fp16_roof.peak_tflops == pytest.approx(150.0)
    assert fp16_roof.peak_tflops != pytest.approx(2_250.0)
    assert deep_report.pick_precision(fp64) == "fp64"
    assert fp64_roof is None
    assert deep_report.pick_precision(mixed) == "mixed"
    assert deep_report.compute_flops(mixed) is None
    assert mixed_roof is None


def test_zero_flop_memory_operation_remains_a_valid_memory_roofline_point() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("copy")
    for name, value, unit in (
        ("gpu__time_duration.sum", 1.0, "ms"),
        ("flop_count_sp", 0.0, "flop"),
        ("dram__bytes.sum", 1_000_000.0, "byte"),
    ):
        metrics.metrics[name] = deep_report.RawMetric(name, value, unit)

    roofline, _duration, flops, _bytes, precision = deep_report.derive_roofline(
        metrics, deep_report.get_architecture_specs_for_profile("b200")
    )

    assert precision == "operation_free"
    assert flops == 0.0
    assert roofline is not None
    assert roofline.achieved_tflops == 0.0
    assert roofline.arithmetic_intensity == 0.0
    assert roofline.is_memory_bound is True
    assert roofline.peak_tflops is None
    assert roofline.ridge_point is None
    assert roofline.compute_utilization_pct is None


def test_default_ncu_producer_emits_complete_reporter_counter_families() -> None:
    metrics_config = importlib.import_module("core.scripts.harness.metrics_config")
    metrics = set(metrics_config.BASE_NCU_METRICS)

    assert {"dram__bytes_read.sum", "dram__bytes_write.sum"} <= metrics
    for prefix in ("f", "h", "d"):
        assert {
            f"smsp__sass_thread_inst_executed_op_{prefix}fma_pred_on.sum",
            f"smsp__sass_thread_inst_executed_op_{prefix}add_pred_on.sum",
            f"smsp__sass_thread_inst_executed_op_{prefix}mul_pred_on.sum",
        } <= metrics
    assert "gpu__time_duration.sum" in metrics
    assert "sm__sass_thread_inst_executed_op_fp32_pred_on.sum" not in metrics
    assert "sm__sass_thread_inst_executed_op_fp16_pred_on.sum" not in metrics


def test_default_ncu_zero_sass_families_form_operation_free_memory_point() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics_config = importlib.import_module("core.scripts.harness.metrics_config")
    metrics = deep_report.KernelMetrics("producer_copy")
    for name in metrics_config.BASE_NCU_METRICS:
        if "sass_thread_inst_executed_op_" in name:
            metrics.metrics[name] = deep_report.RawMetric(name, 0.0, "instruction")
    metrics.metrics["gpu__time_duration.sum"] = deep_report.RawMetric(
        "gpu__time_duration.sum", 1.0, "ms"
    )
    metrics.metrics["dram__bytes_read.sum"] = deep_report.RawMetric(
        "dram__bytes_read.sum", 100.0, "byte"
    )
    metrics.metrics["dram__bytes_write.sum"] = deep_report.RawMetric(
        "dram__bytes_write.sum", 0.0, "byte"
    )

    roofline, _duration, flops, hbm_bytes, precision = deep_report.derive_roofline(
        metrics, deep_report.get_architecture_specs_for_profile("b200")
    )

    assert precision == "operation_free"
    assert flops == 0.0
    assert hbm_bytes == pytest.approx(100.0)
    assert roofline is not None
    assert roofline.arithmetic_intensity == 0.0
    assert roofline.peak_tflops is None


def test_explicit_percent_unit_preserves_sub_one_percentage() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("small_percentage")
    metric_name = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    metrics.metrics[metric_name] = deep_report.RawMetric(metric_name, 0.5, "%")
    specs = deep_report.get_architecture_specs_for_profile("b200")

    advisory = deep_report.build_advisory(metrics, specs)

    assert advisory.sm_util_pct == pytest.approx(0.5)
    assert deep_report.safe_pct(deep_report.RawMetric("unitless", 0.5)) == pytest.approx(50.0)


def test_l1_only_bytes_cannot_feed_hbm_roofline() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("l1_only")
    for name, value, unit in (
        ("gpu__time_duration.sum", 2.0, "ms"),
        ("flop_count_sp", 100.0, "flop"),
        ("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", 60.0, "byte"),
        ("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum", 40.0, "byte"),
    ):
        metrics.metrics[name] = deep_report.RawMetric(name, value, unit)
    specs = deep_report.get_architecture_specs_for_profile("b200")

    roofline, _, flops, hbm_bytes, _ = deep_report.derive_roofline(metrics, specs)

    assert deep_report.compute_bytes(metrics) == pytest.approx(100.0)
    assert deep_report.compute_hbm_bytes(metrics) is None
    assert flops == pytest.approx(100.0)
    assert hbm_bytes is None
    assert roofline is None


def test_negative_byte_component_and_percentage_fail_closed() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("invalid_counters")
    metrics.metrics["dram__bytes_read.sum"] = deep_report.RawMetric(
        "dram__bytes_read.sum", -100.0, "byte"
    )
    metrics.metrics["dram__bytes_write.sum"] = deep_report.RawMetric(
        "dram__bytes_write.sum", 200.0, "byte"
    )

    assert deep_report.compute_hbm_bytes(metrics) is None
    assert deep_report.safe_pct(deep_report.RawMetric("negative", -0.5, "%")) is None
    assert deep_report.safe_pct(deep_report.RawMetric("nan", float("nan"), "%")) is None


def test_invalid_authoritative_hbm_counter_blocks_lower_family_fallback() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("invalid_authoritative_hbm")
    for name, value in (
        ("dram__bytes.sum", -1.0),
        ("dram__bytes_read.sum", 60.0),
        ("dram__bytes_write.sum", 40.0),
    ):
        metrics.metrics[name] = deep_report.RawMetric(name, value, "byte")

    assert deep_report.compute_hbm_bytes(metrics) is None


def test_invalid_recognized_sass_counter_blocks_positive_family_fallback() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("invalid_sass")
    for operation, value in (("ffma", 10.0), ("fadd", 1.0), ("fmul", 1.0)):
        name = f"smsp__sass_thread_inst_executed_op_{operation}_pred_on.sum"
        metrics.metrics[name] = deep_report.RawMetric(name, value, "instruction")
    invalid_name = "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"
    metrics.metrics[invalid_name] = deep_report.RawMetric(
        invalid_name, -1.0, "instruction"
    )

    assert deep_report.pick_precision(metrics) == "unknown"
    assert deep_report.compute_flops(metrics) is None


def test_tensor_instruction_count_without_shape_metadata_has_no_flop_value() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("tensor_instruction_only")
    for name, value, unit in (
        ("gpu__time_duration.sum", 2.0, "ms"),
        ("smsp__inst_executed_pipe_tensor.sum", 10.0, "instruction"),
        ("dram__bytes.sum", 100.0, "byte"),
    ):
        metrics.metrics[name] = deep_report.RawMetric(name, value, unit)
    specs = deep_report.get_architecture_specs_for_profile("b200")

    roofline, _, flops, hbm_bytes, _ = deep_report.derive_roofline(metrics, specs)

    assert deep_report.compute_flops(metrics) is None
    assert flops is None
    assert hbm_bytes == pytest.approx(100.0)
    assert roofline is None


def test_deep_report_rejects_header_only_input_with_per_input_status(
    tmp_path,
    capsys,
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    csv_path = tmp_path / "header_only.csv"
    csv_path.write_text('"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n')

    result = deep_report.main(["--ncu-csv", str(csv_path)])
    captured = capsys.readouterr()

    assert result == 2
    assert f"[ncu-input] {csv_path}: status=empty_or_unrecognized" in captured.err
    assert "No Nsight Compute CSV input contained usable kernel metrics." in captured.err


def test_deep_report_records_partial_missing_input_and_returns_nonzero(
    tmp_path,
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    good = tmp_path / "good.csv"
    missing = tmp_path / "missing.csv"
    output = tmp_path / "report.json"
    good.write_text(
        '"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        '"1","kernel","gpu__time_duration.sum","1","ms"\n'
    )

    result = deep_report.main(
        [
            "--ncu-csv",
            str(good),
            "--ncu-csv",
            str(missing),
            "--output-json",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())

    assert result == 3
    assert payload["success"] is False
    assert [item["status"] for item in payload["inputs"]] == ["parsed", "missing"]
    assert payload["inputs"][1]["capture_id"] is None


def test_deep_report_records_existing_input_read_failure_in_json(
    tmp_path,
    monkeypatch,
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    csv_path = tmp_path / "unreadable.csv"
    output = tmp_path / "report.json"
    csv_path.write_text("fixture")

    def fail_read(_path):
        raise OSError("fixture read failure")

    monkeypatch.setattr(deep_report, "parse_ncu_csv", fail_read)

    result = deep_report.main(["--ncu-csv", str(csv_path), "--output-json", str(output)])
    payload = json.loads(output.read_text())

    assert result == 2
    assert payload["success"] is False
    assert payload["inputs"][0]["status"] == "error"
    assert payload["inputs"][0]["capture_id"] is None
    assert "fixture read failure" in payload["inputs"][0]["error"]


def test_deep_report_rejects_output_collision_without_mutating_input(tmp_path) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    csv_path = tmp_path / "metrics.csv"
    original = (
        b'"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        b'"1","kernel","gpu__time_duration.sum","1","ms"\n'
    )
    csv_path.write_bytes(original)

    result = deep_report.main(["--ncu-csv", str(csv_path), "--output-json", str(csv_path)])

    assert result == 1
    assert csv_path.read_bytes() == original


def test_deep_report_rejects_hard_link_output_alias_without_mutating_input(tmp_path) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    csv_path = tmp_path / "metrics.csv"
    output_alias = tmp_path / "report.json"
    original = (
        b'"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        b'"1","kernel","gpu__time_duration.sum","1","ms"\n'
    )
    csv_path.write_bytes(original)
    os.link(csv_path, output_alias)

    result = deep_report.main(
        ["--ncu-csv", str(csv_path), "--output-json", str(output_alias)]
    )

    assert result == 1
    assert csv_path.read_bytes() == original
    assert output_alias.read_bytes() == original


def test_deep_report_rejects_symlink_output_alias_of_nsys_input(tmp_path) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    csv_path = tmp_path / "metrics.csv"
    nsys_path = tmp_path / "capture.nsys-rep"
    output_alias = tmp_path / "output.json"
    csv_path.write_text(
        '"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        '"1","kernel","gpu__time_duration.sum","1","ms"\n'
    )
    original = b"profile evidence"
    nsys_path.write_bytes(original)
    output_alias.symlink_to(nsys_path)

    result = deep_report.main(
        [
            "--ncu-csv",
            str(csv_path),
            "--nsys-report",
            str(nsys_path),
            "--output-json",
            str(output_alias),
        ]
    )

    assert result == 1
    assert nsys_path.read_bytes() == original


def test_nsys_summary_returns_nonzero_without_input(capsys) -> None:
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")

    result = nsys_summary.main([])

    assert result == 1
    assert "No Nsight Systems reports found." in capsys.readouterr().err


def test_nsys_summary_returns_nonzero_when_all_reports_fail(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")
    report_path = tmp_path / "broken.nsys-rep"
    report_path.touch()

    def fail_summary(*_args, **_kwargs):
        raise RuntimeError("mocked nsys stats failure")

    monkeypatch.setattr(
        nsys_summary.NsightSystemsReportParser,
        "summarize_report",
        staticmethod(fail_summary),
    )

    result = nsys_summary.main(["--report", str(report_path)])
    output = capsys.readouterr().out

    assert result == 2
    assert str(report_path.resolve()) in output
    assert "Source receipt:" in output
    assert '"status": "error"' in output
    assert "Failed to summarise: mocked nsys stats failure" in output


def test_nsys_summary_persists_parser_source_receipt(tmp_path, monkeypatch) -> None:
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")
    report_path = tmp_path / "capture.nsys-rep"
    output_path = tmp_path / "summary.txt"
    report_path.write_bytes(b"profile evidence")
    source = {
        "requested_path": str(report_path),
        "resolved_path": str(report_path.resolve()),
        "sha256": "a" * 64,
        "size_bytes": len(b"profile evidence"),
        "status": "parsed",
        "error": None,
    }
    monkeypatch.setattr(
        nsys_summary.NsightSystemsReportParser,
        "summarize_report",
        staticmethod(lambda *_args, **_kwargs: {"kernels": [], "source": source}),
    )

    result = nsys_summary.main(
        ["--report", str(report_path), "--output", str(output_path)]
    )

    assert result == 0
    text = output_path.read_text()
    assert "Source receipt:" in text
    assert f'"sha256": "{"a" * 64}"' in text
    assert '"status": "parsed"' in text


def test_nsys_summary_returns_nonzero_when_stats_times_out(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    core_nsys = importlib.import_module("core.profiling.nsight_systems")
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")
    report_path = tmp_path / "slow.nsys-rep"
    report_path.write_bytes(b"fixture")

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nsys stats", 60)

    monkeypatch.setattr(core_nsys.subprocess, "run", timeout)

    result = nsys_summary.main(["--report", str(report_path)])

    assert result == 2
    assert "timed out after 60s" in capsys.readouterr().out


def test_nsys_summary_returns_partial_failure_when_only_some_reports_parse(
    tmp_path,
    monkeypatch,
) -> None:
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")
    good = tmp_path / "good.nsys-rep"
    bad = tmp_path / "bad.nsys-rep"
    good.touch()
    bad.touch()

    def selective_summary(path, **_kwargs):
        if Path(path).name == "bad.nsys-rep":
            raise RuntimeError("broken")
        return {"kernels": []}

    monkeypatch.setattr(
        nsys_summary.NsightSystemsReportParser,
        "summarize_report",
        staticmethod(selective_summary),
    )

    assert nsys_summary.main(["--report", str(good), "--report", str(bad)]) == 3


def test_nsys_summary_preserves_explicit_missing_report_as_partial_failure(
    tmp_path,
    monkeypatch,
) -> None:
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")
    good = tmp_path / "good.nsys-rep"
    missing = tmp_path / "missing.nsys-rep"
    good.touch()

    def existing_only(path, **_kwargs):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return {"kernels": []}

    monkeypatch.setattr(
        nsys_summary.NsightSystemsReportParser,
        "summarize_report",
        staticmethod(existing_only),
    )

    assert nsys_summary.main(["--report", str(good), "--report", str(missing)]) == 3


def test_nsys_summary_rejects_output_collision_without_mutating_report(
    tmp_path,
) -> None:
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")
    report_path = tmp_path / "capture.nsys-rep"
    original = b"immutable report"
    report_path.write_bytes(original)

    result = nsys_summary.main(["--report", str(report_path), "--output", str(report_path)])

    assert result == 1
    assert report_path.read_bytes() == original


def test_nsys_summary_rejects_hard_link_output_alias_without_mutating_report(
    tmp_path,
) -> None:
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")
    report_path = tmp_path / "capture.nsys-rep"
    output_alias = tmp_path / "summary.txt"
    original = b"immutable report"
    report_path.write_bytes(original)
    os.link(report_path, output_alias)

    result = nsys_summary.main(
        ["--report", str(report_path), "--output", str(output_alias)]
    )

    assert result == 1
    assert report_path.read_bytes() == original
    assert output_alias.read_bytes() == original


@pytest.mark.parametrize("top_k", ["0", "-1"])
def test_profiling_clis_reject_nonpositive_top_k(top_k, capsys) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    nsys_summary = importlib.import_module("core.profiling.nsys_summary")

    assert deep_report.main(["--top-k", top_k]) == 1
    assert nsys_summary.main(["--top-k", top_k]) == 1
    assert "--top-k must be a positive integer" in capsys.readouterr().err


def test_deep_report_caches_nsys_failure_and_returns_nonzero(
    tmp_path,
    monkeypatch,
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    csv_path = tmp_path / "metrics.csv"
    nsys_path = tmp_path / "capture.nsys-rep"
    output_path = tmp_path / "report.json"
    csv_path.write_text(
        '"ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
        '"1","kernel","gpu__time_duration.sum","1","ms"\n'
        '"1","kernel","flop_count_sp","0","flop"\n'
        '"1","kernel","dram__bytes.sum","100","byte"\n'
    )
    nsys_path.touch()
    calls = 0

    def fail_once(_path, _top_k):
        nonlocal calls
        calls += 1
        return {"error": "fixture nsys failure"}

    monkeypatch.setattr(deep_report, "summarise_nsys", fail_once)

    result = deep_report.main(
        [
            "--ncu-csv",
            str(csv_path),
            "--nsys-report",
            str(nsys_path),
            "--output-json",
            str(output_path),
        ]
    )

    assert result == 3
    assert calls == 1
    assert json.loads(output_path.read_text())["nsight_systems"] == {
        "error": "fixture nsys failure"
    }


@pytest.mark.parametrize(
    ("value", "unit", "expected_bytes"),
    [
        (2.5, "Kbyte", 2_500.0),
        (1.25, "Gbyte", 1_250_000_000.0),
    ],
)
def test_hbm_byte_counters_normalize_decimal_nsight_units(
    value,
    unit,
    expected_bytes,
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")
    metrics = deep_report.KernelMetrics("scaled_hbm_bytes")
    metrics.metrics["dram__bytes.sum"] = deep_report.RawMetric(
        "dram__bytes.sum",
        value,
        unit,
    )

    assert deep_report.compute_hbm_bytes(metrics) == pytest.approx(expected_bytes)


@pytest.mark.parametrize(
    ("value", "unit", "expected_ms"),
    [
        (2_000_000.0, "nsecond", 2.0),
        (2_000.0, "usecond", 2.0),
    ],
)
def test_duration_metrics_normalize_explicit_nsight_units(
    value,
    unit,
    expected_ms,
) -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")

    assert deep_report.metric_in_unit(
        deep_report.RawMetric("gpu__time_duration.sum", value, unit)
    ) == pytest.approx(expected_ms)


def test_duration_metric_with_unknown_unit_fails_closed() -> None:
    deep_report = importlib.import_module("core.analysis.deep_profiling_report")

    assert (
        deep_report.metric_in_unit(deep_report.RawMetric("gpu__time_duration.sum", 2.0, "cycle"))
        is None
    )
    assert (
        deep_report.metric_in_unit(deep_report.RawMetric("gpu__time_duration.sum", 2.0, None))
        is None
    )
