import json
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import mcp.mcp_server as mcp_server
import pytest
from core.harness.benchmark_harness import lock_gpu_clocks
from tests.integration._strict_gpu_env import skip_if_strict_benchmark_env_invalid

from core import (
    profile_artifacts,
    compile_analysis,
    costs,
    optimization_reports,
    optimization_stack,
    whatif,
    ncu_analysis,
    profile_insights,
)


def _skip_if_live_nsight_unavailable(*tools: str) -> None:
    skip_if_strict_benchmark_env_invalid()
    for tool in tools:
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} required for live profile comparison test")


def test_profile_artifacts_empty(tmp_path: Path):
    # With no traces, loaders return default structures and messages
    empty_root = tmp_path
    assert profile_artifacts.load_flame_graph_data(empty_root).get("message")
    assert profile_artifacts.load_memory_timeline(empty_root).get("message")
    assert profile_artifacts.load_cpu_gpu_timeline(empty_root) is not None
    assert profile_artifacts.load_kernel_breakdown({"children": []})["kernels"] == []


def test_compile_analysis_empty():
    result = compile_analysis.load_compile_analysis(Path.cwd(), [])
    assert "compile_benchmarks" in result
    assert isinstance(result.get("recommendations"), list)


def test_costs_and_roi_empty():
    cost = costs.calculate_costs([], {"name": "H100"})
    assert cost["current_rate"] > 0
    roi_result = optimization_reports.compute_roi([], cost)
    assert "techniques" in roi_result


def test_optimization_stack_fallbacks():
    # These should not raise even if advanced_analysis is missing
    assert optimization_stack.get_all_optimizations()
    assert optimization_stack.get_optimization_playbooks()
    assert optimization_stack.calculate_compound_optimization([], {}) is not None
    assert optimization_stack.get_optimal_optimization_stack(2.0, "medium", {}) is not None


def test_whatif_and_ncu_empty(tmp_path: Path):
    scenarios = whatif.get_scenarios()
    assert scenarios.get("scenarios")
    ncu = ncu_analysis.load_ncu_deepdive(tmp_path)
    assert "available" in ncu


def test_profile_insights_bottlenecks_and_score():
    flame_data = {
        "value": 100.0,
        "children": [
            {"name": "gpu_memcpy", "value": 30},
            {"name": "python_function", "value": 20},
            {"name": "overhead", "value": 6},
        ],
    }
    kernel_data = {
        "summary": {"total_time_us": 80},
        "kernels": [
            {"name": "gemm_kernel", "time_us": 20},
            {"name": "copy_kernel", "time_us": 15},
        ],
    }
    hw_caps = {
        "features": [
            {"name": "TMA Copy", "supported": True, "optimization": "Use async copies"},
            {"name": "FP8 Tensor Cores", "supported": True, "optimization": "Enable FP8"},
        ],
        "architecture": "blackwell",
        "gpu": {"name": "B200"},
    }

    result = profile_insights.detect_bottlenecks(flame_data, kernel_data, hw_caps)
    assert result["bottlenecks"], "Expected bottlenecks from synthetic data"

    score = profile_insights.calculate_optimization_score(hw_caps, result, kernel_data)
    assert 0 <= score["score"] <= 100
    assert score["quick_wins"], "Feature-based quick wins should be suggested"


def test_profile_insights_ncu_comparison_and_recommendations(tmp_path: Path):
    baseline_csv = tmp_path / "demo_baseline_ncu.csv"
    optimized_csv = tmp_path / "demo_optimized_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,50\noccupancy,40\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,70\noccupancy,45\n")

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    assert comparison.get("metrics"), "CSV-based NCU comparison should return metrics"
    staged_pair_dir = comparison.get("staged_pair_dir")
    assert staged_pair_dir, "NCU comparison should surface a concrete staged pair directory"
    assert Path(staged_pair_dir).exists(), "Staged pair directory should exist on disk"

    recs = profile_insights.generate_recommendations_from_profiles(
        {
            "ncu_comparison": comparison,
            "nsys_comparison": {"metrics": [{"name": "dram_util", "delta": -20}]},
        }
    )
    assert recs, "Recommendations should be produced from comparison data"


def test_profile_insights_nsys_comparison(tmp_path: Path):
    _skip_if_live_nsight_unavailable("nsys")
    script = tmp_path / "nvtx_script.py"
    script.write_text(
        (
            "import torch\n"
            "assert torch.cuda.is_available(), 'CUDA required for nsys test'\n"
            "x = torch.randn(1024, device='cuda')\n"
            "with torch.cuda.nvtx.range('nsys_test_range'):\n"
            "    y = x * 2\n"
            "torch.cuda.synchronize()\n"
            "print(float(y[0].item()))\n"
        ),
        encoding="utf-8",
    )

    baseline_prefix = tmp_path / "baseline_test"
    with lock_gpu_clocks():
        subprocess.run(
            [
                "nsys",
                "profile",
                "--force-overwrite=true",
                "-t",
                "cuda,nvtx,osrt",
                "-o",
                str(baseline_prefix),
                sys.executable,
                str(script),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

    baseline_rep = baseline_prefix.with_suffix(".nsys-rep")
    assert baseline_rep.exists()
    optimized_rep = tmp_path / "optimized_test.nsys-rep"
    optimized_rep.write_bytes(baseline_rep.read_bytes())

    comparison = profile_insights.compare_nsys_files(tmp_path)
    assert comparison is not None
    assert comparison.get("metrics"), "nsys comparison should yield metrics when NVTX ranges exist"


def test_profile_insights_nsys_requires_pair_key(tmp_path: Path):
    (tmp_path / "baseline_one.nsys-rep").write_text("stub", encoding="utf-8")
    (tmp_path / "optimized_one.nsys-rep").write_text("stub", encoding="utf-8")
    (tmp_path / "baseline_two.nsys-rep").write_text("stub", encoding="utf-8")
    (tmp_path / "optimized_two.nsys-rep").write_text("stub", encoding="utf-8")

    comparison = profile_insights.compare_nsys_files(tmp_path)
    assert comparison is not None
    assert comparison.get("error")
    assert comparison.get("candidates")


def test_profile_insights_ncu_comparison_from_rep(tmp_path: Path):
    _skip_if_live_nsight_unavailable("ncu")
    script = tmp_path / "ncu_script.py"
    script.write_text(
        (
            "import torch\n"
            "assert torch.cuda.is_available(), 'CUDA required for ncu test'\n"
            "x = torch.randn(512, 512, device='cuda')\n"
            "y = torch.randn(512, 512, device='cuda')\n"
            "z = x @ y\n"
            "torch.cuda.synchronize()\n"
            "print(float(z[0, 0]))\n"
        ),
        encoding="utf-8",
    )

    metrics = ",".join(
        [
            "gpu__time_duration.avg",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    )

    out_prefix = tmp_path / "ncu_test"
    with lock_gpu_clocks():
        subprocess.run(
            [
                "ncu",
                "--metrics",
                metrics,
                "--force-overwrite",
                "-o",
                str(out_prefix),
                sys.executable,
                str(script),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )

    rep = out_prefix.with_suffix(".ncu-rep")
    assert rep.exists()
    (tmp_path / "baseline_test.ncu-rep").write_bytes(rep.read_bytes())
    (tmp_path / "optimized_test.ncu-rep").write_bytes(rep.read_bytes())

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    assert comparison.get("kernel_comparison"), "ncu comparison should yield kernel comparisons"


def test_profile_insights_ncu_requires_pair_key(tmp_path: Path):
    baseline_csv = tmp_path / "pair_one_baseline_ncu.csv"
    optimized_csv = tmp_path / "pair_one_optimized_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,50\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,70\n")

    baseline_csv_two = tmp_path / "pair_two_baseline_ncu.csv"
    optimized_csv_two = tmp_path / "pair_two_optimized_ncu.csv"
    baseline_csv_two.write_text("Metric Name,Metric Value\nsm__throughput,40\n")
    optimized_csv_two.write_text("Metric Name,Metric Value\nsm__throughput,60\n")

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    assert comparison.get("error")
    assert comparison.get("candidates")


def test_profile_insights_ncu_pair_key_selects_csv(tmp_path: Path):
    baseline_csv = tmp_path / "pair_one_baseline_ncu.csv"
    optimized_csv = tmp_path / "pair_one_optimized_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,50\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,70\n")

    baseline_csv_two = tmp_path / "pair_two_baseline_ncu.csv"
    optimized_csv_two = tmp_path / "pair_two_optimized_ncu.csv"
    baseline_csv_two.write_text("Metric Name,Metric Value\nsm__throughput,40\n")
    optimized_csv_two.write_text("Metric Name,Metric Value\nsm__throughput,60\n")

    comparison = profile_insights.compare_ncu_files(
        tmp_path,
        pair_key="pair_one",
        include_ncu_details=True,
    )
    assert comparison is not None
    assert comparison.get("metrics")
    metrics = {m["name"]: m for m in comparison["metrics"]}
    assert metrics["sm__throughput"]["baseline"] == "50"
    assert metrics["sm__throughput"]["optimized"] == "70"
    assert "baseline_sources" not in comparison


def test_ncu_command_supports_nvtx_filters_and_profile_gate(tmp_path: Path):
    from core.profiling.nsight_automation import NsightAutomation

    automation = NsightAutomation(tmp_path)
    cmd = automation.build_ncu_command(
        command=[sys.executable, "-c", "print('ok')"],
        output_path=tmp_path / "demo.ncu-rep",
        workload_type="memory_bound",
        kernel_filter="kernel_cutlass",
        kernel_name_base="demangled",
        nvtx_includes=["cutlass_range"],
        profile_from_start="off",
        metric_set="minimal",
        launch_skip=5,
        launch_count=1,
        replay_mode="kernel",
    )

    assert "--kernel-name-base" in cmd
    assert "demangled" in cmd
    assert "--kernel-name" in cmd
    assert "kernel_cutlass" in cmd
    assert "--nvtx" in cmd
    assert "--nvtx-include" in cmd
    assert "cutlass_range" in cmd
    assert "cutlass_range/" in cmd
    assert "--profile-from-start" in cmd
    assert "off" in cmd


def test_nsight_automation_workdir_defaults_to_repo_root(tmp_path: Path):
    from core.profiling.nsight_automation import NsightAutomation

    automation = NsightAutomation(tmp_path)
    expected_root = Path(__file__).resolve().parents[1]
    assert automation.run_cwd == expected_root


def test_profile_artifact_materializes_symlink(tmp_path: Path):
    target = tmp_path / "real_profile.nsys-rep"
    target.write_text("profile-bytes", encoding="utf-8")
    symlink_path = tmp_path / "baseline_profile.nsys-rep"
    symlink_path.symlink_to(target)

    materialized = profile_insights._materialize_profile_if_needed(symlink_path, root=tmp_path)
    assert materialized != symlink_path
    assert materialized.exists()
    assert not materialized.is_symlink()
    assert materialized.read_text(encoding="utf-8") == "profile-bytes"


def test_profile_artifact_materialization_avoids_name_collisions(tmp_path: Path):
    target_a = tmp_path / "capture_a.ncu-rep"
    target_b = tmp_path / "capture_b.ncu-rep"
    target_a.write_text("A", encoding="utf-8")
    target_b.write_text("B", encoding="utf-8")

    links_a = tmp_path / "dir_a"
    links_b = tmp_path / "dir_b"
    links_a.mkdir(parents=True, exist_ok=True)
    links_b.mkdir(parents=True, exist_ok=True)
    symlink_a = links_a / "baseline.ncu-rep"
    symlink_b = links_b / "baseline.ncu-rep"
    symlink_a.symlink_to(target_a)
    symlink_b.symlink_to(target_b)

    materialized_a = profile_insights._materialize_profile_if_needed(symlink_a, root=tmp_path)
    materialized_b = profile_insights._materialize_profile_if_needed(symlink_b, root=tmp_path)

    assert materialized_a != materialized_b
    assert materialized_a.read_text(encoding="utf-8") == "A"
    assert materialized_b.read_text(encoding="utf-8") == "B"


def test_stage_profile_pair_writes_manifest(tmp_path: Path):
    baseline = tmp_path / "baseline_sample.ncu-rep"
    optimized = tmp_path / "optimized_sample.ncu-rep"
    baseline.write_text("baseline", encoding="utf-8")
    optimized.write_text("optimized", encoding="utf-8")

    staged_baseline, staged_optimized = profile_insights._stage_profile_pair(
        baseline,
        optimized,
        root=tmp_path,
        label="ncu",
    )
    assert staged_baseline.exists()
    assert staged_optimized.exists()
    manifest_path = staged_baseline.parent / "pair_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("pair_status") == "complete"
    assert manifest.get("artifacts", {}).get("baseline", {}).get("exists") is True
    assert manifest.get("artifacts", {}).get("optimized", {}).get("exists") is True


def test_assess_profile_pair_health_reports_missing_nsys(tmp_path: Path):
    (tmp_path / "baseline_only.ncu-rep").write_text("baseline", encoding="utf-8")
    (tmp_path / "optimized_only.ncu-rep").write_text("optimized", encoding="utf-8")

    health = profile_insights.assess_profile_pair_health(tmp_path)
    assert health["ok"] is False
    assert "nsys_pair" in health["missing"]
    assert health["has_any_ncu_pair"] is True


def test_compare_ncu_emits_pair_health_and_root_manifest(tmp_path: Path):
    baseline_csv = tmp_path / "demo_baseline_ncu.csv"
    optimized_csv = tmp_path / "demo_optimized_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,33\n", encoding="utf-8")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,44\n", encoding="utf-8")

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    assert "pair_health" in comparison
    assert comparison["pair_health"]["ok"] is False
    manifest_path = tmp_path / "pair_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("label") == "pair_health"
    assert manifest.get("availability", {}).get("has_ncu_csv_pair") is True


def test_kernel_alias_similarity_cutlass_vs_nvjet():
    baseline_kernel = (
        "cutlass3x_sm100_bstensorop_s256x192x64gemm_block_scaled_ue4m3xf4_"
        "ue4m3xf4_f32_f16_f16_256x192x256_0_tnn_align32_o_vs16_2sm_bias_f16_relu"
    )
    optimized_kernel = "nvjet_sm100_oohsh_128x256_256x5_2x2_2cta_h_bz_Avec16UE4M3_Bvec16UE4M3_TNT"
    assert profile_insights._kernel_similarity(baseline_kernel, optimized_kernel) >= 0.35


def test_parse_ps_for_defunct_launcher():
    from core.profiling.nsight_automation import NsightAutomation

    ps_output = (
        "1001 42 S /opt/nvidia/nsight-systems/target-linux-x64/nsys profile --output demo\n"
        "1002 42 Z [nsys-launcher] <defunct>\n"
        "1003 77 S python worker.py\n"
    )
    assert NsightAutomation._parse_ps_for_defunct_launcher(ps_output, parent_pid=42)
    assert not NsightAutomation._parse_ps_for_defunct_launcher(ps_output, parent_pid=77)


def test_nsight_automation_profile_nsys_defaults():
    from core.profiling.nsight_automation import NsightAutomation

    signature = inspect.signature(NsightAutomation.profile_nsys)
    assert signature.parameters["preset"].default == "light"
    assert signature.parameters["wait_mode"].default == "primary"
    assert signature.parameters["finalize_grace_seconds"].default == 20.0
    assert signature.parameters["sanitize_python_startup"].default is True


def test_nsight_automation_build_env_adds_startup_stub(tmp_path: Path):
    from core.profiling.nsight_automation import NsightAutomation

    automation = NsightAutomation(tmp_path)
    env = automation._build_env(sanitize_python_startup=True)  # type: ignore[attr-defined]
    pythonpath_entries = [entry for entry in env.get("PYTHONPATH", "").split(":") if entry]
    assert pythonpath_entries, "Expected PYTHONPATH to be populated"
    assert "aisp_profile_python_startup" in pythonpath_entries[0]
    assert env.get("PYTHONNOUSERSITE") == "1"
    startup_stub = Path(pythonpath_entries[0]) / "sitecustomize.py"
    user_stub = Path(pythonpath_entries[0]) / "usercustomize.py"
    assert startup_stub.exists()
    assert user_stub.exists()


def test_profile_nsys_timeout_accepts_late_finalized_report(tmp_path: Path, monkeypatch):
    import subprocess
    from core.profiling.nsight_automation import NsightAutomation

    automation = NsightAutomation(tmp_path)
    automation.nsys_available = True

    class _FakeProcess:
        def __init__(self) -> None:
            self.pid = 12345
            self.returncode = 124
            self._timed_out = False

        def communicate(self, timeout=None):
            if not self._timed_out:
                self._timed_out = True
                raise subprocess.TimeoutExpired(cmd="nsys", timeout=timeout, output="", stderr="")
            return ("", "")

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())
    monkeypatch.setattr(
        automation,
        "_finalize_timed_out_nsys",
        lambda process, grace_seconds: {
            "completed": False,
            "stdout": "",
            "stderr": "",
            "signals": [],
            "defunct_launcher_detected": False,
        },
    )

    def _late_report(output_path: Path, settle_seconds: float = 5.0, poll_interval: float = 0.2) -> bool:
        output_path.write_text("rep", encoding="utf-8")
        return True

    monkeypatch.setattr(automation, "_wait_for_output_artifact", _late_report)

    result = automation.profile_nsys(
        command=[sys.executable, "-c", "print('ok')"],
        output_name="late_finalize_demo",
        timeout_seconds=1,
    )

    assert result == tmp_path / "late_finalize_demo.nsys-rep"
    assert result.exists()
    assert automation.last_error is None
    assert automation.last_run["output"] == str(result)


def test_profile_nsys_nonzero_exit_with_usable_report_is_failure(tmp_path: Path, monkeypatch):
    import subprocess
    from core.profiling.nsight_automation import NsightAutomation

    automation = NsightAutomation(tmp_path)
    automation.nsys_available = True

    class _FakeProcess:
        def __init__(self) -> None:
            self.pid = 12345
            self.returncode = 7

        def communicate(self, timeout=None):
            _ = timeout
            return ("", "nsys exited oddly")

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())

    def _usable_report(output_path: Path, settle_seconds: float = 5.0, poll_interval: float = 0.2) -> bool:
        _ = settle_seconds, poll_interval
        output_path.write_text("rep", encoding="utf-8")
        return True

    monkeypatch.setattr(automation, "_wait_for_output_artifact", _usable_report)

    result = automation.profile_nsys(
        command=[sys.executable, "-c", "print('ok')"],
        output_name="nonzero_finalize_demo",
        timeout_seconds=30,
    )

    expected_report = tmp_path / "nonzero_finalize_demo.nsys-rep"
    assert result is None
    assert expected_report.exists()
    assert "returned non-zero status but produced a report artifact" in automation.last_error
    assert automation.last_run["output"] == str(expected_report)
    assert automation.last_run["artifact_on_failure"] == str(expected_report)
    assert automation.last_run["returncode"] == 7


def test_profile_nsys_zero_exit_waits_for_late_report(tmp_path: Path, monkeypatch):
    import subprocess
    from core.profiling.nsight_automation import NsightAutomation

    automation = NsightAutomation(tmp_path)
    automation.nsys_available = True

    class _FakeProcess:
        def __init__(self) -> None:
            self.pid = 12345
            self.returncode = 0

        def communicate(self, timeout=None):
            _ = timeout
            return ("", "")

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())

    def _late_report(output_path: Path, settle_seconds: float = 5.0, poll_interval: float = 0.2) -> bool:
        _ = settle_seconds, poll_interval
        output_path.write_text("rep", encoding="utf-8")
        return True

    monkeypatch.setattr(automation, "_wait_for_output_artifact", _late_report)

    result = automation.profile_nsys(
        command=[sys.executable, "-c", "print('ok')"],
        output_name="zero_exit_late_report",
        timeout_seconds=30,
    )

    assert result == tmp_path / "zero_exit_late_report.nsys-rep"
    assert result.exists()
    assert automation.last_error is None
    assert automation.last_run["output"] == str(result)
    assert automation.last_run["returncode"] == 0


def test_profile_nsys_mcp_schema_includes_safety_defaults():
    schema = mcp_server.TOOLS["profile_nsys"].input_schema
    props = schema["properties"]
    assert props["preset"]["default"] == "light"
    assert props["trace_forks"]["default"] is False
    assert props["wait_mode"]["default"] == "primary"
    assert props["finalize_grace_seconds"]["default"] == 20.0
    assert props["sanitize_python_startup"]["default"] is True


def test_profile_nsys_cli_defaults_match_safety_profile():
    import cli.aisp as aisp

    commands = {command.name: command for command in aisp.profile_app.registered_commands}
    nsys_cmd = commands["nsys"]
    signature = inspect.signature(nsys_cmd.callback)
    assert signature.parameters["preset"].default.default == "light"
    assert signature.parameters["wait_mode"].default.default == "primary"
    assert signature.parameters["finalize_grace_seconds"].default.default == 20.0
    assert signature.parameters["sanitize_python_startup"].default.default is True


def test_harness_nsys_paths_use_nsight_automation():
    from core.harness import run_benchmarks

    python_source = inspect.getsource(run_benchmarks.profile_python_benchmark)
    cuda_source = inspect.getsource(run_benchmarks.profile_cuda_executable)
    assert "NsightAutomation" in python_source
    assert "profile_nsys(" in python_source
    assert "NsightAutomation" in cuda_source
    assert "profile_nsys(" in cuda_source


def test_compare_tools_emit_pair_health_on_missing_pairs(tmp_path: Path):
    nsys_result = mcp_server.tool_compare_nsys({"profiles_dir": str(tmp_path)})
    assert nsys_result.get("success") is False
    assert "pair_health" in nsys_result

    ncu_result = mcp_server.tool_compare_ncu({"profiles_dir": str(tmp_path)})
    assert ncu_result.get("success") is False
    assert "pair_health" in ncu_result


def test_profile_compare_emits_pair_health_on_missing_profiles(tmp_path: Path):
    result = mcp_server.tool_profile_compare({"profiles_dir": str(tmp_path)})
    assert result.get("error")
    assert "pair_health" in result


def test_profile_compare_surfaces_metric_analysis_warning(tmp_path: Path):
    with (
        patch.object(profile_insights, "assess_profile_pair_health", return_value={"status": "ok"}),
        patch.object(profile_insights, "generate_flamegraph_comparison", return_value={"speedup": 1.0}),
        patch.object(profile_insights, "compare_nsys_files", return_value={"metrics": [{"name": "gpu", "delta": 1.0}]}),
        patch.object(profile_insights, "compare_ncu_files", return_value=None),
        patch.object(profile_insights, "generate_side_by_side_report", return_value={"success": True}),
        patch("core.perf_core_base.PerformanceCoreBase.compare_profiles", side_effect=RuntimeError("metric boom")),
    ):
        result = mcp_server.tool_profile_compare({"chapter": "ch11", "profiles_dir": str(tmp_path)})

    assert result.get("chapter") == "ch11"
    assert result.get("warnings")
    assert any("Metric analysis unavailable for profile_compare chapter=ch11: metric boom" in warning for warning in result["warnings"])


def test_collect_profile_role_files_materializes_role_symlinks_with_same_target(tmp_path: Path):
    target = tmp_path / "shared_capture.ncu-rep"
    target.write_text("shared", encoding="utf-8")

    baseline_link = tmp_path / "case_baseline.ncu-rep"
    optimized_link = tmp_path / "case_optimized.ncu-rep"
    baseline_link.symlink_to(target)
    optimized_link.symlink_to(target)

    baseline_files, optimized_files = profile_insights._collect_profile_role_files(  # type: ignore[attr-defined]
        tmp_path,
        ".ncu-rep",
    )
    assert baseline_files, "baseline role should be detected from symlink artifact names"
    assert optimized_files, "optimized role should be detected from symlink artifact names"
    assert all(not p.is_symlink() for p in baseline_files)
    assert all(not p.is_symlink() for p in optimized_files)


def test_profile_insights_ncu_csv_symlink_layout_materializes_and_compares(tmp_path: Path):
    baseline_target = tmp_path / "capture_a.csv"
    optimized_target = tmp_path / "capture_b.csv"
    baseline_target.write_text("Metric Name,Metric Value\nsm__throughput,12\n", encoding="utf-8")
    optimized_target.write_text("Metric Name,Metric Value\nsm__throughput,20\n", encoding="utf-8")

    baseline_link = tmp_path / "case_baseline_ncu.csv"
    optimized_link = tmp_path / "case_optimized_ncu.csv"
    baseline_link.symlink_to(baseline_target)
    optimized_link.symlink_to(optimized_target)

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    assert comparison.get("metrics")
    metrics = {m["name"]: m for m in comparison["metrics"]}
    assert metrics["sm__throughput"]["baseline"] == "12"
    assert metrics["sm__throughput"]["optimized"] == "20"


def test_profile_insights_ncu_role_aliases_base_opt(tmp_path: Path):
    baseline_csv = tmp_path / "tiny_case_base_ncu.csv"
    optimized_csv = tmp_path / "tiny_case_opt_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,11\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,19\n")

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    metrics = {m["name"]: m for m in comparison["metrics"]}
    assert metrics["sm__throughput"]["baseline"] == "11"
    assert metrics["sm__throughput"]["optimized"] == "19"


def test_profile_insights_ncu_single_role_pair_fallback(tmp_path: Path):
    baseline_csv = tmp_path / "alpha_baseline_ncu.csv"
    optimized_csv = tmp_path / "beta_optimized_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,37\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,53\n")

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    metrics = {m["name"]: m for m in comparison["metrics"]}
    assert metrics["sm__throughput"]["baseline"] == "37"
    assert metrics["sm__throughput"]["optimized"] == "53"


def test_profile_insights_ncu_parent_dir_role_detection(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    optimized_dir = tmp_path / "optimized"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    optimized_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = baseline_dir / "capture_a_ncu.csv"
    optimized_csv = optimized_dir / "capture_b_ncu.csv"
    baseline_csv.write_text("Metric Name,Metric Value\nsm__throughput,21\n")
    optimized_csv.write_text("Metric Name,Metric Value\nsm__throughput,29\n")

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    metrics = {m["name"]: m for m in comparison["metrics"]}
    assert metrics["sm__throughput"]["baseline"] == "21"
    assert metrics["sm__throughput"]["optimized"] == "29"


def test_profile_insights_ncu_two_file_mtime_fallback(tmp_path: Path):
    first = tmp_path / "capture_a_ncu.csv"
    second = tmp_path / "capture_b_ncu.csv"
    first.write_text("Metric Name,Metric Value\nsm__throughput,17\n")
    second.write_text("Metric Name,Metric Value\nsm__throughput,31\n")
    # Equal mtimes model filesystems whose timestamp resolution cannot
    # distinguish consecutive captures. The path tie-break stays stable.
    shared_mtime_ns = first.stat().st_mtime_ns
    os.utime(first, ns=(shared_mtime_ns, shared_mtime_ns))
    os.utime(second, ns=(shared_mtime_ns, shared_mtime_ns))

    comparison = profile_insights.compare_ncu_files(tmp_path)
    assert comparison is not None
    metrics = {m["name"]: m for m in comparison["metrics"]}
    assert metrics["sm__throughput"]["baseline"] == "17"
    assert metrics["sm__throughput"]["optimized"] == "31"


def test_profile_insights_ncu_role_fallback_prefers_pair_key_overlap(tmp_path: Path):
    baseline_alpha = tmp_path / "run_a_baseline_ncu.csv"
    baseline_beta = tmp_path / "run_b_baseline_ncu.csv"
    optimized_alpha = tmp_path / "candidate_a_optimized_ncu.csv"
    optimized_beta = tmp_path / "candidate_b_optimized_ncu.csv"

    baseline_alpha.write_text("Metric Name,Metric Value\nsm__throughput,10\n")
    baseline_beta.write_text("Metric Name,Metric Value\nsm__throughput,40\n")
    optimized_alpha.write_text("Metric Name,Metric Value\nsm__throughput,20\n")
    optimized_beta.write_text("Metric Name,Metric Value\nsm__throughput,80\n")

    comparison = profile_insights.compare_ncu_files(tmp_path, pair_key="run_b")
    assert comparison is not None
    assert comparison.get("pair_key") == "run_b"
    metrics = {m["name"]: m for m in comparison["metrics"]}
    assert metrics["sm__throughput"]["baseline"] == "40"
    assert metrics["sm__throughput"]["optimized"] == "80"


def test_profile_insights_role_detection_ignores_pair_dir_bias(tmp_path: Path):
    pair_dir = tmp_path / "pair__optimized_demo"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "example__baseline.ncu-rep").write_text("baseline", encoding="utf-8")
    (pair_dir / "example__optimized.ncu-rep").write_text("optimized", encoding="utf-8")

    baseline_files, optimized_files = profile_insights._collect_profile_role_files(pair_dir, ".ncu-rep")
    assert len(baseline_files) == 1
    assert len(optimized_files) == 1


def test_mcp_compare_tools_include_metrics(tmp_path: Path):
    _skip_if_live_nsight_unavailable("nsys", "ncu")
    script = tmp_path / "compare_script.py"
    script.write_text(
        (
            "import torch\n"
            "assert torch.cuda.is_available(), 'CUDA required for compare tool test'\n"
            "x = torch.randn(1024, device='cuda')\n"
            "with torch.cuda.nvtx.range('compare_tool_range'):\n"
            "    y = x * 2\n"
            "torch.cuda.synchronize()\n"
            "print(float(y[0].item()))\n"
        ),
        encoding="utf-8",
    )

    baseline_prefix = tmp_path / "baseline_compare"
    with lock_gpu_clocks():
        subprocess.run(
            [
                "nsys",
                "profile",
                "--force-overwrite=true",
                "-t",
                "cuda,nvtx,osrt",
                "-o",
                str(baseline_prefix),
                sys.executable,
                str(script),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    baseline_nsys = baseline_prefix.with_suffix(".nsys-rep")
    assert baseline_nsys.exists()
    optimized_nsys = tmp_path / "optimized_compare.nsys-rep"
    optimized_nsys.write_bytes(baseline_nsys.read_bytes())

    metrics = ",".join(
        [
            "gpu__time_duration.avg",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    )
    out_prefix = tmp_path / "ncu_compare"
    with lock_gpu_clocks():
        subprocess.run(
            [
                "ncu",
                "--metrics",
                metrics,
                "--force-overwrite",
                "-o",
                str(out_prefix),
                sys.executable,
                str(script),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    rep = out_prefix.with_suffix(".ncu-rep")
    assert rep.exists()
    (tmp_path / "baseline_compare.ncu-rep").write_bytes(rep.read_bytes())
    (tmp_path / "optimized_compare.ncu-rep").write_bytes(rep.read_bytes())

    nsys_result = mcp_server.tool_compare_nsys({"profiles_dir": str(tmp_path)})
    if (tmp_path / "baseline_compare.nsys-rep").exists() and (tmp_path / "optimized_compare.nsys-rep").exists():
        assert nsys_result.get("metrics"), "nsys comparison should include metrics"
    ncu_from_nsys = nsys_result.get("ncu_comparison")
    if (tmp_path / "baseline_compare.ncu-rep").exists() and (tmp_path / "optimized_compare.ncu-rep").exists():
        assert ncu_from_nsys, "nsys comparison should include ncu metrics when captured"
        assert ncu_from_nsys.get("kernel_comparison") or ncu_from_nsys.get("metrics")

    ncu_result = mcp_server.tool_compare_ncu({"profiles_dir": str(tmp_path)})
    if (tmp_path / "baseline_compare.ncu-rep").exists() and (tmp_path / "optimized_compare.ncu-rep").exists():
        assert ncu_result.get("kernel_comparison") or ncu_result.get("metrics"), "ncu comparison should include metrics"
    nsys_from_ncu = ncu_result.get("nsys_comparison")
    if (tmp_path / "baseline_compare.nsys-rep").exists() and (tmp_path / "optimized_compare.nsys-rep").exists():
        assert nsys_from_ncu, "ncu comparison should include nsys metrics when captured"
        assert nsys_from_ncu.get("metrics")

    profile_result = mcp_server.tool_profile_compare({"profiles_dir": str(tmp_path)})
    if (tmp_path / "baseline_compare.nsys-rep").exists() and (tmp_path / "optimized_compare.nsys-rep").exists():
        assert profile_result.get("nsys_comparison"), "profile compare should include nsys metrics when captured"
        assert profile_result["nsys_comparison"].get("metrics"), "profile compare should include nsys metric entries"
    if (tmp_path / "baseline_compare.ncu-rep").exists() and (tmp_path / "optimized_compare.ncu-rep").exists():
        assert profile_result.get("ncu_comparison"), "profile compare should include ncu metrics when captured"
        assert profile_result["ncu_comparison"].get("kernel_comparison") or profile_result["ncu_comparison"].get("metrics")


def test_profile_insights_normalizes_repeated_names():
    name = (
        "optimized_precisionfp8_pad_inner_matmul_optimized_"
        "optimized_precisionfp8_pad_inner_matmul.nsys-rep"
    )
    normalized = profile_insights._normalize_profile_name(name)
    assert normalized == "precisionfp8_pad_inner_matmul"
