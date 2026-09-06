from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from core.utils.python_entrypoints import (
    build_python_entry_command,
    build_repo_python_env,
    build_torchrun_entry_command,
    install_local_module_override,
    load_module_from_path,
)


def test_build_repo_python_env_prepends_repo_root_and_dedupes() -> None:
    repo_root = Path("/tmp/repo")
    env = build_repo_python_env(
        repo_root,
        base_env={"PYTHONPATH": f"/tmp/repo{os.pathsep}/existing"},
        extra_pythonpath=["/extra", "/existing"],
    )

    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(repo_root.resolve()),
        str(Path("/extra").resolve()),
        str(Path("/existing").resolve()),
    ]


def test_build_python_entry_command_supports_module_launch() -> None:
    cmd = build_python_entry_command(module_name="ch11.stream_overlap_demo", argv=["--help"])
    assert cmd == [sys.executable, "-m", "ch11.stream_overlap_demo", "--help"]


def test_build_python_entry_command_supports_script_launch(tmp_path: Path) -> None:
    script_path = tmp_path / "tool.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")

    cmd = build_python_entry_command(script_path=script_path, argv=["--flag"])
    assert cmd == [sys.executable, str(script_path.resolve()), "--flag"]


def test_build_python_entry_command_requires_exactly_one_target(tmp_path: Path) -> None:
    script_path = tmp_path / "tool.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")

    with pytest.raises(ValueError):
        build_python_entry_command()
    with pytest.raises(ValueError):
        build_python_entry_command(module_name="json.tool", script_path=script_path)


def test_build_torchrun_entry_command_supports_module_launch() -> None:
    cmd = build_torchrun_entry_command(
        module_name="ch15.tensor_parallel_demo",
        argv=["--batch", "1"],
        nproc_per_node=2,
        nnodes=1,
    )
    assert cmd == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--nnodes",
        "1",
        "-m",
        "ch15.tensor_parallel_demo",
        "--batch",
        "1",
    ]


def test_real_torchrun_workers_keep_python_when_path_has_another_launcher(tmp_path: Path) -> None:
    fake_bin = tmp_path / "other-environment-bin"
    fake_bin.mkdir()
    other_launcher = fake_bin / "torchrun"
    other_launcher.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    other_launcher.chmod(0o755)
    probe = tmp_path / "worker_identity.py"
    probe.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1], os.environ['RANK'] + '.json').write_text("
        "json.dumps({'executable': sys.executable, 'prefix': sys.prefix}))\n",
        encoding="utf-8",
    )
    with socket.socket() as rendezvous_socket:
        rendezvous_socket.bind(("127.0.0.1", 0))
        rendezvous_port = rendezvous_socket.getsockname()[1]
    command = build_torchrun_entry_command(
        script_path=probe, argv=[str(tmp_path)], nproc_per_node=2, nnodes=1,
        rdzv_backend="static", rdzv_endpoint=f"127.0.0.1:{rendezvous_port}",
    )
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ.get("PATH", ""))
    env.pop("PYTHON_EXEC", None)
    process = subprocess.Popen(
        command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"Distributed worker identity probe timed out:\n{output}")
    assert process.returncode == 0, output
    for rank in (0, 1):
        identity = json.loads((tmp_path / f"{rank}.json").read_text())
        assert identity == {"executable": sys.executable, "prefix": sys.prefix}


def test_install_local_module_override_loads_package_without_sys_path_mutation(tmp_path: Path) -> None:
    package_dir = tmp_path / "numba"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")

    original_sys_path = list(sys.path)
    sys.modules.pop("numba", None)
    try:
        module = install_local_module_override("numba", package_dir)
        assert module.VALUE == 7
        assert sys.path == original_sys_path
    finally:
        sys.modules.pop("numba", None)


def test_load_module_from_path_registers_module_for_dataclass_introspection(tmp_path: Path) -> None:
    module_path = tmp_path / "dataclass_module.py"
    module_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from dataclasses import dataclass",
                "",
                "@dataclass",
                "class Payload:",
                "    value: int",
                "",
                "RESULT = Payload(7)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    module_name = "tmp_dataclass_module"
    sys.modules.pop(module_name, None)
    try:
        module = load_module_from_path(module_name, module_path)
        assert module.RESULT.value == 7
        assert sys.modules[module_name] is module
    finally:
        sys.modules.pop(module_name, None)


def test_ch01_ch02_ch03_ch04_ch05_ch06_ch07_ch08_ch09_ch10_ch11_ch12_ch13_ch14_ch15_ch16_ch17_ch18_ch19_and_ch20_are_free_of_local_sys_path_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    offenders = []
    for chapter in ("ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "ch09", "ch10", "ch11", "ch12", "ch13", "ch14", "ch15", "ch16", "ch17", "ch18", "ch19", "ch20"):
        for path in sorted((repo_root / chapter).glob("*.py")):
            if "sys.path.insert" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(repo_root)))
    assert offenders == []


def test_selected_public_entrypoints_no_longer_carry_local_arch_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    paths = [
        "ch05/gpudirect_storage_example.py",
        "ch09/compare.py",
        "ch10/compare.py",
        "ch10/analyze_scaling.py",
        "ch10/cufile_gds_example.py",
        "ch13/compare.py",
        "ch13/custom_allocator.py",
        "ch13/memory_profiling.py",
        "ch13/fsdp_example.py",
        "ch13/train_deepseek_coder.py",
        "ch14/compare.py",
        "ch14/torch_compiler_examples.py",
        "ch14/triton_examples.py",
        "ch09/fusion_pytorch.py",
        "ch16/compare.py",
        "ch16/gpt_large_benchmark.py",
        "ch16/inference_profiling.py",
        "ch16/inference_server_load_test.py",
        "ch16/inference_serving_multigpu.py",
        "ch16/inference_optimizations_blackwell.py",
        "ch16/multi_gpu_validation.py",
        "ch16/perplexity_eval.py",
        "ch16/radix_attention_example.py",
        "ch16/synthetic_moe_inference_benchmark.py",
        "ch16/test_fp8_quantization_real.py",
        "ch16/vllm_monitoring.py",
        "ch17/blackwell_profiling_guide.py",
        "ch17/compare.py",
        "ch17/dynamic_routing.py",
        "ch17/early_rejection.py",
        "ch18/speculative_decode/spec_config_sweep.py",
        "ch15/moe_validation/moe_validation.py",
    ]
    for relpath in paths:
        text = (repo_root / relpath).read_text(encoding="utf-8")
        assert "sys.path.insert" not in text
        assert "from arch_config import ArchitectureConfig" not in text


def test_selected_public_lab_families_are_free_of_local_sys_path_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    offenders = []
    for lab in (
        "block_scaling",
        "blackwell_gemm_optimizations",
        "blackwell_matmul",
        "cudnn_sdpa_bench",
        "custom_vs_cublas",
        "decode_optimization",
        "dynamic_router",
        "flexattention",
        "flashattention4",
        "flashinfer_attention",
        "fullstack_cluster",
        "kv_optimization",
        "moe_cuda",
        "moe_optimization_journey",
        "moe_parallelism",
        "nanochat_fullstack",
        "nvfp4_dual_gemm",
        "nvfp4_gemm",
        "nvfp4_group_gemm",
        "occupancy_tuning",
        "parameterized_cuda_graphs",
        "real_world_models",
        "async_input_pipeline",
        "persistent_decode",
        "speculative_decode",
        "training_hotpath",
        "train_distributed",
        "trtllm_phi_3_5_moe",
        "uma_memory",
    ):
        for path in sorted((repo_root / "labs" / lab).rglob("*.py")):
            if "sys.path.insert" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(repo_root)))
    assert offenders == []


def test_selected_public_examples_and_monitoring_tools_are_free_of_local_sys_path_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    paths = [
        "examples/profiling_examples.py",
        "examples/mcp_client_example.py",
        "examples/optimize_examples.py",
        "monitoring/prometheus_exporter.py",
        "scripts/generate_mcp_docs.py",
        "tests/test_mcp_docs.py",
    ]
    for relpath in paths:
        text = (repo_root / relpath).read_text(encoding="utf-8")
        assert "sys.path.insert" not in text


def test_selected_infra_entrypoints_are_free_of_local_sys_path_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    paths = [
        "cli/aisp.py",
        "mcp/mcp_server.py",
        "scripts/linting/check_benchmarks.py",
        "tools/linting/check_benchmarks.py",
    ]
    for relpath in paths:
        text = (repo_root / relpath).read_text(encoding="utf-8")
        assert "sys.path.insert" not in text


def test_cluster_scripts_are_free_of_local_sys_path_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in sorted((repo_root / "cluster" / "scripts").glob("*.py")):
        if "sys.path.insert" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(repo_root)))
    assert offenders == []


def test_cluster_scripts_support_direct_and_module_help_entrypoints() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    direct = subprocess.run(
        [sys.executable, "cluster/scripts/lock_and_run.py", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert direct.returncode == 0
    assert "Lock GPU clocks and run a command" in direct.stdout

    module = subprocess.run(
        [sys.executable, "-m", "cluster.scripts.torchrun_connectivity_probe", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert module.returncode == 0
    assert "Fast torchrun NCCL connectivity probe" in module.stdout


def test_selected_test_modules_are_free_of_local_sys_path_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    paths = [
        "tests/test_benchmark_report_llm_transparency.py",
        "tests/test_discovery.py",
        "tests/test_benchmark_metrics.py",
        "tests/test_seed_config_immutability.py",
        "tests/test_benchmark_comparison.py",
        "tests/test_benchmark_correctness.py",
        "tests/test_benchmark_defaults.py",
        "tests/test_benchmark_harness.py",
        "tests/test_benchmark_regression.py",
        "tests/test_benchmark_verification.py",
        "tests/test_blackwell_optimizations.py",
        "tests/test_blackwell_stack.py",
        "tests/test_differential_profile_analyzer.py",
        "tests/test_get_benchmark_presence.py",
        "tests/test_harness_protection_failures.py",
        "tests/test_hta_analyzer.py",
        "tests/test_interface_consistency.py",
        "tests/test_metrics_extractor.py",
        "tests/test_microbench.py",
        "tests/test_occupancy_tuning.py",
        "tests/test_profiler_config.py",
        "tests/test_warp_specialization.py",
        "tests/test_run_benchmarks_cleaning.py",
        "tests/test_run_benchmarks_config_merge.py",
        "tests/test_profile_harness_entrypoints.py",
        "tests/test_update_custom_metrics.py",
        "tests/integration/test_bench_commands_manifest.py",
        "tests/integration/test_benchmark_discovery.py",
        "tests/integration/test_benchmark_execution.py",
        "tests/integration/test_ch20_multiple_all_techniques.py",
        "tests/integration/test_comparison_workflow.py",
        "tests/integration/test_metrics_collection.py",
    ]
    for relpath in paths:
        text = (repo_root / relpath).read_text(encoding="utf-8")
        assert "sys.path.insert" not in text


def test_selected_core_package_modules_are_free_of_local_sys_path_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    paths = [
        "core/analysis/analyze_results.py",
        "core/analysis/check_benchmark_alignment.py",
        "core/analysis/compare_benchmark_pairs.py",
        "core/analysis/deep_profiling_report.py",
        "core/analysis/differential_profile_analyzer.py",
        "core/analysis/hta_analyzer.py",
        "core/analysis/llm_profile_analyzer.py",
        "core/analysis/reporting/verification_report.py",
        "core/analysis/roofline_automation.py",
        "core/benchmark/async_input_pipeline_sweep.py",
        "core/optimization/auto/optimizer.py",
        "core/harness/benchmark_harness.py",
        "core/harness/isolated_runner.py",
        "core/harness/run_benchmarks.py",
        "core/harness/torchrun_wrapper.py",
        "core/profiling/memory_profiler.py",
        "core/profiling/cutlass_profiler_integration.py",
        "core/profiling/extract_ncu_metrics.py",
        "core/profiling/metrics_extractor.py",
        "core/profiling/profiler_wrapper.py",
        "core/profiling/profiling_runner.py",
        "core/profiling/nsys_summary.py",
        "core/scripts/alert_dependency_updates.py",
        "core/scripts/audit_verification_compliance.py",
        "core/scripts/ci/generate_quarantine_report.py",
        "core/scripts/generate_concept_mapping.py",
        "core/scripts/generate_file_registry.py",
        "core/scripts/harness/master_profile.py",
        "core/scripts/harness/profile_harness.py",
        "core/scripts/linting/check_benchmarks.py",
        "core/scripts/test_protection_effectiveness.py",
        "scripts/utilities/dump_hardware_capabilities.py",
        "scripts/utilities/precompile_cuda_extensions.py",
        "core/scripts/validate_imports.py",
        "core/profiling/nsight_automation.py",
        "core/utils/chapter_compare_template.py",
        "core/utils/extension_loader_template.py",
        "core/utils/extension_prewarm.py",
        "core/verification/review_baseline_optimized_pairs.py",
        "core/verification/verify_cutlass.py",
        "core/verification/verify_tma_sm121.py",
        "core/verification/verify_triton_blackwell_features.py",
        "core/verification/verify_triton_sm121_patch.py",
        "core/verification/verify_working_optimizations.py",
    ]
    for relpath in paths:
        text = (repo_root / relpath).read_text(encoding="utf-8")
        assert "sys.path.insert" not in text
