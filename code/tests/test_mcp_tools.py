from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List
from unittest.mock import patch

import pytest

import json

import mcp.mcp_server as mcp_server


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
ARTIFACT_DIR = ARTIFACT_RUNS_DIR
E2E_RUNS_DIR = REPO_ROOT / "artifacts" / "e2e_runs"
MICROBENCH_DIR = REPO_ROOT / "artifacts" / "mcp-microbench"
REPORT_OUTPUT = REPO_ROOT / "artifacts" / "mcp-report.pdf"
EXPORT_OUTPUT = REPO_ROOT / "artifacts" / "mcp-export.csv"
BENCH_FILE = REPO_ROOT / "benchmark_test_results.json"
PROFILE_FIXTURE_DIR = ARTIFACT_RUNS_DIR / "mcp-fixtures" / "profiles" / "bench" / "ch04"
QUEUE_DIR = REPO_ROOT / "artifacts" / "parallel_runs"
E2E_STATUS_FIXTURE_RUN_ID = "mcp_e2e_status_smoke"
NSYS_SAMPLE = PROFILE_FIXTURE_DIR / "baseline_nccl_baseline.nsys-summary.csv"
NCU_SAMPLE = PROFILE_FIXTURE_DIR / "baseline_nvlink_baseline.ncu-rep"
NSYS_REP_FIXTURE = PROFILE_FIXTURE_DIR / "baseline_fixture.nsys-rep"
NCU_REP_FIXTURE = PROFILE_FIXTURE_DIR / "baseline_fixture.ncu-rep"
NCU_RAW_CSV_SAMPLE = PROFILE_FIXTURE_DIR / "baseline_nvlink_baseline.ncu-raw.csv"


@dataclass(frozen=True)
class ToolCase:
    name: str
    params: Dict[str, Any]
    category: str
    slow: bool = False
    timeout: int = 15


CATEGORY_TOOLS: Dict[str, List[str]] = {
    "gpu": [
        "gpu_info",
        "gpu_bandwidth",
        "gpu_topology",
        "gpu_topology_matrix",
        "gpu_power",
    ],
    "system": [
        "system_software",
        "system_dependencies",
        "system_context",
        "system_capabilities",
        "system_parameters",
        "system_container",
        "system_cpu_memory",
        "system_env",
        "system_network",
        "system_full",
        "clock_lock_check",
    ],
    "info": [
        "info_features",
    ],
    "benchmarking": [
        "benchmark_contracts",
        "render_benchmark_run",
        "benchmark_targets",
        "list_chapters",
        "run_benchmarks",
        "benchmark_e2e_sweep",
        "benchmark_e2e_status",
        "benchmark_e2e_watch",
        "benchmark_variants",
        "benchmark_explore",
        "benchmark_deep_dive_compare",
        "benchmark_llm_patch_loop",
        "benchmark_report",
        "benchmark_export",
        "benchmark_compare_runs",
        "benchmark_triage",
        "benchmark_data",
        "benchmark_overview",
        "benchmark_opportunities",
        "benchmark_history",
        "benchmark_trends",
        "benchmark_compare",
    ],
    "analysis": [
        "analyze_bottlenecks",
        "analyze_pareto",
        "analyze_scaling",
        "analyze_stacking",
        "analyze_whatif",
        "analyze_comm_overlap",
        "analyze_memory_patterns",
        "analyze_dataloader",
        "analyze_energy",
        "predict_scaling",
    ],
    "optimization": [
        "optimize",
        "recommend",
        "optimize_roi",
        "optimize_techniques",
    ],
    "distributed": [
        "distributed_plan",
        "distributed_nccl",
        "launch_plan",
    ],
    "inference": [
        "inference_vllm",
        "inference_quantization",
        "inference_deploy",
        "inference_estimate",
    ],
    "ai_llm": [
        "ask",
        "explain",
        "ai_status",
        "ai_troubleshoot",
    ],
    "profiling": [
        "profile_flame",
        "profile_memory",
        "profile_kernels",
        "profile_roofline",
        "profile_nsys",
        "profile_ncu",
        "profile_torch",
        "profile_hta",
        "profile_compare",
        "compare_nsys",
        "compare_ncu",
        "nsys_summary",
        "ncu_summary",
        "profile_list_profiles",
        "profile_compile_analysis",
    ],
    "exports": [
        "export_csv",
        "export_pdf",
        "export_html",
    ],
    "hw": [
        "hw_speed",
        "hw_roofline",
        "hw_disk",
        "hw_pcie",
        "hw_cache",
        "hw_tc",
        "hw_network",
        "hw_ib",
        "hw_nccl",
        "hw_p2p",
    ],
    "huggingface": [
        "hf",
    ],
    "cluster_cost": [
        "cluster_slurm",
        "cost_estimate",
        "cluster_eval_suite",
        "cluster_common_eval",
        "cluster_fabric_eval",
        "cluster_nmx_partition_lab",
        "cluster_build_canonical_package",
        "cluster_promote_run",
        "cluster_watch_promote",
        "cluster_validate_field_report",
    ],
    "tools": [
        "tools_kv_cache",
        "tools_cost_per_token",
        "tools_compare_precision",
        "tools_detect_cutlass",
        "tools_dump_hw",
        "tools_probe_hw",
    ],
    "utility": [
        "status",
        "context_summary",
        "context_full",
        "triage",
        "job_status",
        "suggest_tools",
    ],
}

SLOW_TOOLS = {
    "gpu_bandwidth",
    "run_benchmarks",
    "optimize",
    "benchmark_variants",
    "benchmark_deep_dive_compare",
    "benchmark_llm_patch_loop",
    "profile_nsys",
    "profile_ncu",
    "profile_torch",
    "profile_hta",
    "profile_flame",
    "profile_memory",
    "profile_kernels",
    "profile_roofline",
    "compare_nsys",
    "compare_ncu",
    "hw_speed",
    "hw_roofline",
    "hw_disk",
    "hw_pcie",
    "hw_cache",
    "hw_tc",
    "hw_nccl",
    "hw_ib",
    "hw_p2p",
    "cluster_fabric_eval",
}

BENCHMARK_SLOW_TOOLS = {
    "run_benchmarks",
    "optimize",
    "benchmark_variants",
    "benchmark_explore",
    "benchmark_deep_dive_compare",
    "benchmark_llm_patch_loop",
}

TOOL_PARAMS: Dict[str, Dict[str, Any]] = {
    "optimize": {
        "target": "ch10:atomic_reduction",
        "profile": "minimal",
        "iterations": 1,
        "warmup": 5,
        "llm_analysis": False,
        "force_llm": False,
        "apply_patches": False,
        "rebenchmark_llm_patches": False,
    },
    "run_benchmarks": {
        "targets": ["ch10:atomic_reduction"],
        "profile": "minimal",
        "iterations": 1,
        "warmup": 5,
        "llm_analysis": False,
    },
    "benchmark_opportunities": {
        "data_file": str(BENCH_FILE),
        "top": 3,
        "include_context": False,
    },
    "benchmark_e2e_sweep": {
        "run_tier1": False,
        "run_full_sweep": False,
        "run_cluster": False,
        "run_fabric": False,
        "dry_run": True,
    },
    "benchmark_e2e_status": {
        "run_id": "mcp_e2e_status_smoke",
        "recent_events": 2,
    },
    "benchmark_e2e_watch": {
        "run_id": "mcp_e2e_status_smoke",
        "poll_interval_seconds": 5,
        "max_auto_resumes": 0,
    },
    "benchmark_variants": {
        "targets": ["ch10:atomic_reduction"],
        "profile": "minimal",
        "iterations": 1,
        "warmup": 5,
        "llm_analysis": False,
        "force_llm": False,
        "apply_patches": False,
        "rebenchmark_llm_patches": False,
    },
    "benchmark_explore": {
        "path": "ch10/baseline_atomic_reduction.py",
        "deep_dive": "never",
        "iterations": 1,
        "warmup": 5,
        "timeout_seconds": 900,
    },
    "benchmark_report": {
        "data_file": str(BENCH_FILE),
        "output": str(REPORT_OUTPUT),
        "format": "pdf",
        "title": "MCP Report",
        "author": "MCP Tests",
    },
    "benchmark_export": {
        "data_file": str(BENCH_FILE),
        "format": "csv",
        "output": str(EXPORT_OUTPUT),
    },
    "benchmark_compare_runs": {
        "baseline": str(BENCH_FILE),
        "candidate": str(BENCH_FILE),
        "top": 3,
    },
    "analyze_whatif": {"max_vram_gb": 24, "max_latency_ms": 50, "include_context": False},
    "recommend": {"model_size": 7, "gpus": 1, "goal": "throughput", "include_context": False},
    "distributed_plan": {"model_size": 7, "gpus": 4, "nodes": 1, "include_context": False},
    "distributed_nccl": {"nodes": 1, "gpus": 4, "include_context": False},
    "launch_plan": {"nodes": 1, "gpus_per_node": 2, "script": "train.py"},
    "inference_vllm": {"model": "7b", "model_size": 7, "target": "throughput", "include_context": False},
    "inference_deploy": {"model": "7b", "model_size": 7, "goal": "throughput", "include_context": False},
    "inference_estimate": {"model": "7b", "model_size": 7, "goal": "throughput", "include_context": False},
    "inference_quantization": {"model_size": 7, "include_context": False},
    "ask": {"question": "What is tensor parallelism?", "include_context": False},
    "explain": {"concept": "warp divergence", "include_context": False},
    "profile_nsys": {
        "command": ["python", "-c", "print('nsys')"],
        "output_name": "mcp_nsys_test",
        "output_dir": str(ARTIFACT_RUNS_DIR),
        "run_id": "mcp_nsys_test",
        "preset": "light",
        "full_timeline": False,
        "trace_forks": False,
        "include_context": False,
    },
    "profile_ncu": {
        "command": ["python", "-c", "print('ncu')"],
        "output_name": "mcp_ncu_test",
        "output_dir": str(ARTIFACT_RUNS_DIR),
        "run_id": "mcp_ncu_test",
        "workload_type": "memory_bound",
        "include_context": False,
    },
    "compare_nsys": {"profiles_dir": str(PROFILE_FIXTURE_DIR), "include_context": False},
    "compare_ncu": {"profiles_dir": str(PROFILE_FIXTURE_DIR), "include_context": False},
    "nsys_summary": {"report_path": str(NSYS_SAMPLE), "include_context": False},
    "ncu_summary": {"report_path": str(NCU_RAW_CSV_SAMPLE), "top_k": 3, "include_context": False},
    "profile_list_profiles": {"include_context": False},
    "profile_compile_analysis": {"include_context": False},
    "export_csv": {"detailed": False, "include_context": False},
    "export_pdf": {"include_context": False},
    "export_html": {"include_context": False},
    "hw_speed": {"gemm_size": 256, "mem_size_mb": 8, "mem_stride": 64, "include_context": False},
    "hw_roofline": {"size_mb": 8, "strides": [64, 128]},
    "hw_disk": {"file_size_mb": 8, "block_size_kb": 128, "tmp_dir": str(MICROBENCH_DIR)},
    "hw_pcie": {"size_mb": 8, "iters": 1},
    "hw_cache": {"size_mb": 8, "stride": 64},
    "hw_tc": {"size": 512, "precision": "fp16"},
    "hw_ib": {"size_mb": 64},
    "hw_nccl": {"collective": "all_reduce", "gpus": 2},
    "hw_p2p": {"size_mb": 64},
    "info_features": {},
    "profile_compare": {"chapter": "ch11"},
    "benchmark_deep_dive_compare": {
        "path": "ch10/baseline_atomic_reduction.py",
        "output_dir": str(REPO_ROOT / "artifacts" / "mcp-deep-dive-tests"),
        "iterations": 1,
        "warmup": 5,
        "timeout_seconds": 900,
        "validity_profile": "portable",
    },
    "benchmark_llm_patch_loop": {
        "targets": ["ch10:atomic_reduction"],
        "output_dir": str(REPO_ROOT / "artifacts" / "mcp-llm-loop-tests"),
        "compare_output_dir": str(REPO_ROOT / "artifacts" / "mcp-llm-loop-compare-tests"),
        "iterations": 1,
        "warmup": 5,
        "compare_iterations": 1,
        "compare_warmup": 5,
        "force_llm": True,
        "llm_explain": True,
    },
    "profile_torch": {
        "script": str(REPO_ROOT / "tests" / "fixtures" / "mcp_torch_profile_target.py"),
        "output_dir": str(ARTIFACT_RUNS_DIR),
        "run_id": "mcp_torch_test",
    },
    "profile_hta": {
        "command": ["python", "-c", "print('hta')"],
        "output_dir": str(ARTIFACT_RUNS_DIR),
        "run_id": "mcp_hta_test",
    },
    "hf": {"action": "search", "query": "llama", "limit": 3},
    "cluster_slurm": {"model": "7b", "nodes": 1, "gpus": 2},
    "cost_estimate": {"gpu_type": "h100", "num_gpus": 1, "hours_per_day": 1},
    "cluster_eval_suite": {"mode": "smoke", "run_id": "mcp_cluster_smoke_test", "include_context": False},
    "cluster_common_eval": {
        "preset": "multinode-readiness",
        "run_id": "mcp_cluster_smoke_test",
        "hosts": ["localhost"],
        "labels": ["localhost"],
        "include_context": False,
    },
    "cluster_fabric_eval": {
        "run_id": "mcp_cluster_smoke_test",
        "hosts": ["localhost"],
        "labels": ["localhost"],
        "extra_args": [
            "--skip-bootstrap-nodes",
            "--disable-fp4",
            "--health-suite",
            "off",
            "--skip-vllm-multinode",
            "--model",
            "openai-community/gpt2",
            "--tp",
            "1",
            "--isl",
            "128",
            "--osl",
            "64",
            "--concurrency-range",
            "1 2",
            "--vllm-request-rate-range",
            "1 2",
            "--vllm-request-rate-max-concurrency",
            "4",
            "--vllm-request-rate-num-prompts",
            "40",
            "--fio-runtime",
            "5",
            "--nvbandwidth-quick",
            "--skip-render-localhost-report",
        ],
        "include_context": False,
    },
    "cluster_nmx_partition_lab": {
        "nmx_url": "https://nmx.example",
        "alpha_name": "AlphaPartition",
        "beta_name": "BetaPartition",
        "include_context": False,
    },
    "cluster_build_canonical_package": {
        "canonical_run_id": "mcp_cluster_smoke_test",
        "output_dir": str(ARTIFACT_DIR / "cluster_pkg_smoke"),
        "include_context": False,
    },
    "cluster_promote_run": {
        "run_id": "mcp_cluster_smoke_test",
        "skip_render_localhost_report": True,
        "skip_validate_localhost_report": True,
        "include_context": False,
    },
    "cluster_watch_promote": {
        "run_id": "mcp_cluster_smoke_test_missing_run",
        "pid": 999999,
        "include_context": False,
    },
    "cluster_validate_field_report": {"canonical_run_id": "mcp_cluster_smoke_test", "include_context": False},
    "clock_lock_check": {"devices": [0], "include_context": False},
    "suggest_tools": {"query": "profile this model", "llm_routing": False},
    "job_status": {"job_id": "test_job_missing"},
    "benchmark_data": {"page": 1, "page_size": 10},
    "benchmark_overview": {},
    "benchmark_history": {},
    "benchmark_trends": {},
    "benchmark_compare": {"baseline": str(BENCH_FILE), "candidate": str(BENCH_FILE), "top": 3},
    "benchmark_contracts": {},
    "render_benchmark_run": {
        "name": "mcp-lockstep-run",
        "benchmarkClass": "publication_grade",
        "workloadType": "inference",
        "cadence": "nightly",
    },
    "ai_troubleshoot": {"issue": "NCCL timeout", "symptoms": ["timeout"], "config": {"gpus": 4}},
}


def _build_cases() -> List[ToolCase]:
    cases: List[ToolCase] = []
    for category, tools in CATEGORY_TOOLS.items():
        for name in tools:
            params = TOOL_PARAMS.get(name, {})
            cases.append(
                ToolCase(
                    name=name,
                    params=params,
                    category=category,
                    slow=name in SLOW_TOOLS,
                    timeout=900 if name in BENCHMARK_SLOW_TOOLS else 600 if name in SLOW_TOOLS else 60,
                )
            )
    return cases


ALL_TOOL_CASES = _build_cases()
SLOW_TOOL_CASES = [case for case in ALL_TOOL_CASES if case.slow]


@pytest.fixture(scope="module", autouse=True)
def prepare_artifacts() -> None:
    generated_copy_globs = [
        REPO_ROOT / "ch10" / "baseline_atomic_reduction_mcp_copy*.py",
        REPO_ROOT / "ch10" / "baseline_atomic_reduction_mcp_copy*.cu",
    ]
    existing_generated_copies = {
        path.resolve()
        for pattern in generated_copy_globs
        for path in pattern.parent.glob(pattern.name)
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    E2E_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MICROBENCH_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    e2e_run_dir = E2E_RUNS_DIR / E2E_STATUS_FIXTURE_RUN_ID
    e2e_run_dir.mkdir(parents=True, exist_ok=True)
    (e2e_run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": E2E_STATUS_FIXTURE_RUN_ID,
                "run_state": "completed",
                "overall_status": "succeeded",
                "updated_at": "2026-03-27T00:00:00Z",
                "resume_available": False,
                "stages": [],
                "contract": {
                    "run_tier1": False,
                    "run_full_sweep": False,
                    "run_cluster": False,
                    "run_fabric": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (e2e_run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "run_id": E2E_STATUS_FIXTURE_RUN_ID,
                "run_state": "completed",
                "overall_status": "succeeded",
                "updated_at": "2026-03-27T00:00:00Z",
                "resume_available": False,
                "stages": [],
                "contract": {
                    "run_tier1": False,
                    "run_full_sweep": False,
                    "run_cluster": False,
                    "run_fabric": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (e2e_run_dir / "progress.json").write_text(
        json.dumps(
            {
                "run_id": E2E_STATUS_FIXTURE_RUN_ID,
                "current": {
                    "timestamp": "2026-03-27T00:00:00+00:00",
                    "metrics": {
                        "run_state": "completed",
                        "overall_status": "succeeded",
                        "stages": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (e2e_run_dir / "events.jsonl").write_text(
        json.dumps({"ts": "2026-03-27T00:00:00Z", "event": "run_finished", "run_id": E2E_STATUS_FIXTURE_RUN_ID})
        + "\n",
        encoding="utf-8",
    )
    nsys_csv = PROFILE_FIXTURE_DIR / "baseline_nccl_baseline.nsys-summary.csv"
    if not nsys_csv.exists():
        nsys_csv.write_text(
            "Section,Metric Name,Metric Value,Time (%)\n"
            "NVTX,range0,123,45.6\n"
        )
    ncu_raw_csv = PROFILE_FIXTURE_DIR / "baseline_nvlink_baseline.ncu-raw.csv"
    if not ncu_raw_csv.exists():
        ncu_raw_csv.write_text(
            "\"ID\",\"Kernel Name\",\"Block Size\",\"Grid Size\",\"gpu__time_duration.avg\",\"gpu__time_duration.sum\","
            "\"sm__throughput.avg.pct_of_peak_sustained_elapsed\",\"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed\","
            "\"lts__throughput.avg.pct_of_peak_sustained_elapsed\",\"sm__warps_active.avg.pct_of_peak_sustained_active\","
            "\"launch__registers_per_thread\",\"launch__shared_mem_per_block\",\"launch__occupancy_limit_blocks\","
            "\"launch__occupancy_limit_registers\",\"launch__occupancy_limit_shared_mem\",\"launch__occupancy_limit_warps\"\n"
            "\"\",\"\",\"\",\"\",\"us\",\"us\",\"%\",\"%\",\"%\",\"%\",\"register/thread\",\"Kbyte/block\",\"block\",\"block\",\"block\",\"block\"\n"
            "\"0\",\"kernelA\",\"(256,1,1)\",\"(1024,1,1)\",\"10.0\",\"100.0\",\"80.0\",\"70.0\",\"60.0\",\"50.0\",\"64\",\"12.0\",\"16\",\"8\",\"8\",\"4\"\n"
            "\"1\",\"kernelB\",\"(128,1,1)\",\"(2048,1,1)\",\"5.0\",\"50.0\",\"60.0\",\"50.0\",\"40.0\",\"30.0\",\"80\",\"8.0\",\"8\",\"4\",\"4\",\"2\"\n"
        )
    yield

    current_generated_copies = {
        path.resolve()
        for pattern in generated_copy_globs
        for path in pattern.parent.glob(pattern.name)
    }
    for path in sorted(current_generated_copies - existing_generated_copies):
        path.unlink(missing_ok=True)


@pytest.fixture()
def server():
    return mcp_server.MCPServer()


def _payload_from_result(result: mcp_server.ToolResult) -> Dict[str, Any]:
    assert result.content, "Tool response must include content"
    entry = result.content[0]
    ctype = entry.get("type")
    if ctype == "text":
        payload = json.loads(entry.get("text"))
    elif ctype == "application/json":
        payload = entry.get("json")
    else:
        raise AssertionError(f"Unexpected content type: {ctype}")
    assert isinstance(payload, dict), "Payload must be a JSON object"
    return payload


def _call_with_timeout(server: mcp_server.MCPServer, case: ToolCase) -> mcp_server.ToolResult:
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(server.call_tool, case.name, case.params)
        return fut.result(timeout=case.timeout)


def _wait_for_job_terminal_status(
    server: mcp_server.MCPServer,
    job_id: str,
    timeout_seconds: float = 120.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = server.call_tool("job_status", {"job_id": job_id})
        payload = _payload_from_result(result)
        record = payload.get("result", {})
        if isinstance(record, dict) and record.get("status") in {"completed", "error", "not_found"}:
            return record
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach terminal state within {timeout_seconds}s")


def _case_ids(cases: Iterable[ToolCase]) -> List[str]:
    return [case.name for case in cases]


def test_expected_tool_registration_matches_catalog():
    expected = {case.name for case in ALL_TOOL_CASES}
    registered = set(mcp_server.TOOLS.keys())
    assert expected == registered, "Tool catalog must mirror MCP server registry"
    assert 100 <= len(expected) <= 120


def test_optimize_path_resolution():
    from mcp.mcp_server import _resolve_benchmark_target_from_path

    path = REPO_ROOT / "ch10" / "baseline_atomic_reduction.py"
    target, err = _resolve_benchmark_target_from_path(str(path))
    assert err is None
    assert target == "ch10:atomic_reduction"


def test_profile_command_normalizes_repo_relative_script():
    expected_script = REPO_ROOT / "tests" / "fixtures" / "mcp_torch_profile_target.py"
    command, error = mcp_server._normalize_profile_command(  # type: ignore[attr-defined]
        ["python", "tests/fixtures/mcp_torch_profile_target.py"]
    )
    assert error is None
    assert command[1] == str(expected_script)


def test_profile_command_rejects_missing_repo_relative_script():
    command, error = mcp_server._normalize_profile_command(  # type: ignore[attr-defined]
        ["python", "tests/fixtures/does_not_exist.py"]
    )
    assert command[1] == "tests/fixtures/does_not_exist.py"
    assert error is not None
    assert "command script not found" in error


def test_tool_list_protocol_matches_registration(server: mcp_server.MCPServer):
    tool_list = server.get_tool_list()
    names = {tool["name"] for tool in tool_list}
    expected = {case.name for case in ALL_TOOL_CASES}
    assert names == expected


@pytest.mark.parametrize(
    ("query", "expected_tool"),
    [
        ("optimize ch10/baseline_atomic_reduction.py", "optimize"),
        ("tune cuda kernel occupancy", "profile_ncu"),
        ("torch.compile graph breaks", "profile_torch"),
        ("autotune tile sizes", "benchmark_variants"),
        ("compare nsys reports", "compare_nsys"),
        ("compare benchmark runs", "benchmark_compare_runs"),
        ("memory coalescing analysis", "analyze_memory_patterns"),
            ("slurm script for training", "cluster_slurm"),
            ("check cuda version", "system_software"),
            ("kv cache size", "tools_kv_cache"),
            ("benchmark history", "benchmark_history"),
            ("performance trends over time", "benchmark_trends"),
            ("gpu temperature", "gpu_power"),
            ("huggingface search llama", "hf"),
            ("export csv here", "export_csv"),
            ("network status ib", "system_network"),
            ("topology matrix", "gpu_topology_matrix"),
            ("cloud cost estimate", "cost_estimate"),
            ("llm status", "ai_status"),
            ("profile and compare @ch09/baseline_cublas_gemm_fp4_perchannel.py and optimized version", "benchmark_deep_dive_compare"),
    ],
)
def test_suggest_tools_common_intents(server: mcp_server.MCPServer, query: str, expected_tool: str):
    result = server.call_tool("suggest_tools", {"query": query, "llm_routing": False})
    payload = _payload_from_result(result)
    tool_result = payload.get("result") or {}
    suggestions = tool_result.get("suggestions") or []
    tools = {entry.get("tool") for entry in suggestions if isinstance(entry, dict)}
    assert expected_tool in tools


def test_suggest_tools_llm_fallback_warns(server: mcp_server.MCPServer) -> None:
    result = server.call_tool("suggest_tools", {"query": "profile and compare", "llm_routing": True})
    payload = _payload_from_result(result)
    tool_result = payload.get("result") or {}
    routing = tool_result.get("routing")
    warning = tool_result.get("warning", "")
    suggestions = tool_result.get("suggestions") or []

    assert routing in {"llm", "heuristic"}
    assert suggestions, "suggest_tools should return at least one suggestion"
    if routing == "heuristic":
        assert "WARNING" in warning


def test_suggest_tools_heuristic_warns(server: mcp_server.MCPServer) -> None:
    result = server.call_tool("suggest_tools", {"query": "profile this model", "llm_routing": False})
    payload = _payload_from_result(result)
    tool_result = payload.get("result") or {}
    warning = tool_result.get("warning", "")

    assert "WARNING" in warning


def test_tool_response_is_text_only(server: mcp_server.MCPServer):
    """MCP responses must emit only text content to satisfy clients that reject other types."""
    result = server.call_tool("status", {})
    assert isinstance(result.content, list)
    assert len(result.content) == 1, "MCP content should contain exactly one text entry"
    entry = result.content[0]
    assert entry["type"] == "text"
    payload = json.loads(entry["text"])
    assert isinstance(payload, dict)


def test_nsys_summary_uses_fixture_csv(server: mcp_server.MCPServer):
    result = server.call_tool(
        "nsys_summary",
        {"report_path": str(NSYS_SAMPLE), "include_context": False},
    )
    payload = _payload_from_result(result)
    assert payload["tool"] == "nsys_summary"
    assert payload["status"] == "ok"
    tool_result = payload["result"]
    assert tool_result.get("success") is True
    assert tool_result.get("metrics")


def test_ncu_summary_uses_fixture_csv(server: mcp_server.MCPServer):
    result = server.call_tool(
        "ncu_summary",
        {"report_path": str(NCU_RAW_CSV_SAMPLE), "top_k": 3, "include_context": False},
    )
    payload = _payload_from_result(result)
    assert payload["tool"] == "ncu_summary"
    assert payload["status"] == "ok"
    tool_result = payload["result"]
    assert tool_result.get("success") is True
    assert tool_result.get("kernels")


FAST_TOOL_CASES = [case for case in ALL_TOOL_CASES if not case.slow]


@pytest.mark.parametrize("case", FAST_TOOL_CASES, ids=_case_ids(FAST_TOOL_CASES))
def test_tool_call_returns_json_envelope(server: mcp_server.MCPServer, case: ToolCase):
    result = server.call_tool(case.name, case.params)
    payload = _payload_from_result(result)
    assert payload["tool"] == case.name
    assert payload["status"] in {"ok", "error"}
    assert "result" in payload
    assert "context_summary" in payload
    if payload["status"] == "error":
        tool_result = payload["result"]
        assert (
            tool_result.get("error")
            or tool_result.get("stderr")
            or tool_result.get("message")
            or tool_result.get("success") is False
        ), f"{case.name} returned error status without structured failure details"


def test_known_good_tool_returns_success(server: mcp_server.MCPServer) -> None:
    result = server.call_tool("status", {})
    payload = _payload_from_result(result)

    assert payload["tool"] == "status"
    assert payload["status"] == "ok"
    assert payload["result"]["success"] is True


def test_known_bad_tool_request_returns_structured_error(server: mcp_server.MCPServer) -> None:
    result = server.call_tool("benchmark_compare", {})
    payload = _payload_from_result(result)

    assert payload["tool"] == "benchmark_compare"
    assert payload["status"] == "error"
    assert payload["result"]["success"] is False
    assert payload["result"]["error"]
    assert payload["result"]["error_type"]


def test_benchmark_report_returns_structured_error_when_output_dir_init_fails() -> None:
    with patch.object(mcp_server, "_ensure_dir", side_effect=OSError("mkdir boom")):
        result = mcp_server.tool_benchmark_report(
            {
                "data_file": str(BENCH_FILE),
                "output": str(REPO_ROOT / "artifacts" / "blocked" / "report.pdf"),
                "format": "pdf",
            }
        )

    assert result["success"] is False
    assert "Failed to prepare output path for benchmark report" in result["error"]
    assert result["output"].endswith("report.pdf")


def test_emit_stdio_json_logs_transport_warning(server: mcp_server.MCPServer, capsys: pytest.CaptureFixture[str]) -> None:
    with patch("builtins.print", side_effect=RuntimeError("stdout boom")):
        emitted = server._emit_stdio_json({"jsonrpc": "2.0"}, failure_context="test payload")

    captured = capsys.readouterr()
    assert emitted is False
    assert "Failed to emit test payload: stdout boom" in captured.err


def test_mcp_protocol_round_trip(server: mcp_server.MCPServer):
    async def _exercise():
        init = await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init and init["result"]["protocolVersion"] == "2024-11-05"

        tool_list = await server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        assert tool_list and "tools" in tool_list["result"]

        sample_tool = "status"
        call = await server.handle_message(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": sample_tool, "arguments": {}}}
        )
        assert call
        entry = call["result"]["content"][0]
        if entry["type"] == "text":
            payload = json.loads(entry["text"])
        elif entry["type"] == "application/json":
            payload = entry["json"]
        else:
            raise AssertionError(f"Unexpected content type: {entry['type']}")
        assert payload["tool"] == sample_tool

    asyncio.run(_exercise())


@pytest.mark.parametrize("case", SLOW_TOOL_CASES, ids=_case_ids(SLOW_TOOL_CASES))
def test_slow_tools_opt_in_execution(server: mcp_server.MCPServer, case: ToolCase):
    result = _call_with_timeout(server, case)
    payload = _payload_from_result(result)
    assert payload["tool"] == case.name
    assert payload["status"] in {"ok", "error"}
    if payload["status"] == "error":
        tool_result = payload["result"]
        assert (
            tool_result.get("error")
            or tool_result.get("stderr")
            or tool_result.get("message")
            or tool_result.get("success") is False
        ), f"{case.name} returned error status without structured failure details"
    if case.name == "benchmark_deep_dive_compare":
        tool_result = payload["result"]
        if payload["status"] == "ok":
            assert tool_result.get("success") is True
            assert Path(tool_result["analysis_json"]).exists()
        else:
            assert tool_result.get("error")
            assert tool_result.get("bench_result")
    if case.name == "run_benchmarks":
        tool_result = payload["result"]
        if tool_result.get("returncode", 1) == 0 and tool_result.get("results_json"):
            assert "triage" in tool_result
        queue_log = QUEUE_DIR / "queue.log"
        assert queue_log.exists()
        assert "RUN_START" in queue_log.read_text()
    if case.name == "profile_nsys":
        tool_result = payload["result"]
        if tool_result.get("success"):
            assert "nsys_metrics" in tool_result
            assert isinstance(tool_result["nsys_metrics"], dict)
    if case.name == "profile_ncu":
        tool_result = payload["result"]
        if tool_result.get("success"):
            assert "ncu_metrics" in tool_result
            assert isinstance(tool_result["ncu_metrics"], dict)
    if case.name == "profile_hta":
        tool_result = payload["result"]
        if tool_result.get("success"):
            assert "nsys_metrics" in tool_result
            assert isinstance(tool_result["nsys_metrics"], dict)
    if case.name == "profile_torch":
        tool_result = payload["result"]
        if tool_result.get("success"):
            assert "torch_metrics" in tool_result
            assert "report" in tool_result
    if case.name == "compare_nsys":
        tool_result = payload["result"]
        if NSYS_REP_FIXTURE.exists():
            assert tool_result.get("metrics")
            assert len(tool_result["metrics"]) > 0
        ncu_comparison = tool_result.get("ncu_comparison")
        if NCU_REP_FIXTURE.exists():
            assert ncu_comparison
            assert not ncu_comparison.get("error")
            assert ncu_comparison.get("kernel_comparison") or ncu_comparison.get("metrics")
            if ncu_comparison.get("kernel_comparison") is not None:
                assert len(ncu_comparison["kernel_comparison"]) > 0
    if case.name == "compare_ncu":
        tool_result = payload["result"]
        if NCU_REP_FIXTURE.exists():
            assert not tool_result.get("error")
            assert tool_result.get("kernel_comparison") or tool_result.get("metrics")
            if tool_result.get("kernel_comparison") is not None:
                assert len(tool_result["kernel_comparison"]) > 0
        nsys_comparison = tool_result.get("nsys_comparison")
        if NSYS_REP_FIXTURE.exists():
            assert nsys_comparison
            assert not nsys_comparison.get("error")
            assert nsys_comparison.get("metrics")
            assert len(nsys_comparison["metrics"]) > 0


def test_benchmark_export_runs_inprocess(server: mcp_server.MCPServer, tmp_path: Path):
    # Ensure a minimal benchmark file exists for the export tool.
    BENCH_FILE.write_text(json.dumps({"benchmarks": []}))
    output_path = tmp_path / "export.json"
    params = {"data_file": str(BENCH_FILE), "format": "json", "output": str(output_path)}
    result = server.call_tool("benchmark_export", params)
    payload = _payload_from_result(result)
    assert payload["tool"] == "benchmark_export"
    assert payload["result"].get("output") == str(output_path)
    assert output_path.exists()


def test_run_benchmarks_rejects_flag_like_targets(server: mcp_server.MCPServer):
    result = server.call_tool("run_benchmarks", {"targets": ["--auto-analyze"]})
    payload = _payload_from_result(result)
    assert payload["tool"] == "run_benchmarks"
    tool_result = payload["result"]
    assert tool_result.get("success") is False
    assert "Invalid benchmark target" in (tool_result.get("error") or "")


def test_run_benchmarks_async_returns_job_ticket_immediately(server: mcp_server.MCPServer):
    params = {
        "targets": ["ch99:missing_target"],
        "profile": "minimal",
        "iterations": 1,
        "warmup": 1,
        "llm_analysis": False,
        "auto_analyze": False,
        "auto_report": False,
        "async": True,
        "timeout_seconds": 60,
    }
    started = time.monotonic()
    result = server.call_tool("run_benchmarks", params)
    elapsed = time.monotonic() - started

    payload = _payload_from_result(result)
    assert payload["tool"] == "run_benchmarks"
    tool_result = payload["result"]
    assert elapsed < 10.0
    assert tool_result.get("job_id")
    assert tool_result.get("status") == "queued"

    record = _wait_for_job_terminal_status(server, str(tool_result["job_id"]), timeout_seconds=90.0)
    assert record.get("status") in {"completed", "error"}


def test_benchmark_deep_dive_async_returns_job_ticket_immediately(server: mcp_server.MCPServer):
    params = {
        "targets": ["ch99:missing_target"],
        "iterations": 1,
        "warmup": 1,
        "async": True,
        "timeout_seconds": 60,
    }
    started = time.monotonic()
    result = server.call_tool("benchmark_deep_dive_compare", params)
    elapsed = time.monotonic() - started

    payload = _payload_from_result(result)
    assert payload["tool"] == "benchmark_deep_dive_compare"
    tool_result = payload["result"]
    assert elapsed < 10.0
    assert tool_result.get("job_id")
    assert tool_result.get("status") == "queued"

    record = _wait_for_job_terminal_status(server, str(tool_result["job_id"]), timeout_seconds=120.0)
    assert record.get("status") in {"completed", "error"}


def test_queue_active_run_filter_ignores_queue_helpers():
    records = [
        {"pid": 101, "ppid": 1, "cmd": "/usr/bin/bash /tmp/artifacts/parallel_runs/queued_cmd.ABC123.sh"},
        {"pid": 102, "ppid": 1, "cmd": "/usr/bin/bash /tmp/artifacts/parallel_runs/queued_wrapper.DEF456.sh"},
        {"pid": 103, "ppid": 1, "cmd": "/usr/bin/bash /tmp/artifacts/parallel_runs/run_queue.sh --run"},
        {"pid": 104, "ppid": 1, "cmd": "/usr/bin/python -m cli.aisp bench run --targets ch10:atomic_reduction"},
    ]
    active = mcp_server._active_run_processes(records, ignore_pids=set())  # type: ignore[attr-defined]
    assert len(active) == 1
    assert active[0]["pid"] == 104
