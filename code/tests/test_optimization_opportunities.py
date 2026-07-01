from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from core.analysis.optimization_opportunities import (
    analyze_opportunity_file,
    apply_novelty_validation_feedback,
    apply_run_queue_feedback,
    discover_benchmark_target_catalog,
    normalize_candidates,
    rank_opportunities,
    render_novelty_next_wave_shell,
    render_novelty_validation_shell,
    render_opportunities_markdown,
    render_run_queue_shell,
    summarize_run_queue_root,
)
from core.api import handlers
from core.api.handlers import benchmark_opportunities
from core.api.registry import get_routes
from core.benchmark.bench_commands import app
from core.benchmark.contracts_surface import render_benchmark_run_yaml
from mcp.mcp_server import MCPServer


def _write_benchmark_pair(
    root: Path,
    chapter: str,
    name: str,
    *,
    source: str | None = None,
    optimized_source: str | None = None,
) -> None:
    chapter_dir = root / chapter
    chapter_dir.mkdir(parents=True, exist_ok=True)
    benchmark_source = source or "def get_benchmark():\n    return None\n"
    (chapter_dir / f"baseline_{name}.py").write_text(benchmark_source, encoding="utf-8")
    (chapter_dir / f"optimized_{name}.py").write_text(
        optimized_source or benchmark_source, encoding="utf-8"
    )


def _write_run_queue_job(
    root: Path,
    dirname: str,
    payload: dict,
    *,
    done: bool = False,
    approved: bool = False,
    manual: bool = False,
    stdout: str | None = None,
    stderr: str | None = None,
    review: str | None = None,
) -> None:
    job_dir = root / dirname
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(payload), encoding="utf-8")
    if done:
        (job_dir / "DONE").write_text("2026-07-01T00:00:00Z\n", encoding="utf-8")
    if approved:
        (job_dir / "APPROVED").write_text("approved after evidence review\n", encoding="utf-8")
    if manual:
        (job_dir / "MANUAL_REVIEW_REQUIRED").write_text("2026-07-01T00:00:00Z\n", encoding="utf-8")
    if stdout is not None:
        (job_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    if stderr is not None:
        (job_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    if review is not None:
        (job_dir / "promotion_review.md").write_text(review, encoding="utf-8")


def _write_required_contract_files(root: Path, job: dict) -> None:
    job_dir = root / str(job["id"])
    job_dir.mkdir(parents=True, exist_ok=True)
    contract = job.get("artifact_contract") or {}
    stage_contracts = contract.get("stage_contracts") or [contract]
    for stage_contract in stage_contracts:
        if not isinstance(stage_contract, dict):
            continue
        if stage_contract.get("job_id") != job["id"] and stage_contract.get("stage") != job.get(
            "stage"
        ):
            continue
        for filename in stage_contract.get("required_files", []) or []:
            path = job_dir / str(filename)
            if not path.exists():
                path.write_text(f"{filename}\n", encoding="utf-8")


def test_opportunity_radar_ranks_flat_slow_targets_first(tmp_path: Path) -> None:
    result_path = tmp_path / "benchmark_test_results.json"
    result_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-06-30T00:00:00Z",
                "results": [
                    {
                        "chapter": "ch15",
                        "benchmarks": [
                            {
                                "example": "kv_decode_cache",
                                "status": "succeeded",
                                "baseline_time_ms": 1250.0,
                                "best_speedup": 1.01,
                                "optimization_goal": "performance",
                                "baseline_file": "ch15/baseline_kv_decode_cache.py",
                            }
                        ],
                    },
                    {
                        "chapter": "ch09",
                        "benchmarks": [
                            {
                                "example": "cutlass_fp8_gemm",
                                "status": "succeeded",
                                "baseline_time_ms": 6.0,
                                "best_speedup": 2.4,
                                "optimization_goal": "performance",
                                "baseline_file": "ch09/baseline_cutlass_fp8_gemm.py",
                                "ncu_json": "artifacts/gemm.json",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = analyze_opportunity_file(result_path, top_n=2)

    opportunities = result["opportunities"]
    assert opportunities[0]["target"] == "ch15:kv_decode_cache"
    assert opportunities[0]["opportunity_type"] == "rework_flat_optimization"
    assert opportunities[0]["priority"] == "high"
    assert "KV-cache" in " ".join(opportunities[0]["recommended_experiments"])
    assert opportunities[0]["benchmark_run"]["overrides"]["workloadType"] == "inference"
    assert opportunities[0]["benchmark_run"]["overrides"]["benchmarkClass"] == "publication_grade"
    assert "benchmark-run-render" in opportunities[0]["benchmark_run"]["render_command"]
    assert opportunities[1]["target"] == "ch09:cutlass_fp8_gemm"
    assert result["execution_plan"]["phases"][0]["name"] == "deep_profile_headroom"
    assert result["execution_plan"]["next_commands"][0].endswith(
        "ch15:kv_decode_cache --profile deep_dive --verify-output"
    )


def test_opportunity_radar_accepts_tier1_summary_targets() -> None:
    payload = {
        "suite_name": "tier1",
        "targets": [
            {
                "target": "labs/kv_optimization:kv_standard",
                "category": "inference",
                "status": "succeeded",
                "baseline_time_ms": 900.0,
                "best_speedup": 1.3,
                "optimization_goal": "memory",
                "best_memory_savings_pct": 2.0,
                "artifacts": {},
            },
            {
                "target": "labs/block_scaling:block_scaling",
                "category": "kernel",
                "status": "succeeded",
                "baseline_time_ms": 0.2,
                "best_speedup": 1.9,
                "optimization_goal": "performance",
                "best_memory_savings_pct": 0.0,
                "artifacts": {"ncu_json": "artifacts/block.json"},
            },
        ],
    }

    candidates = normalize_candidates(payload)
    result = rank_opportunities(candidates, top_n=2)

    assert result["summary"]["total_candidates"] == 2
    assert result["opportunities"][0]["target"] == "labs/kv_optimization:kv_standard"
    assert result["opportunities"][0]["opportunity_type"] == "memory_pressure_probe"
    assert result["opportunities"][1]["target"] == "labs/block_scaling:block_scaling"


def test_bench_opportunities_cli_outputs_json(tmp_path: Path) -> None:
    data_file = tmp_path / "summary.json"
    data_file.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "chapter": "ch04",
                        "name": "gradient_fusion",
                        "status": "failed",
                        "baseline_time_ms": 20.0,
                        "speedup": 0.0,
                        "optimization_goal": "performance",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["opportunities", "--data-file", str(data_file), "--json", "--top", "1"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["opportunities"][0]["target"] == "ch04:gradient_fusion"
    assert payload["opportunities"][0]["opportunity_type"] == "restore_benchmark_evidence"
    assert "python -m cli.aisp bench run" in payload["opportunities"][0]["next_command"]


def test_bench_opportunities_cli_treats_empty_valid_input_as_success(tmp_path: Path) -> None:
    data_file = tmp_path / "summary.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["opportunities", "--data-file", str(data_file), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["total_candidates"] == 0
    assert payload["opportunities"] == []


def test_bench_opportunities_cli_uses_catalog_for_frontier_targets(tmp_path: Path) -> None:
    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    catalog_file = tmp_path / "targets.json"
    catalog_file.write_text(
        json.dumps(
            [
                {
                    "target": "labs/flexattention:flex_prefill",
                    "category": "attention",
                    "rationale": "frontier attention target",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "opportunities",
            "--data-file",
            str(data_file),
            "--catalog-file",
            str(catalog_file),
            "--json",
            "--top",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["frontier_candidates"] == 1
    assert payload["opportunities"][0]["target"] == "labs/flexattention:flex_prefill"
    assert payload["opportunities"][0]["opportunity_type"] == "novel_frontier_probe"
    assert payload["opportunities"][0]["next_command"].endswith(
        "labs/flexattention:flex_prefill --profile minimal --verify-output"
    )


def test_bench_opportunities_cli_writes_executable_run_queue_script(tmp_path: Path) -> None:
    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    catalog_file = tmp_path / "targets.json"
    catalog_file.write_text(
        json.dumps([{"target": "labs/flexattention:flex_prefill", "category": "attention"}]),
        encoding="utf-8",
    )
    output_script = tmp_path / "run_queue.sh"

    result = CliRunner().invoke(
        app,
        [
            "opportunities",
            "--data-file",
            str(data_file),
            "--catalog-file",
            str(catalog_file),
            "--top",
            "1",
            "--output-run-queue-sh",
            str(output_script),
        ],
    )

    assert result.exit_code == 0, result.output
    script = output_script.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash")
    assert "AISP_RUN_QUEUE_ROOT" in script
    assert "labs/flexattention:flex_prefill" in script
    assert "MANUAL_REVIEW_REQUIRED" in script
    assert output_script.stat().st_mode & 0o111


def test_bench_opportunities_cli_writes_executable_novelty_validation_script(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    catalog_file = tmp_path / "targets.json"
    catalog_file.write_text(
        json.dumps(
            [
                {
                    "target": "labs/distributed_decode:tp_decode_overlap",
                    "category": "communication_overlap",
                    "rationale": "distributed serving decode target with exposed collective time",
                    "source_terms": ["decode", "serving", "nccl", "allreduce"],
                    "frontier_signal_matches": [
                        {
                            "signal": "serving_decode_hotpath",
                            "matched_terms": ["decode", "serving"],
                        },
                        {
                            "signal": "distributed_fabric",
                            "matched_terms": ["nccl", "allreduce"],
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_script = tmp_path / "novelty_validation.sh"

    result = CliRunner().invoke(
        app,
        [
            "opportunities",
            "--data-file",
            str(data_file),
            "--catalog-file",
            str(catalog_file),
            "--top",
            "1",
            "--output-novelty-validation-sh",
            str(output_script),
        ],
    )

    assert result.exit_code == 0, result.output
    script = output_script.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash")
    assert "AISP_NOVELTY_QUEUE_ROOT" in script
    assert "labs/distributed_decode:tp_decode_overlap" in script
    assert "Novelty Review" in script
    assert "MANUAL_REVIEW_REQUIRED" in script
    assert output_script.stat().st_mode & 0o111


def test_bench_opportunities_cli_writes_executable_novelty_next_wave_script(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    target = {
        "target": "labs/distributed_decode:tp_decode_overlap",
        "category": "communication_overlap",
        "source_terms": ["decode", "serving", "nccl", "allreduce"],
        "frontier_signal_matches": [
            {"signal": "serving_decode_hotpath", "matched_terms": ["decode"]},
            {"signal": "distributed_fabric", "matched_terms": ["nccl"]},
        ],
    }
    catalog_file = tmp_path / "targets.json"
    catalog_file.write_text(json.dumps([target]), encoding="utf-8")
    seed = rank_opportunities(
        normalize_candidates({"benchmarks": [], "target_catalog": [target]}),
        top_n=1,
    )
    queue_root = tmp_path / "novelty"
    control_job, candidate_job = seed["novelty_validation_plan"]["jobs"][:2]
    _write_run_queue_job(queue_root, control_job["id"], control_job, done=True)
    _write_run_queue_job(
        queue_root,
        candidate_job["id"],
        candidate_job,
        stderr="Zymtrace CUDA_INJECTION64_PATH injection library not found\n",
    )
    output_script = tmp_path / "novelty_next_wave.sh"

    result = CliRunner().invoke(
        app,
        [
            "opportunities",
            "--data-file",
            str(data_file),
            "--catalog-file",
            str(catalog_file),
            "--novelty-queue-root",
            str(queue_root),
            "--top",
            "1",
            "--output-novelty-next-wave-sh",
            str(output_script),
        ],
    )

    assert result.exit_code == 0, result.output
    script = output_script.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash")
    assert "AISP_NOVELTY_NEXT_WAVE_ROOT" in script
    assert "novelty_next_wave_plan.json" in script
    assert "recover_blocked_leads" in script
    assert "CUDA_INJECTION64_PATH" in script
    assert "recovery_command.txt" in script
    assert "activate_backups" in script
    assert output_script.stat().st_mode & 0o111


def test_bench_opportunities_cli_can_discover_frontier_targets(tmp_path: Path) -> None:
    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    bench_root = tmp_path / "bench"
    _write_benchmark_pair(bench_root, "ch01", "frontier_probe")

    result = CliRunner().invoke(
        app,
        [
            "opportunities",
            "--data-file",
            str(data_file),
            "--include-discovered-targets",
            "--bench-root",
            str(bench_root),
            "--json",
            "--top",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["frontier_candidates"] == 1
    assert payload["opportunities"][0]["target"] == "ch01:frontier_probe"
    assert payload["opportunities"][0]["opportunity_type"] == "novel_frontier_probe"
    assert payload["discovered_target_source"] == str(bench_root.resolve())


def test_source_mined_catalog_extracts_frontier_signals_from_benchmark_files(
    tmp_path: Path,
) -> None:
    bench_root = tmp_path / "bench"
    _write_benchmark_pair(
        bench_root,
        "ch01",
        "plain_probe",
        source=(
            "# FlashAttention KV cache decode benchmark with CUDA graph replay\n"
            "def get_benchmark():\n"
            "    return None\n"
        ),
    )

    catalog = discover_benchmark_target_catalog(bench_root)

    assert catalog["target_count"] == 1
    entry = catalog["targets"][0]
    assert entry["target"] == "ch01:plain_probe"
    assert entry["category"] == "attention_kv_layout"
    assert "attention" in entry["source_terms"]
    assert "kv" in entry["source_terms"]
    signals = {match["signal"] for match in entry["frontier_signal_matches"]}
    assert {"attention_or_kv", "runtime_launch"} <= signals


def test_source_mined_catalog_detects_optimized_side_primitives(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    _write_benchmark_pair(
        bench_root,
        "ch01",
        "decode_probe",
        source="# baseline decode loop\n\ndef get_benchmark():\n    return None\n",
        optimized_source=(
            "# optimized decode loop with CUDA Graph replay and KV cache page size sweep\n"
            "def get_benchmark():\n"
            "    return None\n"
        ),
    )

    catalog = discover_benchmark_target_catalog(bench_root)

    entry = catalog["targets"][0]
    assert "cuda graph" in entry["source_delta_terms"]
    primitives = {item["primitive"]: item for item in entry["optimization_primitives"]}
    assert primitives["cuda_graph_replay"]["introduced"] is True
    assert "cuda graph" in primitives["cuda_graph_replay"]["introduced_terms"]
    assert primitives["kv_cache_layout"]["introduced"] is True


def test_bench_opportunity_catalog_cli_outputs_reusable_json(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    _write_benchmark_pair(
        bench_root,
        "ch01",
        "plain_probe",
        source="# NVLink allreduce benchmark\n\ndef get_benchmark():\n    return None\n",
    )
    output_file = tmp_path / "catalog.json"

    result = CliRunner().invoke(
        app,
        [
            "opportunity-catalog",
            "--bench-root",
            str(bench_root),
            "--output-json",
            str(output_file),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["targets"][0]["target"] == "ch01:plain_probe"
    assert output_file.exists()
    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved["targets"][0]["frontier_signal_matches"][0]["signal"] == "distributed_fabric"


def test_bench_opportunities_cli_uses_source_mined_catalog_for_discovery(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    bench_root = tmp_path / "bench"
    _write_benchmark_pair(
        bench_root,
        "ch01",
        "plain_probe",
        source="# KV decode prefill benchmark with FlashAttention\n\ndef get_benchmark():\n    return None\n",
    )

    result = CliRunner().invoke(
        app,
        [
            "opportunities",
            "--data-file",
            str(data_file),
            "--include-discovered-targets",
            "--bench-root",
            str(bench_root),
            "--json",
            "--top",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    opportunity = payload["opportunities"][0]
    assert opportunity["target"] == "ch01:plain_probe"
    assert opportunity["frontier_motif"] == "attention_kv_layout"
    assert "attention_or_kv" in opportunity["frontier_signals"]
    assert "kv" in opportunity["source_terms"]


def test_source_transfer_map_links_source_mined_patterns_to_recipients() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/flexattention:flex_prefill",
                "category": "attention_kv_layout",
                "rationale": "source-mined matched terms attention kv decode",
                "source_terms": ["attention", "kv", "decode"],
                "frontier_signal_matches": [
                    {"signal": "attention_or_kv", "matched_terms": ["attention", "kv"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "kv_cache_layout",
                        "matched_terms": ["kv cache"],
                        "introduced_terms": ["kv cache"],
                        "introduced": True,
                        "transfer_question": "Can the recipient validate KV cache layout?",
                    }
                ],
                "baseline_file": "labs/flexattention/baseline_flex_prefill.py",
                "optimized_files": ["labs/flexattention/optimized_flex_prefill.py"],
                "catalog_source": "benchmark_tree",
            },
            {
                "target": "labs/paged_decode:paged_decode",
                "category": "attention_kv_layout",
                "rationale": "source-mined matched terms kv decode prefill",
                "source_terms": ["kv", "decode", "prefill"],
                "frontier_signal_matches": [
                    {"signal": "attention_or_kv", "matched_terms": ["kv", "decode"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "kv_cache_layout",
                        "matched_terms": ["kv cache"],
                        "introduced_terms": [],
                        "introduced": False,
                        "transfer_question": "Can the recipient validate KV cache layout?",
                    }
                ],
                "baseline_file": "labs/paged_decode/baseline_paged_decode.py",
                "optimized_files": ["labs/paged_decode/optimized_paged_decode.py"],
                "catalog_source": "benchmark_tree",
            },
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=2)

    transfer = result["source_transfer_map"]
    assert transfer["pattern_count"] >= 1
    assert all(row["pattern"] != "catalog_category" for row in transfer["patterns"])
    pattern = next(row for row in transfer["patterns"] if row["pattern"] == "attention_or_kv")
    assert pattern["source_target"] in {
        "labs/flexattention:flex_prefill",
        "labs/paged_decode:paged_decode",
    }
    assert pattern["recipient_count"] == 1
    assert pattern["recipient_targets"][0] in {
        "labs/flexattention:flex_prefill",
        "labs/paged_decode:paged_decode",
    }
    assert pattern["recipient_targets"][0] != pattern["source_target"]
    assert "kv" in pattern["source_terms"]
    assert pattern["blueprint_ids"][0].endswith("kv-page-layout-sweep")
    primitive_pattern = next(
        row for row in transfer["patterns"] if row["pattern"] == "primitive:kv_cache_layout"
    )
    assert primitive_pattern["pattern_type"] == "optimization_primitive"
    assert primitive_pattern["source_terms"] == ["kv cache"]
    assert primitive_pattern["recipient_count"] == 1
    opportunity = result["opportunities"][0]
    assert opportunity["catalog_source"] == "benchmark_tree"
    assert opportunity["source_files"]
    assert opportunity["optimization_primitives"]


def test_compound_primitive_hypotheses_pair_source_backed_partials() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/decode:kv_layout_probe",
                "category": "attention_kv_layout",
                "rationale": "source-mined decode target with KV cache layout pressure",
                "source_terms": ["kv cache", "decode"],
                "frontier_signal_matches": [
                    {"signal": "attention_or_kv", "matched_terms": ["kv cache", "decode"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "kv_cache_layout",
                        "matched_terms": ["kv cache"],
                        "introduced_terms": ["kv cache"],
                        "introduced": True,
                        "transfer_question": "Can the recipient validate KV cache layout?",
                    }
                ],
                "baseline_file": "labs/decode/baseline_kv_layout_probe.py",
                "optimized_files": ["labs/decode/optimized_kv_layout_probe.py"],
                "catalog_source": "benchmark_tree",
            },
            {
                "target": "labs/runtime:graph_replay_probe",
                "category": "runtime_launch_reduction",
                "rationale": "source-mined static-shape launch target with CUDA Graph replay",
                "source_terms": ["cuda graph", "launch"],
                "frontier_signal_matches": [
                    {"signal": "runtime_launch", "matched_terms": ["cuda graph", "launch"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "cuda_graph_replay",
                        "matched_terms": ["cuda graph"],
                        "introduced_terms": ["cuda graph"],
                        "introduced": True,
                        "transfer_question": "Can the recipient expose a static replay region?",
                    }
                ],
                "baseline_file": "labs/runtime/baseline_graph_replay_probe.py",
                "optimized_files": ["labs/runtime/optimized_graph_replay_probe.py"],
                "catalog_source": "benchmark_tree",
            },
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=1)

    assert result["summary"]["returned"] == 1
    compounds = result["compound_primitive_hypotheses"]
    assert compounds["hypothesis_count"] >= 1
    required = {"kv_cache_layout", "cuda_graph_replay"}
    hypothesis = next(row for row in compounds["hypotheses"] if set(row["primitives"]) == required)
    assert hypothesis["target"] in {
        "labs/decode:kv_layout_probe",
        "labs/runtime:graph_replay_probe",
    }
    assert set(hypothesis["present_primitives"]) | set(hypothesis["missing_primitives"]) == required
    assert len(hypothesis["missing_primitives"]) == 1
    assert hypothesis["support_targets"]
    assert "bench run --targets" in hypothesis["prototype_command"]
    assert "both primitives" in hypothesis["acceptance_gate"]

    markdown = render_opportunities_markdown(result)
    assert "## Compound Primitive Hypotheses" in markdown
    assert "Graph-stabilized KV decode loop" in markdown


def test_primitive_pair_synthesis_finds_untried_source_backed_pairs() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/decode:kv_layout_probe",
                "category": "attention_kv_layout",
                "rationale": "decode target with KV cache layout pressure",
                "source_terms": ["kv cache", "decode"],
                "frontier_signal_matches": [
                    {"signal": "attention_or_kv", "matched_terms": ["kv cache", "decode"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "kv_cache_layout",
                        "matched_terms": ["kv cache"],
                        "introduced_terms": ["kv cache"],
                        "introduced": True,
                        "transfer_question": "Can the recipient validate KV cache layout?",
                    }
                ],
                "baseline_file": "labs/decode/baseline_kv_layout_probe.py",
                "optimized_files": ["labs/decode/optimized_kv_layout_probe.py"],
                "catalog_source": "benchmark_tree",
            },
            {
                "target": "labs/runtime:persistent_decode_probe",
                "category": "launch_graph_persistence",
                "rationale": "decode-adjacent runtime target with persistent kernel residency",
                "source_terms": ["persistent kernel", "decode", "launch"],
                "frontier_signal_matches": [
                    {"signal": "runtime_launch", "matched_terms": ["persistent", "launch"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "persistent_kernel",
                        "matched_terms": ["persistent kernel"],
                        "introduced_terms": ["persistent kernel"],
                        "introduced": True,
                        "transfer_question": "Can the recipient justify persistent residency?",
                    }
                ],
                "baseline_file": "labs/runtime/baseline_persistent_decode_probe.py",
                "optimized_files": ["labs/runtime/optimized_persistent_decode_probe.py"],
                "catalog_source": "benchmark_tree",
            },
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=2)

    synthesis = result["novelty_primitive_pair_synthesis_plan"]
    assert synthesis["synthesis_count"] >= 1
    pair = {"kv_cache_layout", "persistent_kernel"}
    item = next(row for row in synthesis["syntheses"] if set(row["pair"]) == pair)
    assert item["pair_state"] == "not_in_known_compound_specs"
    assert item["support_targets"]
    assert item["candidate_transfer_question"]
    assert "combined pair beats the best isolated primitive" in item["acceptance_gate"]
    assert "untried primitive-pair hypothesis" in item["claim_boundary"]

    markdown = render_opportunities_markdown(result)
    assert "## Novelty Primitive Pair Synthesis Plan" in markdown
    assert "kv_cache_layout + persistent_kernel" in markdown


def test_coverage_gap_map_surfaces_underintroduced_primitives() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/decode:kv_layout_probe",
                "category": "attention_kv_layout",
                "rationale": "decode target where KV cache appears in both baseline and optimized source",
                "source_terms": ["kv cache", "decode"],
                "frontier_signal_matches": [
                    {"signal": "attention_or_kv", "matched_terms": ["kv cache", "decode"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "kv_cache_layout",
                        "matched_terms": ["kv cache"],
                        "introduced_terms": [],
                        "introduced": False,
                        "transfer_question": "Can the recipient validate KV cache layout?",
                    }
                ],
                "baseline_file": "labs/decode/baseline_kv_layout_probe.py",
                "optimized_files": ["labs/decode/optimized_kv_layout_probe.py"],
                "catalog_source": "benchmark_tree",
            }
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=1)

    gap_map = result["coverage_gap_map"]
    assert gap_map["gap_count"] >= 1
    primitive_gap = next(
        gap
        for gap in gap_map["gaps"]
        if gap["gap_type"] == "optimization_primitive" and gap["primitive"] == "kv_cache_layout"
    )
    assert primitive_gap["coverage_state"] == "not_introduced"
    assert primitive_gap["matched_target_count"] == 1
    assert primitive_gap["introduced_target_count"] == 0
    assert primitive_gap["sample_targets"][0]["target"] == "labs/decode:kv_layout_probe"
    assert primitive_gap["first_probe_command"].endswith(
        "labs/decode:kv_layout_probe --profile minimal --verify-output"
    )
    assert "one optimized variant" in primitive_gap["recommended_action"]

    signal_gap = next(gap for gap in gap_map["gaps"] if gap["gap_type"] == "frontier_signal")
    assert signal_gap["coverage_state"] in {"missing", "thin"}

    markdown = render_opportunities_markdown(result)
    assert "## Coverage Gap Map" in markdown
    assert "Coverage gaps are negative-space leads" in markdown


def test_novelty_queue_merges_frontier_transfer_compound_and_gap_leads() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/decode:kv_layout_probe",
                "category": "attention_kv_layout",
                "rationale": "decode target with KV cache layout pressure",
                "source_terms": ["kv cache", "decode"],
                "frontier_signal_matches": [
                    {"signal": "attention_or_kv", "matched_terms": ["kv cache", "decode"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "kv_cache_layout",
                        "matched_terms": ["kv cache"],
                        "introduced_terms": ["kv cache"],
                        "introduced": True,
                        "transfer_question": "Can the recipient validate KV cache layout?",
                    }
                ],
                "baseline_file": "labs/decode/baseline_kv_layout_probe.py",
                "optimized_files": ["labs/decode/optimized_kv_layout_probe.py"],
                "catalog_source": "benchmark_tree",
            },
            {
                "target": "labs/decode:prefill_probe",
                "category": "attention_kv_layout",
                "rationale": "prefill target with attention pressure but no KV layout variant",
                "source_terms": ["attention", "prefill"],
                "frontier_signal_matches": [
                    {"signal": "attention_or_kv", "matched_terms": ["attention", "prefill"]}
                ],
                "baseline_file": "labs/decode/baseline_prefill_probe.py",
                "optimized_files": ["labs/decode/optimized_prefill_probe.py"],
                "catalog_source": "benchmark_tree",
            },
            {
                "target": "labs/runtime:graph_replay_probe",
                "category": "launch_graph_persistence",
                "rationale": "runtime launch target with CUDA Graph replay",
                "source_terms": ["cuda graph", "launch"],
                "frontier_signal_matches": [
                    {"signal": "runtime_launch", "matched_terms": ["cuda graph", "launch"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "cuda_graph_replay",
                        "matched_terms": ["cuda graph"],
                        "introduced_terms": ["cuda graph"],
                        "introduced": True,
                        "transfer_question": "Can the recipient expose a static replay region?",
                    }
                ],
                "baseline_file": "labs/runtime/baseline_graph_replay_probe.py",
                "optimized_files": ["labs/runtime/optimized_graph_replay_probe.py"],
                "catalog_source": "benchmark_tree",
            },
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=2)

    queue = result["novelty_queue"]
    assert queue["lead_count"] >= 4
    lead_types = set(queue["lead_type_counts"])
    assert {
        "frontier_probe",
        "source_transfer",
        "compound_primitive",
        "coverage_gap",
    } <= lead_types
    assert all(lead["evidence_gate"] for lead in queue["leads"])
    assert any(lead["command"] for lead in queue["leads"])

    markdown = render_opportunities_markdown(result)
    assert "## Novelty Queue" in markdown
    assert "Use the novelty queue to choose the next experiment lead" in markdown


def test_cross_lane_bridge_map_promotes_multi_signal_experiments() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/distributed_decode:tp_decode_overlap",
                "category": "communication_overlap",
                "rationale": "distributed serving decode target with exposed collective time",
                "source_terms": ["decode", "serving", "nccl", "allreduce", "kv cache"],
                "frontier_signal_matches": [
                    {
                        "signal": "serving_decode_hotpath",
                        "matched_terms": ["decode", "serving"],
                    },
                    {
                        "signal": "distributed_fabric",
                        "matched_terms": ["nccl", "allreduce"],
                    },
                    {"signal": "attention_or_kv", "matched_terms": ["kv cache"]},
                ],
                "optimization_primitives": [
                    {
                        "primitive": "communication_overlap",
                        "matched_terms": ["overlap", "allreduce"],
                        "introduced_terms": ["overlap"],
                        "introduced": True,
                        "transfer_question": "Can the recipient prove lower exposed collective time?",
                    },
                    {
                        "primitive": "kv_cache_layout",
                        "matched_terms": ["kv cache"],
                        "introduced_terms": ["kv cache"],
                        "introduced": True,
                        "transfer_question": "Can the recipient validate KV cache layout?",
                    },
                ],
                "baseline_file": "labs/distributed_decode/baseline_tp_decode_overlap.py",
                "optimized_files": ["labs/distributed_decode/optimized_tp_decode_overlap.py"],
                "catalog_source": "benchmark_tree",
            }
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=1)

    bridge_map = result["cross_lane_bridge_map"]
    assert bridge_map["bridge_count"] >= 1
    bridge = next(row for row in bridge_map["bridges"] if row["bridge"] == "decode_fabric_overlap")
    assert bridge["prototype_target"] == "labs/distributed_decode:tp_decode_overlap"
    assert bridge["direct_support_count"] == 1
    assert {"communication_overlap", "kv_cache_layout"} <= set(bridge["present_primitives"])
    assert bridge["validation_commands"][1].endswith(
        "labs/distributed_decode:tp_decode_overlap --profile deep_dive --verify-output"
    )
    assert "decode metrics improve" in bridge["acceptance_gate"]

    queue = result["novelty_queue"]
    assert "cross_lane_bridge" in queue["lead_type_counts"]

    markdown = render_opportunities_markdown(result)
    assert "## Cross-Lane Bridge Map" in markdown
    assert "Decode/fabric overlap bridge" in markdown


def test_novelty_validation_plan_builds_dependency_ordered_jobs() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/distributed_decode:tp_decode_overlap",
                "category": "communication_overlap",
                "rationale": "distributed serving decode target with exposed collective time",
                "source_terms": ["decode", "serving", "nccl", "allreduce", "kv cache"],
                "frontier_signal_matches": [
                    {
                        "signal": "serving_decode_hotpath",
                        "matched_terms": ["decode", "serving"],
                    },
                    {
                        "signal": "distributed_fabric",
                        "matched_terms": ["nccl", "allreduce"],
                    },
                ],
                "optimization_primitives": [
                    {
                        "primitive": "communication_overlap",
                        "matched_terms": ["overlap", "allreduce"],
                        "introduced_terms": ["overlap"],
                        "introduced": True,
                        "transfer_question": "Can the recipient prove lower exposed collective time?",
                    }
                ],
                "baseline_file": "labs/distributed_decode/baseline_tp_decode_overlap.py",
                "optimized_files": ["labs/distributed_decode/optimized_tp_decode_overlap.py"],
                "catalog_source": "benchmark_tree",
            },
            {
                "target": "labs/decode:kv_layout_probe",
                "category": "attention_kv_layout",
                "rationale": "decode target with KV cache layout pressure",
                "source_terms": ["kv cache", "decode"],
                "frontier_signal_matches": [
                    {"signal": "attention_or_kv", "matched_terms": ["kv cache", "decode"]}
                ],
                "optimization_primitives": [
                    {
                        "primitive": "kv_cache_layout",
                        "matched_terms": ["kv cache"],
                        "introduced_terms": ["kv cache"],
                        "introduced": True,
                        "transfer_question": "Can the recipient validate KV cache layout?",
                    }
                ],
                "baseline_file": "labs/decode/baseline_kv_layout_probe.py",
                "optimized_files": ["labs/decode/optimized_kv_layout_probe.py"],
                "catalog_source": "benchmark_tree",
            },
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=2)

    playbook_map = result["novelty_experiment_playbooks"]
    assert playbook_map["playbook_count"] >= 1
    bridge_playbook = next(
        playbook
        for playbook in playbook_map["playbooks"]
        if playbook["lead_type"] == "cross_lane_bridge"
    )
    assert bridge_playbook["variant_ladder"][1]["variant"] == "lane_a_isolated"
    assert bridge_playbook["variant_ladder"][3]["variant"] == "bridge_combined"
    assert "primary_lane_metric" in bridge_playbook["primary_metrics"]
    assert "both_lane_metrics_present" in bridge_playbook["guardrail_metrics"]
    assert "nsys" in bridge_playbook["profiler_tools"]

    mutation_plan = result["novelty_mutation_plan"]
    assert mutation_plan["lead_count"] == playbook_map["playbook_count"]
    assert mutation_plan["mutation_count"] >= mutation_plan["lead_count"]
    bridge_mutations = next(
        item
        for item in mutation_plan["lead_mutations"]
        if item["lead_type"] == "cross_lane_bridge"
    )
    bridge_operators = {item["operator"] for item in bridge_mutations["mutations"]}
    assert {"lane_a_only", "lane_b_only", "bridge_interaction_toggle"}.issubset(
        bridge_operators
    )
    first_bridge_mutation = bridge_mutations["mutations"][0]
    assert "one-variable mutation" in first_bridge_mutation["isolation_rule"]
    assert first_bridge_mutation["promotion_gate"]

    budget = result["novelty_budget_plan"]
    assert budget["selected_count"] >= 1
    assert budget["selected_cost_units"] <= budget["max_cost_units"]
    selected_targets = [item["target"] for item in budget["selected"] if item.get("target")]
    if len(selected_targets) != len(set(selected_targets)):
        assert any(
            item.get("selection_phase") == "lead_type_diversity_target_repeat"
            for item in budget["selected"]
        )
    selected_types = {item["lead_type"] for item in budget["selected"]}
    assert "cross_lane_bridge" in selected_types
    assert all("expected_value_score" in item for item in budget["selected"])
    assert all("cost_units" in item for item in budget["selected"])
    assert all("risk_mitigation_steps" in item for item in budget["selected"])
    assert all("selection_reason" in item for item in budget["selected"])
    assert budget["backlog_count"] == len(budget["backlog"])
    if budget["backlog"]:
        assert budget["deferral_reason_counts"]
        assert all("deferral_reason" in item for item in budget["backlog"])
        assert all("next_unlock" in item for item in budget["backlog"])

    mutation_budget = result["novelty_mutation_budget_plan"]
    assert mutation_budget["mutation_count"] == mutation_plan["mutation_count"]
    assert mutation_budget["selected_count"] >= 1
    assert mutation_budget["selected_cost_units"] <= mutation_budget["max_cost_units"]
    assert mutation_budget["selected_operator_count"] >= 1
    assert mutation_budget["next_mutation"]["mutation_id"]
    assert all("information_gain_score" in item for item in mutation_budget["selected"])
    assert all("required_evidence" in item for item in mutation_budget["selected"])
    assert all("selection_reason" in item for item in mutation_budget["selected"])
    if mutation_budget["backlog"]:
        assert mutation_budget["deferral_reason_counts"]
        assert all("deferral_reason" in item for item in mutation_budget["backlog"])
        assert all("next_unlock" in item for item in mutation_budget["backlog"])

    decision = result["novelty_decision_frontier"]
    assert decision["selected_lead_count"] == budget["selected_count"]
    assert decision["backlog_lead_count"] == budget["backlog_count"]
    decision_lanes = {lane["lane"]: lane for lane in decision["lanes"]}
    assert {"quick_proofs", "high_upside", "de_risk_first", "pareto_frontier"}.issubset(
        decision_lanes
    )
    assert decision_lanes["high_upside"]["leads"]
    assert all(
        "selection_state" in item for lane in decision_lanes.values() for item in lane["leads"]
    )
    assert all(
        "decision_reason" in item for lane in decision_lanes.values() for item in lane["leads"]
    )
    assert any(
        item["selection_state"] == "backlog" for item in decision_lanes["deferred_unlocks"]["leads"]
    )

    falsification = result["novelty_falsification_plan"]
    assert falsification["lead_count"] == budget["selected_count"]
    assert falsification["check_count"] >= falsification["lead_count"]
    bridge_falsification = next(
        item for item in falsification["lead_checks"] if item["lead_type"] == "cross_lane_bridge"
    )
    assert "bridge is not real" in bridge_falsification["null_hypothesis"]
    assert any(
        "only one bridged lane" in check for check in bridge_falsification["falsification_checks"]
    )
    assert "Claim a bridge" in bridge_falsification["claim_boundary"]

    ablation = result["novelty_ablation_plan"]
    assert ablation["lead_count"] == budget["selected_count"]
    assert ablation["control_count"] >= ablation["lead_count"]
    bridge_ablation = next(
        item for item in ablation["lead_controls"] if item["lead_type"] == "cross_lane_bridge"
    )
    assert any(control["control_type"] == "lane_a_only" for control in bridge_ablation["controls"])
    assert any("ablation control summary" in item for item in bridge_ablation["required_evidence"])

    reproducibility = result["novelty_reproducibility_plan"]
    assert reproducibility["lead_count"] == budget["selected_count"]
    assert reproducibility["repeat_count_total"] >= reproducibility["lead_count"] * 3
    bridge_reproducibility = next(
        item
        for item in reproducibility["lead_profiles"]
        if item["lead_type"] == "cross_lane_bridge"
    )
    assert bridge_reproducibility["repeat_count"] >= 5
    assert bridge_reproducibility["variance_threshold_pct"] > 0
    assert "median_wall_time_ms" in bridge_reproducibility["stability_metrics"]
    assert any(
        "repeat-run manifest" in item for item in bridge_reproducibility["required_evidence"]
    )

    instrumentation = result["novelty_instrumentation_plan"]
    assert instrumentation["lead_count"] == budget["selected_count"]
    assert instrumentation["tool_counts"]
    bridge_instrumentation = next(
        item
        for item in instrumentation["lead_profiles"]
        if item["lead_type"] == "cross_lane_bridge"
    )
    assert "nsys" in bridge_instrumentation["required_profiler_tools"]
    assert bridge_instrumentation["preflight_checks"]
    assert any(
        "profiler preflight manifest" in item
        for item in bridge_instrumentation["required_evidence"]
    )
    if "zymtrace" in bridge_instrumentation["required_profiler_tools"]:
        assert "CUDA_INJECTION64_PATH" in bridge_instrumentation["launch_environment"]
        assert any(
            check["check"] == "zymtrace_cuda_injection_ready"
            for check in bridge_instrumentation["preflight_checks"]
        )

    artifact_contract_plan = result["novelty_artifact_contract_plan"]
    assert artifact_contract_plan["lead_count"] == budget["selected_count"]
    assert artifact_contract_plan["required_file_count"] >= budget["selected_count"] * 4
    bridge_contract = next(
        item
        for item in artifact_contract_plan["lead_contracts"]
        if item["lead_type"] == "cross_lane_bridge"
    )
    assert bridge_contract["package_manifest"] == "artifact_contract_manifest.json"
    bridge_stage_contracts = {
        item["stage"]: item for item in bridge_contract["stage_contracts"]
    }
    assert {"control", "bridge_variant", "deep_profile", "manual_review"}.issubset(
        bridge_stage_contracts
    )
    assert "repeat_run_manifest.json" in bridge_stage_contracts["bridge_variant"][
        "required_files"
    ]
    assert "profiler_preflight_manifest.json" in bridge_stage_contracts["deep_profile"][
        "required_files"
    ]
    assert any(
        "artifact contract manifest" in item for item in bridge_contract["required_evidence"]
    )

    claim_packet_plan = result["novelty_claim_packet_plan"]
    assert claim_packet_plan["lead_count"] == budget["selected_count"]
    assert claim_packet_plan["required_section_count"] >= budget["selected_count"]
    assert claim_packet_plan["disallowed_claim_count"] >= budget["selected_count"]
    bridge_claim_packet = next(
        item
        for item in claim_packet_plan["lead_packets"]
        if item["lead_type"] == "cross_lane_bridge"
    )
    assert "Claim a bridge" in bridge_claim_packet["allowed_claim_scope"]
    assert any(
        section["section"] == "artifact_packet"
        for section in bridge_claim_packet["required_sections"]
    )
    assert any("do not claim a bridge" in item for item in bridge_claim_packet["disallowed_claims"])
    assert bridge_claim_packet["packet_path"].endswith("/claim_packet.md")
    assert any("claim packet links" in item for item in bridge_claim_packet["required_evidence"])

    plan = result["novelty_validation_plan"]
    assert plan["selected_lead_count"] == budget["selected_count"]
    assert plan["job_count"] == plan["selected_lead_count"] * 4
    assert plan["dispatch_groups"][0]["stage_mix"] == {"control": plan["selected_lead_count"]}
    bridge_lead = next(
        lead for lead in plan["selected_leads"] if lead["lead_type"] == "cross_lane_bridge"
    )
    selected_bridge_playbook = next(
        playbook
        for playbook in playbook_map["playbooks"]
        if playbook["lead_id"] == bridge_lead["lead_id"]
    )
    lead_jobs = [job for job in plan["jobs"] if job["lead_id"] == bridge_lead["lead_id"]]
    assert [job["stage"] for job in lead_jobs] == [
        "control",
        "bridge_variant",
        "deep_profile",
        "manual_review",
    ]
    assert lead_jobs[1]["depends_on"] == [lead_jobs[0]["id"]]
    assert lead_jobs[2]["depends_on"] == [lead_jobs[1]["id"]]
    assert lead_jobs[3]["depends_on"] == [lead_jobs[2]["id"]]
    assert lead_jobs[1]["experiment_playbook_id"] == selected_bridge_playbook["playbook_id"]
    assert "first lane variant" in lead_jobs[1]["experiment_variables"]
    assert "primary_lane_metric" in lead_jobs[1]["primary_metrics"]
    assert "both_lane_metrics_present" in lead_jobs[1]["guardrail_metrics"]
    assert "multi_variable_isolation" in lead_jobs[1]["risk_flags"]
    assert lead_jobs[1]["risk_mitigation_steps"]
    assert lead_jobs[1]["falsification_checks"]
    assert any(
        control["control_type"] == "lane_a_only" for control in lead_jobs[1]["ablation_controls"]
    )
    assert lead_jobs[1]["repeat_count"] >= 5
    assert "median_wall_time_ms" in lead_jobs[1]["stability_metrics"]
    assert lead_jobs[1]["variance_threshold_pct"] > 0
    assert "nsys" in lead_jobs[1]["required_profiler_tools"]
    assert lead_jobs[1]["instrumentation_preflight"]
    assert lead_jobs[1]["artifact_contract"]["job_id"] == lead_jobs[1]["id"]
    assert "nsys" in lead_jobs[2]["profiler_tools"]
    assert lead_jobs[2]["instrumentation_preflight"]
    assert lead_jobs[2]["artifact_contract"]["job_id"] == lead_jobs[2]["id"]
    assert lead_jobs[3]["stop_conditions"]
    assert "Claim a bridge" in lead_jobs[3]["claim_boundary"]
    assert lead_jobs[3]["ablation_controls"]
    assert lead_jobs[3]["reproducibility_gate"]
    assert lead_jobs[3]["instrumentation_preflight"]
    assert lead_jobs[3]["artifact_contract"]["contract_id"] == bridge_contract["contract_id"]
    assert lead_jobs[3]["claim_packet"]["packet_id"] == bridge_claim_packet["packet_id"]
    assert "same workload contract" in " ".join(lead_jobs[3]["required_evidence"])
    assert "isolated-variable evidence" in " ".join(lead_jobs[3]["required_evidence"])
    assert "resolved falsification checklist" in " ".join(lead_jobs[3]["required_evidence"])
    assert "ablation control summary" in " ".join(lead_jobs[3]["required_evidence"])
    assert "repeat-run manifest" in " ".join(lead_jobs[3]["required_evidence"])
    assert "profiler preflight manifest" in " ".join(lead_jobs[3]["required_evidence"])
    assert "artifact contract manifest" in " ".join(lead_jobs[3]["required_evidence"])
    assert "claim packet links" in " ".join(lead_jobs[3]["required_evidence"])
    assert "output verification record" in lead_jobs[0]["expected_artifacts"]

    markdown = render_opportunities_markdown(result)
    assert "## Novelty Experiment Playbooks" in markdown
    assert "## Novelty Mutation Plan" in markdown
    assert "First mutation:" in markdown
    assert "## Novelty Mutation Budget Plan" in markdown
    assert "Deferred mutations:" in markdown
    assert "## Novelty Budget Plan" in markdown
    assert "Mitigate:" in markdown
    assert "Deferred:" in markdown
    assert "## Novelty Decision Frontier" in markdown
    assert "`high_upside`" in markdown
    assert "## Novelty Falsification Plan" in markdown
    assert "Disprove if:" in markdown
    assert "## Novelty Ablation Plan" in markdown
    assert "Control:" in markdown
    assert "## Novelty Reproducibility Plan" in markdown
    assert "Stability metrics:" in markdown
    assert "## Novelty Instrumentation Plan" in markdown
    assert "Profiler tools:" in markdown
    assert "## Novelty Artifact Contract Plan" in markdown
    assert "artifact_contract_manifest.json" in markdown
    assert "## Novelty Claim Packet Plan" in markdown
    assert "First blocked overclaim:" in markdown
    assert "## Novelty Validation Plan" in markdown
    assert "manual novelty review" in markdown


def test_frontier_ranking_prefers_high_leverage_signals_over_alphabetical_order() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            "ch01:plain_probe",
            "ch01:gemm",
            "ch04:nvlink_topology_aware_multigpu",
            "ch15:kv_decode_cache",
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=4)

    targets = [row["target"] for row in result["opportunities"]]
    assert targets[:3] == [
        "ch15:kv_decode_cache",
        "ch04:nvlink_topology_aware_multigpu",
        "ch01:gemm",
    ]
    assert result["opportunities"][0]["frontier_motif"] == "attention_kv_layout"
    assert "serving_decode_hotpath" in result["opportunities"][0]["frontier_signals"]
    assert result["opportunities"][0]["score"] > result["opportunities"][-1]["score"]
    first_blueprint = result["opportunities"][0]["experiment_blueprints"][0]
    assert first_blueprint["name"] == "prefill_decode_phase_split"
    assert first_blueprint["profiler_recipe"]["profile_mode"] == "deep_dive"
    assert "nsys" in first_blueprint["profiler_recipe"]["tools"]
    assert "zymtrace" in first_blueprint["profiler_recipe"]["tools"]
    assert any(
        "CUDA injection library resolves" in check
        for check in first_blueprint["profiler_recipe"]["preflight_checks"]
    )

    frontier_map = result["frontier_discovery_map"]
    assert frontier_map["frontier_candidate_count"] == 4
    assert frontier_map["lane_count"] >= 3
    assert frontier_map["diversity_queue"][0]["lane"] == "attention_kv_layout"
    attention_lane = next(
        lane for lane in frontier_map["lanes"] if lane["lane"] == "attention_kv_layout"
    )
    assert attention_lane["experiment_blueprints"][0]["name"] == "kv_page_layout_sweep"
    assert "zymtrace_launch_manifest.json" in " ".join(
        attention_lane["experiment_blueprints"][0]["profiler_recipe"]["artifact_expectations"]
    )
    assert attention_lane["top_targets"][0]["blueprint_ids"][0].endswith("kv-page-layout-sweep")
    assert {lane["lane"] for lane in frontier_map["lanes"]} >= {
        "attention_kv_layout",
        "communication_overlap",
        "precision_tile_autotune",
    }


def test_benchmark_opportunities_api_handler_accepts_data_file(tmp_path: Path) -> None:
    data_file = tmp_path / "tier1_summary.json"
    data_file.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target": "labs/persistent_decode:persistent_decode",
                        "status": "succeeded",
                        "baseline_time_ms": 42.0,
                        "best_speedup": 1.04,
                        "optimization_goal": "performance",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = benchmark_opportunities({"data_file": str(data_file), "top": 1})

    assert result["source"] == str(data_file.resolve())
    assert result["opportunities"][0]["target"] == "labs/persistent_decode:persistent_decode"
    assert result["opportunities"][0]["opportunity_type"] == "rework_flat_optimization"


def test_benchmark_opportunities_api_handler_accepts_catalog_file(tmp_path: Path) -> None:
    data_file = tmp_path / "summary.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(
        json.dumps({"available_targets": ["labs/moe_cuda:router_vectorized"]}),
        encoding="utf-8",
    )

    result = benchmark_opportunities(
        {"data_file": str(data_file), "catalog_file": str(catalog_file), "top": 1}
    )

    assert result["target_catalog_source"] == str(catalog_file.resolve())
    assert result["opportunities"][0]["target"] == "labs/moe_cuda:router_vectorized"
    assert result["opportunities"][0]["opportunity_type"] == "novel_frontier_probe"


def test_benchmark_opportunities_api_can_include_discovered_targets(
    tmp_path: Path, monkeypatch
) -> None:
    data_file = tmp_path / "summary.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")

    class _BenchmarkDomain:
        def targets(self) -> dict:
            return {"targets": ["labs/flexattention:flex_prefill"]}

    class _Engine:
        benchmark = _BenchmarkDomain()

    monkeypatch.setattr(handlers, "get_engine", lambda: _Engine())

    result = benchmark_opportunities(
        {"data_file": str(data_file), "include_discovered_targets": True, "top": 1}
    )

    assert result["discovered_target_source"] == "benchmark discovery"
    assert result["summary"]["frontier_candidates"] == 1
    assert result["opportunities"][0]["target"] == "labs/flexattention:flex_prefill"
    assert result["opportunities"][0]["opportunity_type"] == "novel_frontier_probe"


def test_benchmark_opportunities_api_can_source_mine_discovered_targets(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "summary.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    bench_root = tmp_path / "bench"
    _write_benchmark_pair(
        bench_root,
        "ch01",
        "plain_probe",
        source="# FP8 NVFP4 CUTLASS matmul benchmark\n\ndef get_benchmark():\n    return None\n",
    )

    result = benchmark_opportunities(
        {
            "data_file": str(data_file),
            "include_discovered_targets": True,
            "bench_root": str(bench_root),
            "top": 1,
        }
    )

    assert result["summary"]["frontier_candidates"] == 1
    assert result["opportunities"][0]["target"] == "ch01:plain_probe"
    assert result["opportunities"][0]["frontier_motif"] == "precision_tile_autotune"
    assert "emerging_precision" in result["opportunities"][0]["frontier_signals"]


def test_benchmark_opportunities_route_and_mcp_tool_are_registered(tmp_path: Path) -> None:
    routes = {route.name: route for route in get_routes()}
    route = routes["benchmark.opportunities"]
    assert route.path == "/api/benchmark/opportunities"
    assert route.mcp_tool == "benchmark_opportunities"

    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "chapter": "ch04",
                        "benchmarks": [
                            {
                                "example": "gradient_fusion",
                                "status": "failed",
                                "baseline_time_ms": 20.0,
                                "best_speedup": 0.0,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    response = MCPServer().call_tool(
        "benchmark_opportunities",
        {"data_file": str(data_file), "top": 1, "include_context": False},
    )
    payload = json.loads(response.content[0]["text"])
    assert payload["tool"] == "benchmark_opportunities"
    assert payload["status"] == "ok"
    assert payload["result"]["opportunities"][0]["target"] == "ch04:gradient_fusion"


def test_opportunity_execution_plan_orders_evidence_repair_before_experiments() -> None:
    payload = {
        "benchmarks": [
            {
                "chapter": "ch04",
                "name": "gradient_fusion",
                "status": "failed",
                "baseline_time_ms": 20.0,
                "speedup": 0.0,
            },
            {
                "chapter": "ch15",
                "name": "kv_decode_cache",
                "status": "succeeded",
                "baseline_time_ms": 1200.0,
                "speedup": 1.02,
            },
        ]
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=2)

    phases = result["execution_plan"]["phases"]
    assert [phase["name"] for phase in phases[:2]] == [
        "restore_evidence",
        "deep_profile_headroom",
    ]
    assert phases[0]["items"][0]["command"].endswith(
        "ch04:gradient_fusion --profile minimal --verify-output"
    )
    assert phases[0]["items"][0]["benchmark_run"]["overrides"]["workloadType"] == "training"
    assert phases[1]["items"][0]["command"].endswith(
        "ch15:kv_decode_cache --profile deep_dive --verify-output"
    )

    markdown = render_opportunities_markdown(result)
    assert "## Execution Plan" in markdown
    assert "BenchmarkRun:" in markdown
    assert "Completion gate:" in markdown


def test_opportunity_benchmark_run_overrides_render_with_existing_contract() -> None:
    payload = {
        "benchmarks": [
            {
                "chapter": "ch09",
                "name": "cutlass_fp8_gemm",
                "status": "succeeded",
                "baseline_time_ms": 6.0,
                "speedup": 1.2,
                "baseline_file": "ch09/baseline_cutlass_fp8_gemm.py",
            }
        ]
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=1)
    overrides = result["opportunities"][0]["benchmark_run"]["overrides"]
    rendered = render_benchmark_run_yaml(overrides)

    assert rendered["applied_values"]["name"].startswith("opportunity-ch09-cutlass-fp8-gemm")
    assert rendered["applied_values"]["precision"] == "fp8"
    assert "kind: BenchmarkRun" in rendered["rendered_yaml"]


def test_opportunity_radar_clusters_transferable_innovation_hypotheses() -> None:
    payload = {
        "benchmarks": [
            {
                "chapter": "ch15",
                "name": "kv_decode_cache",
                "status": "succeeded",
                "baseline_time_ms": 1200.0,
                "speedup": 1.01,
                "baseline_file": "ch15/baseline_kv_decode_cache.py",
            },
            {
                "chapter": "labs/persistent_decode",
                "name": "persistent_decode",
                "status": "succeeded",
                "baseline_time_ms": 88.0,
                "speedup": 1.25,
                "rationale": "decode launch and KV-cache pressure remain visible",
            },
            {
                "chapter": "ch09",
                "name": "cutlass_fp8_gemm",
                "status": "succeeded",
                "baseline_time_ms": 6.0,
                "speedup": 1.2,
                "baseline_file": "ch09/baseline_cutlass_fp8_gemm.py",
            },
        ]
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=3)

    hypotheses = result["innovation_hypotheses"]
    attention = hypotheses["clusters"][0]
    assert hypotheses["cluster_count"] >= 2
    assert attention["motif"] == "attention_kv_layout"
    assert attention["prototype_target"] == "ch15:kv_decode_cache"
    assert attention["support_count"] == 2
    assert "KV page/block size" in " ".join(attention["transfer_experiments"])
    assert attention["validation_commands"][0].endswith(
        "ch15:kv_decode_cache --profile minimal --verify-output"
    )

    markdown = render_opportunities_markdown(result)
    assert "## Innovation Hypotheses" in markdown
    assert "## Experiment Matrix" in markdown
    assert "## Portfolio Plan" in markdown
    assert "## Promotion Gates" in markdown
    assert "## Run Queue" in markdown
    assert "Critical path groups:" in markdown
    assert "Transfer attention and KV-cache layout experiments" in markdown

    matrix = result["experiment_matrix"]
    kv_card = next(card for card in matrix["cards"] if card["target"] == "ch15:kv_decode_cache")
    assert kv_card["motif"] == "attention_kv_layout"
    assert kv_card["primary_metric"] == "median_wall_time_ms"
    assert kv_card["variants"][0]["name"] == "control_verified_current"
    assert any("kv-page-block-size" in variant["name"] for variant in kv_card["variants"])

    portfolio = result["portfolio_plan"]
    assert portfolio["budget_slots"] == 5
    selected_motifs = {item["motif"] for item in portfolio["selected"]}
    assert {"attention_kv_layout", "precision_tile_autotune"} <= selected_motifs
    assert portfolio["waves"][0]["name"] == "diverse_optimization_batch"

    gates = result["promotion_gates"]
    assert gates["blocked_count"] == gates["gate_count"]
    kv_gate = next(gate for gate in gates["gates"] if gate["target"] == "ch15:kv_decode_cache")
    assert kv_gate["promotion_state"] == "blocked_until_variant_validated"
    assert kv_gate["claim_allowed"] is False
    assert "support target reproduces" in " ".join(kv_gate["required_evidence"])
    assert kv_gate["benchmark_run"]["mcp_tool"] == "render_benchmark_run"

    queue = result["run_queue"]
    assert queue["job_count"] >= 3
    assert queue["critical_path_groups"] >= 3
    assert queue["dispatch_groups"][0]["stage_mix"] == {"control": len(queue["ready_job_ids"])}
    assert all(job_id.endswith("control") for job_id in queue["ready_job_ids"])
    kv_jobs = [job for job in queue["jobs"] if job["target"] == "ch15:kv_decode_cache"]
    assert [job["stage"] for job in kv_jobs[:3]] == [
        "control",
        "candidate",
        "promotion_review",
    ]
    assert kv_jobs[1]["depends_on"] == [kv_jobs[0]["id"]]
    assert kv_jobs[2]["depends_on"] == [kv_jobs[1]["id"]]
    assert "output verification record" in kv_jobs[0]["expected_artifacts"]
    assert "primary and guardrail metrics are captured" in kv_jobs[1]["success_criteria"]
    assert "support target reproduces" in " ".join(kv_jobs[2]["success_criteria"])


def test_opportunity_radar_adds_frontier_targets_from_catalog() -> None:
    payload = {
        "benchmarks": [
            {
                "chapter": "ch15",
                "name": "kv_decode_cache",
                "status": "succeeded",
                "baseline_time_ms": 1200.0,
                "speedup": 1.01,
            }
        ],
        "target_catalog": [
            "ch15:kv_decode_cache",
            {
                "target": "labs/flexattention:flex_prefill",
                "category": "attention",
                "rationale": "unmeasured attention frontier",
            },
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=5)

    frontier = [
        row for row in result["opportunities"] if row["opportunity_type"] == "novel_frontier_probe"
    ]
    assert result["summary"]["frontier_candidates"] == 1
    assert frontier[0]["target"] == "labs/flexattention:flex_prefill"
    assert frontier[0]["priority"] == "high"
    assert "first clean evidence" in " ".join(frontier[0]["recommended_experiments"])
    assert result["execution_plan"]["phases"][0]["name"] == "explore_frontier"
    markdown = render_opportunities_markdown(result)
    assert "## Frontier Discovery Map" in markdown
    assert "Blueprint:" in markdown

    matrix = result["experiment_matrix"]
    frontier_card = next(
        card for card in matrix["cards"] if card["target"] == "labs/flexattention:flex_prefill"
    )
    assert frontier_card["experiment_blueprints"][0]["name"] == "kv_page_layout_sweep"
    assert frontier_card["primary_metric"] == "verified_status"
    assert "baseline_optimized_pair_exists" in frontier_card["guardrail_metrics"]
    assert [variant["name"] for variant in frontier_card["variants"][:3]] == [
        "control_verified_current",
        "frontier_minimal_smoke",
        "frontier_deep_dive_followup",
    ]

    portfolio = result["portfolio_plan"]
    assert portfolio["waves"][0]["name"] == "evidence_and_frontier"
    assert portfolio["selected"][0]["target"] == "labs/flexattention:flex_prefill"
    assert portfolio["selected"][0]["first_variant"] == "frontier_minimal_smoke"
    assert portfolio["selected"][0]["experiment_blueprint_ids"][0].endswith("kv-page-layout-sweep")

    gate = result["promotion_gates"]["gates"][0]
    assert gate["target"] == "labs/flexattention:flex_prefill"
    assert gate["promotion_state"] == "blocked_until_first_evidence"
    assert gate["claim_allowed"] is False
    assert "first verified minimal-profile run succeeds" in gate["required_evidence"]

    queue = result["run_queue"]
    frontier_jobs = [
        job for job in queue["jobs"] if job["target"] == "labs/flexattention:flex_prefill"
    ]
    assert [job["stage"] for job in frontier_jobs[:4]] == [
        "control",
        "candidate",
        "profile_followup",
        "promotion_review",
    ]
    assert frontier_jobs[2]["depends_on"] == [frontier_jobs[1]["id"]]
    assert frontier_jobs[3]["depends_on"] == [frontier_jobs[2]["id"]]
    assert frontier_jobs[1]["experiment_blueprint_ids"][0].endswith("kv-page-layout-sweep")
    assert frontier_jobs[1]["experiment_blueprint"]["name"] == "kv_page_layout_sweep"
    assert "deep-dive profiler trace or kernel summary" in frontier_jobs[2]["expected_artifacts"]
    assert any(
        group["stage_mix"].get("profile_followup") == 1 for group in queue["dispatch_groups"]
    )
    assert queue["critical_path_groups"] >= 4


def test_render_run_queue_shell_preserves_dependencies_and_manual_gates() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [{"target": "labs/flexattention:flex_prefill", "category": "attention"}],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=1)
    script = render_run_queue_shell(result)

    assert "require_dependency labs-flexattention-flex-prefill-control" in script
    assert (
        "bash -lc 'python -m cli.aisp bench run --targets labs/flexattention:flex_prefill --profile minimal --verify-output'"
        in script
    )
    assert "promotion_review.md" in script
    assert "first verified minimal-profile run succeeds" in script
    assert "MANUAL_REVIEW_REQUIRED" in script
    assert "APPROVED" in script
    assert "skip: DONE already exists" in script
    assert "manual review pending" in script


def test_render_novelty_validation_shell_preserves_dependencies_and_review_gates() -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/distributed_decode:tp_decode_overlap",
                "category": "communication_overlap",
                "rationale": "distributed serving decode target with exposed collective time",
                "source_terms": ["decode", "serving", "nccl", "allreduce"],
                "frontier_signal_matches": [
                    {
                        "signal": "serving_decode_hotpath",
                        "matched_terms": ["decode", "serving"],
                    },
                    {
                        "signal": "distributed_fabric",
                        "matched_terms": ["nccl", "allreduce"],
                    },
                ],
            }
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=1)
    script = render_novelty_validation_shell(result)

    assert "AISP_NOVELTY_QUEUE_ROOT" in script
    assert "Novelty validation root" in script
    assert "require_dependency" in script
    assert (
        "bash -lc 'python -m cli.aisp bench run --targets labs/distributed_decode:tp_decode_overlap --profile minimal --verify-output'"
        in script
    )
    assert "Novelty Review" in script
    assert "Claim boundary:" in script
    assert "Falsification checks:" in script
    assert "Risk mitigations:" in script
    assert "Ablation controls:" in script
    assert "Reproducibility gate:" in script
    assert "Instrumentation preflight:" in script
    assert "Claim packet:" in script
    assert "claim_packet.json" in script
    assert "claim_packet.md" in script
    assert "Disallowed claims:" in script
    assert "Artifact contract:" in script
    assert "artifact_contract.json" in script
    assert "artifact_contract_manifest.json" in script
    assert "resolved falsification checklist" in script
    assert "ablation control summary" in script
    assert "repeat-run manifest" in script
    assert "profiler preflight manifest" in script
    assert "same workload contract" in script
    assert "distributed topology" in script
    assert "Manual reviews remain evidence gates" in script
    assert "APPROVED" in script
    assert "skip: DONE already exists" in script


def test_render_novelty_next_wave_shell_preserves_recovery_and_learning_actions(
    tmp_path: Path,
) -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/distributed_decode:tp_decode_overlap",
                "category": "communication_overlap",
                "source_terms": ["decode", "serving", "nccl", "allreduce"],
                "frontier_signal_matches": [
                    {"signal": "serving_decode_hotpath", "matched_terms": ["decode"]},
                    {"signal": "distributed_fabric", "matched_terms": ["nccl"]},
                ],
            }
        ],
    }

    result = rank_opportunities(normalize_candidates(payload), top_n=1)
    queue_root = tmp_path / "novelty"
    control_job, candidate_job = result["novelty_validation_plan"]["jobs"][:2]
    _write_run_queue_job(queue_root, control_job["id"], control_job, done=True)
    _write_run_queue_job(
        queue_root,
        candidate_job["id"],
        candidate_job,
        stderr="Zymtrace CUDA_INJECTION64_PATH injection library not found\n",
    )
    overlay = apply_novelty_validation_feedback(result, summarize_run_queue_root(queue_root))

    script = render_novelty_next_wave_shell(overlay)

    assert "AISP_NOVELTY_NEXT_WAVE_ROOT" in script
    assert "Novelty Next Wave Action" in script
    assert "recover_blocked_leads" in script
    assert "CUDA_INJECTION64_PATH" in script
    assert "recovery_stdout.log" in script
    assert candidate_job["command"] in script
    assert "MANUAL_ACTION_REQUIRED" in script
    assert "apply_learning_before_rerank" in script


def test_summarize_run_queue_root_classifies_runbook_evidence(tmp_path: Path) -> None:
    root = tmp_path / "queue"
    root.mkdir()
    _write_run_queue_job(
        root,
        "01_control",
        {
            "id": "target-control",
            "target": "labs/flexattention:flex_prefill",
            "stage": "control",
            "command": "echo control",
        },
        done=True,
    )
    _write_run_queue_job(
        root,
        "02_candidate",
        {
            "id": "target-candidate",
            "target": "labs/flexattention:flex_prefill",
            "stage": "candidate",
            "command": "false",
            "depends_on": ["target-control"],
        },
        stdout="started\n",
        stderr="Zymtrace CUDA_INJECTION64_PATH injection library not found\n",
    )
    _write_run_queue_job(
        root,
        "03_profile",
        {
            "id": "target-profile",
            "target": "labs/flexattention:flex_prefill",
            "stage": "profile_followup",
            "command": "echo profile",
            "depends_on": ["target-candidate"],
        },
    )
    _write_run_queue_job(
        root,
        "04_review",
        {
            "id": "target-review",
            "target": "labs/flexattention:flex_prefill",
            "stage": "promotion_review",
            "depends_on": ["target-control"],
            "promotion_gate": "frontier_first_evidence_required",
            "required_evidence": ["first verified minimal-profile run succeeds"],
        },
        manual=True,
        review="# Promotion Review\n",
    )
    _write_run_queue_job(
        root,
        "05_approved_review",
        {
            "id": "target-approved-review",
            "target": "labs/flexattention:flex_prefill",
            "stage": "promotion_review",
            "depends_on": ["target-control"],
            "promotion_gate": "frontier_first_evidence_required",
        },
        approved=True,
        review="# Promotion Review\n",
    )

    summary = summarize_run_queue_root(root)

    assert summary["exists"] is True
    assert summary["job_count"] == 5
    assert summary["status_counts"]["completed"] == 1
    assert summary["status_counts"]["failed_or_incomplete"] == 1
    assert summary["status_counts"]["blocked_by_dependency"] == 1
    assert summary["status_counts"]["manual_review_required"] == 1
    assert summary["status_counts"]["approved"] == 1
    assert summary["promotion_summary"]["approved"] == ["target-approved-review"]
    assert summary["promotion_summary"]["manual_review_required"] == ["target-review"]
    assert summary["promotion_summary"]["claim_allowed_count"] == 1
    assert summary["jobs"][1]["stderr_tail"].endswith("injection library not found\n")
    assert summary["jobs"][1]["diagnostic_signals"][0]["signal"] == "zymtrace_injection_missing"
    assert "zymtrace_injection_missing" in summary["next_actions"][0]


def test_bench_opportunity_run_summary_cli_outputs_json(tmp_path: Path) -> None:
    root = tmp_path / "queue"
    _write_run_queue_job(
        root,
        "01_control",
        {
            "id": "target-control",
            "target": "labs/flexattention:flex_prefill",
            "stage": "control",
            "command": "echo control",
        },
        done=True,
    )

    result = CliRunner().invoke(
        app,
        ["opportunity-run-summary", "--run-queue-root", str(root), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["job_count"] == 1
    assert payload["status_counts"] == {"completed": 1}


def test_apply_run_queue_feedback_unlocks_approved_promotions_and_resume_plan(
    tmp_path: Path,
) -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [{"target": "labs/flexattention:flex_prefill", "category": "attention"}],
    }
    result = rank_opportunities(normalize_candidates(payload), top_n=1)
    queue_root = tmp_path / "queue"
    for job in result["run_queue"]["jobs"]:
        status_kwargs = {"done": True}
        if job["stage"] == "promotion_review":
            status_kwargs = {"approved": True, "review": "# Promotion Review\n"}
        _write_run_queue_job(queue_root, job["id"], job, **status_kwargs)

    overlay = apply_run_queue_feedback(result, summarize_run_queue_root(queue_root))

    gate = overlay["promotion_gates"]["gates"][0]
    assert gate["claim_allowed"] is True
    assert gate["promotion_state"] == "approved_after_manual_review"
    assert gate["run_queue_promotion_status"] == "approved"
    assert overlay["promotion_gates"]["claim_allowed_count"] == 1
    assert overlay["run_queue"]["resume_plan"]["ready_job_ids"] == []
    assert (
        overlay["run_queue_feedback"]["targets"]["labs/flexattention:flex_prefill"][
            "promotion_status"
        ]
        == "approved"
    )
    assert all(
        job["resume_action"] in {"skip_completed", "skip_approved_promotion"}
        for job in overlay["run_queue"]["jobs"]
    )


def test_apply_run_queue_feedback_finds_next_runnable_job_after_partial_run(
    tmp_path: Path,
) -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [{"target": "labs/flexattention:flex_prefill", "category": "attention"}],
    }
    result = rank_opportunities(normalize_candidates(payload), top_n=1)
    queue_root = tmp_path / "queue"
    control_job = result["run_queue"]["jobs"][0]
    _write_run_queue_job(queue_root, control_job["id"], control_job, done=True)

    overlay = apply_run_queue_feedback(result, summarize_run_queue_root(queue_root))

    candidate = result["run_queue"]["jobs"][1]
    profile = result["run_queue"]["jobs"][2]
    resume_plan = overlay["run_queue"]["resume_plan"]
    assert resume_plan["ready_job_ids"] == [candidate["id"]]
    assert candidate["command"] in resume_plan["next_commands"]
    assert profile["id"] in resume_plan["blocked_job_ids"]
    annotated_candidate = next(
        job for job in overlay["run_queue"]["jobs"] if job["id"] == candidate["id"]
    )
    assert annotated_candidate["resume_action"] == "run_next"


def test_apply_novelty_validation_feedback_finds_next_runnable_job_after_partial_run(
    tmp_path: Path,
) -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/distributed_decode:tp_decode_overlap",
                "category": "communication_overlap",
                "source_terms": ["decode", "serving", "nccl", "allreduce"],
                "frontier_signal_matches": [
                    {"signal": "serving_decode_hotpath", "matched_terms": ["decode"]},
                    {"signal": "distributed_fabric", "matched_terms": ["nccl"]},
                ],
            }
        ],
    }
    result = rank_opportunities(normalize_candidates(payload), top_n=1)
    queue_root = tmp_path / "novelty"
    control_job = result["novelty_validation_plan"]["jobs"][0]
    _write_run_queue_job(queue_root, control_job["id"], control_job, done=True)
    control_dir = queue_root / control_job["id"]
    for filename in control_job["artifact_contract"]["required_files"]:
        path = control_dir / filename
        if not path.exists():
            path.write_text(f"{filename}\n", encoding="utf-8")

    overlay = apply_novelty_validation_feedback(result, summarize_run_queue_root(queue_root))

    candidate = result["novelty_validation_plan"]["jobs"][1]
    profile = result["novelty_validation_plan"]["jobs"][2]
    lead_id = control_job["lead_id"]
    resume_plan = overlay["novelty_validation_plan"]["resume_plan"]
    assert candidate["id"] in resume_plan["ready_job_ids"]
    assert candidate["command"] in resume_plan["next_commands"]
    assert profile["id"] in resume_plan["blocked_job_ids"]
    assert (
        overlay["novelty_validation_feedback"]["leads"][lead_id]["validation_status"]
        == "in_progress"
    )
    assert overlay["novelty_validation_plan"]["selected_leads"][0]["validation_feedback"][
        "completed_job_ids"
    ] == [control_job["id"]]
    assert (
        overlay["novelty_budget_plan"]["selected"][0]["validation_feedback"]["lead_id"] == lead_id
    )
    evidence_audit = overlay["novelty_evidence_audit_plan"]
    lead_audit = next(
        item for item in evidence_audit["lead_audits"] if item["lead_id"] == lead_id
    )
    control_stage_audit = next(
        item for item in lead_audit["stage_audits"] if item["stage"] == "control"
    )
    candidate_stage_audit = next(
        item for item in lead_audit["stage_audits"] if item["job_id"] == candidate["id"]
    )
    assert control_stage_audit["audit_status"] == "complete"
    assert candidate_stage_audit["audit_status"] == "missing_job"
    assert lead_audit["audit_status"] == "evidence_packet_incomplete"
    assert lead_audit["missing_file_count"] > 0
    assert overlay["novelty_validation_feedback"]["leads"][lead_id][
        "evidence_audit_status"
    ] == "evidence_packet_incomplete"
    adaptive_decision = next(
        item
        for item in overlay["novelty_adaptive_decision_plan"]["selected_decisions"]
        if item["lead_id"] == lead_id
    )
    assert adaptive_decision["disposition"] == "run_next_validation_job"
    assert candidate["id"] in adaptive_decision["ready_job_ids"]
    assert overlay["novelty_budget_plan"]["selected"][0]["adaptive_decision"][
        "lead_id"
    ] == lead_id
    learning_feedback = overlay["novelty_budget_plan"]["selected"][0]["learning_feedback"]
    assert learning_feedback["rerank_action"] == "continue_selected_validation"
    assert learning_feedback["expected_value_adjustment"] > 0
    assert overlay["novelty_learning_plan"]["portfolio_guidance"]["continue_count"] >= 1
    next_wave = overlay["novelty_next_wave_plan"]
    continue_wave = next(
        wave for wave in next_wave["waves"] if wave["name"] == "continue_active_validation"
    )
    assert continue_wave["items"][0]["job_id"] == candidate["id"]
    assert continue_wave["items"][0]["command"] == candidate["command"]
    mutation_wave = next(
        wave for wave in next_wave["waves"] if wave["name"] == "run_selected_mutations"
    )
    assert mutation_wave["items"][0]["mutation_id"]
    assert mutation_wave["items"][0]["source"] == "novelty_mutation_budget_plan.selected"
    assert mutation_wave["items"][0]["isolation_rule"]
    mutation_budget_rows = [
        *overlay["novelty_mutation_budget_plan"]["selected"],
        *overlay["novelty_mutation_budget_plan"]["backlog"],
    ]
    mutation_budget_for_lead = next(
        item for item in mutation_budget_rows if item["lead_id"] == lead_id
    )
    assert mutation_budget_for_lead["validation_feedback"]["lead_id"] == lead_id
    assert (
        overlay["novelty_validation_feedback"]["mutation_budget_summary"]["selected_count"]
        >= 1
    )
    assert (
        overlay["novelty_validation_feedback"]["next_wave_summary"]["action_count"] >= 1
    )
    contract_with_audit = next(
        item
        for item in overlay["novelty_artifact_contract_plan"]["lead_contracts"]
        if item["lead_id"] == lead_id
    )
    packet_with_audit = next(
        item
        for item in overlay["novelty_claim_packet_plan"]["lead_packets"]
        if item["lead_id"] == lead_id
    )
    assert contract_with_audit["evidence_audit"]["lead_id"] == lead_id
    assert packet_with_audit["evidence_audit"]["lead_id"] == lead_id


def test_apply_novelty_validation_feedback_harvests_approved_audited_claims(
    tmp_path: Path,
) -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/flexattention:flex_prefill",
                "category": "attention",
                "source_terms": ["attention", "prefill", "flash"],
                "frontier_signal_matches": [
                    {"signal": "attention_or_kv", "matched_terms": ["attention"]},
                ],
            }
        ],
    }
    result = rank_opportunities(normalize_candidates(payload), top_n=1)
    queue_root = tmp_path / "novelty"
    lead_id = result["novelty_validation_plan"]["selected_leads"][0]["lead_id"]
    lead_jobs = [
        job for job in result["novelty_validation_plan"]["jobs"] if job["lead_id"] == lead_id
    ]
    review_job = next(job for job in lead_jobs if job["stage"] == "manual_review")
    for job in lead_jobs:
        _write_run_queue_job(
            queue_root,
            job["id"],
            job,
            done=job["stage"] != "manual_review",
            approved=job["stage"] == "manual_review",
            manual=job["stage"] == "manual_review",
            review="approved claim packet\n" if job["stage"] == "manual_review" else None,
        )
        _write_required_contract_files(queue_root, job)

    overlay = apply_novelty_validation_feedback(result, summarize_run_queue_root(queue_root))

    adaptive_decision = next(
        item
        for item in overlay["novelty_adaptive_decision_plan"]["selected_decisions"]
        if item["lead_id"] == lead_id
    )
    assert adaptive_decision["disposition"] == "claim_ready"
    harvest = overlay["novelty_harvest_plan"]
    assert harvest["pattern_count"] == 1
    assert harvest["blocked_harvest_count"] == 0
    pattern = harvest["patterns"][0]
    assert pattern["source_lead_id"] == lead_id
    assert pattern["claim_packet_id"]
    assert review_job["id"] in pattern["evidence_job_ids"]
    assert pattern["disallowed_claims"]
    followup = harvest["followup_experiments"][0]
    assert followup["source_pattern_id"] == pattern["pattern_id"]
    assert followup["followup_type"] in {
        "deep_dive_after_first_evidence",
        "coverage_gap_replication",
        "mine_related_targets",
    }
    if followup["followup_type"] == "deep_dive_after_first_evidence":
        assert followup["target"] == "labs/flexattention:flex_prefill"
        assert "--profile deep_dive" in followup["command"]
    elif followup["followup_type"] == "mine_related_targets":
        assert followup["target"] is None
        assert followup["command"] is None
        assert "mine the target catalog" in followup["reason"]
    else:
        assert followup["target"]
        assert followup["command"]
    assert overlay["novelty_validation_feedback"]["harvest_summary"]["harvest_count"] == 1
    harvest_wave = next(
        wave for wave in overlay["novelty_next_wave_plan"]["waves"] if wave["name"] == "run_harvest_followups"
    )
    assert harvest_wave["items"][0]["followup_id"] == followup["followup_id"]
    assert harvest_wave["items"][0]["command"] == followup["command"]
    assert overlay["novelty_validation_feedback"]["next_wave_summary"]["wave_item_counts"][
        "run_harvest_followups"
    ] == 1
    assert "## Novelty Harvest Plan" in render_opportunities_markdown(overlay)


def test_apply_novelty_validation_feedback_builds_recovery_plan_for_failed_jobs(
    tmp_path: Path,
) -> None:
    payload = {
        "benchmarks": [],
        "target_catalog": [
            {
                "target": "labs/distributed_decode:tp_decode_overlap",
                "category": "communication_overlap",
                "source_terms": ["decode", "serving", "nccl", "allreduce"],
                "frontier_signal_matches": [
                    {"signal": "serving_decode_hotpath", "matched_terms": ["decode"]},
                    {"signal": "distributed_fabric", "matched_terms": ["nccl"]},
                ],
            }
        ],
    }
    result = rank_opportunities(normalize_candidates(payload), top_n=1)
    queue_root = tmp_path / "novelty"
    control_job, candidate_job = result["novelty_validation_plan"]["jobs"][:2]
    _write_run_queue_job(queue_root, control_job["id"], control_job, done=True)
    _write_run_queue_job(
        queue_root,
        candidate_job["id"],
        candidate_job,
        stdout="started candidate\n",
        stderr="Zymtrace CUDA_INJECTION64_PATH injection library not found\n",
    )

    overlay = apply_novelty_validation_feedback(result, summarize_run_queue_root(queue_root))

    recovery = overlay["novelty_recovery_plan"]
    zymtrace_action = next(
        action for action in recovery["actions"] if action["issue_type"] == "zymtrace_injection_missing"
    )
    assert recovery["action_count"] >= 1
    assert recovery["blocking_action_count"] >= 1
    assert zymtrace_action["job_id"] == candidate_job["id"]
    assert "CUDA_INJECTION64_PATH" in zymtrace_action["recovery_command"]
    assert zymtrace_action["rerun_after_recovery"] == candidate_job["command"]
    adaptive_decision = next(
        item
        for item in overlay["novelty_adaptive_decision_plan"]["selected_decisions"]
        if item["lead_id"] == candidate_job["lead_id"]
    )
    assert adaptive_decision["disposition"] == "recover_failed_evidence"
    assert adaptive_decision["slot_state"] == "blocked"
    assert zymtrace_action["action_id"] in adaptive_decision["recovery_action_ids"]
    learning_adjustment = next(
        item
        for item in overlay["novelty_learning_plan"]["lead_adjustments"]
        if item["lead_id"] == candidate_job["lead_id"]
    )
    assert learning_adjustment["rerank_action"] == "hold_selected_slot_until_recovery"
    assert learning_adjustment["expected_value_adjustment"] < 0
    assert "zymtrace_injection_required" in learning_adjustment["risk_updates"]
    assert "profiler_preflight_required" in learning_adjustment["risk_updates"]
    next_wave = overlay["novelty_next_wave_plan"]
    assert next_wave["first_wave"] == "recover_blocked_leads"
    recover_wave = next(
        wave for wave in next_wave["waves"] if wave["name"] == "recover_blocked_leads"
    )
    assert any(item["action_id"] == zymtrace_action["action_id"] for item in recover_wave["items"])
    backup_wave = next(wave for wave in next_wave["waves"] if wave["name"] == "activate_backups")
    assert backup_wave["items"][0]["replacement_for"] == candidate_job["lead_id"]
    assert (
        overlay["novelty_validation_feedback"]["next_wave_summary"]["first_wave"]
        == "recover_blocked_leads"
    )
    annotated_candidate = next(
        job
        for job in overlay["novelty_validation_plan"]["jobs"]
        if job["id"] == candidate_job["id"]
    )
    assert annotated_candidate["diagnostic_signals"][0]["signal"] == "zymtrace_injection_missing"
    lead_feedback = overlay["novelty_validation_feedback"]["leads"][candidate_job["lead_id"]]
    assert zymtrace_action["action_id"] in lead_feedback["recovery_action_ids"]
    assert (
        overlay["novelty_validation_feedback"]["recovery_summary"]["blocking_action_count"] >= 1
    )
    assert (
        overlay["novelty_validation_feedback"]["adaptive_decision_summary"]["blocked_count"]
        >= 1
    )


def test_bench_opportunities_cli_overlays_run_queue_feedback(tmp_path: Path) -> None:
    data_file = tmp_path / "summary.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    catalog_file = tmp_path / "targets.json"
    catalog_file.write_text(
        json.dumps([{"target": "labs/flexattention:flex_prefill", "category": "attention"}]),
        encoding="utf-8",
    )
    seed = rank_opportunities(
        normalize_candidates(
            {
                "benchmarks": [],
                "target_catalog": [
                    {"target": "labs/flexattention:flex_prefill", "category": "attention"}
                ],
            }
        ),
        top_n=1,
    )
    queue_root = tmp_path / "queue"
    control_job = seed["run_queue"]["jobs"][0]
    _write_run_queue_job(queue_root, control_job["id"], control_job, done=True)

    result = CliRunner().invoke(
        app,
        [
            "opportunities",
            "--data-file",
            str(data_file),
            "--catalog-file",
            str(catalog_file),
            "--run-queue-root",
            str(queue_root),
            "--json",
            "--top",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_queue"]["resume_plan"]["ready_job_ids"] == [
        seed["run_queue"]["jobs"][1]["id"]
    ]
    assert payload["opportunities"][0]["run_queue_feedback"]["completed_job_ids"] == [
        control_job["id"]
    ]


def test_bench_opportunities_cli_overlays_novelty_validation_feedback(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "summary.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    catalog_file = tmp_path / "targets.json"
    target = {
        "target": "labs/distributed_decode:tp_decode_overlap",
        "category": "communication_overlap",
        "source_terms": ["decode", "serving", "nccl", "allreduce"],
        "frontier_signal_matches": [
            {"signal": "serving_decode_hotpath", "matched_terms": ["decode"]},
            {"signal": "distributed_fabric", "matched_terms": ["nccl"]},
        ],
    }
    catalog_file.write_text(json.dumps([target]), encoding="utf-8")
    seed = rank_opportunities(
        normalize_candidates({"benchmarks": [], "target_catalog": [target]}),
        top_n=1,
    )
    queue_root = tmp_path / "novelty"
    control_job = seed["novelty_validation_plan"]["jobs"][0]
    _write_run_queue_job(queue_root, control_job["id"], control_job, done=True)

    result = CliRunner().invoke(
        app,
        [
            "opportunities",
            "--data-file",
            str(data_file),
            "--catalog-file",
            str(catalog_file),
            "--novelty-queue-root",
            str(queue_root),
            "--json",
            "--top",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert (
        seed["novelty_validation_plan"]["jobs"][1]["id"]
        in payload["novelty_validation_plan"]["resume_plan"]["ready_job_ids"]
    )
    lead_id = control_job["lead_id"]
    assert payload["novelty_validation_feedback"]["leads"][lead_id]["completed_job_ids"] == [
        control_job["id"]
    ]
    assert payload["novelty_evidence_audit_plan"]["lead_count"] >= 1
    assert (
        payload["novelty_validation_feedback"]["evidence_audit_summary"][
            "missing_file_count"
        ]
        > 0
    )
    assert payload["novelty_recovery_plan"]["action_count"] > 0
    assert payload["novelty_validation_feedback"]["recovery_summary"]["action_count"] > 0
    assert payload["novelty_adaptive_decision_plan"]["selected_count"] >= 1
    assert payload["novelty_validation_feedback"]["adaptive_decision_summary"][
        "slot_state_counts"
    ]
    assert payload["novelty_learning_plan"]["lead_count"] >= 1
    assert payload["novelty_validation_feedback"]["learning_summary"]["learning_state_counts"]
    assert payload["novelty_next_wave_plan"]["wave_count"] >= 1
    assert payload["novelty_validation_feedback"]["next_wave_summary"]["action_count"] >= 1
    assert payload["novelty_mutation_budget_plan"]["selected_count"] >= 1
    assert (
        payload["novelty_validation_feedback"]["mutation_budget_summary"]["selected_count"]
        >= 1
    )
    assert payload["novelty_harvest_plan"]["harvest_count"] >= 0
    assert "harvest_summary" in payload["novelty_validation_feedback"]


def test_benchmark_opportunities_api_overlays_run_queue_feedback(tmp_path: Path) -> None:
    data_file = tmp_path / "summary.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    catalog_file = tmp_path / "targets.json"
    catalog_file.write_text(
        json.dumps([{"target": "labs/flexattention:flex_prefill", "category": "attention"}]),
        encoding="utf-8",
    )
    seed = rank_opportunities(
        normalize_candidates(
            {
                "benchmarks": [],
                "target_catalog": [
                    {"target": "labs/flexattention:flex_prefill", "category": "attention"}
                ],
            }
        ),
        top_n=1,
    )
    queue_root = tmp_path / "queue"
    control_job = seed["run_queue"]["jobs"][0]
    _write_run_queue_job(queue_root, control_job["id"], control_job, done=True)

    payload = benchmark_opportunities(
        {
            "data_file": str(data_file),
            "catalog_file": str(catalog_file),
            "run_queue_root": str(queue_root),
            "top": 1,
        }
    )

    assert payload["run_queue_feedback"]["target_count"] == 1
    assert payload["run_queue"]["resume_plan"]["ready_job_ids"] == [
        seed["run_queue"]["jobs"][1]["id"]
    ]


def test_benchmark_opportunities_api_overlays_novelty_validation_feedback(
    tmp_path: Path,
) -> None:
    data_file = tmp_path / "summary.json"
    data_file.write_text(json.dumps({"benchmarks": []}), encoding="utf-8")
    catalog_file = tmp_path / "targets.json"
    target = {
        "target": "labs/distributed_decode:tp_decode_overlap",
        "category": "communication_overlap",
        "source_terms": ["decode", "serving", "nccl", "allreduce"],
        "frontier_signal_matches": [
            {"signal": "serving_decode_hotpath", "matched_terms": ["decode"]},
            {"signal": "distributed_fabric", "matched_terms": ["nccl"]},
        ],
    }
    catalog_file.write_text(json.dumps([target]), encoding="utf-8")
    seed = rank_opportunities(
        normalize_candidates({"benchmarks": [], "target_catalog": [target]}),
        top_n=1,
    )
    queue_root = tmp_path / "novelty"
    control_job = seed["novelty_validation_plan"]["jobs"][0]
    _write_run_queue_job(queue_root, control_job["id"], control_job, done=True)

    payload = benchmark_opportunities(
        {
            "data_file": str(data_file),
            "catalog_file": str(catalog_file),
            "novelty_queue_root": str(queue_root),
            "top": 1,
        }
    )

    assert payload["novelty_validation_feedback"]["lead_count"] >= 1
    assert payload["novelty_validation_feedback"]["leads"][control_job["lead_id"]][
        "completed_job_ids"
    ] == [control_job["id"]]
    assert payload["novelty_evidence_audit_plan"]["lead_count"] >= 1
    assert (
        payload["novelty_validation_feedback"]["evidence_audit_summary"][
            "missing_file_count"
        ]
        > 0
    )
    assert payload["novelty_recovery_plan"]["action_count"] > 0
    assert payload["novelty_validation_feedback"]["recovery_summary"]["action_count"] > 0
    assert payload["novelty_adaptive_decision_plan"]["selected_count"] >= 1
    assert payload["novelty_validation_feedback"]["adaptive_decision_summary"][
        "slot_state_counts"
    ]
    assert payload["novelty_learning_plan"]["lead_count"] >= 1
    assert payload["novelty_validation_feedback"]["learning_summary"]["learning_state_counts"]
    assert payload["novelty_next_wave_plan"]["wave_count"] >= 1
    assert payload["novelty_validation_feedback"]["next_wave_summary"]["action_count"] >= 1
    assert payload["novelty_mutation_budget_plan"]["selected_count"] >= 1
    assert (
        payload["novelty_validation_feedback"]["mutation_budget_summary"]["selected_count"]
        >= 1
    )
    assert payload["novelty_harvest_plan"]["harvest_count"] >= 0
    assert "harvest_summary" in payload["novelty_validation_feedback"]
    assert (
        seed["novelty_validation_plan"]["jobs"][1]["id"]
        in payload["novelty_validation_plan"]["resume_plan"]["ready_job_ids"]
    )
