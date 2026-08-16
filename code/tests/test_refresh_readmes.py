from __future__ import annotations

from pathlib import Path

import json

from core.scripts.refresh_readmes import (
    ENTRIES,
    REPO_ROOT,
    _format_markdown,
    _render_current_representative_deltas_body,
    main,
)


PRIORITY_EVIDENCE_DOCS = (
    "ch01",
    "ch02",
    "ch03",
    "ch05",
    "ch06",
    "ch07",
    "ch08",
    "ch09",
    "ch04",
    "ch10",
    "ch11",
    "ch12",
    "ch13",
    "ch14",
    "ch15",
    "ch16",
    "ch17",
    "ch18",
    "ch19",
    "ch20",
    "labs/block_scaling",
    "labs/blackwell_gemm_optimizations",
    "labs/blackwell_matmul",
    "labs/async_input_pipeline",
    "labs/custom_vs_cublas",
    "labs/cudnn_sdpa_bench",
    "labs/decode_optimization",
    "labs/flexattention",
    "labs/flashinfer_attention",
    "labs/flashattention_gluon",
    "labs/fullstack_cluster",
    "labs/flashattention4",
    "labs/kv_cache_compression",
    "labs/kv_optimization",
    "labs/moe_cuda",
    "labs/moe_optimization_journey",
    "labs/nanochat_fullstack",
    "labs/nvfp4_dual_gemm",
    "labs/nvfp4_gemm",
    "labs/nvfp4_gemv",
    "labs/nvfp4_group_gemm",
    "labs/occupancy_tuning",
    "labs/parameterized_cuda_graphs",
    "labs/persistent_decode",
    "labs/real_world_models",
    "labs/speculative_decode",
    "labs/training_hotpath",
    "labs/train_distributed",
    "labs/trtllm_phi_3_5_moe",
)

GENERATED_SPECIAL_DOCS = (
    "labs/README.md",
    "labs/moe_decode_blackwell_matrix",
    "labs/nanochat_fullstack/rustbpe",
    "labs/python_concurrency",
    "labs/vllm-deepseek-tuning",
)


def _assert_evidence_sections(markdown: str) -> None:
    assert "## Problem" in markdown
    assert "## Baseline Path" in markdown
    assert "## Optimized Path" in markdown
    assert "## Measured Delta" in markdown
    assert "## Profiler Evidence" in markdown
    assert "## Repro Commands" in markdown
    assert markdown.index("## Problem") < markdown.index("## Learning Goals")


def _output_path(slug: str) -> Path:
    if slug.endswith(".md"):
        return REPO_ROOT / slug
    return REPO_ROOT / slug / "README.md"


def test_root_readme_preserves_evidence_first_sections() -> None:
    markdown = _format_markdown(ENTRIES["README.md"])

    assert "## Tier-1 Canonical Suite" in markdown
    assert "## Current Representative Deltas" in markdown
    assert "## Profiler Evidence" in markdown
    assert "## Optimization Opportunity Radar" in markdown
    assert "## Lab Navigation" in markdown
    assert "`docs/autoresearch_campaigns.md`" in markdown
    assert "`labs/parameterized_cuda_graphs`" in markdown
    assert markdown.index("## Tier-1 Canonical Suite") < markdown.index("## Learning Goals")


def test_ch10_and_priority_labs_render_custom_evidence_sections() -> None:
    ch01_markdown = _format_markdown(ENTRIES["ch01"])
    ch02_markdown = _format_markdown(ENTRIES["ch02"])
    ch03_markdown = _format_markdown(ENTRIES["ch03"])
    ch05_markdown = _format_markdown(ENTRIES["ch05"])
    ch06_markdown = _format_markdown(ENTRIES["ch06"])
    ch04_markdown = _format_markdown(ENTRIES["ch04"])
    ch07_markdown = _format_markdown(ENTRIES["ch07"])
    ch08_markdown = _format_markdown(ENTRIES["ch08"])
    ch09_markdown = _format_markdown(ENTRIES["ch09"])
    ch10_markdown = _format_markdown(ENTRIES["ch10"])
    ch11_markdown = _format_markdown(ENTRIES["ch11"])
    ch12_markdown = _format_markdown(ENTRIES["ch12"])
    ch13_markdown = _format_markdown(ENTRIES["ch13"])
    ch14_markdown = _format_markdown(ENTRIES["ch14"])
    ch15_markdown = _format_markdown(ENTRIES["ch15"])
    ch16_markdown = _format_markdown(ENTRIES["ch16"])
    ch17_markdown = _format_markdown(ENTRIES["ch17"])
    ch18_markdown = _format_markdown(ENTRIES["ch18"])
    ch19_markdown = _format_markdown(ENTRIES["ch19"])
    ch20_markdown = _format_markdown(ENTRIES["ch20"])
    blackwell_grouped_gemm_markdown = _format_markdown(ENTRIES["labs/blackwell_gemm_optimizations"])
    blackwell_matmul_markdown = _format_markdown(ENTRIES["labs/blackwell_matmul"])
    async_input_pipeline_markdown = _format_markdown(ENTRIES["labs/async_input_pipeline"])
    block_scaling_markdown = _format_markdown(ENTRIES["labs/block_scaling"])
    custom_vs_cublas_markdown = _format_markdown(ENTRIES["labs/custom_vs_cublas"])
    cudnn_sdpa_markdown = _format_markdown(ENTRIES["labs/cudnn_sdpa_bench"])
    decode_optimization_markdown = _format_markdown(ENTRIES["labs/decode_optimization"])
    flexattention_markdown = _format_markdown(ENTRIES["labs/flexattention"])
    flashinfer_attention_markdown = _format_markdown(ENTRIES["labs/flashinfer_attention"])
    flashattention_gluon_markdown = _format_markdown(ENTRIES["labs/flashattention_gluon"])
    fullstack_cluster_markdown = _format_markdown(ENTRIES["labs/fullstack_cluster"])
    kv_cache_compression_markdown = _format_markdown(ENTRIES["labs/kv_cache_compression"])
    kv_markdown = _format_markdown(ENTRIES["labs/kv_optimization"])
    moe_cuda_markdown = _format_markdown(ENTRIES["labs/moe_cuda"])
    moe_journey_markdown = _format_markdown(ENTRIES["labs/moe_optimization_journey"])
    nanochat_fullstack_markdown = _format_markdown(ENTRIES["labs/nanochat_fullstack"])
    nvfp4_dual_gemm_markdown = _format_markdown(ENTRIES["labs/nvfp4_dual_gemm"])
    nvfp4_gemm_markdown = _format_markdown(ENTRIES["labs/nvfp4_gemm"])
    nvfp4_gemv_markdown = _format_markdown(ENTRIES["labs/nvfp4_gemv"])
    nvfp4_group_gemm_markdown = _format_markdown(ENTRIES["labs/nvfp4_group_gemm"])
    occupancy_tuning_markdown = _format_markdown(ENTRIES["labs/occupancy_tuning"])
    parameterized_cuda_graphs_markdown = _format_markdown(ENTRIES["labs/parameterized_cuda_graphs"])
    models_markdown = _format_markdown(ENTRIES["labs/real_world_models"])
    speculative_decode_markdown = _format_markdown(ENTRIES["labs/speculative_decode"])
    training_hotpath_markdown = _format_markdown(ENTRIES["labs/training_hotpath"])
    train_distributed_markdown = _format_markdown(ENTRIES["labs/train_distributed"])
    trtllm_phi_markdown = _format_markdown(ENTRIES["labs/trtllm_phi_3_5_moe"])

    for markdown in (
        ch01_markdown,
        ch02_markdown,
        ch03_markdown,
        ch04_markdown,
        ch05_markdown,
        ch06_markdown,
        ch07_markdown,
        ch08_markdown,
        ch09_markdown,
        ch10_markdown,
        ch11_markdown,
        ch12_markdown,
        ch13_markdown,
        ch14_markdown,
        ch15_markdown,
        ch16_markdown,
        ch17_markdown,
        ch18_markdown,
        ch19_markdown,
        ch20_markdown,
    ):
        _assert_evidence_sections(markdown)

    assert "## Running the Lab" in block_scaling_markdown
    assert "## Recommended Knobs" in block_scaling_markdown
    assert "## Harness vs Microbenchmark" in block_scaling_markdown

    for markdown in (
        async_input_pipeline_markdown,
        blackwell_grouped_gemm_markdown,
        blackwell_matmul_markdown,
        custom_vs_cublas_markdown,
        flexattention_markdown,
        flashinfer_attention_markdown,
        flashattention_gluon_markdown,
        fullstack_cluster_markdown,
        kv_cache_compression_markdown,
        kv_markdown,
        cudnn_sdpa_markdown,
        decode_optimization_markdown,
        moe_cuda_markdown,
        moe_journey_markdown,
        nanochat_fullstack_markdown,
        nvfp4_dual_gemm_markdown,
        nvfp4_gemm_markdown,
        nvfp4_gemv_markdown,
        nvfp4_group_gemm_markdown,
        occupancy_tuning_markdown,
        parameterized_cuda_graphs_markdown,
        models_markdown,
        speculative_decode_markdown,
        training_hotpath_markdown,
        train_distributed_markdown,
        trtllm_phi_markdown,
    ):
        _assert_evidence_sections(markdown)


def test_priority_readmes_match_generated_content() -> None:
    slugs = ("README.md",) + PRIORITY_EVIDENCE_DOCS + GENERATED_SPECIAL_DOCS

    for slug in slugs:
        expected = _format_markdown(ENTRIES[slug]).rstrip() + "\n"
        actual = _output_path(slug).read_text(encoding="utf-8")
        assert actual == expected, f"{slug} is out of sync with core/scripts/refresh_readmes.py"


def test_playbook_and_matrix_lab_docs_render_honest_nonpair_sections() -> None:
    labs_index_markdown = _format_markdown(ENTRIES["labs/README.md"])
    moe_decode_blackwell_matrix_markdown = _format_markdown(ENTRIES["labs/moe_decode_blackwell_matrix"])
    rustbpe_markdown = _format_markdown(ENTRIES["labs/nanochat_fullstack/rustbpe"])
    python_concurrency_markdown = _format_markdown(ENTRIES["labs/python_concurrency"])
    vllm_tuning_markdown = _format_markdown(ENTRIES["labs/vllm-deepseek-tuning"])

    assert "## Lab Index" in labs_index_markdown
    assert "Benchmark-pair labs" in labs_index_markdown
    assert "honest workflow/component docs" in labs_index_markdown
    assert "matrix/playbook labs" in labs_index_markdown
    assert "`labs/parameterized_cuda_graphs/`" in labs_index_markdown
    assert "`labs/moe_decode_blackwell_matrix/`" in labs_index_markdown

    assert "## Why This Exists" in moe_decode_blackwell_matrix_markdown
    assert "## Why This Is Not A Benchmark Pair" in moe_decode_blackwell_matrix_markdown
    assert "## Playbooks" in moe_decode_blackwell_matrix_markdown
    assert "## Artifact Layout" in moe_decode_blackwell_matrix_markdown
    assert "## Profiler-Backed Comparison" in moe_decode_blackwell_matrix_markdown
    assert "shared-doc Blackwell MoE decode entry" in moe_decode_blackwell_matrix_markdown

    assert "## What This Component Is" in rustbpe_markdown
    assert "## Why This Is Not A Benchmark Pair" in rustbpe_markdown
    assert "## How It Fits Into NanoChat" in rustbpe_markdown
    assert "Rust Edition 2021" in rustbpe_markdown

    assert "## What This Lab Is" in python_concurrency_markdown
    assert "## What A Proper Benchmark Pair Would Look Like" in python_concurrency_markdown
    assert "script-first, not harness-first" in python_concurrency_markdown

    assert "## Current Artifact State" in vllm_tuning_markdown
    assert "## What Proper Benchmark Pairs Would Look Like" in vllm_tuning_markdown
    assert "not currently a clean benchmark-pair lab" in vllm_tuning_markdown


def test_current_representative_deltas_prefer_tier1_history_when_available(tmp_path: Path) -> None:
    history_root = tmp_path / "artifacts" / "history" / "tier1" / "20260308_070000_manual"
    history_root.mkdir(parents=True)
    summary_path = history_root / "summary.json"
    summary_payload = {
        "run_id": "20260308_070000_manual",
        "summary": {
            "avg_speedup": 5.0,
            "median_speedup": 5.0,
            "geomean_speedup": 4.0,
            "representative_speedup": 4.0,
        },
        "targets": [
            {
                "target": "labs/block_scaling:block_scaling",
                "status": "succeeded",
                "baseline_time_ms": 0.2,
                "best_speedup": 2.0,
            },
            {
                "target": "ch04:gradient_fusion",
                "status": "succeeded",
                "baseline_time_ms": 4.0,
                "best_speedup": 8.0,
            },
        ],
    }
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
    index_path = tmp_path / "artifacts" / "history" / "tier1" / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "runs": [
                    {
                        "run_id": "20260308_070000_manual",
                        "summary_path": str(summary_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    body = _render_current_representative_deltas_body(tmp_path)

    assert "latest canonical tier-1 history summary" in body
    assert "Representative suite speedup" in body
    assert "`labs/block_scaling:block_scaling`" in body
    assert "`0.200 ms`" in body
    assert "`0.100 ms`" in body
    assert "artifacts/history/tier1/20260308_070000_manual/summary.json" in body


def test_current_representative_deltas_surface_tier1_history_warnings_when_falling_back(tmp_path: Path) -> None:
    index_path = tmp_path / "artifacts" / "history" / "tier1" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("{not-json", encoding="utf-8")

    body = _render_current_representative_deltas_body(tmp_path)

    assert "fall back to stored representative rows" in body
    assert "Warnings:" in body
    assert str(index_path) in body


def test_current_representative_deltas_surface_summary_shape_warnings_when_falling_back(tmp_path: Path) -> None:
    history_root = tmp_path / "artifacts" / "history" / "tier1" / "20260308_070000_manual"
    history_root.mkdir(parents=True)
    summary_path = history_root / "summary.json"
    summary_path.write_text(json.dumps({"targets": {}}), encoding="utf-8")
    index_path = tmp_path / "artifacts" / "history" / "tier1" / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "runs": [
                    {
                        "run_id": "20260308_070000_manual",
                        "summary_path": str(summary_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    body = _render_current_representative_deltas_body(tmp_path)

    assert "fall back to stored representative rows" in body
    assert "Expected targets list in tier-1 summary artifact" in body
    assert str(summary_path) in body


def test_refresh_readmes_requires_explicit_write_scope(capsys) -> None:
    try:
        main(["--write"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("main(['--write']) should reject implicit full-repo writes")

    captured = capsys.readouterr()
    assert "Refusing to write without an explicit scope" in captured.err


def test_refresh_readmes_help_mode_is_read_only(capsys) -> None:
    rc = main([])

    captured = capsys.readouterr()
    assert rc == 0
    assert "usage:" in captured.out.lower()
    assert "Wrote " not in captured.out


def test_refresh_readmes_write_target_only_updates_selected_readme(tmp_path: Path, capsys) -> None:
    rc = main(["--write", "--target", "ch03", "--repo-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Wrote ch03/README.md" in captured.out
    assert (tmp_path / "ch03" / "README.md").exists()
    assert not (tmp_path / "ch10" / "README.md").exists()


def test_refresh_readmes_check_reports_selected_mismatch(tmp_path: Path, capsys) -> None:
    readme_path = tmp_path / "ch03" / "README.md"
    readme_path.parent.mkdir(parents=True)
    readme_path.write_text("stale\n", encoding="utf-8")

    rc = main(["--check", "--target", "ch03", "--repo-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "README targets out of sync:" in captured.out
    assert "out_of_sync: ch03/README.md" in captured.out
