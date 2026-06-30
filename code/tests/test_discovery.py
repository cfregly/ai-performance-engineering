"""Unit tests for benchmark discovery functionality.

Tests discovery of Python and CUDA benchmark pairs as specified in Part 2.10.
"""

import pytest
import warnings
from pathlib import Path

repo_root = Path(__file__).parent.parent

from core.discovery import (
    discover_benchmarks,
    discover_cuda_benchmarks,
    discover_all_chapters,
    discover_benchmark_pairs,
)

_DUMMY_BENCH_SOURCE = "def get_benchmark():\n    return None\n"


class TestPythonBenchmarkDiscovery:
    """Test discovery of Python benchmark pairs."""
    
    def test_discover_benchmarks_finds_baseline_optimized_pairs(self, tmp_path):
        """Test that discover_benchmarks finds baseline/optimized pairs."""
        # Create a test chapter directory
        chapter_dir = tmp_path / "ch01"
        chapter_dir.mkdir()
        
        # Create baseline file
        baseline_file = chapter_dir / "baseline_attention.py"
        baseline_file.write_text(_DUMMY_BENCH_SOURCE)
        
        # Create optimized file
        optimized_file = chapter_dir / "optimized_attention.py"
        optimized_file.write_text(_DUMMY_BENCH_SOURCE)
        
        pairs = discover_benchmarks(chapter_dir)
        
        assert len(pairs) == 1
        baseline_path, optimized_paths, example_name = pairs[0]
        assert baseline_path.name == "baseline_attention.py"
        assert len(optimized_paths) == 1
        assert optimized_paths[0].name == "optimized_attention.py"
        assert example_name == "attention"
    
    def test_discover_benchmarks_finds_multiple_optimizations(self, tmp_path):
        """Test that discover_benchmarks finds multiple optimized variants."""
        chapter_dir = tmp_path / "ch01"
        chapter_dir.mkdir()
        
        baseline_file = chapter_dir / "baseline_moe.py"
        baseline_file.write_text(_DUMMY_BENCH_SOURCE)
        
        optimized1 = chapter_dir / "optimized_moe_sparse.py"
        optimized1.write_text(_DUMMY_BENCH_SOURCE)
        
        optimized2 = chapter_dir / "optimized_moe_dense.py"
        optimized2.write_text(_DUMMY_BENCH_SOURCE)
        
        pairs = discover_benchmarks(chapter_dir)

        names = {example_name for _, _, example_name in pairs}
        assert {"moe", "moe_sparse", "moe_dense"} <= names

        primary = next(p for p in pairs if p[2] == "moe")
        baseline_path, optimized_paths, _ = primary
        assert len(optimized_paths) == 2
        assert any(p.name == "optimized_moe_sparse.py" for p in optimized_paths)
        assert any(p.name == "optimized_moe_dense.py" for p in optimized_paths)
    
    def test_discover_benchmarks_handles_no_baseline(self, tmp_path):
        """Test that discover_benchmarks handles missing baseline files."""
        chapter_dir = tmp_path / "ch01"
        chapter_dir.mkdir()
        
        # Only optimized file, no baseline
        optimized_file = chapter_dir / "optimized_attention.py"
        optimized_file.write_text(_DUMMY_BENCH_SOURCE)
        
        pairs = discover_benchmarks(chapter_dir)
        
        assert len(pairs) == 0
    
    def test_discover_benchmarks_handles_no_optimized(self, tmp_path):
        """Test that discover_benchmarks handles missing optimized files."""
        chapter_dir = tmp_path / "ch01"
        chapter_dir.mkdir()
        
        # Only baseline file, no optimized
        baseline_file = chapter_dir / "baseline_attention.py"
        baseline_file.write_text(_DUMMY_BENCH_SOURCE)
        
        pairs = discover_benchmarks(chapter_dir)
        
        assert len(pairs) == 0
    
    def test_discover_benchmarks_extracts_example_name_correctly(self, tmp_path):
        """Test that example name is extracted correctly from baseline filename."""
        chapter_dir = tmp_path / "ch01"
        chapter_dir.mkdir()
        
        baseline_file = chapter_dir / "baseline_speculative_decoding.py"
        baseline_file.write_text(_DUMMY_BENCH_SOURCE)
        
        optimized_file = chapter_dir / "optimized_speculative_decoding.py"
        optimized_file.write_text(_DUMMY_BENCH_SOURCE)
        
        pairs = discover_benchmarks(chapter_dir)
        
        assert len(pairs) == 1
        _, _, example_name = pairs[0]
        assert example_name == "speculative_decoding"

    def test_discover_benchmarks_prefers_specific_baseline_over_variant_alias(self, tmp_path):
        """A real baseline should suppress the alias pair from a broader baseline."""
        chapter_dir = tmp_path / "ch01"
        chapter_dir.mkdir()

        (chapter_dir / "baseline_performance.py").write_text(_DUMMY_BENCH_SOURCE)
        (chapter_dir / "baseline_performance_fp16.py").write_text(_DUMMY_BENCH_SOURCE)
        (chapter_dir / "optimized_performance.py").write_text(_DUMMY_BENCH_SOURCE)
        (chapter_dir / "optimized_performance_fp16.py").write_text(_DUMMY_BENCH_SOURCE)

        pairs = discover_benchmarks(chapter_dir)
        lookup = {example_name: (baseline_path.name, [p.name for p in optimized_paths]) for baseline_path, optimized_paths, example_name in pairs}

        assert lookup["performance"] == ("baseline_performance.py", ["optimized_performance.py"])
        assert lookup["performance_fp16"] == ("baseline_performance_fp16.py", ["optimized_performance_fp16.py"])

    def test_discover_benchmarks_pairs_pageable_copy_with_normalized_name(self, tmp_path):
        """Discovery should pair the pageable-copy benchmark without alias logic."""
        chapter_dir = tmp_path / "ch03"
        chapter_dir.mkdir()

        (chapter_dir / "baseline_pageable_copy.py").write_text(_DUMMY_BENCH_SOURCE)
        (chapter_dir / "optimized_pageable_copy.py").write_text(_DUMMY_BENCH_SOURCE)

        pairs = discover_benchmarks(chapter_dir)
        lookup = {example_name: (baseline_path.name, [p.name for p in optimized_paths]) for baseline_path, optimized_paths, example_name in pairs}

        assert lookup["pageable_copy"] == (
            "baseline_pageable_copy.py",
            ["optimized_pageable_copy.py"],
        )
    
    def test_discover_benchmarks_real_chapter(self):
        """Test discovery on a real chapter directory if it exists."""
        ch01_dir = repo_root / "ch01"
        if not ch01_dir.exists():
            pytest.skip("ch01 directory not found")
        
        pairs = discover_benchmarks(ch01_dir)
        
        # Should find at least some pairs if chapter exists
        assert isinstance(pairs, list)
        for baseline_path, optimized_paths, example_name in pairs:
            assert baseline_path.exists()
            assert baseline_path.name.startswith("baseline_")
            assert len(optimized_paths) > 0
            for opt_path in optimized_paths:
                assert opt_path.exists()
                assert opt_path.name.startswith("optimized_")
            assert isinstance(example_name, str)
            assert len(example_name) > 0

    def test_discover_benchmarks_ch05_uses_host_staged_reduction_target(self):
        """Chapter 5 should expose the precise single-GPU reduction target name."""
        ch05_dir = repo_root / "ch05"
        if not ch05_dir.exists():
            pytest.skip("ch05 directory not found")

        pairs = discover_benchmarks(ch05_dir)
        names = {example_name for _, _, example_name in pairs}

        assert "host_staged_reduction" in names
        assert "distributed" not in names

    def test_discover_benchmarks_ch06_hides_duplicate_add_aliases(self):
        """Chapter 6 should expose only the canonical add targets."""
        ch06_dir = repo_root / "ch06"
        if not ch06_dir.exists():
            pytest.skip("ch06 directory not found")

        pairs = discover_benchmarks(ch06_dir)
        names = {example_name for _, _, example_name in pairs}

        assert "add" in names
        assert "add_cuda" in names
        assert "add_tensors" not in names
        assert "add_tensors_cuda" not in names
        assert "add_cuda_parallel" not in names

    def test_discover_benchmarks_suppresses_add_cuda_parallel_variant_alias(self, tmp_path):
        """The canonical add_cuda target should not surface a parallel alias target."""
        chapter_dir = tmp_path / "ch06"
        chapter_dir.mkdir()

        (chapter_dir / "baseline_add_cuda.py").write_text(_DUMMY_BENCH_SOURCE)
        (chapter_dir / "optimized_add_cuda_parallel.py").write_text(_DUMMY_BENCH_SOURCE)

        pairs = discover_benchmarks(chapter_dir)
        names = {example_name for _, _, example_name in pairs}

        assert "add_cuda" in names
        assert "add_cuda_parallel" not in names

    def test_discover_benchmarks_ch14_hides_auxiliary_bench_variants(self):
        """Chapter 14 should expose only canonical paired benchmark targets."""
        ch14_dir = repo_root / "ch14"
        if not ch14_dir.exists():
            pytest.skip("ch14 directory not found")

        pairs = discover_benchmarks(ch14_dir)
        names = {example_name for _, _, example_name in pairs}

        assert "sliding_window" in names
        assert "triton_persistent" in names
        assert "cublas_vs_cutlass" in names
        assert "model_compile_reduced_precision" in names
        assert "sliding_window_bench" not in names
        assert "triton_persistent_bench" not in names
        assert "cutlass" not in names
        assert "model_compile_bf16" not in names

    def test_discover_benchmarks_ch04_and_ch15_expose_renamed_control_targets(self):
        chapter_expectations = {
            "ch04": ({"pcie_staging"}, {"nvlink"}),
            "ch15": ({"single_gpu_kv_handoff"}, {"disaggregated_inference"}),
        }

        for chapter, (present, absent) in chapter_expectations.items():
            chapter_dir = repo_root / chapter
            if not chapter_dir.exists():
                pytest.skip(f"{chapter} directory not found")
            pairs = discover_benchmarks(chapter_dir)
            names = {example_name for _, _, example_name in pairs}
            assert present <= names
            assert not (absent & names)

    def test_discover_benchmarks_ch10_removes_off_theme_matmul_pair(self):
        """Chapter 10 auto-discovery should stay aligned with the book-native target set."""
        ch10_dir = repo_root / "ch10"
        if not ch10_dir.exists():
            pytest.skip("ch10 directory not found")

        pairs = discover_benchmarks(ch10_dir)
        names = {example_name for _, _, example_name in pairs}

        assert "double_buffered_pipeline" in names
        assert "warp_specialized_pipeline" in names
        assert "cluster_group" in names
        assert "matmul" not in names

    def test_discover_benchmarks_keeps_intentional_variant_alias_targets(self):
        """Optimized suffix variants should stay discoverable as explicit targets when intended."""
        chapter_expectations = {
            "ch10": {"dsmem_reduction_cluster_atomic"},
            "ch12": {
                "cuda_graphs_router",
                "kernel_fusion_llm_reuse_static_tensor_and_simplify_setup",
            },
            "ch16": {"dense_attention_flash_blackwell_variant"},
            "ch18": {"flexdecoding_graphs"},
        }

        for chapter, expected_names in chapter_expectations.items():
            chapter_dir = repo_root / chapter
            if not chapter_dir.exists():
                pytest.skip(f"{chapter} directory not found")
            pairs = discover_benchmarks(chapter_dir)
            names = {example_name for _, _, example_name in pairs}
            assert expected_names <= names

    def test_discover_benchmarks_surfaces_split_targets_and_drops_removed_aliases(self):
        """Renamed and split targets should be visible, and removed aliases should be gone."""
        chapter_expectations = {
            "ch10": ({"matmul_tcgen05_vs_cublas"}, {"tmem_cutlass", "tmem_tcgen05"}),
            "ch15": ({"moe_dispatch", "moe_routing_topology_aware"}, {"moe_routing_simple"}),
            "ch18": ({"paged_attn_backend", "paged_attn_layout"}, {"paged_attn"}),
        }

        for chapter, (expected_names, removed_names) in chapter_expectations.items():
            chapter_dir = repo_root / chapter
            if not chapter_dir.exists():
                pytest.skip(f"{chapter} directory not found")
            pairs = discover_benchmarks(chapter_dir)
            names = {example_name for _, _, example_name in pairs}
            assert expected_names <= names
            assert not (removed_names & names)

    def test_discover_benchmarks_surfaces_new_first_class_coverage_targets(self):
        """New benchmark families should be discoverable as first-class targets."""
        chapter_expectations = {
            "ch13": {"sequence_parallel_multigpu"},
            "ch15": {
                "medusa_eagle_speculative",
                "medusa_eagle_speculative_medusa",
                "medusa_eagle_speculative_eagle",
                "moe_comm_exchange",
                "moe_comm_exchange_overlap",
                "moe_comm_exchange_hierarchical",
            },
            "ch16": {
                "awq_gptq_smoothquant",
                "awq_gptq_smoothquant_awq",
                "awq_gptq_smoothquant_gptq",
                "awq_gptq_smoothquant_smoothquant",
            },
            "labs/moe_cuda": {"kv_transfer_direct_graphs"},
        }

        for chapter, expected_names in chapter_expectations.items():
            chapter_dir = repo_root / chapter
            if not chapter_dir.exists():
                pytest.skip(f"{chapter} directory not found")
            pairs = discover_benchmarks(chapter_dir)
            names = {example_name for _, _, example_name in pairs}
            assert expected_names <= names

    def test_discover_benchmarks_ch12_graph_conditional_runtime_keeps_python_entrypoint(self):
        """Discovery should keep the optimized Python graph-conditional runtime benchmark."""
        ch12_dir = repo_root / "ch12"
        if not ch12_dir.exists():
            pytest.skip("ch12 directory not found")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pairs = discover_benchmarks(ch12_dir)

        names = {example_name for _, _, example_name in pairs}
        assert "graph_conditional_runtime" in names
        assert not any(
            "optimized_graph_conditional_runtime.py" in str(warning.message)
            and "missing get_benchmark()" in str(warning.message)
            for warning in caught
        )



class TestCudaBenchmarkDiscovery:
    """Test discovery of CUDA benchmarks."""
    
    def test_discover_cuda_benchmarks_finds_cu_files(self, tmp_path):
        """Test that discover_cuda_benchmarks finds .cu files."""
        ch01_dir = tmp_path / "ch01"
        ch01_dir.mkdir()
        
        cuda_file = ch01_dir / "test.cu"
        cuda_file.write_text("// CUDA code")
        
        cuda_benchmarks = discover_cuda_benchmarks(tmp_path)
        
        assert len(cuda_benchmarks) == 1
        assert cuda_benchmarks[0].name == "test.cu"
    
    def test_discover_cuda_benchmarks_finds_in_subdir(self, tmp_path):
        """Test that discover_cuda_benchmarks finds .cu files in cuda/ subdir."""
        ch01_dir = tmp_path / "ch01"
        ch01_dir.mkdir()
        
        cuda_subdir = ch01_dir / "cuda"
        cuda_subdir.mkdir()
        
        cuda_file = cuda_subdir / "test.cu"
        cuda_file.write_text("// CUDA code")
        
        cuda_benchmarks = discover_cuda_benchmarks(tmp_path)
        
        assert len(cuda_benchmarks) == 1
        assert cuda_benchmarks[0].name == "test.cu"
    
    def test_discover_cuda_benchmarks_handles_no_cuda_files(self, tmp_path):
        """Test that discover_cuda_benchmarks handles no CUDA files."""
        ch01_dir = tmp_path / "ch01"
        ch01_dir.mkdir()
        
        cuda_benchmarks = discover_cuda_benchmarks(tmp_path)
        
        assert len(cuda_benchmarks) == 0
    
    def test_discover_cuda_benchmarks_returns_sorted(self, tmp_path):
        """Test that discover_cuda_benchmarks returns sorted list."""
        ch01_dir = tmp_path / "ch01"
        ch01_dir.mkdir()
        
        cuda_file1 = ch01_dir / "z_test.cu"
        cuda_file1.write_text("// CUDA code")
        
        cuda_file2 = ch01_dir / "a_test.cu"
        cuda_file2.write_text("// CUDA code")
        
        cuda_benchmarks = discover_cuda_benchmarks(tmp_path)
        
        assert len(cuda_benchmarks) == 2
        # Should be sorted
        assert cuda_benchmarks[0].name < cuda_benchmarks[1].name

    def test_discover_cuda_benchmarks_ignores_generated_copies(self, tmp_path):
        """Generated benchmark copies should not leak into CUDA discovery."""
        ch01_dir = tmp_path / "ch01"
        ch01_dir.mkdir()

        (ch01_dir / "baseline_atomic_reduction_mcp_copy_3.cu").write_text("// generated")
        (ch01_dir / "baseline_atomic_reduction.cu").write_text("// canonical")

        cuda_benchmarks = discover_cuda_benchmarks(tmp_path)

        assert [path.name for path in cuda_benchmarks] == ["baseline_atomic_reduction.cu"]


class TestChapterDiscovery:
    """Test discovery of all chapters."""
    
    def test_discover_all_chapters_finds_ch_directories(self, tmp_path):
        """Test that discover_all_chapters finds ch* directories."""
        ch01 = tmp_path / "ch01"
        ch01.mkdir()
        
        ch02 = tmp_path / "ch02"
        ch02.mkdir()
        
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        
        chapters = discover_all_chapters(tmp_path)
        
        assert len(chapters) == 2
        chapter_names = [ch.name for ch in chapters]
        assert "ch01" in chapter_names
        assert "ch02" in chapter_names
        assert "other" not in chapter_names
    
    def test_discover_all_chapters_filters_by_number(self, tmp_path):
        """Test that discover_all_chapters only finds ch* with numbers."""
        ch01 = tmp_path / "ch01"
        ch01.mkdir()
        
        ch_abc = tmp_path / "chabc"
        ch_abc.mkdir()
        
        chapters = discover_all_chapters(tmp_path)
        
        assert len(chapters) == 1
        assert chapters[0].name == "ch01"
    
    def test_discover_all_chapters_returns_sorted(self, tmp_path):
        """Test that discover_all_chapters returns sorted list."""
        ch10 = tmp_path / "ch10"
        ch10.mkdir()
        
        ch02 = tmp_path / "ch02"
        ch02.mkdir()
        
        ch01 = tmp_path / "ch01"
        ch01.mkdir()
        
        chapters = discover_all_chapters(tmp_path)
        
        assert len(chapters) == 3
        # Should be sorted
        assert chapters[0].name == "ch01"
        assert chapters[1].name == "ch02"
        assert chapters[2].name == "ch10"

class TestBenchmarkPairDiscovery:
    """Test discovery of benchmark pairs across chapters."""
    
    def test_discover_benchmark_pairs_all_chapters(self, tmp_path):
        """Test that discover_benchmark_pairs finds pairs across all chapters."""
        ch01 = tmp_path / "ch01"
        ch01.mkdir()
        
        baseline1 = ch01 / "baseline_test.py"
        baseline1.write_text(_DUMMY_BENCH_SOURCE)
        optimized1 = ch01 / "optimized_test.py"
        optimized1.write_text(_DUMMY_BENCH_SOURCE)
        
        ch02 = tmp_path / "ch02"
        ch02.mkdir()
        
        baseline2 = ch02 / "baseline_test.py"
        baseline2.write_text(_DUMMY_BENCH_SOURCE)
        optimized2 = ch02 / "optimized_test.py"
        optimized2.write_text(_DUMMY_BENCH_SOURCE)
        
        pairs = discover_benchmark_pairs(tmp_path, chapter="all")
        
        assert len(pairs) == 2
    
    def test_discover_benchmark_pairs_specific_chapter(self, tmp_path):
        """Test that discover_benchmark_pairs finds pairs in specific chapter."""
        ch01 = tmp_path / "ch01"
        ch01.mkdir()
        
        baseline1 = ch01 / "baseline_test.py"
        baseline1.write_text(_DUMMY_BENCH_SOURCE)
        optimized1 = ch01 / "optimized_test.py"
        optimized1.write_text(_DUMMY_BENCH_SOURCE)
        
        ch02 = tmp_path / "ch02"
        ch02.mkdir()
        
        baseline2 = ch02 / "baseline_test.py"
        baseline2.write_text(_DUMMY_BENCH_SOURCE)
        optimized2 = ch02 / "optimized_test.py"
        optimized2.write_text(_DUMMY_BENCH_SOURCE)
        
        pairs = discover_benchmark_pairs(tmp_path, chapter="ch01")
        
        assert len(pairs) == 1
        assert pairs[0][0].parent.name == "ch01"
    
    def test_discover_benchmark_pairs_normalizes_chapter_name(self, tmp_path):
        """Test that discover_benchmark_pairs normalizes chapter name."""
        ch01 = tmp_path / "ch01"
        ch01.mkdir()
        
        baseline1 = ch01 / "baseline_test.py"
        baseline1.write_text(_DUMMY_BENCH_SOURCE)
        optimized1 = ch01 / "optimized_test.py"
        optimized1.write_text(_DUMMY_BENCH_SOURCE)
        
        # Test with number only
        pairs1 = discover_benchmark_pairs(tmp_path, chapter="1")
        assert len(pairs1) == 1
        
        # Test with ch prefix
        pairs2 = discover_benchmark_pairs(tmp_path, chapter="ch01")
        assert len(pairs2) == 1
        
        # Test with just number string
        pairs3 = discover_benchmark_pairs(tmp_path, chapter="1")
        assert len(pairs3) == 1
    
    def test_discover_benchmark_pairs_handles_nonexistent_chapter(self, tmp_path):
        """Test that discover_benchmark_pairs handles nonexistent chapter."""
        pairs = discover_benchmark_pairs(tmp_path, chapter="ch999")
        
        assert len(pairs) == 0
