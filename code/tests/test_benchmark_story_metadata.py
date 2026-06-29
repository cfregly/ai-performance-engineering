from __future__ import annotations

import json
from pathlib import Path

import torch

from ch03.baseline_gemm import BaselineGemmBenchmark
from ch03.optimized_gemm import OptimizedGemmBenchmark
import ch08.tiling_benchmark_base_tcgen05 as tiling_tcgen05_base
from ch07.baseline_lookup import BaselineLookupBenchmark
from ch07.optimized_lookup import OptimizedLookupBenchmark
from ch08.baseline_nvfp4_mlp import BaselineChapter8NVFP4MLPBenchmark
from ch08.baseline_thresholdtma import BaselineThresholdTMABenchmark
from ch08.baseline_tiling import BaselineTilingBenchmark
from ch08.baseline_tiling_tcgen05 import BaselineTilingBenchmarkTCGen05
from ch08.baseline_tcgen05_custom_vs_cublas import BaselineTcgen05CustomVsCublasBenchmark
from ch08.optimized_nvfp4_mlp import OptimizedChapter8NVFP4MLPBenchmark
from ch08.optimized_thresholdtma import OptimizedThresholdTMABenchmark
from ch08.optimized_tiling import OptimizedTilingBenchmark
from ch08.optimized_tiling_tcgen05 import OptimizedTilingBenchmarkTCGen05
from ch08.optimized_tcgen05_custom_vs_cublas import OptimizedTcgen05CustomVsCublasBenchmark
from ch04.baseline_pcie_staging import BaselinePcieStagingBenchmark
from ch04.optimized_pcie_staging import OptimizedPcieStagingBenchmark
from ch04.baseline_symmetric_memory_perf import BaselineSymmetricMemoryPerfBenchmark
from ch04.optimized_symmetric_memory_perf import OptimizedSymmetricMemoryPerfBenchmark
from ch13.baseline_context_parallel_multigpu import BaselineContextParallelMultigpuBenchmark
from ch13.baseline_expert_parallel_multigpu import BaselineExpertParallelMultigpuBenchmark
from ch13.baseline_sequence_parallel_multigpu import BaselineSequenceParallelMultigpuBenchmark
from ch13.optimized_context_parallel_multigpu import OptimizedContextParallelMultigpuBenchmark
from ch13.optimized_expert_parallel_multigpu import OptimizedExpertParallelMultigpuBenchmark
from ch13.optimized_sequence_parallel_multigpu import OptimizedSequenceParallelMultigpuBenchmark
from ch17.baseline_inference_full import BaselineInferenceFullBenchmark
from ch17.optimized_inference_full import OptimizedInferenceFullBenchmark
from ch15.baseline_single_gpu_kv_handoff import get_benchmark as get_baseline_single_gpu_kv_handoff
from ch15.optimized_single_gpu_kv_handoff import get_benchmark as get_optimized_single_gpu_kv_handoff
from ch15.baseline_disaggregated_inference_multigpu import BaselineDisaggregatedInferenceMultiGPUBenchmark
from ch15.optimized_disaggregated_inference_multigpu import OptimizedDisaggregatedInferenceMultiGPUBenchmark
from core.analysis.performance_analyzer import load_benchmark_data
from core.analysis.reporting.generator import generate_report
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode


class _StoryMetadataSmokeBenchmark(BaseBenchmark):
    allow_cpu = True
    story_metadata = {
        "pair_role": "comparison",
        "chapter_alignment": "supplementary",
        "chapter_native_exemplar": False,
        "comparison_reason": "unit-test smoke benchmark",
    }

    def setup(self) -> None:
        self.tensor = torch.ones(16, device=self.device)

    def benchmark_fn(self) -> None:
        self.tensor = self.tensor + 1

    def teardown(self) -> None:
        self.tensor = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5, validity_profile="portable")


def test_ch03_gemm_reports_comparison_story_explicitly() -> None:
    baseline = BaselineGemmBenchmark()
    optimized = OptimizedGemmBenchmark()

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()
    baseline_story = baseline.get_story_metadata()
    optimized_story = optimized.get_story_metadata()

    assert baseline_metrics is not None
    assert optimized_metrics is not None
    assert baseline_story is not None
    assert optimized_story is not None
    assert baseline_metrics["story.comparison_pair"] == 1.0
    assert baseline_metrics["story.chapter_native_exemplar"] == 0.0
    assert baseline_metrics["launch.gemm_calls_per_iteration"] == 8.0
    assert baseline_metrics["launch.block_k"] == 256.0
    assert baseline_story["pair_role"] == "comparison"
    assert baseline_story["chapter_alignment"] == "supplementary"
    assert baseline_story["chapter_native_exemplar"] is False
    assert baseline_story["execution_pattern"] == "fragmented_gemm_launches"
    assert baseline_story["chapter_native_targets"] == [
        "pageable_copy",
        "rack_prep",
        "pinned_prefetch_mlp",
        "double_buffered_batch_provisioning",
    ]
    assert optimized_metrics["story.comparison_pair"] == 1.0
    assert optimized_metrics["story.chapter_native_exemplar"] == 0.0
    assert optimized_metrics["launch.gemm_calls_per_iteration"] == 1.0
    assert optimized_metrics["launch.block_k"] == 2048.0
    assert optimized_story["pair_role"] == "comparison"
    assert optimized_story["chapter_alignment"] == "supplementary"
    assert optimized_story["chapter_native_exemplar"] is False
    assert optimized_story["execution_pattern"] == "single_compiled_gemm_launch"
    assert optimized_story["optimization_mechanism"] == 'use torch.compile(mode="reduce-overhead") to amortize launch fragmentation while keeping math fixed'


def test_ch07_lookup_reports_layout_transform_explicitly() -> None:
    baseline = BaselineLookupBenchmark()
    optimized = OptimizedLookupBenchmark()

    assert baseline.friendly_name == "Baseline Lookup (Scattered Reads)"
    assert optimized.friendly_name == "Optimized Lookup (Pretransposed Paths)"
    assert baseline.get_custom_metrics() == {
        "reads_per_output": 64.0,
        "layout_pretransposed": 0.0,
        "pointer_chase_in_kernel": 1.0,
    }
    assert optimized.get_custom_metrics() == {
        "reads_per_output": 64.0,
        "layout_pretransposed": 1.0,
        "pointer_chase_in_kernel": 0.0,
    }


def test_ch17_inference_comparison_reports_active_layer_delta() -> None:
    baseline = BaselineInferenceFullBenchmark()
    optimized = OptimizedInferenceFullBenchmark()

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()
    baseline_story = baseline.get_story_metadata()
    optimized_story = optimized.get_story_metadata()

    assert baseline_metrics is not None
    assert optimized_metrics is not None
    assert baseline_story is not None
    assert optimized_story is not None
    assert baseline_metrics["configured_layers"] == 24.0
    assert baseline_metrics["active_layers"] == 24.0
    assert baseline_metrics["identity_layers_skipped"] == 0.0
    assert baseline_metrics["story.comparison_pair"] == 1.0
    assert baseline_metrics["story.chapter_native_exemplar"] == 0.0
    assert baseline_story["pair_role"] == "comparison"
    assert baseline_story["chapter_alignment"] == "supplementary"
    assert baseline_story["chapter_native_exemplar"] is False
    assert baseline_story["execution_pattern"] == "full_depth_inference"
    assert optimized_metrics["configured_layers"] == 24.0
    assert optimized_metrics["active_layers"] == 6.0
    assert optimized_metrics["identity_layers_skipped"] == 18.0
    assert optimized_metrics["story.comparison_pair"] == 1.0
    assert optimized_metrics["story.chapter_native_exemplar"] == 0.0


def test_ch08_bridge_comparisons_report_story_explicitly(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tiling_tcgen05_base, "_check_tcgen05_extension_available", lambda: (True, None))

    benches = [
        BaselineThresholdTMABenchmark(),
        OptimizedThresholdTMABenchmark(),
        BaselineTilingBenchmark(),
        OptimizedTilingBenchmark(),
        BaselineTilingBenchmarkTCGen05(),
        OptimizedTilingBenchmarkTCGen05(),
        BaselineTcgen05CustomVsCublasBenchmark(),
        OptimizedTcgen05CustomVsCublasBenchmark(),
        BaselineChapter8NVFP4MLPBenchmark(),
        OptimizedChapter8NVFP4MLPBenchmark(),
    ]

    for bench in benches:
        metrics = bench.get_custom_metrics()
        assert metrics is not None
        assert metrics["story.comparison_pair"] == 1.0
        assert metrics["story.chapter_native_exemplar"] == 0.0


def test_ch04_symmetric_memory_perf_story_metadata_marks_compound_optimized_path() -> None:
    baseline_story = BaselineSymmetricMemoryPerfBenchmark.story_metadata
    optimized_story = OptimizedSymmetricMemoryPerfBenchmark.story_metadata

    assert baseline_story["comparison_axis"] == "allocation_blocking_copy_vs_preallocated_async_copy"
    assert baseline_story["optimization_mechanism"] == "per_iteration_allocation_plus_blocking_copy"
    assert baseline_story["compound_optimization"] is False
    assert optimized_story["comparison_axis"] == "allocation_blocking_copy_vs_preallocated_async_copy"
    assert optimized_story["optimization_mechanism"] == "preallocated_buffer_plus_nonblocking_copy"
    assert optimized_story["compound_optimization"] is True


def test_ch04_pcie_staging_reports_comparison_story_explicitly() -> None:
    baseline = BaselinePcieStagingBenchmark()
    optimized = OptimizedPcieStagingBenchmark()

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()
    baseline_story = baseline.get_story_metadata()
    optimized_story = optimized.get_story_metadata()

    assert baseline_metrics is not None
    assert optimized_metrics is not None
    assert baseline_story is not None
    assert optimized_story is not None
    assert baseline_metrics["story.comparison_pair"] == 1.0
    assert baseline_metrics["story.chapter_native_exemplar"] == 0.0
    assert baseline_metrics["transfer.type"] == 0.0
    assert baseline_metrics["pcie.host_buffer_pinned"] == 0.0
    assert optimized_metrics["story.comparison_pair"] == 1.0
    assert optimized_metrics["story.chapter_native_exemplar"] == 0.0
    assert optimized_metrics["transfer.type"] == 0.0
    assert optimized_metrics["pcie.host_buffer_pinned"] == 1.0
    assert baseline_story["pair_role"] == "comparison"
    assert baseline_story["chapter_alignment"] == "supplementary"
    assert optimized_story["chapter_alignment"] == "supplementary"
    assert optimized_story["optimization_mechanism"] == "pinned host buffer plus nonblocking copies"


def test_ch15_single_gpu_kv_handoff_wrappers_expose_comparison_story_metadata() -> None:
    baseline = get_baseline_single_gpu_kv_handoff()
    optimized = get_optimized_single_gpu_kv_handoff()

    baseline_story = baseline.get_story_metadata()
    optimized_story = optimized.get_story_metadata()
    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    assert baseline_story is not None
    assert optimized_story is not None
    assert baseline_metrics is not None
    assert optimized_metrics is not None
    assert baseline_story["pair_role"] == "comparison"
    assert baseline_story["chapter_alignment"] == "supplementary"
    assert baseline_story["chapter_native_exemplar"] is False
    assert optimized_story["pair_role"] == "comparison"
    assert optimized_story["chapter_alignment"] == "supplementary"
    assert optimized_story["chapter_native_exemplar"] is False
    assert baseline_metrics["story.comparison_pair"] == 1.0
    assert optimized_metrics["story.comparison_pair"] == 1.0
    assert baseline_metrics["single_gpu_kv_handoff.host_staged_kv"] == 1.0
    assert optimized_metrics["single_gpu_kv_handoff.device_resident_kv"] == 1.0


def test_ch13_multigpu_wrappers_expose_verification_and_launch_modes_in_story_metadata() -> None:
    benches = [
        BaselineContextParallelMultigpuBenchmark,
        OptimizedContextParallelMultigpuBenchmark,
        BaselineExpertParallelMultigpuBenchmark,
        OptimizedExpertParallelMultigpuBenchmark,
        BaselineSequenceParallelMultigpuBenchmark,
        OptimizedSequenceParallelMultigpuBenchmark,
    ]

    for bench_cls in benches:
        story = bench_cls.story_metadata
        assert story["pair_role"] == "canonical"
        assert story["chapter_alignment"] == "native"
        assert story["chapter_native_exemplar"] is True
        assert story["timed_launch_mode"] == "torchrun_multi_gpu"
        assert story["verification_mode"] == "single_process_surrogate"

    assert BaselineContextParallelMultigpuBenchmark.story_metadata["optimization_mechanism"] == "all_gather_attention"
    assert OptimizedContextParallelMultigpuBenchmark.story_metadata["optimization_mechanism"] == "ring_attention"
    assert BaselineExpertParallelMultigpuBenchmark.story_metadata["optimization_mechanism"] == "reused_list_all_to_all_buffers"
    assert OptimizedExpertParallelMultigpuBenchmark.story_metadata["optimization_mechanism"] == "preallocated_all_to_all_single"
    assert (
        BaselineSequenceParallelMultigpuBenchmark.story_metadata["optimization_mechanism"]
        == "all_gather_full_sequence_after_tp_all_reduce"
    )
    assert (
        OptimizedSequenceParallelMultigpuBenchmark.story_metadata["optimization_mechanism"]
        == "keep_tp_activations_sequence_sharded_between_layers"
    )


def test_ch15_disagg_multigpu_wrappers_expose_baseline_owned_shared_harness_metadata() -> None:
    baseline_story = BaselineDisaggregatedInferenceMultiGPUBenchmark.story_metadata
    optimized_story = OptimizedDisaggregatedInferenceMultiGPUBenchmark.story_metadata

    for story in (baseline_story, optimized_story):
        assert story["pair_role"] == "canonical"
        assert story["chapter_alignment"] == "native"
        assert story["chapter_native_exemplar"] is True
        assert story["timed_launch_mode"] == "torchrun_multi_gpu"
        assert story["verification_mode"] == "local_multi_device_surrogate"
        assert story["shared_harness_layout"] == "baseline_owned_shared_base"
        assert story["shared_harness_owner"] == "ch15/baseline_disaggregated_inference_multigpu.py"

    assert baseline_story["execution_pattern"] == "serialized_prefill_then_decode"
    assert optimized_story["execution_pattern"] == "overlapped_prefill_decode_pipeline"


def test_harness_result_carries_story_metadata() -> None:
    config = BenchmarkConfig(
        iterations=1,
        warmup=5,
        validity_profile="portable",
        allow_foreign_gpu_processes=True,
    )
    config.use_subprocess = False
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(_StoryMetadataSmokeBenchmark())

    assert result.story_metadata is not None
    assert result.story_metadata["pair_role"] == "comparison"
    assert result.story_metadata["chapter_alignment"] == "supplementary"
    assert result.story_metadata["chapter_native_exemplar"] is False


def test_aggregated_data_surfaces_story_metadata_and_story_note(tmp_path: Path) -> None:
    raw_results = {
        "timestamp": "2026-03-17 00:00:00",
        "results": [
            {
                "chapter": "ch17",
                "benchmarks": [
                    {
                        "example": "inference_full",
                        "best_speedup": 3.2,
                        "baseline_time_ms": 12.5,
                        "status": "succeeded",
                        "baseline_story_metadata": {
                            "pair_role": "comparison",
                            "chapter_alignment": "supplementary",
                            "chapter_native_exemplar": False,
                            "comparison_reason": "Model-side work reduction comparison benchmark",
                            "chapter_native_targets": [
                                "prefill_decode_disagg_ttft",
                                "prefill_decode_disagg_overlap",
                                "prefill_decode_disagg_batched",
                            ],
                        },
                        "optimizations": [
                            {
                                "file": "optimized_inference_full.py",
                                "technique": "early exit",
                                "time_ms": 3.9,
                                "speedup": 3.2,
                                "status": "succeeded",
                                "story_metadata": {
                                    "pair_role": "comparison",
                                    "chapter_alignment": "supplementary",
                                    "chapter_native_exemplar": False,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(json.dumps(raw_results), encoding="utf-8")

    data = load_benchmark_data(data_file=data_file)
    bench = data["benchmarks"][0]

    assert bench["pair_role"] == "comparison"
    assert bench["chapter_alignment"] == "supplementary"
    assert bench["chapter_native_exemplar"] is False
    assert "Supplementary comparison pair." in bench["story_note"]
    assert "prefill_decode_disagg_ttft" in bench["story_note"]
    assert bench["optimizations"][0]["story_metadata"]["pair_role"] == "comparison"


def test_generate_report_from_raw_results_file_renders_story_note(tmp_path: Path) -> None:
    raw_results = {
        "timestamp": "2026-03-17 00:00:00",
        "results": [
            {
                "chapter": "ch03",
                "benchmarks": [
                    {
                        "example": "gemm",
                        "best_speedup": 2.9,
                        "baseline_time_ms": 0.548,
                        "status": "succeeded",
                        "baseline_story_metadata": {
                            "pair_role": "comparison",
                            "chapter_alignment": "supplementary",
                            "chapter_native_exemplar": False,
                            "comparison_reason": "Host/runtime overhead comparison benchmark",
                            "chapter_native_targets": ["pageable_copy", "rack_prep"],
                        },
                        "optimizations": [
                            {
                                "file": "optimized_gemm.py",
                                "technique": "compiled matmul",
                                "time_ms": 0.189,
                                "speedup": 2.9,
                                "status": "succeeded",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(json.dumps(raw_results), encoding="utf-8")
    output_path = tmp_path / "story_report.html"

    generate_report(data_file, output_path, format="html")
    html = output_path.read_text(encoding="utf-8")

    assert "Supplementary comparison pair." in html
    assert "Chapter-native targets: pageable_copy, rack_prep." in html
