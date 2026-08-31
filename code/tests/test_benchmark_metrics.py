#!/usr/bin/env python3
"""Unit tests for benchmark_metrics.py helper functions.

Run with: pytest tests/test_benchmark_metrics.py -v
"""

import pytest

from core.benchmark.metrics import (
    # Hardware specs
    HardwareSpecs,
    BLACKWELL_B200,
    HOPPER_H100,
    detect_hardware_specs,
    # Chapter-specific helpers
    compute_memory_transfer_metrics,
    compute_kernel_fundamentals_metrics,
    compute_memory_access_metrics,
    compute_optimization_metrics,
    compute_roofline_metrics,
    compute_stream_metrics,
    compute_graph_metrics,
    compute_precision_metrics,
    compute_inference_metrics,
    compute_speculative_decoding_metrics,
    # New helpers
    compute_environment_metrics,
    compute_system_config_metrics,
    compute_distributed_metrics,
    compute_storage_io_metrics,
    compute_pipeline_metrics,
    compute_triton_metrics,
    compute_ai_optimization_metrics,
    compute_moe_metrics,
    # Generic
    compute_speedup_metrics,
    validate_metrics,
)


class TestHardwareSpecs:
    """Test hardware specification dataclasses."""
    
    def test_blackwell_specs(self):
        """Pin the HGX B200 per-GPU dense peaks, not sparse eight-GPU totals.

        NVIDIA HGX: 72 PF/s FP8 and 36 PF/s BF16 across eight GPUs,
        with sparsity; dense is half. FP32 is 600 TF/s across eight.
        https://www.nvidia.com/en-us/data-center/hgx/
        """
        assert BLACKWELL_B200.name == "NVIDIA B200"
        assert BLACKWELL_B200.hbm_bandwidth_gbps == 8000.0
        assert BLACKWELL_B200.fp8_tflops == 4500.0
        assert BLACKWELL_B200.tensor_tflops == 2250.0
        assert BLACKWELL_B200.fp32_tflops == 75.0
    
    def test_hopper_specs(self):
        """Test Hopper H100 specs are reasonable."""
        assert HOPPER_H100.name == "NVIDIA H100"
        assert HOPPER_H100.hbm_bandwidth_gbps >= 3000.0
        assert HOPPER_H100.num_sms > 100
        # H100 SXM is already dense: published sparse FP8 is 3958 TF/s.
        # Do not apply the audit's erroneous extra factor of one-half.
        assert HOPPER_H100.fp8_tflops == 1979.0
        assert HOPPER_H100.tensor_tflops == pytest.approx(1979.0 / 2, abs=1.0)
    
    def test_detect_hardware_returns_specs(self):
        """Test that detect_hardware_specs returns valid specs."""
        specs = detect_hardware_specs()
        assert isinstance(specs, HardwareSpecs)
        assert specs.hbm_bandwidth_gbps > 0


class TestMemoryTransferMetrics:
    """Test compute_memory_transfer_metrics."""
    
    def test_basic_transfer(self):
        """Test basic memory transfer metrics."""
        metrics = compute_memory_transfer_metrics(
            bytes_transferred=1e9,  # 1 GB
            elapsed_ms=100.0,  # 100 ms
            transfer_type="hbm"
        )
        
        assert "transfer.bytes" in metrics
        assert "transfer.achieved_gbps" in metrics
        assert "transfer.efficiency_pct" in metrics
        
        # 1 GB in 100 ms = 10 GB/s
        assert abs(metrics["transfer.achieved_gbps"] - 10.0) < 0.01
    
    def test_transfer_types(self):
        """Test different transfer types."""
        for t_type in ["pcie", "nvlink", "hbm"]:
            metrics = compute_memory_transfer_metrics(
                bytes_transferred=1e9,
                elapsed_ms=100.0,
                transfer_type=t_type
            )
            assert metrics["transfer.theoretical_peak_gbps"] > 0
    
    def test_zero_time_handling(self):
        """Test handling of zero elapsed time."""
        metrics = compute_memory_transfer_metrics(
            bytes_transferred=1e9,
            elapsed_ms=0.0,
        )
        # Should not crash, should return reasonable values
        assert metrics["transfer.achieved_gbps"] > 0

    def test_missing_elapsed_time_omits_derived_bandwidth(self):
        """Missing timing should not fabricate throughput metrics."""
        metrics = compute_memory_transfer_metrics(
            bytes_transferred=1e9,
            elapsed_ms=None,
        )
        assert metrics["transfer.bytes"] == 1e9
        assert "transfer.achieved_gbps" not in metrics
        assert "transfer.efficiency_pct" not in metrics


class TestKernelFundamentalsMetrics:
    """Test compute_kernel_fundamentals_metrics."""
    
    def test_basic_kernel(self):
        """Test basic kernel metrics."""
        metrics = compute_kernel_fundamentals_metrics(
            num_elements=1024,
            num_iterations=10
        )
        
        assert metrics["kernel.elements"] == 1024.0
        assert metrics["kernel.iterations"] == 10.0
    
    def test_bank_conflict_severity(self):
        """Test bank conflict severity calculation."""
        # No conflicts
        metrics = compute_kernel_fundamentals_metrics(
            num_elements=1024,
            expected_bank_conflicts_per_warp=0.0
        )
        assert metrics["kernel.bank_conflict_severity"] == 0.0
        
        # Worst case (32-way)
        metrics = compute_kernel_fundamentals_metrics(
            num_elements=1024,
            expected_bank_conflicts_per_warp=32.0
        )
        assert metrics["kernel.bank_conflict_severity"] == 1.0


class TestMemoryAccessMetrics:
    """Test compute_memory_access_metrics."""
    
    def test_perfect_coalescing(self):
        """Test perfect coalescing (100% efficiency)."""
        metrics = compute_memory_access_metrics(
            bytes_requested=1024,
            bytes_actually_transferred=1024,
            num_transactions=32,
            optimal_transactions=32
        )
        
        assert metrics["memory.efficiency_pct"] == 100.0
        assert metrics["memory.transaction_efficiency_pct"] == 100.0
    
    def test_poor_coalescing(self):
        """Test poor coalescing (50% efficiency)."""
        metrics = compute_memory_access_metrics(
            bytes_requested=1024,
            bytes_actually_transferred=2048,
            num_transactions=64,
            optimal_transactions=32
        )
        
        assert metrics["memory.efficiency_pct"] == 50.0
        assert metrics["memory.transaction_efficiency_pct"] == 50.0


class TestOptimizationMetrics:
    """Test compute_optimization_metrics."""
    
    def test_2x_speedup(self):
        """Test 2x speedup calculation."""
        metrics = compute_optimization_metrics(
            baseline_ms=10.0,
            optimized_ms=5.0,
            technique="test"
        )
        
        assert metrics["optimization.speedup"] == 2.0
        assert metrics["optimization.improvement_pct"] == 50.0
    
    def test_no_improvement(self):
        """Test no improvement case."""
        metrics = compute_optimization_metrics(
            baseline_ms=10.0,
            optimized_ms=10.0,
            technique="test"
        )
        
        assert metrics["optimization.speedup"] == 1.0
        assert metrics["optimization.improvement_pct"] == 0.0


class TestRooflineMetrics:
    """Test compute_roofline_metrics."""
    
    def test_memory_bound(self):
        """Test memory-bound kernel detection."""
        # Low arithmetic intensity = memory bound
        metrics = compute_roofline_metrics(
            total_flops=1e9,  # 1 GFLOP
            total_bytes=1e9,  # 1 GB
            elapsed_ms=1.0,
            precision="fp16"
        )
        
        # AI = 1 FLOP/byte, which is typically memory bound
        assert metrics["roofline.arithmetic_intensity"] == 1.0
        assert metrics["roofline.is_compute_bound"] == 0.0
    
    def test_compute_bound(self):
        """Test compute-bound kernel detection."""
        # High arithmetic intensity = compute bound
        metrics = compute_roofline_metrics(
            total_flops=1e15,  # 1 PFLOP
            total_bytes=1e6,   # 1 MB
            elapsed_ms=1.0,
            precision="fp16"
        )
        
        # AI = 1e9 FLOP/byte, definitely compute bound
        assert metrics["roofline.arithmetic_intensity"] > 1e6
        assert metrics["roofline.is_compute_bound"] == 1.0

    def test_missing_elapsed_time_omits_achieved_throughput(self):
        """Missing timing should keep only static roofline characteristics."""
        metrics = compute_roofline_metrics(
            total_flops=1e9,
            total_bytes=1e8,
            elapsed_ms=None,
            precision="fp16",
        )
        assert metrics["roofline.arithmetic_intensity"] == 10.0
        assert "roofline.achieved_tflops" not in metrics
        assert "roofline.efficiency_pct" not in metrics


class TestStreamMetrics:
    """Test compute_stream_metrics."""
    
    def test_perfect_overlap(self):
        """Test perfect stream overlap."""
        metrics = compute_stream_metrics(
            sequential_time_ms=100.0,
            overlapped_time_ms=25.0,
            num_streams=4,
            num_operations=4
        )
        
        assert metrics["stream.time_saved_ms"] == 75.0
        assert metrics["stream.overlap_efficiency_pct"] == 75.0
    
    def test_no_overlap(self):
        """Test no stream overlap."""
        metrics = compute_stream_metrics(
            sequential_time_ms=100.0,
            overlapped_time_ms=100.0,
            num_streams=4,
            num_operations=4
        )
        
        assert metrics["stream.time_saved_ms"] == 0.0
        assert metrics["stream.overlap_efficiency_pct"] == 0.0


class TestGraphMetrics:
    """Test compute_graph_metrics."""
    
    def test_graph_overhead_reduction(self):
        """Test CUDA graph overhead reduction."""
        metrics = compute_graph_metrics(
            baseline_launch_overhead_us=10.0,
            graph_launch_overhead_us=2.0,
            num_nodes=100,
            num_iterations=1000
        )
        
        assert metrics["graph.overhead_reduction_us"] == 8.0
        assert metrics["graph.overhead_reduction_pct"] == 80.0
        assert metrics["graph.total_overhead_saved_us"] == 8000.0


class TestDistributedAndTritonMetrics:
    """Test helpers that now tolerate missing timing."""

    def test_distributed_metrics_omit_derived_bandwidth_without_elapsed_time(self):
        metrics = compute_distributed_metrics(
            world_size=4,
            bytes_transferred=1e9,
            elapsed_ms=None,
            collective_type="allreduce",
        )
        assert metrics["distributed.world_size"] == 4.0
        assert "distributed.achieved_gbps" not in metrics
        assert "distributed.efficiency_pct" not in metrics

    def test_triton_metrics_omit_throughput_without_elapsed_time(self):
        metrics = compute_triton_metrics(
            num_elements=4096,
            elapsed_ms=None,
            block_size=256,
            num_warps=8,
        )
        assert metrics["triton.num_elements"] == 4096.0
        assert metrics["triton.block_size"] == 256.0
        assert "triton.elements_per_second" not in metrics


class TestPrecisionMetrics:
    """Test compute_precision_metrics."""
    
    def test_fp8_speedup(self):
        """Test FP8 precision speedup."""
        metrics = compute_precision_metrics(
            fp32_time_ms=10.0,
            reduced_precision_time_ms=2.5,
            precision_type="fp8"
        )
        
        assert metrics["precision.speedup"] == 4.0
        assert metrics["precision.memory_reduction_factor"] == 4.0
        assert metrics["precision.speedup_efficiency_pct"] == 100.0
    
    def test_fp16_speedup(self):
        """Test FP16 precision speedup."""
        metrics = compute_precision_metrics(
            fp32_time_ms=10.0,
            reduced_precision_time_ms=5.0,
            precision_type="fp16"
        )
        
        assert metrics["precision.speedup"] == 2.0
        assert metrics["precision.memory_reduction_factor"] == 2.0

    def test_int8_speedup(self):
        """Test INT8 precision speedup."""
        metrics = compute_precision_metrics(
            fp32_time_ms=12.0,
            reduced_precision_time_ms=3.0,
            precision_type="int8",
        )

        assert metrics["precision.speedup"] == 4.0
        assert metrics["precision.memory_reduction_factor"] == 4.0
        assert metrics["precision.speedup_efficiency_pct"] == 100.0

    def test_int4_reports_eight_x_memory_reduction(self):
        """INT4 should report the expected post-training quantization storage gain."""
        metrics = compute_precision_metrics(
            fp32_time_ms=12.0,
            reduced_precision_time_ms=2.0,
            precision_type="int4",
        )

        assert metrics["precision.speedup"] == 6.0
        assert metrics["precision.memory_reduction_factor"] == 8.0

    def test_fp32_is_neutral_precision(self):
        """FP32 should report no storage reduction versus itself."""
        metrics = compute_precision_metrics(
            fp32_time_ms=10.0,
            reduced_precision_time_ms=None,
            precision_type="fp32",
        )

        assert metrics["precision.fp32_ms"] == 10.0
        assert metrics["precision.memory_reduction_factor"] == 1.0


class TestInferenceMetrics:
    """Test compute_inference_metrics."""
    
    def test_basic_inference(self):
        """Test basic inference metrics."""
        metrics = compute_inference_metrics(
            ttft_ms=50.0,
            tpot_ms=10.0,
            total_tokens=100,
            total_requests=10,
            batch_size=8,
            max_batch_size=32
        )
        
        assert metrics["inference.ttft_ms"] == 50.0
        assert metrics["inference.tpot_ms"] == 10.0
        assert metrics["inference.batch_utilization_pct"] == 25.0


class TestSpeculativeDecodingMetrics:
    """Test compute_speculative_decoding_metrics."""
    
    def test_high_acceptance(self):
        """Test high acceptance rate."""
        metrics = compute_speculative_decoding_metrics(
            draft_tokens=100,
            accepted_tokens=90,
            draft_time_ms=5.0,
            verify_time_ms=10.0,
            num_rounds=10
        )
        
        assert metrics["speculative.acceptance_rate_pct"] == 90.0
        assert metrics["speculative.waste_pct"] == 10.0
        assert metrics["speculative.avg_accepted_per_round"] == 9.0
    
    def test_low_acceptance(self):
        """Test low acceptance rate."""
        metrics = compute_speculative_decoding_metrics(
            draft_tokens=100,
            accepted_tokens=20,
            draft_time_ms=5.0,
            verify_time_ms=10.0,
            num_rounds=10
        )
        
        assert metrics["speculative.acceptance_rate_pct"] == 20.0
        assert metrics["speculative.waste_pct"] == 80.0


class TestEnvironmentMetrics:
    """Test compute_environment_metrics."""
    
    def test_single_gpu(self):
        """Test single GPU environment."""
        metrics = compute_environment_metrics(
            gpu_count=1,
            gpu_memory_gb=80.0
        )
        
        assert metrics["env.gpu_count"] == 1.0
        assert metrics["env.is_multi_gpu"] == 0.0
        assert metrics["env.total_memory_gb"] == 80.0
    
    def test_multi_gpu(self):
        """Test multi-GPU environment."""
        metrics = compute_environment_metrics(
            gpu_count=8,
            gpu_memory_gb=80.0
        )
        
        assert metrics["env.gpu_count"] == 8.0
        assert metrics["env.is_multi_gpu"] == 1.0
        assert metrics["env.total_memory_gb"] == 640.0


class TestSystemConfigMetrics:
    """Test compute_system_config_metrics."""

    def test_unknown_numa_topology_uses_zero_without_fake_per_node_split(self):
        """Unknown NUMA topology should stay explicit in metrics."""
        metrics = compute_system_config_metrics(
            numa_nodes=0,
            cpu_cores=64,
        )

        assert metrics["system.numa_nodes"] == 0.0
        assert metrics["system.cores_per_numa"] == 0.0


class TestDistributedMetrics:
    """Test compute_distributed_metrics."""
    
    def test_allreduce(self):
        """Test AllReduce metrics."""
        metrics = compute_distributed_metrics(
            world_size=8,
            bytes_transferred=1e9,
            elapsed_ms=10.0,
            collective_type="allreduce"
        )
        
        assert metrics["distributed.world_size"] == 8.0
        assert metrics["distributed.achieved_gbps"] == 100.0  # 1GB in 10ms = 100 GB/s


class TestStorageIOMetrics:
    """Test compute_storage_io_metrics."""
    
    def test_read_write(self):
        """Test read/write metrics."""
        metrics = compute_storage_io_metrics(
            bytes_read=1e9,
            bytes_written=1e9,
            read_time_ms=100.0,
            write_time_ms=100.0
        )
        
        assert metrics["storage.read_gbps"] == 10.0
        assert metrics["storage.write_gbps"] == 10.0
        assert metrics["storage.read_write_ratio"] == 1.0


class TestPipelineMetrics:
    """Test compute_pipeline_metrics."""
    
    def test_balanced_pipeline(self):
        """Test balanced pipeline stages."""
        metrics = compute_pipeline_metrics(
            num_stages=4,
            stage_times_ms=[10.0, 10.0, 10.0, 10.0],
            bubble_time_ms=0.0,
            microbatches=4
        )
        
        assert metrics["pipeline.load_imbalance"] == 1.0
        assert metrics["pipeline.bubble_fraction"] == 0.0
    
    def test_imbalanced_pipeline(self):
        """Test imbalanced pipeline stages."""
        metrics = compute_pipeline_metrics(
            num_stages=4,
            stage_times_ms=[10.0, 20.0, 10.0, 10.0],
            bubble_time_ms=10.0,
            microbatches=4
        )
        
        # max=20, min=10, imbalance = max/min = 2.0
        assert metrics["pipeline.load_imbalance"] == 2.0


class TestTritonMetrics:
    """Test compute_triton_metrics."""
    
    def test_basic_triton(self):
        """Test basic Triton kernel metrics."""
        metrics = compute_triton_metrics(
            num_elements=1e6,
            elapsed_ms=1.0,
            block_size=1024,
            num_warps=4
        )
        
        assert metrics["triton.num_elements"] == 1e6
        assert metrics["triton.threads_per_block"] == 128.0  # 4 warps * 32


class TestMoEMetrics:
    """Test compute_moe_metrics."""
    
    def test_balanced_experts(self):
        """Test balanced expert load."""
        metrics = compute_moe_metrics(
            num_experts=8,
            active_experts=2,
            tokens_per_expert=[100, 100, 100, 100, 100, 100, 100, 100],
            routing_time_ms=1.0,
            expert_compute_time_ms=10.0
        )
        
        assert metrics["moe.load_imbalance"] == 1.0
        assert metrics["moe.expert_utilization_pct"] == 100.0
    
    def test_imbalanced_experts(self):
        """Test imbalanced expert load."""
        metrics = compute_moe_metrics(
            num_experts=8,
            active_experts=2,
            tokens_per_expert=[200, 100, 50, 50, 50, 50, 0, 0],
            routing_time_ms=1.0,
            expert_compute_time_ms=10.0
        )
        
        # 6 experts used out of 8
        assert metrics["moe.expert_utilization_pct"] == 75.0
        # max=200, avg=62.5, imbalance=3.2
        assert metrics["moe.load_imbalance"] > 1.0


class TestSpeedupMetrics:
    """Test compute_speedup_metrics."""
    
    def test_basic_speedup(self):
        """Test basic speedup calculation."""
        metrics = compute_speedup_metrics(
            baseline_ms=10.0,
            optimized_ms=2.0,
            name="test"
        )
        
        assert metrics["test.speedup"] == 5.0
        assert metrics["test.improvement_pct"] == 80.0
    
    def test_no_name(self):
        """Test speedup without name prefix."""
        metrics = compute_speedup_metrics(
            baseline_ms=10.0,
            optimized_ms=5.0
        )
        
        assert "speedup" in metrics
        assert "improvement_pct" in metrics


class TestValidateMetrics:
    """Test validate_metrics helper."""
    
    def test_valid_metrics(self):
        """Test validation of valid metrics."""
        result = validate_metrics({
            "transfer.bytes": 1000.0,
            "transfer.gbps": 10.0,
        })
        
        assert result["valid"] is True
        assert len(result["issues"]) == 0
    
    def test_none_metrics(self):
        """Test validation of None."""
        result = validate_metrics(None)
        
        assert result["valid"] is False
        assert "Metrics is None" in result["issues"]
    
    def test_empty_metrics(self):
        """Test validation of empty dict."""
        result = validate_metrics({})
        
        assert result["valid"] is False
        assert "Metrics dict is empty" in result["issues"]
    
    def test_bad_naming(self):
        """Test validation catches bad naming."""
        result = validate_metrics({
            "bytes": 1000.0,  # Missing category prefix
        })
        
        assert result["valid"] is False
        assert any("naming convention" in issue for issue in result["issues"])
    
    def test_all_zeros(self):
        """Test validation warns about all zeros."""
        result = validate_metrics({
            "transfer.bytes": 0.0,
            "transfer.gbps": 0.0,
        })
        
        assert result["valid"] is False
        assert any("zero" in issue.lower() for issue in result["issues"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
