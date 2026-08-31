def test_cuda_event_timing_waits_on_terminal_event_not_whole_device() -> None:
    event_sync_files = {
        "ch02/hardware_info.py": "end.synchronize()",
        "ch02/nvlink_c2c_bandwidth_benchmark.py": "end.synchronize()",
        "ch12/bias_relu_residual_fusion_benchmark.py": "end.synchronize()",
        "ch13/fp8_static_demo.py": "end.synchronize()",
        "ch13/optimized_fp8_static.py": "end.synchronize()",
        "ch13/fp8_perchannel_demo.py": "end.synchronize()",
        "ch14/optimized_flex_attention_sparse.py": "end.synchronize()",
        "ch14/triton_persistent_demo.py": "end.synchronize()",
        "ch14/sliding_window_demo.py": "end.synchronize()",
        "ch14/flex_attention_sparse_demo.py": "end.synchronize()",
        "ch14/training_large_model_1_5x.py": "end.synchronize()",
        "ch16/inference_optimizations_blackwell.py": "end.synchronize()",
        "ch16/gpt_quick_test.py": "end.synchronize()",
        "ch16/test_fp8_quantization_real.py": "end.synchronize()",
        "ch16/moe_performance_benchmark.py": "end_event.synchronize()",
        "ch16/synthetic_moe_inference_benchmark.py": "end_event.synchronize()",
        "ch18/flex_attention_native.py": "end.synchronize()",
        "ch18/flex_attention_enhanced.py": "end.synchronize()",
        "ch18/flex_attention_large_model.py": "end.synchronize()",
        "ch19/fp8_compiled_matmul.py": "end.synchronize()",
        "ch19/native_fp4_quantization.py": "end.synchronize()",
        "ch19/native_fp6_quantization.py": "end.synchronize()",
        "ch19/native_fp8_training.py": "end.synchronize()",
        "ch20/ai_kernel_generator.py": "end.synchronize()",
        "labs/flexattention/flex_attention_cute.py": "end.synchronize()",
        "labs/cutlass_profiler_kernel_selector/run_triton_matmul.py": "end.synchronize()",
        "labs/moe_decode_blackwell_matrix/runner.py": "end_event.synchronize()",
        "labs/nvfp4_dual_gemm/env_probe_b200.py": "end.synchronize()",
        "labs/nvfp4_dual_gemm/local_eval.py": "end.synchronize()",
        "labs/nvfp4_dual_gemm/official_semantics_eval.py": "end.synchronize()",
        "labs/nvfp4_gemm/local_eval_official597.py": "end_event.synchronize()",
        "labs/nvfp4_gemm/local_eval_submission.py": "end.synchronize()",
    }

    global_wait_after_event = re.compile(
        r"end(?:_event)?\.record\(\)\n\s*torch\.cuda\.synchronize\("
    )
    for filename, expected_sync in event_sync_files.items():
        source = (REPO_ROOT / filename).read_text(encoding="utf-8")
        assert expected_sync in source
        assert global_wait_after_event.search(source) is None

    # This experiment no longer executes or reports timing. Keep an actual
    # failure contract rather than requiring a dead CUDA-event spelling.
    from labs.moe_optimization_journey.triton_fused_moe import benchmark_triton_moe
    with pytest.raises(NotImplementedError, match="Retired incomplete Triton MoE"):
        benchmark_triton_moe()
