.F.........                                                              [100%]
=================================== FAILURES ===================================
_________ test_ch10_and_priority_labs_render_custom_evidence_sections __________

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
>           _assert_evidence_sections(markdown)

code/tests/test_refresh_readmes.py:211: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

markdown = '# Lab - Quantized Projection Compute with BF16 KV Cache\n\n## Summary\nThis lab compares per-tensor delayed-scaling F...\n- CPU source checks do not qualify CUDA accuracy, Transformer Engine kernels, memory measurements, or performance.\n'

    def _assert_evidence_sections(markdown: str) -> None:
>       assert "## Problem" in markdown
E       AssertionError: assert '## Problem' in '# Lab - Quantized Projection Compute with BF16 KV Cache\n\n## Summary\nThis lab compares per-tensor delayed-scaling F...\n- CPU source checks do not qualify CUDA accuracy, Transformer Engine kernels, memory measurements, or performance.\n'

code/tests/test_refresh_readmes.py:78: AssertionError
=========================== short test summary info ============================
FAILED code/tests/test_refresh_readmes.py::test_ch10_and_priority_labs_render_custom_evidence_sections
1 failed, 10 passed in 0.10s
