"""Repeat only the16 named existing CPU checks; no tokenizer build or GPU claim."""
import runpy
import labs.nanochat_fullstack

ns = runpy.run_path("labs/nanochat_fullstack/tests/test_engine.py")
names = ['test_kv_cache_resize', 'test_kv_cache_reuses_batch_index_buffer_for_padded_inserts', 'test_kv_cache_dense_row_pos_insert_skips_materialized_true_mask', 'test_sample_next_token_reuses_workspace_outputs', 'test_sample_batch_tokens_reuses_sampler_workspace', 'test_sample_batch_tokens_reuses_sparse_active_logits_buffer', 'test_build_attention_mask_reuses_position_buffer', 'test_batch_row_index_buffer_reuses_arange_storage', 'test_lengths_by_batch_buffer_reuses_device_storage', 'test_full_active_mask_reuses_device_buffer', 'test_attention_reuses_causal_mask_buffers', 'test_attention_reuses_padded_mask_buffer_and_skips_decode_causal_mask', 'test_apply_rotary_emb_inference_matches_reference', 'test_attention_reuses_rotary_buffers_for_inference', 'test_apply_rotary_emb_training_path_keeps_gradients', 'test_mlp_relu_square_reuses_buffer_without_grad_and_preserves_backward']
for name in names:
    ns[name]()
    print("PASS", name)
print(f"{len(names)} existing CPU function checks passed; no tokenizer/CUDA qualification")
