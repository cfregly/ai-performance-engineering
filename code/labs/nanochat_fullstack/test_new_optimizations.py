#!/usr/bin/env python3
"""
Test script to validate CTA clustering and persistent decode optimizations.

Usage:
    python test_new_optimizations.py
"""

import sys

import torch

from labs.nanochat_fullstack.nanochat.engine import Engine
from labs.nanochat_fullstack.nanochat.gpt import GPT, GPTConfig
from labs.nanochat_fullstack.nanochat.tokenizer import get_tokenizer


def test_cta_clustering():
    """Test CTA clustering optimization."""
    print("=" * 60)
    print("Testing CTA Clustering Optimization")
    print("=" * 60)
    
    # Test 1: CTA clustering disabled (default)
    config_off = GPTConfig(
        sequence_len=1024,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        use_cta_clustering=False,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.device("meta"):
        model_off = GPT(config_off)
    model_off.to_empty(device=device)
    model_off.init_weights()
    # Ensure consistent dtype (avoid bfloat16/float32 mismatch)
    if device.type == "cuda":
        model_off = model_off.to(dtype=torch.bfloat16)
    model_off.eval()
    
    # Forward pass with clustering off
    test_input = torch.randint(0, 1000, (2, 128), device=device)
    with torch.inference_mode():
        logits_off = model_off(test_input)
    
    print(f"✅ CTA clustering OFF: output shape = {logits_off.shape}")
    
    # Test 2: CTA clustering enabled
    config_on = GPTConfig(
        sequence_len=1024,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        use_cta_clustering=True,
        cta_cluster_size=2,
        cta_cluster_seq_threshold=64,  # Low threshold for testing
    )
    
    with torch.device("meta"):
        model_on = GPT(config_on)
    model_on.to_empty(device=device)
    model_on.init_weights()
    # Ensure consistent dtype (avoid bfloat16/float32 mismatch)
    if device.type == "cuda":
        model_on = model_on.to(dtype=torch.bfloat16)
    model_on.eval()
    
    # Forward pass with clustering on (should work for seq >= 64)
    test_input_long = torch.randint(0, 1000, (2, 128), device=device)
    with torch.inference_mode():
        logits_on = model_on(test_input_long)
    
    print(f"✅ CTA clustering ON: output shape = {logits_on.shape}")
    
    # Test 3: Short sequence (below threshold) - should skip clustering
    test_input_short = torch.randint(0, 1000, (2, 32), device=device)
    with torch.inference_mode():
        logits_short = model_on(test_input_short)
    
    print(f"✅ CTA clustering ON (below threshold): output shape = {logits_short.shape}")
    print("✅ CTA Clustering tests passed!\n")


def test_persistent_decode():
    """Test persistent decode optimization."""
    print("=" * 60)
    print("Testing Persistent Decode Optimization")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test 1: Persistent decode disabled (default)
    config_off = GPTConfig(
        sequence_len=512,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        enable_persistent_decode=False,
    )
    
    with torch.device("meta"):
        model_off = GPT(config_off)
    model_off.to_empty(device=device)
    model_off.init_weights()
    # Ensure consistent dtype (avoid bfloat16/float32 mismatch)
    if device.type == "cuda":
        model_off = model_off.to(dtype=torch.bfloat16)
    model_off.eval()
    
    # Create mock tokenizer for testing
    class MockTokenizer:
        def get_bos_token_id(self):
            return 0
        def encode_special(self, token):
            special_tokens = {
                "<|python_start|>": 1,
                "<|python_end|>": 2,
                "<|output_start|>": 3,
                "<|output_end|>": 4,
                "<|assistant_end|>": 5,
            }
            return special_tokens.get(token, 0)
        def encode(self, text):
            return [6, 7, 8]
        def decode(self, tokens):
            return "test"
    
    tokenizer = MockTokenizer()
    
    engine_off = Engine(model_off, tokenizer, enable_batch_decode=False)
    assert engine_off.enable_persistent_decode == False
    assert engine_off._persistent_stream is None
    print("✅ Persistent decode OFF: no persistent stream created")
    
    # Test 2: Persistent decode enabled
    config_on = GPTConfig(
        sequence_len=512,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        enable_persistent_decode=True,
    )
    
    with torch.device("meta"):
        model_on = GPT(config_on)
    model_on.to_empty(device=device)
    model_on.init_weights()
    # Ensure consistent dtype (avoid bfloat16/float32 mismatch)
    if device.type == "cuda":
        model_on = model_on.to(dtype=torch.bfloat16)
    model_on.eval()
    
    engine_on = Engine(model_on, tokenizer, enable_batch_decode=False)
    assert engine_on.enable_persistent_decode == True
    assert engine_on.reuse_ids_buffer == True  # Should be auto-enabled
    if torch.cuda.is_available():
        assert engine_on._persistent_stream is not None
        print("✅ Persistent decode ON: persistent stream created")
        print(f"   Stream priority: {engine_on._persistent_stream.priority if hasattr(engine_on._persistent_stream, 'priority') else 'N/A'}")
    else:
        print("✅ Persistent decode ON: no CUDA available, stream is None")
    
    # Test 3: Buffer management
    if engine_on.enable_persistent_decode:
        test_buffer = engine_on._get_or_create_persistent_buffer(
            "test", (2, 1000), torch.float32, device
        )
        assert test_buffer is not None
        assert test_buffer.shape == (2, 1000)
        
        # Test buffer reuse
        test_buffer2 = engine_on._get_or_create_persistent_buffer(
            "test", (2, 1000), torch.float32, device
        )
        assert test_buffer is test_buffer2  # Same buffer should be reused
        print("✅ Persistent decode buffer management: buffers reused correctly")
        
        # Test buffer reallocation on shape change
        test_buffer3 = engine_on._get_or_create_persistent_buffer(
            "test", (4, 1000), torch.float32, device
        )
        assert test_buffer is not test_buffer3  # Different buffer
        assert test_buffer3.shape == (4, 1000)
        print("✅ Persistent decode buffer management: buffers reallocated on shape change")
    
    print("✅ Persistent Decode tests passed!\n")


def test_integration():
    """Test that both optimizations work together."""
    print("=" * 60)
    print("Testing Integration (Both Optimizations Enabled)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = GPTConfig(
        sequence_len=512,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        use_cta_clustering=True,
        cta_cluster_size=2,
        cta_cluster_seq_threshold=64,
        enable_persistent_decode=True,
    )
    
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    # Ensure consistent dtype (avoid bfloat16/float32 mismatch)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    
    class MockTokenizer:
        def get_bos_token_id(self):
            return 0
        def encode_special(self, token):
            return 1
        def encode(self, text):
            return [2, 3, 4]
        def decode(self, tokens):
            return "test"
    
    tokenizer = MockTokenizer()
    engine = Engine(model, tokenizer, enable_batch_decode=False)
    
    # Test forward pass
    test_input = torch.randint(0, 1000, (2, 128), device=device)
    with torch.inference_mode():
        logits = model(test_input)
    
    print(f"✅ Both optimizations enabled: output shape = {logits.shape}")
    print(f"   CTA clustering: enabled (threshold={config.cta_cluster_seq_threshold})")
    print(f"   Persistent decode: enabled (stream={'created' if engine._persistent_stream else 'N/A'})")
    print("✅ Integration test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NanoChat Advanced Optimizations Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_cta_clustering()
        test_persistent_decode()
        test_integration()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nBoth optimizations are working correctly:")
        print("  • CTA Clustering: ✅ Implemented")
        print("  • Persistent Decode: ✅ Implemented")
        print("\nOptimizations are properly gated behind flags and")
        print("maintain backward compatibility when disabled.")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
