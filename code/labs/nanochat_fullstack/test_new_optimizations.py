#!/usr/bin/env python3
"""Numerical flag checks with separate explicit CUDA capability gates.

Run from code: python -m labs.nanochat_fullstack.test_new_optimizations
CPU checks do not qualify the external CUDA attention backend.
"""
from dataclasses import replace
from unittest import SkipTest

import torch

import labs.nanochat_fullstack  # noqa: F401  (registers the top-level nanochat package)
from nanochat.engine import Engine, KVCache
from nanochat.gpt import GPT, GPTConfig


def _model(config, device):
    torch.manual_seed(718)
    model = GPT(config).to(device).eval()
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
        # Rotary lookup tables intentionally remain bfloat16.
    assert torch.count_nonzero(model.lm_head.weight) > 0
    return model


def _config(**overrides):
    overrides.setdefault("use_flash_attn_varlen", False)
    return GPTConfig(sequence_len=64, vocab_size=64, n_layer=2, n_head=2,
                     n_kv_head=2, n_embd=32, use_flash_sdp=False,
                     **overrides)


def _assert_logits(actual, expected, shape, tolerance=2e-5):
    assert actual.shape == expected.shape == shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    assert actual.abs().max() > 0, "zero-initialized logits cannot validate parity"
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def _require_cuda():
    if not torch.cuda.is_available():
        raise SkipTest("requires a real CUDA device; CPU does not qualify this optimization")
    return torch.device("cuda")


@torch.inference_mode()
def test_flash_attn_varlen_backend():
    device = _require_cuda()
    config = replace(_config(), use_flash_attn_varlen=True)
    model = _model(config, device)
    if any(
        not block.attn.use_flash_attn_varlen
        or block.attn.flash_attn_varlen_fn is None
        or block.attn.flash_attn_varlen_backend != "flash_attn_2_cuda"
        for block in model.transformer.h
    ):
        raise SkipTest("requires the pinned FlashAttention 2 CUDA backend")
    baseline = _model(_config(), device)
    baseline.load_state_dict(model.state_dict())
    inputs = torch.randint(0, 64, (2, 16), device=device)
    _assert_logits(model(inputs), baseline(inputs), (2, 16, 64), tolerance=2e-2)
    print("PASS: FlashAttention 2 varlen numerical parity (performance unmeasured)")


@torch.inference_mode()
def _compare_decode(config):
    device = _require_cuda()
    model = _model(config, device)
    # This check compares side-stream decode with eager decode. Engine compiles by
    # default on Blackwell, which would change the mode to side_stream_torch_compile.
    engine = Engine(model, tokenizer=None, enable_batch_decode=False, allow_compile=False)
    assert engine.enable_persistent_decode and engine._persistent_stream is not None
    assert engine.reuse_ids_buffer
    cached = engine._get_or_create_persistent_buffer("test", (2, 64), torch.float32, device)
    assert engine._get_or_create_persistent_buffer("test", (2, 64), torch.float32, device) is cached
    assert engine._get_or_create_persistent_buffer("test", (3, 64), torch.float32, device) is not cached
    eager = KVCache(**engine._kv_cache_params(2, 16))
    optimized = KVCache(**engine._kv_cache_params(2, 16))
    inputs = torch.randint(0, 64, (2, 10), device=device)
    model(inputs[:, :3], kv_cache=eager)
    model(inputs[:, :3], kv_cache=optimized)
    for position in range(3, 10):
        step = inputs[:, position:position + 1]
        expected = model(step, kv_cache=eager).clone()
        actual = engine._execute_decode(step, optimized)
        _assert_logits(actual, expected, (2, 1, 64), tolerance=2e-2)
        assert engine.decode_execution_mode == "side_stream"
        torch.testing.assert_close(optimized.kv_cache[..., :position + 1, :],
                                   eager.kv_cache[..., :position + 1, :], atol=2e-2, rtol=2e-2)
    print("PASS: repeated side-stream decode matches eager logits and KV entries")


def test_persistent_decode():
    _compare_decode(_config(enable_persistent_decode=True))


def test_integration():
    # This tests the external varlen backend and side-stream buffer reuse together.
    _compare_decode(_config(use_flash_attn_varlen=True, enable_persistent_decode=True))


def main():
    passed = skipped = 0
    for test in (test_flash_attn_varlen_backend, test_persistent_decode, test_integration):
        try:
            test()
            passed += 1
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}")
    # Assertions/errors are not swallowed: the process exits nonzero.
    print(f"{passed} checks passed; {skipped} capability checks skipped")
    if skipped:
        print("CUDA qualification is incomplete; skipped checks are not passes")


if __name__ == "__main__":
    main()
