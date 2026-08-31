#!/usr/bin/env python3
"""Numerical flag checks, with separate explicit CUDA capability gates.

Run from code: python -m labs.nanochat_fullstack.test_new_optimizations
CPU hint parity is not evidence that a clustered CUDA kernel executed.
"""
from dataclasses import replace
from unittest import SkipTest

import torch

import labs.nanochat_fullstack
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
    return GPTConfig(sequence_len=64, vocab_size=64, n_layer=2, n_head=2,
                     n_kv_head=2, n_embd=32, use_flash_sdp=False,
                     use_flash3=False, **overrides)


def _assert_logits(actual, expected, shape, tolerance=2e-5):
    assert actual.shape == expected.shape == shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    assert actual.abs().max() > 0, "zero-initialized logits cannot validate parity"
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@torch.inference_mode()
def test_cta_clustering():
    """The optional hint must preserve output on the ordinary SDPA path."""
    device = torch.device("cpu")
    off = _model(_config(), device)
    on = _model(_config(use_cta_clustering=True, cta_cluster_seq_threshold=8), device)
    on.load_state_dict(off.state_dict())
    for length in (4, 16):
        inputs = torch.randint(0, 64, (2, length), device=device)
        _assert_logits(on(inputs), off(inputs), (2, length, 64))
    print("PASS: identical-weight CTA hint parity on CPU SDPA; no clustered kernel claimed")


def _require_cuda():
    if not torch.cuda.is_available():
        raise SkipTest("requires a real CUDA device; CPU does not qualify this optimization")
    return torch.device("cuda")


@torch.inference_mode()
def test_cta_backend():
    device = _require_cuda()
    config = replace(_config(use_cta_clustering=True, cta_cluster_seq_threshold=8),
                     use_flash3=True)
    model = _model(config, device)
    if any(not b.attn.use_flash3 or b.attn.flash3_fn is None or not b.attn._flash3_accepts_clusters
           for b in model.transformer.h):
        raise SkipTest("requires a real FA3 backend accepting num_sm_clusters")
    baseline = _model(_config(), device)
    baseline.load_state_dict(model.state_dict())
    inputs = torch.randint(0, 64, (2, 16), device=device)
    _assert_logits(model(inputs), baseline(inputs), (2, 16, 64), tolerance=2e-2)
    print("PASS: cluster-capable FA3 numerical parity (performance unmeasured)")


@torch.inference_mode()
def _compare_decode(config):
    device = _require_cuda()
    model = _model(config, device)
    engine = Engine(model, tokenizer=None, enable_batch_decode=False)
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
    # This tests the hint + actual side stream together, not resident kernels.
    _compare_decode(_config(use_cta_clustering=True, cta_cluster_seq_threshold=8,
                            enable_persistent_decode=True))


def main():
    passed = skipped = 0
    for test in (test_cta_clustering, test_cta_backend, test_persistent_decode, test_integration):
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
