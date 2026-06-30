from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from ch19.baseline_dynamic_precision import BaselineDynamicPrecisionBenchmark
from ch19.dynamic_precision_benchmark_common import (
    DynamicPrecisionBenchmarkConfig,
    FixedDecodeWorkspace,
    HighConfidenceDecoder,
    build_model,
    build_prompt,
    decode_dynamic_precision,
    decode_fixed_precision,
    decode_host_policy_baseline,
)
from ch19.dynamic_precision_switching import (
    DynamicPrecisionWorkspace,
    compute_entropy,
    decode_with_dynamic_precision,
    should_use_low_precision,
)
from ch19.optimized_dynamic_precision import OptimizedDynamicPrecisionBenchmark


def test_high_confidence_decoder_applies_target_bias_without_full_tensor() -> None:
    class_source = inspect.getsource(HighConfidenceDecoder)
    source = inspect.getsource(HighConfidenceDecoder.forward)

    assert "torch.full_like" not in source
    assert "bias.scatter_" not in source
    assert "self._target_boost_views: dict[tuple[int, torch.device], torch.Tensor] = {}" in class_source
    assert "logits.add_(-4.0)" in source
    assert "logits.scatter_add_(" in source
    assert "target_boost = self._target_boost_views.get(boost_key)" in source
    assert "self._target_boost_views[boost_key] = target_boost" in source
    assert "self._target_boost.expand(next_id.size(0), 1)" in source

    device = torch.device("cpu")
    model = HighConfidenceDecoder(16, 8, dtype=torch.float32, device=device).eval()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 15]], dtype=torch.int64)

    with torch.no_grad():
        x = model.embed(input_ids)
        x = x.mean(dim=1)
        x = F.gelu(model.proj_in(x))
        expected = model.proj_out(x).to(torch.float32)
        next_id = (input_ids[:, -1] + 1) % model.vocab_size
        expected.add_(-4.0)
        expected[torch.arange(input_ids.size(0)), next_id] += 16.0

        assert model._target_boost_views == {}
        actual = model(input_ids=input_ids)
        cached_boost = model._target_boost_views[(input_ids.size(0), device)]
        actual_again = model(input_ids=input_ids)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_again, expected)
    assert model._target_boost_views[(input_ids.size(0), device)] is cached_boost


def test_dynamic_precision_decode_matches_fixed_precision_on_cpu() -> None:
    cfg = DynamicPrecisionBenchmarkConfig(batch_size=2, prompt_len=8, max_steps=8, vocab_size=64, hidden_dim=64)
    device = torch.device("cpu")
    prompt = build_prompt(cfg, device)
    baseline_model = build_model(cfg, device, dtype=torch.float32)
    optimized_model = build_model(cfg, device, dtype=torch.float32)

    baseline_tokens = decode_fixed_precision(baseline_model, prompt, max_steps=cfg.max_steps, device=device)
    optimized_tokens, stats = decode_dynamic_precision(optimized_model, prompt, max_steps=cfg.max_steps, device=device)

    assert torch.equal(baseline_tokens, optimized_tokens)
    assert stats is not None
    assert stats.total_tokens > 0


def test_host_policy_baseline_matches_fixed_precision_on_cpu() -> None:
    cfg = DynamicPrecisionBenchmarkConfig(batch_size=2, prompt_len=8, max_steps=4, vocab_size=64, hidden_dim=64)
    device = torch.device("cpu")
    prompt = build_prompt(cfg, device)
    fixed_model = build_model(cfg, device, dtype=torch.float32)
    host_policy_model = build_model(cfg, device, dtype=torch.float32)

    fixed_tokens = decode_fixed_precision(fixed_model, prompt, max_steps=cfg.max_steps, device=device)
    host_policy_tokens = decode_host_policy_baseline(
        host_policy_model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
    )

    assert torch.equal(fixed_tokens, host_policy_tokens)


def test_low_precision_policy_handles_confident_and_flat_logits() -> None:
    confident_logits = torch.zeros(2, 32)
    confident_logits[:, 7] = 12.0
    flat_logits = torch.zeros(2, 32)

    assert should_use_low_precision(
        confident_logits,
        entropy_threshold=0.5,
        max_prob_threshold=0.8,
    )
    assert not should_use_low_precision(
        flat_logits,
        entropy_threshold=0.5,
        max_prob_threshold=0.8,
    )


def test_compute_entropy_reuses_log_softmax_probabilities() -> None:
    source = inspect.getsource(compute_entropy)

    assert "log_probs = torch.log_softmax(logits, dim=dim)" in source
    assert "probs = log_probs.exp()" in source
    assert "torch.softmax(logits" not in source

    logits = torch.tensor([[1.0, 2.0, -1.0], [0.25, 0.25, 0.25]])
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = -(log_probs.exp() * log_probs).sum(dim=-1)

    torch.testing.assert_close(compute_entropy(logits), expected)


def test_dynamic_precision_decoders_reuse_selection_buffers() -> None:
    fixed_source = inspect.getsource(decode_fixed_precision)
    dynamic_source = inspect.getsource(decode_with_dynamic_precision)

    assert "next_token = torch.empty((batch_size, 1)" in fixed_source
    assert "torch.max(last_step_logits, dim=-1, keepdim=True, out=(next_token_values, next_token))" in fixed_source
    assert "torch.argmax(last_step_logits" not in fixed_source
    assert "next_token = torch.empty((batch_size, 1)" in dynamic_source
    assert "top2_values = torch.empty(" in dynamic_source
    assert "top2_indices = torch.empty(" in dynamic_source
    assert "torch.topk(last, k=2, dim=topk_dim, out=(top2_values, top2_indices))" in dynamic_source
    assert "torch.max(last_step_logits, dim=-1, keepdim=True, out=(next_token_values, next_token))" in dynamic_source
    assert "next_token = torch.argmax(last_step_logits" not in dynamic_source


def test_dynamic_precision_decoders_accept_reusable_workspaces_on_cpu() -> None:
    cfg = DynamicPrecisionBenchmarkConfig(batch_size=2, prompt_len=8, max_steps=4, vocab_size=64, hidden_dim=64)
    device = torch.device("cpu")
    prompt = build_prompt(cfg, device)
    output_shape = (cfg.batch_size, cfg.prompt_len + cfg.max_steps)
    token_shape = (cfg.batch_size, 1)

    fixed_model = build_model(cfg, device, dtype=torch.float32)
    fixed_workspace = FixedDecodeWorkspace(
        generated=torch.empty(output_shape, device=device, dtype=prompt.dtype),
        next_token=torch.empty(token_shape, device=device, dtype=prompt.dtype),
        next_token_values=torch.empty(token_shape, device=device, dtype=torch.float32),
    )
    fixed_generated_ptr = fixed_workspace.generated.data_ptr()
    fixed_values_ptr = fixed_workspace.next_token_values.data_ptr()

    fixed_tokens = decode_fixed_precision(
        fixed_model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        workspace=fixed_workspace,
    )

    assert fixed_tokens.data_ptr() == fixed_generated_ptr
    assert fixed_workspace.next_token_values is not None
    assert fixed_workspace.next_token_values.data_ptr() == fixed_values_ptr

    host_model = build_model(cfg, device, dtype=torch.float32)
    host_workspace = FixedDecodeWorkspace(
        generated=torch.empty(output_shape, device=device, dtype=prompt.dtype),
        next_token=torch.empty(token_shape, device=device, dtype=prompt.dtype),
        next_token_values=torch.empty(token_shape, device=device, dtype=torch.float32),
        host_logits_buffer=torch.empty((cfg.batch_size, cfg.vocab_size), device="cpu", dtype=torch.float32),
        policy_metrics_buffer=torch.empty(4, device="cpu", dtype=torch.float32),
        policy_metric_values=[0.0] * 4,
        policy_top2_values=torch.empty((cfg.batch_size, 2), device="cpu", dtype=torch.float32),
        policy_top2_indices=torch.empty((cfg.batch_size, 2), device="cpu", dtype=torch.long),
    )
    host_generated_ptr = host_workspace.generated.data_ptr()
    host_logits_ptr = host_workspace.host_logits_buffer.data_ptr()
    host_policy_ptr = host_workspace.policy_metrics_buffer.data_ptr()
    host_policy_values = host_workspace.policy_metric_values
    host_top2_values_ptr = host_workspace.policy_top2_values.data_ptr()
    host_top2_indices_ptr = host_workspace.policy_top2_indices.data_ptr()

    host_tokens = decode_host_policy_baseline(
        host_model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        workspace=host_workspace,
    )

    assert host_tokens.data_ptr() == host_generated_ptr
    assert host_workspace.host_logits_buffer is not None
    assert host_workspace.host_logits_buffer.data_ptr() == host_logits_ptr
    assert host_workspace.policy_metrics_buffer is not None
    assert host_workspace.policy_metrics_buffer.data_ptr() == host_policy_ptr
    assert host_workspace.policy_metric_values is host_policy_values
    assert host_workspace.policy_top2_values is not None
    assert host_workspace.policy_top2_values.data_ptr() == host_top2_values_ptr
    assert host_workspace.policy_top2_indices is not None
    assert host_workspace.policy_top2_indices.data_ptr() == host_top2_indices_ptr
    assert any(value != 0.0 for value in host_workspace.policy_metric_values)

    dynamic_model = build_model(cfg, device, dtype=torch.float32)
    dynamic_workspace = DynamicPrecisionWorkspace(
        generated=torch.empty(output_shape, device=device, dtype=prompt.dtype),
        next_token=torch.empty(token_shape, device=device, dtype=prompt.dtype),
        next_token_values=torch.empty(token_shape, device=device, dtype=torch.float32),
        top2_values=torch.empty((cfg.batch_size, 2), device=device, dtype=torch.float32),
        top2_indices=torch.empty((cfg.batch_size, 2), device=device, dtype=torch.long),
        margin_values=torch.empty(cfg.batch_size, device=device, dtype=torch.float32),
        margin_mean=torch.empty((), device=device, dtype=torch.float32),
        ema_conf=torch.empty((), device=device, dtype=torch.float32),
    )
    dynamic_generated_ptr = dynamic_workspace.generated.data_ptr()
    dynamic_top2_ptr = dynamic_workspace.top2_values.data_ptr()

    dynamic_tokens, stats = decode_with_dynamic_precision(
        dynamic_model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        enable_fp8=False,
        enable_fp4=False,
        reeval_interval=1,
        workspace=dynamic_workspace,
    )

    assert dynamic_tokens.data_ptr() == dynamic_generated_ptr
    assert dynamic_workspace.top2_values is not None
    assert dynamic_workspace.top2_values.data_ptr() == dynamic_top2_ptr
    assert stats is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for chapter 19 dynamic-precision benchmark pair")
def test_dynamic_precision_benchmark_pair_matches_on_gpu() -> None:
    cfg = DynamicPrecisionBenchmarkConfig(batch_size=2, prompt_len=8, max_steps=8, vocab_size=64, hidden_dim=64)
    baseline = BaselineDynamicPrecisionBenchmark(cfg=cfg)
    optimized = OptimizedDynamicPrecisionBenchmark(cfg=cfg)

    baseline.setup()
    optimized.setup()
    try:
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        assert baseline.output is not None
        assert optimized.output is not None
        assert torch.equal(baseline.output.cpu(), optimized.output.cpu())
    finally:
        baseline.teardown()
        optimized.teardown()
