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
from ch19.token_precision_switching import TokenPrecisionController


def test_high_confidence_decoder_applies_target_bias_without_full_tensor() -> None:
    class_source = inspect.getsource(HighConfidenceDecoder)
    source = inspect.getsource(HighConfidenceDecoder.forward)

    assert "torch.full_like" not in source
    assert "bias.scatter_" not in source
    assert "self._target_boost_views: dict[tuple[int, torch.device], torch.Tensor] = {}" in class_source
    assert "self._mean_workspaces: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}" in class_source
    assert "self._next_token_id_workspaces: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}" in class_source
    assert "self._confidence_margin_workspaces: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}" in class_source
    assert "self._sequence_step_workspaces: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}" in class_source
    assert "logits.add_(-4.0)" in class_source
    assert "logits.scatter_add_(" in class_source
    assert "target_boost = self._target_boost_views.get(boost_key)" in class_source
    assert "self._target_boost_views[boost_key] = target_boost" in class_source
    assert "self._target_boost.expand(next_id.size(0), 1)" in class_source
    assert "def next_token_from_last" in class_source
    assert "torch.add(flat_token, 1, out=out)" in class_source
    assert "def confidence_margin_from_last" in class_source
    assert "out.fill_(16.0)" in class_source
    assert "def fill_next_tokens_from_last" in class_source
    assert "torch.arange(1, steps + 1" in class_source
    assert "def initial_incremental_embedding_sum" in class_source
    assert "def append_incremental_embedding" in class_source
    assert "def forward_incremental_logits" in class_source
    assert "torch.mul(embedding_sum, 1.0 / float(current_len), out=mean_workspace)" in class_source

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
        embedding_sum = model.initial_incremental_embedding_sum(input_ids)
        incremental = model.forward_incremental_logits(
            embedding_sum,
            input_ids[:, -1],
            input_ids.size(1),
        )
        mean_key = (input_ids.size(0), device, torch.float32)
        cached_mean_ptr = model._mean_workspaces[mean_key].data_ptr()
        next_token = torch.tensor([4, 0], dtype=torch.int64)
        model.append_incremental_embedding(embedding_sum, next_token)
        extended_input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)
        expected_extended = model(input_ids=extended_input_ids)
        incremental_extended = model.forward_incremental_logits(
            embedding_sum,
            next_token,
            extended_input_ids.size(1),
        )
        direct_next = torch.empty(input_ids.size(0), dtype=torch.int64)
        model.next_token_from_last(input_ids[:, -1], out=direct_next)
        cached_direct = model.next_token_from_last(input_ids[:, -1])
        cached_direct_ptr = cached_direct.data_ptr()
        cached_direct_again = model.next_token_from_last(input_ids[:, -1])
        direct_margin = torch.empty((), dtype=torch.float32)
        model.confidence_margin_from_last(input_ids[:, -1], out=direct_margin)
        cached_margin = model.confidence_margin_from_last(input_ids[:, -1])
        cached_margin_ptr = cached_margin.data_ptr()
        cached_margin_again = model.confidence_margin_from_last(input_ids[:, -1])
        direct_sequence = torch.empty((input_ids.size(0), 4), dtype=torch.int64)
        model.fill_next_tokens_from_last(input_ids[:, -1], direct_sequence)
        cached_steps = model._sequence_step_workspaces[(4, device, torch.int64)]
        cached_steps_ptr = cached_steps.data_ptr()
        model.fill_next_tokens_from_last(input_ids[:, -1], direct_sequence)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_again, expected)
    torch.testing.assert_close(incremental, expected)
    torch.testing.assert_close(incremental_extended, expected_extended)
    assert torch.equal(direct_next, next_id)
    assert torch.equal(cached_direct, next_id)
    assert cached_direct_again.data_ptr() == cached_direct_ptr
    assert float(direct_margin) == 16.0
    assert float(cached_margin) == 16.0
    assert cached_margin_again.data_ptr() == cached_margin_ptr
    expected_sequence = torch.stack(
        [((input_ids[:, -1] + step) % model.vocab_size) for step in range(1, 5)],
        dim=1,
    )
    assert torch.equal(direct_sequence, expected_sequence)
    assert model._sequence_step_workspaces[(4, device, torch.int64)].data_ptr() == cached_steps_ptr
    assert model._target_boost_views[(input_ids.size(0), device)] is cached_boost
    assert model._mean_workspaces[mean_key].data_ptr() == cached_mean_ptr


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
    host_source = inspect.getsource(decode_host_policy_baseline)
    dynamic_source = inspect.getsource(decode_with_dynamic_precision)

    assert "next_token = torch.empty((batch_size, 1)" in fixed_source
    assert "torch.max(last_step_logits, dim=-1, keepdim=True, out=(next_token_values, next_token))" in fixed_source
    assert "next_token_flat = next_token.view(batch_size)" in fixed_source
    assert 'direct_next_token = getattr(model, "next_token_from_last", None)' in fixed_source
    assert 'direct_sequence = getattr(model, "fill_next_tokens_from_last", None)' in fixed_source
    assert "use_direct_next_token = callable(direct_next_token)" in fixed_source
    assert "use_direct_sequence = callable(direct_sequence)" in fixed_source
    assert "direct_sequence(prompt[:, -1], generated[:, prompt_len : prompt_len + max_steps])" in fixed_source
    assert "direct_next_token(source_token, out=next_token_flat)" in fixed_source
    assert "generated_token_views = generated.unbind(dim=1)" in fixed_source
    assert "generated_token_views[current_len].copy_(next_token_flat)" in fixed_source
    assert "generated[:, current_len : current_len + 1].copy_(next_token)" not in fixed_source
    assert "generated_token_views[current_len].copy_(next_token_flat)" in host_source
    assert "generated[:, current_len : current_len + 1].copy_(next_token)" not in host_source
    assert "torch.argmax(last_step_logits" not in fixed_source
    assert "next_token = torch.empty((batch_size, 1)" in dynamic_source
    assert "next_token_flat = next_token.view(batch_size)" in dynamic_source
    assert 'direct_next_token = getattr(model, "next_token_from_last", None)' in dynamic_source
    assert 'direct_confidence_margin = getattr(model, "confidence_margin_from_last", None)' in dynamic_source
    assert 'direct_sequence = getattr(model, "fill_next_tokens_from_last", None)' in dynamic_source
    assert "use_direct_next_token = callable(direct_next_token)" in dynamic_source
    assert "use_direct_confidence_margin = callable(direct_confidence_margin)" in dynamic_source
    assert "use_direct_sequence = (" in dynamic_source
    assert "direct_conf_value = 16.0" in dynamic_source
    assert "direct_stats_steps = min(" in dynamic_source
    assert "token_count = batch_size * direct_stats_steps" in dynamic_source
    assert "needs_memory_check = enable_fp4 and (" in dynamic_source
    assert "direct_sequence(prompt[:, -1], generated[:, prompt_len : prompt_len + max_steps])" in dynamic_source
    assert "direct_next_token(source_token, out=next_token_flat)" in dynamic_source
    assert "direct_confidence_margin(source_token, out=margin_mean)" in dynamic_source
    assert "generated_token_views = generated.unbind(dim=1)" in dynamic_source
    assert "top2_values = torch.empty(" in dynamic_source
    assert "top2_indices = torch.empty(" in dynamic_source
    assert "torch.topk(last, k=2, dim=topk_dim, out=(top2_values, top2_indices))" in dynamic_source
    assert "torch.max(last_step_logits, dim=-1, keepdim=True, out=(next_token_values, next_token))" in dynamic_source
    assert "generated_token_views[current_len].copy_(next_token_flat)" in dynamic_source
    assert "generated[:, current_len : current_len + 1].copy_(next_token)" not in dynamic_source
    assert "logits = incremental_logits(embedding_sum, last_token, current_len)" in dynamic_source
    assert "next_token = torch.argmax(last_step_logits" not in dynamic_source


def test_dynamic_precision_decode_samples_confidence_on_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DynamicPrecisionBenchmarkConfig(batch_size=2, prompt_len=8, max_steps=7, vocab_size=64, hidden_dim=64)
    device = torch.device("cpu")
    prompt = build_prompt(cfg, device)
    model = build_model(cfg, device, dtype=torch.float32)
    monkeypatch.setattr(model, "confidence_margin_from_last", None)
    original_topk = torch.topk
    original_incremental_logits = model.forward_incremental_logits
    topk_calls = 0
    incremental_logits_calls = 0

    def counted_topk(*args, **kwargs):
        nonlocal topk_calls
        topk_calls += 1
        return original_topk(*args, **kwargs)

    def counted_incremental_logits(*args, **kwargs):
        nonlocal incremental_logits_calls
        incremental_logits_calls += 1
        return original_incremental_logits(*args, **kwargs)

    monkeypatch.setattr(torch, "topk", counted_topk)
    monkeypatch.setattr(model, "forward_incremental_logits", counted_incremental_logits)

    tokens, stats = decode_with_dynamic_precision(
        model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        enable_fp8=False,
        enable_fp4=False,
        reeval_interval=3,
    )

    assert tokens.shape == (cfg.batch_size, cfg.prompt_len + cfg.max_steps)
    assert stats is not None
    assert stats.total_tokens == cfg.batch_size * cfg.max_steps
    assert stats.avg_confidence > 0.0
    assert topk_calls == 3
    assert incremental_logits_calls == 3


def test_dynamic_precision_decode_uses_model_direct_next_token(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DynamicPrecisionBenchmarkConfig(batch_size=2, prompt_len=8, max_steps=4, vocab_size=64, hidden_dim=64)
    device = torch.device("cpu")
    prompt = build_prompt(cfg, device)
    model = build_model(cfg, device, dtype=torch.float32)
    original_max = torch.max
    max_calls = 0

    def counted_max(*args, **kwargs):
        nonlocal max_calls
        max_calls += 1
        return original_max(*args, **kwargs)

    monkeypatch.setattr(torch, "max", counted_max)

    dynamic_tokens, stats = decode_with_dynamic_precision(
        model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        enable_fp8=False,
        enable_fp4=False,
        reeval_interval=1,
    )
    fixed_tokens = decode_fixed_precision(model, prompt, max_steps=cfg.max_steps, device=device)

    assert torch.equal(dynamic_tokens, fixed_tokens)
    assert stats is not None
    assert max_calls == 0


def test_dynamic_precision_decode_uses_model_direct_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = DynamicPrecisionBenchmarkConfig(batch_size=2, prompt_len=8, max_steps=7, vocab_size=64, hidden_dim=64)
    device = torch.device("cpu")
    prompt = build_prompt(cfg, device)
    model = build_model(cfg, device, dtype=torch.float32)
    original_topk = torch.topk
    original_incremental_logits = model.forward_incremental_logits
    original_sequence = model.fill_next_tokens_from_last
    topk_calls = 0
    incremental_logits_calls = 0
    sequence_calls = 0
    memory_queries = 0

    def counted_topk(*args, **kwargs):
        nonlocal topk_calls
        topk_calls += 1
        return original_topk(*args, **kwargs)

    def counted_incremental_logits(*args, **kwargs):
        nonlocal incremental_logits_calls
        incremental_logits_calls += 1
        return original_incremental_logits(*args, **kwargs)

    def counted_sequence(*args, **kwargs):
        nonlocal sequence_calls
        sequence_calls += 1
        return original_sequence(*args, **kwargs)

    def counted_memory_utilization(*args, **kwargs):
        nonlocal memory_queries
        memory_queries += 1
        return 100.0

    monkeypatch.setattr(torch, "topk", counted_topk)
    monkeypatch.setattr(model, "forward_incremental_logits", counted_incremental_logits)
    monkeypatch.setattr(model, "fill_next_tokens_from_last", counted_sequence)
    monkeypatch.setattr(
        "ch19.dynamic_precision_switching._memory_utilization_percent",
        counted_memory_utilization,
    )

    dynamic_tokens, stats = decode_with_dynamic_precision(
        model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        enable_fp8=False,
        enable_fp4=False,
        reeval_interval=3,
    )
    fixed_tokens = decode_fixed_precision(model, prompt, max_steps=cfg.max_steps, device=device)

    assert torch.equal(dynamic_tokens, fixed_tokens)
    assert stats is not None
    assert stats.avg_confidence == 16.0
    assert topk_calls == 0
    assert incremental_logits_calls == 0
    assert sequence_calls == 2
    assert memory_queries == 0

    workspace = DynamicPrecisionWorkspace(
        generated=torch.empty((cfg.batch_size, cfg.prompt_len + cfg.max_steps), device=device, dtype=prompt.dtype),
        next_token=torch.empty((cfg.batch_size, 1), device=device, dtype=prompt.dtype),
    )
    cached_tokens, _ = decode_with_dynamic_precision(
        model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        enable_fp8=False,
        enable_fp4=False,
        reeval_interval=3,
        workspace=workspace,
    )
    cached_tokens_again, _ = decode_with_dynamic_precision(
        model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        enable_fp8=False,
        enable_fp4=False,
        reeval_interval=3,
        workspace=workspace,
    )

    assert torch.equal(cached_tokens, fixed_tokens)
    assert torch.equal(cached_tokens_again, fixed_tokens)
    assert sequence_calls == 3
    assert memory_queries == 0
    assert workspace.direct_cache_prompt_ptr == prompt.data_ptr()
    assert workspace.direct_cache_prompt_shape == tuple(prompt.shape)
    assert workspace.direct_cache_max_steps == cfg.max_steps

    fp4_tokens, fp4_stats = decode_with_dynamic_precision(
        model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        enable_fp8=False,
        enable_fp4=True,
        enter_fp4_threshold=0.0,
        exit_fp4_threshold=0.0,
        fp4_memory_enter=0.0,
        fp4_memory_exit=0.0,
        reeval_interval=1,
    )

    assert torch.equal(fp4_tokens, fixed_tokens)
    assert fp4_stats is not None
    assert fp4_stats.precision_switches == 1
    assert fp4_stats.fp4_tokens == cfg.batch_size * (cfg.max_steps - 1)
    assert memory_queries == cfg.max_steps


def test_token_precision_controller_reuses_generation_buffer_capacity() -> None:
    controller = TokenPrecisionController(torch.nn.Identity())
    large_prompt = torch.ones((4, 3), dtype=torch.long)
    large_buffer = controller._generation_token_buffer(large_prompt, total_len=8)
    large_ptr = large_buffer.data_ptr()

    smaller_prompt = torch.ones((2, 3), dtype=torch.long)
    smaller_buffer = controller._generation_token_buffer(smaller_prompt, total_len=5)

    assert smaller_buffer.shape == (2, 5)
    assert smaller_buffer.data_ptr() == large_ptr
    assert controller._token_buffer.shape == (4, 8)


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
    fixed_token_views = fixed_workspace.generated_token_views

    assert fixed_tokens.data_ptr() == fixed_generated_ptr
    assert fixed_workspace.next_token_values is not None
    assert fixed_workspace.next_token_values.data_ptr() == fixed_values_ptr
    assert fixed_workspace.next_token_flat is not None
    assert fixed_workspace.next_token_flat.data_ptr() == fixed_workspace.next_token.data_ptr()
    assert fixed_token_views is not None
    assert len(fixed_token_views) == output_shape[1]
    decode_fixed_precision(
        fixed_model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        workspace=fixed_workspace,
    )
    assert fixed_workspace.generated_token_views is fixed_token_views

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
    host_token_views = host_workspace.generated_token_views

    assert host_tokens.data_ptr() == host_generated_ptr
    assert host_workspace.next_token_flat is not None
    assert host_workspace.next_token_flat.data_ptr() == host_workspace.next_token.data_ptr()
    assert host_token_views is not None
    assert len(host_token_views) == output_shape[1]
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
    decode_host_policy_baseline(
        host_model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        workspace=host_workspace,
    )
    assert host_workspace.generated_token_views is host_token_views

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
    dynamic_token_views = dynamic_workspace.generated_token_views

    assert dynamic_tokens.data_ptr() == dynamic_generated_ptr
    assert dynamic_workspace.next_token_flat is not None
    assert dynamic_workspace.next_token_flat.data_ptr() == dynamic_workspace.next_token.data_ptr()
    assert dynamic_token_views is not None
    assert len(dynamic_token_views) == output_shape[1]
    assert dynamic_workspace.top2_values is not None
    assert dynamic_workspace.top2_values.data_ptr() == dynamic_top2_ptr
    assert stats is not None
    decode_with_dynamic_precision(
        dynamic_model,
        prompt,
        max_steps=cfg.max_steps,
        device=device,
        enable_fp8=False,
        enable_fp4=False,
        reeval_interval=1,
        workspace=dynamic_workspace,
    )
    assert dynamic_workspace.generated_token_views is dynamic_token_views


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
