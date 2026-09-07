"""Live-input regression for cache-aware multi-GPU verification payloads."""

from __future__ import annotations

import torch

from ch17.prefill_decode_disagg_multigpu_common import TinyPrefillDecode
from labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_common import (
    CacheAwareDisaggMultiGPUBenchmark,
    CacheAwareDisaggMultiGPUConfig,
    _build_request_plans,
    _split_prompt,
    _verification_prompt_plan,
)


def test_declared_cache_aware_prompt_is_the_full_live_model_input() -> None:
    cfg = CacheAwareDisaggMultiGPUConfig(
        hidden_size=4,
        num_layers=1,
        batch_size=1,
        requests_per_rank=4,
        context_window=4,
        chunk_size=2,
        decode_tokens=2,
        warm_request_ratio=0.75,
        warm_prefix_ratio=0.5,
        prefill_ranks=1,
        dtype=torch.float32,
    )
    benchmark = CacheAwareDisaggMultiGPUBenchmark(
        optimized=True,
        label="cache_aware_live_input_cpu_control",
        cfg=cfg,
    )
    plans = _build_request_plans(cfg, prefill_ranks=1)
    selected_plan = _verification_prompt_plan(plans)
    assert selected_plan.warm_chunks == 0

    torch.manual_seed(42)
    prompt_bank = torch.randn(
        cfg.requests_per_rank,
        cfg.batch_size,
        cfg.context_window,
        cfg.hidden_size,
    )
    benchmark._request_plans = plans
    benchmark._prompts = {0: prompt_bank}
    benchmark._prompt_chunks = {
        (0, request_idx): _split_prompt(prompt_bank[request_idx], cfg.chunk_size)
        for request_idx in range(cfg.requests_per_rank)
    }
    benchmark._bind_live_verification_prompt()

    model = TinyPrefillDecode(
        cfg.hidden_size,
        cfg.num_layers,
        torch.device("cpu"),
        cfg.dtype,
    ).eval()

    def run_selected_prompt() -> torch.Tensor:
        chunks = benchmark._prompt_chunks[
            (selected_plan.prefill_rank, selected_plan.local_request_idx)
        ]
        full_prompt = torch.cat(tuple(chunks), dim=1)
        with torch.inference_mode():
            cache, seed = model.prefill(full_prompt)
            return model.decode(seed, cache, cfg.decode_tokens).clone()

    original_output = run_selected_prompt()
    assert benchmark._verify_prompt is not None
    assert benchmark._verify_prompt.shape == (
        cfg.batch_size,
        cfg.context_window,
        cfg.hidden_size,
    )
    assert (
        benchmark._verify_prompt.data_ptr()
        == prompt_bank[selected_plan.local_request_idx].data_ptr()
    )

    with torch.no_grad():
        benchmark._verify_prompt.add_(0.5)
    perturbed_output = run_selected_prompt()

    assert not torch.equal(original_output, perturbed_output)
    reconstructed = torch.cat(
        tuple(
            benchmark._prompt_chunks[
                (selected_plan.prefill_rank, selected_plan.local_request_idx)
            ]
        ),
        dim=1,
    )
    torch.testing.assert_close(reconstructed, benchmark._verify_prompt, rtol=0.0, atol=0.0)
