from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from labs.flashattention4.baseline_flashattention4_dense import (
    BaselineFlashAttention4DenseBenchmark,
)
from labs.flashattention4.optimized_best_available_attention_alibi import (
    OptimizedBestAvailableAttentionAlibiBenchmark,
)
from labs.flashattention4.flashattention4_common import (
    FlashAttention4Inputs,
    FlashAttention4Config,
    _ALIBI_DISTANCE_CACHE,
    _ALIBI_SLOPE_CACHE,
    _DENSE_MASK_POSITION_CACHE,
    _alibi_distance_for,
    _dense_mask_positions_for,
    _experimental_windowed_skip_reason,
    best_available_candidate_providers,
    build_alibi_slopes,
    build_flashattention4_mode_table_payload,
    build_reference_inputs,
    build_dense_attention_mask,
    count_nonmasked_attention_elements,
    emit_flashattention4_mode_table_artifacts,
    estimate_attention_forward_flops,
    flashattention4_claim_type_id,
    flashattention4_provider_id,
    reference_attention,
    resolve_flashattention4_mode_decision,
    select_lowest_latency_provider,
)


def test_dense_attention_mask_for_windowed_mode_is_causal_and_bounded() -> None:
    _DENSE_MASK_POSITION_CACHE.clear()
    mask = build_dense_attention_mask(
        "windowed",
        seq_len=8,
        window_size=3,
        device=torch.device("cpu"),
    )
    first_q_idx, first_kv_idx = _dense_mask_positions_for(8, torch.device("cpu"))
    second_mask = build_dense_attention_mask(
        "windowed",
        seq_len=8,
        window_size=3,
        device=torch.device("cpu"),
    )
    second_q_idx, second_kv_idx = _dense_mask_positions_for(8, torch.device("cpu"))

    assert mask is not None
    assert second_mask is not None
    assert second_mask.data_ptr() != mask.data_ptr()
    assert second_q_idx.data_ptr() == first_q_idx.data_ptr()
    assert second_kv_idx.data_ptr() == first_kv_idx.data_ptr()
    torch.testing.assert_close(mask, second_mask)
    mask_2d = mask[0, 0]
    assert bool(mask_2d[3, 3])
    assert bool(mask_2d[3, 1])
    assert not bool(mask_2d[3, 0])
    assert not bool(mask_2d[3, 4])


def test_reference_attention_runs_for_softcap_mode_on_cpu() -> None:
    source = inspect.getsource(reference_attention)
    assert "scores.addcmul_(" in source
    assert "scores.div_(inputs.softcap_scale).tanh_().mul_(inputs.softcap_scale)" in source
    assert 'scores.masked_fill_(~inputs.dense_mask, float("-inf"))' in source
    assert "scores = scores - inputs.alibi_slopes" not in source
    assert "scores = inputs.softcap_scale * torch.tanh" not in source
    assert "scores = scores.masked_fill(" not in source

    cfg = FlashAttention4Config(
        batch=1,
        heads=2,
        seq_len=8,
        head_dim=4,
        mode="softcap",
        dtype=torch.float32,
    )
    inputs = build_reference_inputs(cfg, device=torch.device("cpu"), include_block_mask=False)
    output = reference_attention(inputs)
    assert output.shape == (1, 2, 8, 4)
    assert output.dtype == torch.float32


def test_reference_attention_reuses_alibi_distance_cache_on_cpu() -> None:
    cfg = FlashAttention4Config(
        batch=1,
        heads=2,
        seq_len=8,
        head_dim=4,
        mode="alibi",
        dtype=torch.float32,
    )
    _ALIBI_DISTANCE_CACHE.clear()
    _ALIBI_SLOPE_CACHE.clear()
    inputs = build_reference_inputs(cfg, device=torch.device("cpu"), include_block_mask=False)

    first = reference_attention(inputs)
    first_distance = _alibi_distance_for(cfg.seq_len, inputs.q.device)
    first_slopes = build_alibi_slopes(cfg.heads, device=inputs.q.device)
    second = reference_attention(inputs)
    second_distance = _alibi_distance_for(cfg.seq_len, inputs.q.device)
    second_slopes = build_alibi_slopes(cfg.heads, device=inputs.q.device)

    assert second_distance.data_ptr() == first_distance.data_ptr()
    assert second_slopes.data_ptr() == first_slopes.data_ptr()
    assert inputs.alibi_slopes is not None
    assert first_slopes.data_ptr() == inputs.alibi_slopes.data_ptr()
    torch.testing.assert_close(first, second)


def test_attention_flop_count_matches_dense_and_causal_conventions() -> None:
    dense_nonmasked = count_nonmasked_attention_elements(
        "dense",
        q_seq_len=8,
        kv_seq_len=8,
    )
    causal_nonmasked = count_nonmasked_attention_elements(
        "causal",
        q_seq_len=8,
        kv_seq_len=8,
    )
    assert dense_nonmasked == 64
    assert causal_nonmasked == 36

    dense_flops = estimate_attention_forward_flops(
        batch=2,
        heads=4,
        q_seq_len=8,
        kv_seq_len=8,
        head_dim=16,
        mode="dense",
    )
    causal_flops = estimate_attention_forward_flops(
        batch=2,
        heads=4,
        q_seq_len=8,
        kv_seq_len=8,
        head_dim=16,
        mode="causal",
    )
    assert dense_flops == 4 * 2 * 4 * 16 * 64
    assert causal_flops == 4 * 2 * 4 * 16 * 36


def test_nonmasked_attention_count_closed_form_matches_brute_force() -> None:
    def brute_causal(q_seq_len: int, kv_seq_len: int) -> int:
        return sum(min(kv_seq_len, q_idx + 1) for q_idx in range(q_seq_len))

    def brute_windowed(q_seq_len: int, kv_seq_len: int, window_size: int) -> int:
        total = 0
        for q_idx in range(q_seq_len):
            upper = min(kv_seq_len - 1, q_idx)
            lower = max(0, q_idx - window_size + 1)
            total += max(0, upper - lower + 1)
        return total

    for q_seq_len in (0, 1, 2, 8, 9, 16):
        for kv_seq_len in (0, 1, 3, 8, 11, 16):
            assert count_nonmasked_attention_elements(
                "causal",
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
            ) == brute_causal(q_seq_len, kv_seq_len)
            for window_size in (1, 2, 3, 8, 32):
                assert count_nonmasked_attention_elements(
                    "windowed",
                    q_seq_len=q_seq_len,
                    kv_seq_len=kv_seq_len,
                    window_size=window_size,
                ) == brute_windowed(q_seq_len, kv_seq_len, window_size)
                assert count_nonmasked_attention_elements(
                    "alibi_windowed",
                    q_seq_len=q_seq_len,
                    kv_seq_len=kv_seq_len,
                    window_size=window_size,
                ) == brute_windowed(q_seq_len, kv_seq_len, window_size)


def test_windowed_nonmasked_count_respects_window_size() -> None:
    count = count_nonmasked_attention_elements(
        "windowed",
        q_seq_len=8,
        kv_seq_len=8,
        window_size=3,
    )
    assert count == 21


def test_best_available_candidates_include_cudnn_for_dense_and_causal() -> None:
    assert best_available_candidate_providers("dense", include_flash_backend=True) == ("cudnn_sdpa",)
    assert best_available_candidate_providers("causal", include_flash_backend=False) == ("cudnn_sdpa",)


def test_best_available_candidates_exclude_cudnn_for_flex_only_modes() -> None:
    assert best_available_candidate_providers("alibi", include_flash_backend=True) == (
        "flash_backend",
        "flex_tma",
        "flex_compiled",
    )
    assert best_available_candidate_providers("softcap", include_flash_backend=False) == (
        "flex_tma",
        "flex_compiled",
    )


def test_select_lowest_latency_provider_prefers_smallest_median() -> None:
    winner = select_lowest_latency_provider(
        {
            "flash_backend": 0.92,
            "cudnn_sdpa": 0.54,
            "flex_tma": 0.81,
        }
    )
    assert winner == "cudnn_sdpa"


def test_provider_id_encoding_is_stable() -> None:
    assert flashattention4_provider_id("cudnn_sdpa") == 1.0
    assert flashattention4_provider_id("flash_backend") == 2.0
    assert flashattention4_provider_id("flex_tma") == 3.0
    assert flashattention4_provider_id("flex_compiled") == 4.0
    assert flashattention4_provider_id("eager_flex") == 5.0
    assert flashattention4_provider_id("unknown") == 0.0


def test_claim_type_encoding_and_mode_decision_payload() -> None:
    assert flashattention4_claim_type_id("educational") == 1.0
    assert flashattention4_claim_type_id("absolute") == 2.0
    assert flashattention4_claim_type_id("reproduction") == 3.0
    assert flashattention4_claim_type_id("unknown") == 0.0

    decision = resolve_flashattention4_mode_decision("dense")
    assert decision.recommended_backend == "cudnn_sdpa"
    assert decision.recommended_claim_type == "absolute"

    payload = build_flashattention4_mode_table_payload(
        current_mode="alibi",
        run_claim_type="educational",
        target_label="labs/flashattention4:flashattention4",
        selected_provider="flex_tma",
    )
    assert payload["current_run"]["recommended_backend_for_mode"] == "flex_tma"
    assert payload["current_run"]["run_claim_type"] == "educational"
    assert payload["current_run"]["selected_provider"] == "flex_tma"


def test_emit_mode_table_artifacts_writes_json_and_markdown(tmp_path) -> None:
    config = SimpleNamespace(
        subprocess_stderr_dir=str(tmp_path),
        profiling_output_dir=None,
        target_label="labs/flashattention4:best_available_attention",
    )
    paths = emit_flashattention4_mode_table_artifacts(
        config,
        current_mode="dense",
        run_claim_type="absolute",
        selected_provider="cudnn_sdpa",
    )
    assert paths is not None

    json_path = tmp_path / "flashattention4_mode_table__labs_flashattention4_best_available_attention__dense.json"
    md_path = tmp_path / "flashattention4_mode_table__labs_flashattention4_best_available_attention__dense.md"
    assert paths["json"] == str(json_path)
    assert paths["markdown"] == str(md_path)
    assert json_path.exists()
    assert md_path.exists()
    assert '"recommended_backend_for_mode": "cudnn_sdpa"' in json_path.read_text()
    assert "| dense | absolute | cudnn_sdpa |" in md_path.read_text()


def test_explicit_mode_targets_override_default_mode() -> None:
    dense_bench = BaselineFlashAttention4DenseBenchmark()
    alibi_bench = OptimizedBestAvailableAttentionAlibiBenchmark()
    assert dense_bench.config.mode == "dense"
    assert alibi_bench.config.mode == "alibi"


def test_experimental_windowed_skip_reason_requires_nonfinite_failures() -> None:
    assert _experimental_windowed_skip_reason(
        "windowed",
        {"flash_backend": "failed correctness smoke test: non-finite output"},
    ) is not None
    assert _experimental_windowed_skip_reason(
        "windowed",
        {"flash_backend": "failed correctness smoke test: max_diff=1.23"},
    ) is None
    assert _experimental_windowed_skip_reason(
        "causal",
        {"flash_backend": "failed correctness smoke test: non-finite output"},
    ) is None


def test_flashattention_capture_verification_payload_uses_small_cpu_slice() -> None:
    bench = BaselineFlashAttention4DenseBenchmark()
    bench.inputs = FlashAttention4Inputs(
        q=torch.randn(2, 8, 256, 32),
        k=torch.randn(2, 8, 256, 32),
        v=torch.randn(2, 8, 256, 32),
        dense_mask=None,
        block_mask=None,
        alibi_slopes=None,
        softcap_scale=None,
        mode="dense",
        window_size=256,
        block_size=128,
    )
    bench.output = torch.randn(2, 8, 256, 32)

    bench.capture_verification_payload()

    verify_output = bench.get_verify_output()
    assert verify_output.shape == (1, 1, 128, 16)
    assert verify_output.device.type == "cpu"
