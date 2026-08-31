from __future__ import annotations

import json
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]


def test_disaggregated_decode_consumes_and_verifies_transferred_context() -> None:
    from ch15.baseline_disaggregated_inference_multigpu import (
        DisaggConfig,
        _allocate_decode_outputs,
        _collect_decode_verification_outputs,
        _transferred_kv_context,
        _write_decode_verification_output,
    )
    from core.optimization.moe_inference import MoeInferenceConfig, SimpleMoEGPT

    cache = torch.arange(24, dtype=torch.float32).view(1, 3, 8)
    kv_context, context_fp32 = _transferred_kv_context(cache, context_window=2)
    expected_context = cache[:, :2].mean(dim=1, keepdim=True)
    torch.testing.assert_close(kv_context, expected_context)
    torch.testing.assert_close(context_fp32, expected_context)

    verification = _write_decode_verification_output(
        torch.empty(0),
        torch.tensor([[7]], dtype=torch.long),
        context_fp32,
    )
    torch.testing.assert_close(verification[:, :-1], expected_context.squeeze(1))
    torch.testing.assert_close(verification[:, -1], torch.tensor([7.0]))

    small_cfg = DisaggConfig(batch_size=1, requests_per_rank=2, hidden_size=8)
    decode_outputs = _allocate_decode_outputs(small_cfg, torch.device("cpu"))
    decode_outputs[0].copy_(verification)
    decode_outputs[1].copy_(verification + 1)
    collected = _collect_decode_verification_outputs(
        torch.empty(2, 9),
        decode_outputs,
        batch_size=1,
        hidden_size=8,
    )
    torch.testing.assert_close(collected, torch.cat(decode_outputs))
    try:
        _collect_decode_verification_outputs(
            torch.empty(2, 9),
            [decode_outputs[0], torch.empty(0)],
            batch_size=1,
            hidden_size=8,
        )
    except RuntimeError as exc:
        assert "Decode output 1" in str(exc)
    else:
        raise AssertionError("malformed decode verification output was accepted")

    config = MoeInferenceConfig(
        vocab_size=8,
        hidden_size=8,
        ffn_size=8,
        num_layers=0,
        num_moe_layers=0,
        num_experts=1,
        top_k=1,
        batch_size=1,
        context_window=2,
        decode_tokens=1,
        dtype=torch.float32,
    )
    model = SimpleMoEGPT(config, device=torch.device("cpu")).eval()
    with torch.inference_mode():
        model.embed.weight.zero_()
        model.lm_head.weight.zero_()
        model.lm_head.weight.copy_(torch.eye(8))
        token = torch.tensor([[0]], dtype=torch.long)
        zero_context = torch.zeros(1, 1, 8)
        changed_context = torch.tensor([[[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        _, zero_logits = model.decode(token, kv_context=zero_context)
        _, changed_logits = model.decode(token, kv_context=changed_context)
    assert not torch.equal(zero_logits, changed_logits)

    source = (CODE_ROOT / "ch15" / "baseline_disaggregated_inference_multigpu.py").read_text(
        encoding="utf-8"
    )
    decode_helper = source.split("def _run_decode", 1)[1].split(
        "def _run_torchrun_worker", 1
    )[0]
    torchrun_worker = source.split("def _run_torchrun_worker", 1)[1].split(
        "class _DisaggregatedInferenceMultiGPUBenchmark", 1
    )[0]
    assert "kv_context, context_fp32 = _transferred_kv_context(" in decode_helper
    assert "kv_context=kv_context" in decode_helper
    assert "_write_decode_verification_output(" in decode_helper
    assert "kv_context, context_fp32 = _transferred_kv_context(" in torchrun_worker
    assert "kv_context=kv_context" in torchrun_worker
    assert "_write_decode_verification_output(" in torchrun_worker

    for filename in (
        "expectations_b200.json",
        "expectations_2x_b200.json",
        "expectations_4x_gb200.json",
    ):
        examples = json.loads((CODE_ROOT / "ch15" / filename).read_text(encoding="utf-8"))[
            "examples"
        ]
        assert "disaggregated_inference_multigpu" not in examples


def test_distributed_expert_parallelism_accumulates_weighted_top2_routes() -> None:
    from ch15.expert_parallelism import (
        _accumulate_route_outputs,
        _expert_capacity,
        _route_overflow_mask,
        _weight_route_outputs,
    )

    expert_outputs = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [8.0, 0.0],
            [0.0, 8.0],
        ]
    )
    route_weights = torch.tensor([0.25, 0.75, 0.5, 0.5])
    route_positions = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    weighted = _weight_route_outputs(expert_outputs, route_weights)
    combined = _accumulate_route_outputs(
        torch.empty(2, 2),
        route_positions,
        weighted,
    )
    torch.testing.assert_close(combined, torch.tensor([[1.0, 3.0], [4.0, 4.0]]))
    assert _expert_capacity(8, 2, 4, 1.25) == 5
    overflow = _route_overflow_mask(torch.tensor([0, 1, 0, 0, 1]), capacity=2)
    torch.testing.assert_close(
        overflow,
        torch.tensor([False, False, False, True, False]),
    )

    source = (CODE_ROOT / "ch15" / "expert_parallelism.py").read_text(encoding="utf-8")
    distributed = source.split("def forward_distributed", 1)[1].split("def _parse_args", 1)[0]
    assert "route_expert_ids = flat_idx.reshape(-1)" in distributed
    assert "route_weights = flat_w.reshape(-1)" in distributed
    assert "total_send = route_expert_ids.numel()" in distributed
    assert "route_overflow = _route_overflow_mask(route_expert_ids, cap)" in distributed
    assert "effective_route_weights.masked_fill_(route_overflow, 0.0)" in distributed
    assert '"send_weights"' in distributed
    assert '"recv_weights"' in distributed
    assert "recv_weights[mask]" in distributed
    assert "_accumulate_route_outputs(out, recv_back_pos, recv_back_buf)" in distributed
    assert "top1 = flat_idx[:, 0]" not in distributed
    assert "out[recv_back_pos] = recv_back_buf" not in distributed
