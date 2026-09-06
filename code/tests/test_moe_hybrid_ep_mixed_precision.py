"""CPU math controls for hybrid-EP mixed-precision training state."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from labs.fullstack_cluster.moe_hybrid_ep_common import (
    DeepSeekHybridEPModule,
    ExpertMLP,
    LoadBalancedRouter,
    TopologyInfo,
    _build_fp32_adamw,
)


def _single_rank_module(*, optimized: bool = True) -> DeepSeekHybridEPModule:
    topology = TopologyInfo(
        rank=0,
        world_size=1,
        local_rank=0,
        local_world_size=1,
        node_rank=0,
        num_nodes=1,
        initialized=False,
        local_group=None,
    )
    return DeepSeekHybridEPModule(
        hidden_size=4,
        num_experts=2,
        local_experts=2,
        top_k=1,
        topology=topology,
        route_mode="uniform",
        optimized=optimized,
    ).float()


def test_router_uses_fp32_math_inside_bf16_autocast() -> None:
    torch.manual_seed(41)
    router = LoadBalancedRouter(hidden_size=4, num_experts=5, top_k=2)
    inputs = torch.randn(7, 4, dtype=torch.bfloat16)
    bias = torch.linspace(-0.2, 0.2, 5, dtype=torch.bfloat16)

    expected_logits = F.linear(inputs.float(), router.gate.weight) + bias.float()
    expected_top_logits, expected_indices = torch.topk(expected_logits, 2, dim=-1)
    expected_weights = torch.softmax(expected_top_logits, dim=-1)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        weights, indices, aux = router(inputs, expert_bias=bias)
        loss = weights.square().sum() + aux["balance_loss"]

    assert weights.dtype == torch.float32
    assert aux["balance_loss"].dtype == torch.float32
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)

    loss.backward()
    assert router.gate.weight.grad is not None
    assert router.gate.weight.grad.dtype == torch.float32
    assert torch.isfinite(router.gate.weight.grad).all()


def test_fp32_adamw_rejects_quantized_master_parameters() -> None:
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="requires FP32 master parameters"):
        _build_fp32_adamw([parameter], learning_rate=2e-4)


def test_router_rejects_quantized_trainable_state() -> None:
    router = LoadBalancedRouter(hidden_size=4, num_experts=3, top_k=1).bfloat16()

    with pytest.raises(RuntimeError, match="router parameters must remain FP32"):
        router(torch.ones(2, 4, dtype=torch.bfloat16))


def test_bf16_compute_keeps_real_adamw_parameters_and_moments_fp32() -> None:
    torch.manual_seed(43)
    expert = ExpertMLP(hidden_size=4, ffn_size=8).float()
    optimizer = _build_fp32_adamw(expert.parameters(), learning_rate=2e-4)
    inputs = torch.randn(6, 4, dtype=torch.bfloat16)
    before = [parameter.detach().clone() for parameter in expert.parameters()]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = expert(inputs)
        loss = output.float().square().mean()
    assert output.dtype == torch.bfloat16

    loss.backward()
    optimizer.step()

    parameters = list(expert.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    assert any(not torch.equal(old, new) for old, new in zip(before, parameters, strict=True))
    for parameter in parameters:
        state = optimizer.state[parameter]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32
        assert torch.isfinite(state["exp_avg"]).all()
        assert torch.isfinite(state["exp_avg_sq"]).all()


def test_local_and_roundtrip_paths_preserve_fp32_route_weight_accumulation() -> None:
    torch.manual_seed(47)
    model = _single_rank_module(optimized=True)
    reference_expert = copy.deepcopy(model.experts[0])
    tokens = torch.randn(5, 4, dtype=torch.bfloat16).requires_grad_()
    weights = (
        torch.linspace(0.25, 0.75, 5, dtype=torch.float32)
        .reshape(-1, 1)
        .requires_grad_()
    )
    reference_tokens = tokens.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    expert_ids = torch.zeros(5, dtype=torch.int64)
    token_indices = torch.arange(5, dtype=torch.int64)
    destinations = torch.zeros(5, dtype=torch.int64)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected = reference_expert(reference_tokens).float() * reference_weights
        local = model._apply_local_experts(tokens, expert_ids, weights)
        roundtrip, events = model._roundtrip_routes(
            tokens=tokens.detach(),
            weights=weights.detach(),
            dest_ranks=destinations,
            token_indices=token_indices,
            local_expert_ids=expert_ids,
            group=None,
            group_size=1,
            group_rank=0,
            use_single=True,
            reuse=False,
            event_label="mixed_precision_test",
        )

    assert local.dtype == torch.float32
    assert roundtrip.dtype == torch.float32
    assert events is None
    torch.testing.assert_close(local, expected, rtol=0, atol=0)
    torch.testing.assert_close(roundtrip, expected, rtol=0, atol=0)

    local.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(tokens.grad, reference_tokens.grad)
    torch.testing.assert_close(weights.grad, reference_weights.grad)
    for actual, reference in zip(
        model.experts[0].parameters(),
        reference_expert.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual.grad, reference.grad)


def test_route_payload_and_communication_bytes_follow_promoted_tensors() -> None:
    torch.manual_seed(53)
    model = _single_rank_module()
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)

    expanded_tokens, _indices, _counts, _aux, route_payload = model._route_tokens(hidden)
    expanded_weights = route_payload["expanded_weights"]
    assert expanded_tokens.dtype == torch.bfloat16
    assert expanded_weights.dtype == torch.float32

    packed = torch.cat((expanded_tokens, expanded_weights), dim=1)
    metadata = torch.empty((expanded_tokens.size(0), 2), dtype=torch.int64)
    returned = torch.empty_like(expanded_tokens, dtype=torch.float32)
    model._record_roundtrip_sent_bytes(
        group_size=2,
        dispatch_payload=packed,
        dispatch_metadata=metadata,
        return_payload=returned,
    )

    assert packed.dtype == torch.float32
    assert model._dispatch_payload_sent_bytes == packed.numel() * packed.element_size()
    assert model._dispatch_metadata_sent_bytes == metadata.numel() * metadata.element_size()
    assert model._return_payload_sent_bytes == returned.numel() * returned.element_size()
