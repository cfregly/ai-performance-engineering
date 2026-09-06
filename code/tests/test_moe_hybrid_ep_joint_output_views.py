from __future__ import annotations

import copy

import pytest
import torch

from labs.fullstack_cluster.moe_hybrid_ep_common import (
    DeepSeekHybridEPModule,
    TopologyInfo,
)


def _module(rank: int) -> DeepSeekHybridEPModule:
    topology = TopologyInfo(
        rank=rank,
        world_size=2,
        local_rank=rank,
        local_world_size=2,
        node_rank=0,
        num_nodes=1,
        initialized=False,
        local_group=None,
    )
    return DeepSeekHybridEPModule(
        hidden_size=4,
        num_experts=4,
        local_experts=2,
        top_k=2,
        topology=topology,
        route_mode="uniform",
        optimized=True,
    ).float()


@pytest.mark.parametrize("rank", [0, 1])
def test_two_rank_joint_remote_output_reuses_contiguous_expert_slice(
    rank: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(101)
    model = _module(rank)
    reference = copy.deepcopy(model)
    source_tokens = [torch.randn(3, 4), torch.randn(2, 4)]
    source_weights = [
        torch.linspace(0.2, 0.6, 3).reshape(-1, 1),
        torch.linspace(0.3, 0.7, 2).reshape(-1, 1),
    ]
    source_experts = [
        torch.tensor([1, 0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
    ]
    remote_rank = 1 - rank
    recv_counts = [0, 0]
    recv_counts[remote_rank] = source_tokens[remote_rank].size(0)
    captured: dict[str, torch.Tensor] = {}
    apply_local_experts = model._apply_local_experts

    def record_joint_output(
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        *,
        buffer_namespace: str = "local",
    ) -> torch.Tensor:
        result = apply_local_experts(
            tokens,
            expert_ids,
            weights,
            buffer_namespace=buffer_namespace,
        )
        captured["joint"] = result
        return result

    monkeypatch.setattr(model, "_apply_local_experts", record_joint_output)
    recv_tokens = source_tokens[remote_rank].clone().requires_grad_()
    recv_weights = source_weights[remote_rank].clone().requires_grad_()
    local_tokens = source_tokens[rank].clone().requires_grad_()
    local_weights = source_weights[rank].clone().requires_grad_()
    remote_outputs, local_outputs = model._apply_joint_experts_with_local_routes(
        recv_tokens=recv_tokens,
        recv_weights=recv_weights,
        recv_local_expert_ids=source_experts[remote_rank],
        recv_counts=recv_counts,
        local_tokens=local_tokens,
        local_weights=local_weights,
        local_expert_ids=source_experts[rank],
        group_rank=rank,
        buffer_namespace="two_rank_view_test",
    )

    joint_outputs = captured["joint"]
    assert remote_outputs.is_contiguous()
    assert remote_outputs.untyped_storage().data_ptr() == joint_outputs.untyped_storage().data_ptr()
    assert local_outputs.untyped_storage().data_ptr() == joint_outputs.untyped_storage().data_ptr()

    reference_tokens = torch.cat(source_tokens, dim=0).requires_grad_()
    reference_weights = torch.cat(source_weights, dim=0).requires_grad_()
    reference_outputs = reference._apply_local_experts(
        reference_tokens,
        torch.cat(source_experts, dim=0),
        reference_weights,
    )
    reconstructed = (
        torch.cat((local_outputs, remote_outputs), dim=0)
        if rank == 0
        else torch.cat((remote_outputs, local_outputs), dim=0)
    )
    torch.testing.assert_close(reconstructed, reference_outputs, rtol=0, atol=0)

    coefficients = torch.linspace(0.5, 1.5, reference_outputs.numel()).reshape_as(
        reference_outputs
    )
    (reconstructed * coefficients).sum().backward()
    (reference_outputs * coefficients).sum().backward()
    reference_token_grads = reference_tokens.grad.split([3, 2])
    reference_weight_grads = reference_weights.grad.split([3, 2])
    torch.testing.assert_close(local_tokens.grad, reference_token_grads[rank], rtol=0, atol=0)
    torch.testing.assert_close(recv_tokens.grad, reference_token_grads[remote_rank], rtol=0, atol=0)
    torch.testing.assert_close(local_weights.grad, reference_weight_grads[rank], rtol=0, atol=0)
    torch.testing.assert_close(recv_weights.grad, reference_weight_grads[remote_rank], rtol=0, atol=0)
    for actual, expected in zip(
        model.experts.parameters(),
        reference.experts.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
