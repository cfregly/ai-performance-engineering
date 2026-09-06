"""CPU math controls for hybrid-EP mixed-precision training state."""

from __future__ import annotations

import copy
import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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


@pytest.mark.parametrize("owner_rank", [0, 1])
def test_joint_expert_batch_inserts_local_routes_in_source_order(owner_rank: int) -> None:
    torch.manual_seed(59)
    topology = TopologyInfo(
        rank=owner_rank,
        world_size=2,
        local_rank=owner_rank,
        local_world_size=2,
        node_rank=0,
        num_nodes=1,
        initialized=False,
        local_group=None,
    )
    reference = DeepSeekHybridEPModule(
        hidden_size=4,
        num_experts=4,
        local_experts=2,
        top_k=2,
        topology=topology,
        route_mode="uniform",
        optimized=False,
    ).float()
    candidate = copy.deepcopy(reference)
    candidate.optimized = True

    source_tokens = [
        torch.randn(3, 4, dtype=torch.bfloat16),
        torch.randn(4, 4, dtype=torch.bfloat16),
    ]
    source_weights = [
        torch.linspace(0.2, 0.6, 3, dtype=torch.float32).reshape(-1, 1),
        torch.linspace(0.3, 0.9, 4, dtype=torch.float32).reshape(-1, 1),
    ]
    source_experts = [
        torch.tensor([1, 0, 1], dtype=torch.int64),
        torch.tensor([0, 1, 0, 1], dtype=torch.int64),
    ]
    reference_tokens = torch.cat(source_tokens, dim=0).requires_grad_()
    reference_weights = torch.cat(source_weights, dim=0).requires_grad_()
    reference_experts = torch.cat(source_experts, dim=0)

    remote_rank = 1 - owner_rank
    recv_tokens = source_tokens[remote_rank].clone().requires_grad_()
    recv_weights = source_weights[remote_rank].clone().requires_grad_()
    local_tokens = source_tokens[owner_rank].clone().requires_grad_()
    local_weights = source_weights[owner_rank].clone().requires_grad_()
    recv_counts = [0, 0]
    recv_counts[remote_rank] = recv_tokens.size(0)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        reference_outputs = reference._apply_local_experts(
            reference_tokens,
            reference_experts,
            reference_weights,
        )
        remote_outputs, local_outputs = candidate._apply_joint_experts_with_local_routes(
            recv_tokens=recv_tokens,
            recv_weights=recv_weights,
            recv_local_expert_ids=source_experts[remote_rank],
            recv_counts=recv_counts,
            local_tokens=local_tokens,
            local_weights=local_weights,
            local_expert_ids=source_experts[owner_rank],
            group_rank=owner_rank,
            buffer_namespace="test_joint",
        )

    candidate_outputs = (
        torch.cat((local_outputs, remote_outputs), dim=0)
        if owner_rank == 0
        else torch.cat((remote_outputs, local_outputs), dim=0)
    )
    torch.testing.assert_close(candidate_outputs, reference_outputs, rtol=0, atol=0)

    coefficients = torch.linspace(
        0.25,
        1.25,
        reference_outputs.numel(),
        dtype=torch.float32,
    ).reshape_as(reference_outputs)
    (reference_outputs * coefficients).sum().backward()
    (candidate_outputs * coefficients).sum().backward()
    reference_token_parts = reference_tokens.grad.split(
        [part.size(0) for part in source_tokens]
    )
    reference_weight_parts = reference_weights.grad.split(
        [part.size(0) for part in source_weights]
    )
    torch.testing.assert_close(local_tokens.grad, reference_token_parts[owner_rank], rtol=0, atol=0)
    torch.testing.assert_close(recv_tokens.grad, reference_token_parts[remote_rank], rtol=0, atol=0)
    torch.testing.assert_close(local_weights.grad, reference_weight_parts[owner_rank], rtol=0, atol=0)
    torch.testing.assert_close(recv_weights.grad, reference_weight_parts[remote_rank], rtol=0, atol=0)
    for actual, expected in zip(
        candidate.experts.parameters(),
        reference.experts.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def _gloo_joint_batch_worker(
    rank: int,
    rendezvous: str,
    output_dir: str,
) -> None:
    torch.set_num_threads(1)
    report = {"rank": rank, "backend": "gloo", "cuda_used": False}
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        local_group = dist.new_group(ranks=[0, 1])
        topology = TopologyInfo(
            rank=rank,
            world_size=2,
            local_rank=rank,
            local_world_size=2,
            node_rank=0,
            num_nodes=1,
            initialized=True,
            local_group=local_group,
        )
        torch.manual_seed(67 + rank)
        baseline = DeepSeekHybridEPModule(
            hidden_size=4,
            num_experts=4,
            local_experts=2,
            top_k=2,
            topology=topology,
            route_mode="uniform",
            optimized=False,
        ).float()
        optimized = DeepSeekHybridEPModule(
            hidden_size=4,
            num_experts=4,
            local_experts=2,
            top_k=2,
            topology=topology,
            route_mode="uniform",
            optimized=True,
        ).float()
        optimized.load_state_dict(baseline.state_dict())
        tokens = torch.randn(6, 4, dtype=torch.bfloat16).requires_grad_()
        weights = torch.linspace(0.2, 0.8, 6, dtype=torch.float32).reshape(-1, 1).requires_grad_()
        optimized_tokens = tokens.detach().clone().requires_grad_()
        optimized_weights = weights.detach().clone().requires_grad_()
        destinations = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int64)
        local_expert_ids = torch.tensor([1, 0, 0, 1, 1, 0], dtype=torch.int64)
        token_indices = torch.arange(6, dtype=torch.int64)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            baseline_outputs, baseline_events = baseline._roundtrip_routes(
                tokens=tokens,
                weights=weights,
                dest_ranks=destinations,
                token_indices=token_indices,
                local_expert_ids=local_expert_ids,
                group=None,
                group_size=2,
                group_rank=rank,
                use_single=False,
                reuse=False,
                event_label="baseline_test",
            )
            optimized_outputs, optimized_events = optimized._roundtrip_routes(
                tokens=optimized_tokens,
                weights=optimized_weights,
                dest_ranks=destinations,
                token_indices=token_indices,
                local_expert_ids=local_expert_ids,
                group=local_group,
                group_size=2,
                group_rank=rank,
                use_single=True,
                reuse=True,
                event_label="optimized_test",
                local_bypass_mask=destinations == rank,
            )
        assert baseline_events is None and optimized_events is None
        torch.testing.assert_close(optimized_outputs, baseline_outputs, rtol=0, atol=0)
        cross_routes = int((destinations != rank).sum())
        assert optimized._dispatch_payload_sent_bytes == cross_routes * 5 * 4
        assert optimized._dispatch_metadata_sent_bytes == cross_routes * 2 * 8
        assert optimized._return_payload_sent_bytes == cross_routes * 4 * 4

        coefficients = torch.linspace(
            0.5,
            1.5,
            baseline_outputs.numel(),
            dtype=torch.float32,
        ).reshape_as(baseline_outputs)
        (baseline_outputs * coefficients).sum().backward()
        (optimized_outputs * coefficients).sum().backward()
        torch.testing.assert_close(optimized_tokens.grad, tokens.grad, rtol=0, atol=0)
        torch.testing.assert_close(optimized_weights.grad, weights.grad, rtol=0, atol=0)
        for actual, expected in zip(
            optimized.experts.parameters(),
            baseline.experts.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(report, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Actual Gloo backend required",
)
def test_two_rank_gloo_joint_batch_matches_list_all_to_all(tmp_path: Path) -> None:
    context = mp.spawn(
        _gloo_joint_batch_worker,
        args=(f"file://{tmp_path / 'store'}", str(tmp_path)),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 30
    try:
        while not context.join(timeout=1, grace_period=1):
            if time.monotonic() >= deadline:
                pytest.fail("Actual two-process Gloo joint-batch control exceeded 30 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    reports = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert all(report["status"] == "PASS" for report in reports)


def test_optimized_multi_node_overlap_fails_before_compute() -> None:
    topology = TopologyInfo(
        rank=0,
        world_size=4,
        local_rank=0,
        local_world_size=2,
        node_rank=0,
        num_nodes=2,
        initialized=False,
        local_group=None,
    )
    model = DeepSeekHybridEPModule(
        hidden_size=4,
        num_experts=4,
        local_experts=1,
        top_k=1,
        topology=topology,
        route_mode="uniform",
        optimized=True,
    )
    inputs = torch.randn(2, 4)

    with pytest.raises(RuntimeError, match="source-ordered joint expert batch"):
        model.forward_loss(
            inputs,
            inputs,
            overlap_mode="local_remote",
            aux_loss_scale=0.01,
        )
