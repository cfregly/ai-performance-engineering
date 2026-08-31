"""Real CPU autograd/Gloo controls; no CUDA device or collective is simulated."""
from __future__ import annotations

import copy
from datetime import timedelta
import json
import os
import socket
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from labs.fullstack_cluster.moe_hybrid_ep_common import (
    DeepSeekHybridEPModule, ExpertMLP, HybridEPTrainer, TopologyInfo, _create_local_group,
)


def _module(rank=0, world=1, local=1, group=None, optimized=True, top_k=1):
    topology = TopologyInfo(rank, world, rank % local, local, rank // local, world // local, world > 1, group)
    return DeepSeekHybridEPModule(4, world, 1, top_k, topology, route_mode="uniform", optimized=optimized)


@pytest.mark.parametrize("optimized", [False, True])
def test_expert_outputs_support_real_gradients_and_repeated_steps(optimized):
    torch.manual_seed(53)
    model = _module(optimized=optimized)
    reference = copy.deepcopy(model.experts[0])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        x = torch.randn(3, 4, requires_grad=True)
        weight = torch.tensor([[0.25], [0.5], [0.75]], requires_grad=True)
        ref_x, ref_weight = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
        actual = model._apply_local_experts(x, torch.zeros(3, dtype=torch.int64), weight)
        expected = reference(ref_x) * ref_weight
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(weight.grad, ref_weight.grad)
        for actual_p, expected_p in zip(model.experts[0].parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual_p.grad, expected_p.grad)
        optimizer.step()
        ref_optimizer.step()


def test_inference_branch_storage_does_not_alias_or_overwrite_live_output():
    model = _module()
    with torch.no_grad():
        first = model._apply_local_experts(torch.ones(4, 4), torch.zeros(4, dtype=torch.long), torch.ones(4, 1), buffer_namespace="remote")
        snapshot = first.clone()
        second = model._apply_local_experts(torch.full((2, 4), 2.0), torch.zeros(2, dtype=torch.long), torch.ones(2, 1), buffer_namespace="same_node")
        assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
        torch.testing.assert_close(first, snapshot, rtol=0, atol=0)
        reused = model._apply_local_experts(torch.ones(2, 4), torch.zeros(2, dtype=torch.long), torch.ones(2, 1), buffer_namespace="remote")
        assert reused.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()


def _gloo_worker(rank: int, world: int, rendezvous: str, output_dir: str):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=15))
    report = {"rank": rank, "backend": "gloo", "cuda_used": False, "checks": []}
    try:
        local = _create_local_group(rank, world, 2)
        local_sum = torch.tensor(float(rank))
        dist.all_reduce(local_sum, group=local)
        assert local_sum.item() == (1 if rank < 2 else 5)
        report["checks"].append("ordered_node_groups")
        torch.manual_seed(1000 + rank)
        model = _module(rank, world, 2, local)
        expert_before = [p.detach().clone() for p in model.experts.parameters()]
        before = [p.detach().clone() for p in model.replicated_parameters()]
        signatures = [None] * world
        dist.all_gather_object(signatures, before)
        assert any(not torch.equal(signatures[0][0], values[0]) for values in signatures[1:])
        model.synchronize_replicated_parameters()
        for parameter, expected in zip(model.replicated_parameters(), signatures[0], strict=True):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        for parameter, expected in zip(model.experts.parameters(), expert_before, strict=True):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        report["checks"].append("replica_broadcast_preserves_expert_shards")

        # Run actual replica gradient averaging + optimizer step, then compare all ranks.
        trainer = HybridEPTrainer.__new__(HybridEPTrainer)
        trainer.model, trainer.topology = model, model.topology
        data = torch.full((2, 4), float(rank + 1) / 4)
        loss = model.output_proj(model.input_proj(data)).square().sum() + model.router.gate(data).square().sum()
        loss.backward()
        local_grads = [p.grad.clone() for p in model.replicated_parameters()]
        all_grads = [None] * world
        dist.all_gather_object(all_grads, local_grads)
        trainer._sync_replicated_grads()
        for index, parameter in enumerate(model.replicated_parameters()):
            torch.testing.assert_close(parameter.grad, sum(g[index] for g in all_grads) / world)
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        replicas = [None] * world
        dist.all_gather_object(replicas, [p.detach().clone() for p in model.replicated_parameters()])
        for other in replicas[1:]:
            for actual, expected in zip(other, replicas[0], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report["checks"].append("replica_gradients_and_optimizer_step")

        expert_states = [None] * world
        dist.all_gather_object(expert_states, model.experts[0].state_dict())
        for use_single in (True, False):
            # Gloo's functional list implementation uses scatter, which cannot
            # handle unequal splits (or nonzero-global-rank subgroups in 2.8).
            # NCCL list parity remains a real GPU gate, not a fabricated CPU pass.
            phases = ("remote", "same_node", "all_empty") if use_single else ("balanced_world",)
            for phase in phases:
                model.zero_grad(set_to_none=True)
                if phase == "remote":
                    count = 2 if rank == 1 else 1 if rank == 3 else 0
                    destination = 2 if rank == 1 else 0
                    group, group_size, group_rank, owner = None, world, rank, destination
                elif phase == "same_node":
                    count = 2 if rank % 2 else 0
                    destination = 0
                    group, group_size, group_rank, owner = local, 2, rank % 2, (rank // 2) * 2
                elif phase == "balanced_world":
                    count, destination = world, 0
                    group, group_size, group_rank, owner = None, world, rank, 0
                else:
                    count, destination = 0, 0
                    group, group_size, group_rank, owner = None, world, rank, 0
                x = (torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) / 10 + rank / 10).requires_grad_()
                weight = (torch.arange(count, dtype=torch.float32).reshape(count, 1) / 8 + 0.25).requires_grad_()
                destinations = torch.arange(world - 1, -1, -1) if phase == "balanced_world" else torch.full((count,), destination, dtype=torch.long)
                tokens = torch.arange(count, dtype=torch.long)
                ids = torch.zeros(count, dtype=torch.long)
                print(f"rank={rank} phase={phase} single={use_single} forward", flush=True)
                actual, events = model._roundtrip_routes(
                    tokens=x, weights=weight, dest_ranks=destinations,
                    token_indices=tokens, local_expert_ids=ids, group=group, group_size=group_size,
                    group_rank=group_rank, use_single=use_single, reuse=True, event_label=phase,
                )
                assert events is None  # No GPU timing is invented for CPU collectives.
                reference = ExpertMLP(4, 16)
                reference.load_state_dict(expert_states[owner])
                ref_x, ref_weight = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
                if phase == "balanced_world":
                    pieces = []
                    for row, destination in enumerate(destinations.tolist()):
                        expert = ExpertMLP(4, 16)
                        expert.load_state_dict(expert_states[destination])
                        pieces.append(expert(ref_x[row:row + 1]) * ref_weight[row:row + 1])
                    expected = torch.cat(pieces, dim=0)
                else:
                    expected = reference(ref_x) * ref_weight
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                # Empty senders also execute backward collectives and can own received experts.
                print(f"rank={rank} phase={phase} single={use_single} backward", flush=True)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=1e-5, atol=1e-6)
                requests = [None] * world
                owners = destinations if phase != "same_node" else destinations + (rank // 2) * 2
                dist.all_gather_object(requests, (x.detach(), weight.detach(), owners))
                expert_reference = ExpertMLP(4, 16)
                expert_reference.load_state_dict(expert_states[rank])
                for requested_x, requested_w, requested_owners in requests:
                    selected = requested_owners == rank
                    if selected.any():
                        (expert_reference(requested_x[selected]) * requested_w[selected]).sum().backward()
                for parameter, expected_parameter in zip(model.experts[0].parameters(), expert_reference.parameters(), strict=True):
                    if expected_parameter.grad is None:
                        assert parameter.grad is None
                    else:
                        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=1e-5, atol=1e-6)
                report["checks"].append(f"{phase}_{'single' if use_single else 'list'}_forward_backward")
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Actual Gloo backend required")
def test_four_process_gloo_empty_routes_replicas_and_gradients(tmp_path):
    context = mp.spawn(_gloo_worker, args=(4, f"file://{tmp_path / 'store'}", str(tmp_path)), nprocs=4, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1, grace_period=1):
            if time.monotonic() >= deadline:
                pytest.fail("Actual four-process Gloo validation exceeded 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(4)]
    assert all(report["status"] == "PASS" and len(report["checks"]) == 7 for report in reports)


def _reference_losses(models, inputs, targets):
    """Serial global MoE oracle: explicit token/expert selection, no route helpers."""
    losses, outputs = [], []
    for model, x, target in zip(models, inputs, targets, strict=True):
        hidden = model.input_proj(x)
        logits = model.router.gate(hidden)
        selected_logits, selected = logits.topk(2, dim=-1)
        weights = selected_logits.softmax(dim=-1)
        combined = []
        for token in range(hidden.shape[0]):
            contributions = [
                models[int(selected[token, slot])].experts[0](hidden[token:token + 1]) * weights[token, slot]
                for slot in range(2)
            ]
            combined.append(sum(contributions))
        output = model.output_proj(hidden + torch.cat(combined, dim=0))
        balance = logits.softmax(dim=-1).mean(dim=0).var() * 4
        losses.append(F.mse_loss(output, target) + balance * 0.01)
        outputs.append(output)
    return losses, outputs


def _nccl_worker(rank, world, rendezvous, output_dir, device_index=None):
    """Actual four GPUs; local test uses two logical groups, not a network claim."""
    from core.utils.compile_utils import tf32_override

    assert world == 4
    torch.set_num_threads(1)
    device_index = rank if device_index is None else device_index
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=45))
    report = {"rank": rank, "backend": "nccl", "device_index": device_index,
              "device": torch.cuda.get_device_name(device),
              "capability": list(torch.cuda.get_device_capability(device)),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "logical_node_groups": [[0, 1], [2, 3]], "fabric_qualified": False, "checks": []}
    try:
        hosts = [None] * world
        dist.all_gather_object(hosts, socket.gethostname())
        report["rank_hostnames"] = hosts
        group = _create_local_group(rank, world, 2)
        with tf32_override(enable_matmul=False):
            for pattern in ("skewed", "local_only", "mixed"):
                for optimized, overlap in ((False, "disabled"), (True, "disabled"), (True, "local_remote")):
                    torch.manual_seed(191 + rank)
                    topology = TopologyInfo(rank, world, rank % 2, 2, rank // 2, 2, True, group)
                    model = DeepSeekHybridEPModule(4, 4, 1, 2, topology, route_mode="uniform", optimized=optimized).to(device)
                    with torch.no_grad():
                        model.input_proj.weight.copy_(torch.eye(4, device=device))
                        routing = torch.full((4, 4), -4.0, device=device)
                        for column in range(4):
                            owners = (0, 1) if pattern == "skewed" else (
                                ((column // 2) * 2, (column // 2) * 2 + 1) if pattern == "local_only"
                                else (column, (column + 1) % 4)
                            )
                            routing[owners[0], column], routing[owners[1], column] = 4.0, 3.0
                        model.router.gate.weight.copy_(routing)
                    model.synchronize_replicated_parameters()
                    states = [None] * world
                    dist.all_gather_object(states, {key: value.detach().cpu() for key, value in model.state_dict().items()})
                    references = [_module(rank=index, world=4, top_k=2).double() for index in range(4)]
                    for reference, state in zip(references, states, strict=True):
                        reference.load_state_dict(state)
                    count = (1, 3, 2, 4)[rank]
                    columns = (torch.arange(count) + rank) % 4 if pattern == "mixed" else torch.full((count,), rank)
                    x = torch.eye(4)[columns] * (1 + torch.arange(count).reshape(-1, 1) / 10)
                    target = torch.arange(count * 4).reshape(count, 4).float() / 20
                    cases = [None] * world
                    dist.all_gather_object(cases, (x, target))
                    ref_inputs = [case[0].double().requires_grad_() for case in cases]
                    ref_targets = [case[1].double() for case in cases]
                    actual_input = x.to(device).requires_grad_()
                    actual_target = target.to(device)
                    actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    ref_optimizers = [torch.optim.SGD(reference.parameters(), lr=0.01) for reference in references]
                    trainer = HybridEPTrainer.__new__(HybridEPTrainer)
                    trainer.model, trainer.topology = model, topology
                    observed = []
                    handle = model.output_proj.register_forward_hook(
                        lambda _module, _args, output, observed=observed: observed.append(output.detach().clone())
                    )
                    try:
                        for step in range(3):
                            model.zero_grad(set_to_none=True)
                            actual_input.grad = None
                            for reference, ref_input in zip(references, ref_inputs, strict=True):
                                reference.zero_grad(set_to_none=True)
                                ref_input.grad = None
                            expected_losses, expected_outputs = _reference_losses(references, ref_inputs, ref_targets)
                            loss, _metrics = model.forward_loss(actual_input, actual_target, overlap_mode=overlap, aux_loss_scale=0.01)
                            torch.testing.assert_close(observed.pop().cpu().double(), expected_outputs[rank], rtol=5e-5, atol=2e-6)
                            torch.testing.assert_close(loss.detach().cpu().double(), expected_losses[rank].detach(), rtol=5e-5, atol=2e-6)
                            loss.backward()
                            sum(expected_losses).backward()
                            torch.cuda.synchronize(device)
                            torch.testing.assert_close(actual_input.grad.cpu().double(), ref_inputs[rank].grad, rtol=5e-5, atol=2e-6)
                            for actual, expected in zip(model.parameters(), references[rank].parameters(), strict=True):
                                if expected.grad is None:
                                    assert actual.grad is None
                                else:
                                    torch.testing.assert_close(actual.grad.cpu().double(), expected.grad, rtol=5e-5, atol=2e-6)
                            trainer._sync_replicated_grads()
                            replica_lists = [list(reference.replicated_parameters()) for reference in references]
                            for parameters in zip(*replica_lists, strict=True):
                                mean = sum(parameter.grad for parameter in parameters) / world
                                for parameter in parameters:
                                    parameter.grad = mean.clone()
                            actual_optimizer.step()
                            for optimizer in ref_optimizers:
                                optimizer.step()
                            for actual, expected in zip(model.parameters(), references[rank].parameters(), strict=True):
                                torch.testing.assert_close(actual.cpu().double(), expected, rtol=5e-5, atol=2e-6)
                            report["checks"].append(f"{pattern}/{optimized}/{overlap}/step{step}/outputs_loss_gradients_update")
                            # Inference reuses branch storage; varying token counts
                            # exercises retained capacities while real streams run.
                            with torch.no_grad():
                                for selected_count in (count, 1, count):
                                    # Each rank must use the same per-rank shape rule.
                                    shape_counts = [None] * world
                                    dist.all_gather_object(shape_counts, selected_count)
                                    ref_inference = [value[:n] for value, n in zip(ref_inputs, shape_counts, strict=True)]
                                    ref_inference_targets = [value[:n] for value, n in zip(ref_targets, shape_counts, strict=True)]
                                    expected, expected_output = _reference_losses(references, ref_inference, ref_inference_targets)
                                    inference_loss, _ = model.forward_loss(actual_input[:selected_count], actual_target[:selected_count], overlap_mode=overlap, aux_loss_scale=0.01)
                                    torch.testing.assert_close(observed.pop().cpu().double(), expected_output[rank], rtol=5e-5, atol=2e-6)
                                    torch.testing.assert_close(inference_loss.cpu().double(), expected[rank], rtol=5e-5, atol=2e-6)
                            report["checks"].append(f"{pattern}/{optimized}/{overlap}/step{step}/inference_capacity_reuse")
                    finally:
                        handle.remove()
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        Path(output_dir, f"nccl-rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4 or not dist.is_nccl_available(),
                    reason="Actual four CUDA devices and NCCL required; CPU Gloo is not CUDA acceptance")
def test_four_gpu_nccl_forward_backward_and_reuse_match_serial_reference(tmp_path):
    context = mp.spawn(_nccl_worker, args=(4, f"file://{tmp_path / 'nccl-store'}", str(tmp_path)), nprocs=4, join=False)
    deadline = time.monotonic() + 300
    try:
        while not context.join(timeout=1, grace_period=1):
            if time.monotonic() >= deadline:
                pytest.fail("Actual four-GPU NCCL acceptance exceeded 300 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    reports = [json.loads((tmp_path / f"nccl-rank-{rank}.json").read_text()) for rank in range(4)]
    assert all(report["status"] == "PASS" and len(report["checks"]) == 54 for report in reports)


if __name__ == "__main__":
    # Optional real two-host gate: torchrun --nnodes=2 --nproc-per-node=2 ...
    # Keep host GPU ownership with torchrun's LOCAL_RANK, and require four ranks.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-worker", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ["LOCAL_WORLD_SIZE"]) != 2:
        raise RuntimeError("The torchrun gate requires two hosts with two local ranks each")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _nccl_worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), "env://",
                 str(args.output_dir), int(os.environ["LOCAL_RANK"]))
