from __future__ import annotations

import inspect

import torch

from ch15.dep2_parallel_common import Dep2Config, Dep2Workload


def test_dep2_vectorized_flattens_replicas_without_stack() -> None:
    source = inspect.getsource(Dep2Workload.forward_vectorized)
    moe_source = inspect.getsource(Dep2Workload._moe_vectorized)
    helper_source = inspect.getsource(Dep2Workload._moe_vectorized.__globals__["_sum_weighted_routes_in_place_if_safe"])
    assert "tokens = self.x.reshape(-1, self.cfg.hidden_size)" in source
    assert "torch.relu_(h)" in moe_source
    assert "torch.relu(h)" not in moe_source
    assert "return _sum_weighted_routes_in_place_if_safe(y, weights.unsqueeze(-1))" in moe_source
    assert "reduced = weighted[:, 0, :]" in helper_source
    assert "reduced.add_(weighted[:, route_idx, :])" in helper_source
    assert "return weighted.sum(dim=1)" not in moe_source
    assert "(y * weights.unsqueeze(-1)).sum(dim=1)" not in moe_source
    assert "torch.stack(" not in source
    assert "for replica in range" not in source

    torch.manual_seed(0)
    cfg = Dep2Config(
        dp_replicas=3,
        batch_size=2,
        seq_len=4,
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        top_k=2,
        dtype=torch.float32,
    )
    workload = Dep2Workload(cfg, torch.device("cpu"))

    expected = []
    for replica in range(cfg.dp_replicas):
        tokens = workload.x[replica].reshape(-1, cfg.hidden_size)
        expected.append((tokens @ workload.attn_weight) + workload._moe_vectorized(tokens))
    expected_tensor = torch.stack(expected, dim=0).view(
        cfg.dp_replicas,
        cfg.batch_size,
        cfg.seq_len,
        cfg.hidden_size,
    )

    torch.testing.assert_close(workload.forward_vectorized(), expected_tensor)


def test_dep2_naive_moe_seeds_output_from_first_route() -> None:
    source = inspect.getsource(Dep2Workload._moe_naive)
    assert "out = torch.empty_like(tokens)" in source
    assert "torch.relu_(h)" in source
    assert "torch.relu(h)" not in source
    assert "torch.zeros_like(tokens)" not in source
    assert "for slot in range(self.cfg.top_k):" in source
    assert "token_ids = (expert_ids == expert).nonzero(as_tuple=True)[0]" in source
    assert "if token_ids.numel() == 0:" in source
    assert "weight_factors = weights[token_ids, slot].unsqueeze(-1)" in source
    assert "weighted = _weight_outputs_in_place_if_safe(y, weight_factors)" in source
    assert "if slot == 0:" in source
    assert "out[token_ids] = weighted" in source
    assert "out[token_ids] += weighted" in source
    assert "weighted = y * weights[token_ids, slot].unsqueeze(-1)" not in source
    assert "torch.any(mask)" not in source
    assert "mask.nonzero" not in source

    torch.manual_seed(1)
    cfg = Dep2Config(
        dp_replicas=1,
        batch_size=2,
        seq_len=3,
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        top_k=2,
        dtype=torch.float32,
    )
    workload = Dep2Workload(cfg, torch.device("cpu"))
    tokens = workload.x.reshape(-1, cfg.hidden_size)

    torch.testing.assert_close(workload._moe_naive(tokens), workload._moe_vectorized(tokens))


def test_dep2_naive_forward_reuses_output_buffer_without_stack() -> None:
    class_source = inspect.getsource(Dep2Workload)
    forward_source = inspect.getsource(Dep2Workload.forward_naive)

    assert "self._naive_output: torch.Tensor | None = None" in class_source
    assert "def _naive_output_buffer(self) -> torch.Tensor:" in class_source
    assert "output = self._naive_output_buffer()" in forward_source
    assert "replica_out = output[replica].reshape(-1, self.cfg.hidden_size)" in forward_source
    assert "torch.add(attn, moe, out=replica_out)" in forward_source
    assert "outputs = []" not in forward_source
    assert "outputs.append(" not in forward_source
    assert "torch.stack(outputs" not in forward_source

    torch.manual_seed(2)
    cfg = Dep2Config(
        dp_replicas=2,
        batch_size=2,
        seq_len=3,
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        top_k=2,
        dtype=torch.float32,
    )
    workload = Dep2Workload(cfg, torch.device("cpu"))

    expected = torch.empty_like(workload.x)
    for replica in range(cfg.dp_replicas):
        tokens = workload.x[replica].reshape(-1, cfg.hidden_size)
        expected[replica].view(-1, cfg.hidden_size).copy_(
            (tokens @ workload.attn_weight) + workload._moe_naive(tokens)
        )

    first = workload.forward_naive()
    first_ptr = first.data_ptr()
    second = workload.forward_naive()

    assert second.data_ptr() == first_ptr
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
