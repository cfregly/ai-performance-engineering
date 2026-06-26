from __future__ import annotations

import inspect

import torch

from ch15.dep2_parallel_common import Dep2Config, Dep2Workload


def test_dep2_vectorized_flattens_replicas_without_stack() -> None:
    source = inspect.getsource(Dep2Workload.forward_vectorized)
    assert "tokens = self.x.reshape(-1, self.cfg.hidden_size)" in source
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
