"""Shared helpers for DEP2 (data + expert parallel) single-GPU analogue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class Dep2Config:
    dp_replicas: int = 2
    batch_size: int = 2
    seq_len: int = 32
    hidden_size: int = 256
    intermediate_size: int = 512
    num_experts: int = 8
    top_k: int = 2
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens_per_iter(self) -> int:
        return self.dp_replicas * self.batch_size * self.seq_len


def _topk_gating(x: torch.Tensor, gate_weight: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = x @ gate_weight
    vals, idx = torch.topk(logits, k=top_k, dim=-1)
    weights = torch.softmax(vals, dim=-1)
    return idx, weights


def _weight_outputs_in_place_if_safe(out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and (out.requires_grad or weights.requires_grad):
        return out * weights
    out.mul_(weights)
    return out


def _sum_weighted_routes_in_place_if_safe(out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted = _weight_outputs_in_place_if_safe(out, weights)
    if torch.is_grad_enabled() and (out.requires_grad or weights.requires_grad):
        return weighted.sum(dim=1)
    reduced = weighted[:, 0, :]
    for route_idx in range(1, weighted.shape[1]):
        reduced.add_(weighted[:, route_idx, :])
    return reduced


class Dep2Workload:
    def __init__(self, cfg: Dep2Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.x = torch.randn(
            cfg.dp_replicas,
            cfg.batch_size,
            cfg.seq_len,
            cfg.hidden_size,
            device=device,
            dtype=cfg.dtype,
        )
        self.attn_weight = torch.randn(cfg.hidden_size, cfg.hidden_size, device=device, dtype=cfg.dtype)
        self.gate_weight = torch.randn(cfg.hidden_size, cfg.num_experts, device=device, dtype=cfg.dtype)
        self.w1 = torch.randn(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, device=device, dtype=cfg.dtype)
        self.w2 = torch.randn(cfg.num_experts, cfg.intermediate_size, cfg.hidden_size, device=device, dtype=cfg.dtype)
        self._naive_output: torch.Tensor | None = None

    def _naive_output_buffer(self) -> torch.Tensor:
        shape = (
            self.cfg.dp_replicas,
            self.cfg.batch_size,
            self.cfg.seq_len,
            self.cfg.hidden_size,
        )
        if (
            self._naive_output is None
            or self._naive_output.shape != shape
            or self._naive_output.device != self.x.device
            or self._naive_output.dtype != self.x.dtype
        ):
            self._naive_output = torch.empty(shape, device=self.x.device, dtype=self.x.dtype)
        return self._naive_output

    def _moe_naive(self, tokens: torch.Tensor) -> torch.Tensor:
        idx, weights = _topk_gating(tokens, self.gate_weight, self.cfg.top_k)
        out = torch.empty_like(tokens)
        for slot in range(self.cfg.top_k):
            expert_ids = idx[:, slot]
            for expert in range(self.cfg.num_experts):
                token_ids = (expert_ids == expert).nonzero(as_tuple=True)[0]
                if token_ids.numel() == 0:
                    continue
                x_e = tokens[token_ids]
                h = x_e @ self.w1[expert]
                h = torch.relu_(h)
                y = h @ self.w2[expert]
                weight_factors = weights[token_ids, slot].unsqueeze(-1)
                weighted = _weight_outputs_in_place_if_safe(y, weight_factors)
                if slot == 0:
                    out[token_ids] = weighted
                else:
                    out[token_ids] += weighted
        return out

    def _moe_vectorized(self, tokens: torch.Tensor) -> torch.Tensor:
        idx, weights = _topk_gating(tokens, self.gate_weight, self.cfg.top_k)
        w1_sel = self.w1[idx]
        w2_sel = self.w2[idx]
        x_exp = tokens.unsqueeze(1).expand(-1, self.cfg.top_k, -1)
        h = torch.einsum("tki,tkij->tkj", x_exp, w1_sel)
        h = torch.relu_(h)
        y = torch.einsum("tkj,tkjh->tkh", h, w2_sel)
        return _sum_weighted_routes_in_place_if_safe(y, weights.unsqueeze(-1))

    def forward_naive(self) -> torch.Tensor:
        output = self._naive_output_buffer()
        for replica in range(self.cfg.dp_replicas):
            tokens = self.x[replica].reshape(-1, self.cfg.hidden_size)
            attn = tokens @ self.attn_weight
            moe = self._moe_naive(tokens)
            replica_out = output[replica].reshape(-1, self.cfg.hidden_size)
            torch.add(attn, moe, out=replica_out)
        return output

    def forward_vectorized(self) -> torch.Tensor:
        tokens = self.x.reshape(-1, self.cfg.hidden_size)
        attn = tokens @ self.attn_weight
        moe = self._moe_vectorized(tokens)
        return (attn + moe).view(
            self.cfg.dp_replicas,
            self.cfg.batch_size,
            self.cfg.seq_len,
            self.cfg.hidden_size,
        )
