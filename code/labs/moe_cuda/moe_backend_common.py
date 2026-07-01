"""Shared helpers for MoE backend selection benchmark (FlashInfer-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import time

import torch


@dataclass(frozen=True)
class MoEBackendConfig:
    batch_size: int = 8
    seq_len: int = 32
    hidden_size: int = 128
    intermediate_size: int = 256
    num_experts: int = 8
    top_k: int = 2
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def tokens_per_iter(self) -> int:
        return self.tokens


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


class MoEBackendWorkload:
    def __init__(self, cfg: MoEBackendConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.x = torch.randn(cfg.tokens, cfg.hidden_size, device=device, dtype=cfg.dtype)
        self.gate_weight = torch.randn(cfg.hidden_size, cfg.num_experts, device=device, dtype=cfg.dtype)
        self.w1 = torch.randn(cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, device=device, dtype=cfg.dtype)
        self.w2 = torch.randn(cfg.num_experts, cfg.intermediate_size, cfg.hidden_size, device=device, dtype=cfg.dtype)
        self._naive_out = torch.empty_like(self.x)

    def forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        idx, weights = _topk_gating(x, self.gate_weight, self.cfg.top_k)
        out = self._naive_out
        out.zero_()
        for expert in range(self.cfg.num_experts):
            token_ids, slot_ids = (idx == expert).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            x_e = x[token_ids]
            h = x_e @ self.w1[expert]
            h = torch.relu_(h)
            y = h @ self.w2[expert]
            y.mul_(weights[token_ids, slot_ids].unsqueeze(-1))
            out.index_add_(0, token_ids, y)
        return out

    def forward_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        idx, weights = _topk_gating(x, self.gate_weight, self.cfg.top_k)
        w1_sel = self.w1[idx]
        w2_sel = self.w2[idx]
        x_exp = x.unsqueeze(1).expand(-1, self.cfg.top_k, -1)
        h = torch.einsum("tki,tkij->tkj", x_exp, w1_sel)
        h = torch.relu_(h)
        y = torch.einsum("tkj,tkjh->tkh", h, w2_sel)
        return _sum_weighted_routes_in_place_if_safe(y, weights.unsqueeze(-1))


@torch.inference_mode()
def select_best_backend(
    candidates: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
    x: torch.Tensor,
) -> Tuple[str, Callable[[torch.Tensor], torch.Tensor]]:
    timings: Dict[str, float] = {}
    sync_device = x.device if x.device.type == "cuda" else None
    for name, fn in candidates.items():
        if sync_device is not None:
            torch.cuda.synchronize(sync_device)
        start = time.perf_counter()
        _ = fn(x)
        if sync_device is not None:
            torch.cuda.synchronize(sync_device)
        timings[name] = time.perf_counter() - start
    best = min(timings, key=timings.get)
    return best, candidates[best]
