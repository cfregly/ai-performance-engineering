"""MoE routing guardrails sweep tool (loss/throughput + overflow/Gini/entropy).

This is a chapter tool (not a comparable baseline/optimized benchmark).
Run via `python -m cli.aisp tools moe-validation -- --out artifacts/moe_validation.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from core.optimization.moe_inference import MoeInferenceConfig, SimpleMoEGPT, allocate_kv_cache  # noqa: E402


def compute_gini(counts: torch.Tensor) -> float:
    counts = counts.to(torch.float32)
    if counts.numel() == 0:
        return 0.0
    total = counts.sum()
    if total <= 0:
        return 0.0
    sorted_counts, _ = torch.sort(counts)
    n = counts.numel()
    index = torch.arange(1, n + 1, dtype=torch.float32, device=counts.device)
    gini = 1.0 + 1.0 / n - 2.0 * torch.sum((n + 1 - index) * sorted_counts) / (n * total)
    return float(gini)


class MoEStatsLogger:
    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self.reset()

    def reset(self) -> None:
        self.expert_counts = torch.zeros(self.num_experts, dtype=torch.long)
        self.overflow_tokens = 0
        self.total_tokens = 0
        self.entropy_sum = 0.0
        self.entropy_count = 0
        self._overflow_pending: Optional[torch.Tensor] = None
        self._entropy_pending: Optional[torch.Tensor] = None
        self._entropy_pending_count = 0

    def _accumulate_overflow(self, overflow_mask: torch.Tensor) -> None:
        overflow_delta = overflow_mask.detach().sum().to(dtype=torch.int64).reshape(())
        if (
            self._overflow_pending is not None
            and self._overflow_pending.device != overflow_delta.device
        ):
            self._materialize_pending_scalars()
        if self._overflow_pending is None:
            self._overflow_pending = overflow_delta.clone()
        else:
            self._overflow_pending.add_(overflow_delta)

    def _accumulate_entropy_tensor(self, entropy_val: torch.Tensor) -> None:
        entropy_delta = entropy_val.detach().to(dtype=torch.float32).reshape(())
        if (
            self._entropy_pending is not None
            and self._entropy_pending.device != entropy_delta.device
        ):
            self._materialize_pending_scalars()
        if self._entropy_pending is None:
            self._entropy_pending = entropy_delta.clone()
        else:
            self._entropy_pending.add_(entropy_delta)
        self._entropy_pending_count += 1

    def update(self, stats: Dict[str, torch.Tensor]) -> None:
        if not stats:
            return
        expert_indices = stats.get("expert_indices")
        if expert_indices is not None:
            flat = expert_indices.reshape(-1)
            valid = (flat >= 0) & (flat < self.num_experts)
            self.expert_counts += torch.bincount(
                flat[valid],
                minlength=self.num_experts,
            ).cpu()
            self.total_tokens += int(expert_indices.shape[0])

        overflow_mask = stats.get("overflow_mask")
        if overflow_mask is not None:
            self._accumulate_overflow(overflow_mask)

        entropy_val = stats.get("router_entropy")
        if entropy_val is not None:
            if torch.is_tensor(entropy_val):
                self._accumulate_entropy_tensor(entropy_val)
            else:
                self.entropy_sum += float(entropy_val)
                self.entropy_count += 1

    def _materialize_pending_scalars(self) -> None:
        if self._overflow_pending is not None:
            self.overflow_tokens += int(self._overflow_pending.detach().cpu())
            self._overflow_pending = None
        if self._entropy_pending is not None:
            self.entropy_sum += float(self._entropy_pending.detach().cpu())
            self.entropy_count += self._entropy_pending_count
            self._entropy_pending = None
            self._entropy_pending_count = 0

    def summarize(self) -> Dict[str, float]:
        self._materialize_pending_scalars()
        overflow_rate = self.overflow_tokens / self.total_tokens if self.total_tokens > 0 else 0.0
        gini = compute_gini(self.expert_counts)
        entropy = self.entropy_sum / self.entropy_count if self.entropy_count else 0.0
        return {
            "overflow_rate": float(overflow_rate),
            "gini": float(gini),
            "router_entropy": float(entropy),
        }


def _set_router_config(model: SimpleMoEGPT, top_k: int, capacity_factor: float) -> None:
    for block in model.layers:
        ff = getattr(block, "ff", None)
        if hasattr(ff, "top_k"):
            ff.top_k = top_k  # type: ignore[attr-defined]
        if hasattr(ff, "capacity_factor"):
            ff.capacity_factor = capacity_factor  # type: ignore[attr-defined]


class MoeValidationSweep:
    def __init__(
        self,
        config: MoeInferenceConfig,
        *,
        k_values: List[int],
        capacity_factors: List[float],
        eval_seeds: List[int],
        device: torch.device,
    ) -> None:
        self.config = config
        self.k_values = k_values
        self.capacity_factors = capacity_factors
        self.eval_seeds = eval_seeds
        self.device = device
        self.model: Optional[SimpleMoEGPT] = None
        self._next_token_values: Optional[torch.Tensor] = None
        self._next_token_buffer: Optional[torch.Tensor] = None
        self._loss_readback: Optional[torch.Tensor] = None
        self._kv_cache: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(42)
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats(self.device)
        self.model = SimpleMoEGPT(self.config, device=self.device).eval()
        self._next_token_values = torch.empty(
            (self.config.batch_size, 1),
            device=self.device,
            dtype=self.config.dtype_obj,
        )
        self._next_token_buffer = torch.empty(
            (self.config.batch_size, 1),
            device=self.device,
            dtype=torch.long,
        )
        self._loss_readback = torch.empty(2, device=self.device, dtype=torch.float32)
        total_tokens = self.config.context_window + self.config.decode_tokens
        self._kv_cache = allocate_kv_cache(
            self.config.batch_size,
            total_tokens,
            self.config.hidden_size,
            self.config.dtype_obj,
            self.device,
        )

    def _next_token_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits_last = logits if logits.dim() == 2 else logits[:, -1, :]
        shape = (logits_last.shape[0], 1)
        if (
            self._next_token_values is None
            or self._next_token_values.device != logits_last.device
            or self._next_token_values.dtype != logits_last.dtype
            or tuple(self._next_token_values.shape) != shape
        ):
            self._next_token_values = torch.empty(shape, device=logits_last.device, dtype=logits_last.dtype)
        if (
            self._next_token_buffer is None
            or self._next_token_buffer.device != logits_last.device
            or tuple(self._next_token_buffer.shape) != shape
        ):
            self._next_token_buffer = torch.empty(shape, device=logits_last.device, dtype=torch.long)
        torch.max(logits_last, dim=-1, keepdim=True, out=(self._next_token_values, self._next_token_buffer))
        return self._next_token_buffer

    def _make_batch(self, seed: int) -> Dict[str, torch.Tensor]:
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))
        cfg = self.config
        prompts = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, cfg.context_window),
            device=self.device,
            generator=generator,
        )
        total_tokens = cfg.context_window + cfg.decode_tokens
        labels = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch_size, total_tokens),
            device=self.device,
            generator=generator,
        )
        return {"prompts": prompts, "labels": labels}

    def _run_once(
        self,
        prompts: torch.Tensor,
        labels: torch.Tensor,
        top_k: int,
        capacity_factor: float,
    ) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("setup() must be called before running sweeps")
        _set_router_config(self.model, top_k=top_k, capacity_factor=capacity_factor)
        moe_logger = MoEStatsLogger(num_experts=self.config.num_experts)
        cfg = self.config
        if self._loss_readback is None or self._kv_cache is None:
            raise RuntimeError("setup() must initialize validation buffers")
        loss_readback = self._loss_readback
        kv_cache = self._kv_cache

        decode_loss_count = 0
        with torch.inference_mode():
            loss_readback.zero_()
            start = time.perf_counter()
            _, logits, router_stats = self.model.prefill(
                prompts,
                kv_cache=kv_cache,
                cache_start=0,
                output_router_stats=True,
            )
            token_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                labels[:, : cfg.context_window].reshape(-1),
            )
            for stats in router_stats:
                moe_logger.update(stats)

            seed_tokens = self._next_token_from_logits(logits[:, -1, :])
            loss_readback[0].copy_(token_loss.detach())
            for step in range(cfg.decode_tokens):
                _, decode_logits, decode_stats = self.model.decode(
                    seed_tokens,
                    kv_cache=kv_cache,
                    position=cfg.context_window + step,
                    output_router_stats=True,
                )
                step_loss = F.cross_entropy(
                    decode_logits.reshape(-1, cfg.vocab_size),
                    labels[:, cfg.context_window + step].reshape(-1),
                )
                loss_readback[1].add_(step_loss.detach())
                decode_loss_count += 1
                for stats in decode_stats:
                    moe_logger.update(stats)
                seed_tokens = self._next_token_from_logits(decode_logits[:, -1, :])

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_s = max(time.perf_counter() - start, 1e-6)

        summary = moe_logger.summarize()
        loss_values = loss_readback.detach().cpu()
        avg_decode_loss = (
            float(loss_values[1]) / max(decode_loss_count, 1) if decode_loss_count else 0.0
        )
        avg_loss = float(loss_values[0]) + avg_decode_loss
        record = {
            "top_k": float(top_k),
            "capacity_factor": float(capacity_factor),
            "loss": avg_loss,
            "tokens_per_sec": float(cfg.tokens_per_iteration) / elapsed_s,
            "overflow_rate": summary["overflow_rate"],
            "gini": summary["gini"],
            "router_entropy": summary["router_entropy"],
        }
        return record

    def run(self) -> List[Dict[str, float]]:
        if self.model is None:
            self.setup()
        records: List[Dict[str, float]] = []
        for seed in self.eval_seeds:
            batch = self._make_batch(seed)
            for k in self.k_values:
                for cf in self.capacity_factors:
                    record = self._run_once(batch["prompts"], batch["labels"], k, cf)
                    record["seed"] = float(seed)
                    records.append(record)
        return records


def _parse_csv(raw: str, cast) -> List:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def _summarize(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}

    best = records[0]
    best_tok_s = float(best.get("tokens_per_sec", 0.0))
    overflow_total = 0.0
    gini_total = 0.0
    loss_total = 0.0
    loss_squares = 0.0
    for record in records:
        tokens_per_sec = float(record.get("tokens_per_sec", 0.0))
        if tokens_per_sec > best_tok_s:
            best = record
            best_tok_s = tokens_per_sec
        overflow_total += float(record.get("overflow_rate", 0.0))
        gini_total += float(record.get("gini", 0.0))
        loss = float(record.get("loss", 0.0))
        loss_total += loss
        loss_squares += loss * loss

    count = len(records)
    overflow_mean = overflow_total / count
    gini_mean = gini_total / count
    loss_mean = loss_total / count
    loss_variance = max(0.0, (loss_squares / count) - loss_mean * loss_mean)
    loss_std = math.sqrt(loss_variance) if count > 1 else 0.0
    return {
        "best_tok_s": best_tok_s,
        "best_loss": float(best.get("loss", 0.0)),
        "avg_overflow": float(overflow_mean),
        "avg_gini": float(gini_mean),
        "loss_seed_std": float(loss_std),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE validation sweeps (routing guardrails).")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--ffn-size", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--moe-layers", type=int, default=3)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--moe-frequency", type=int, default=2, help="Every N layers is MoE.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-window", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--router-noise", type=float, default=0.0)
    parser.add_argument("--capacity-factor", type=float, default=0.0)
    parser.add_argument("--k-values", type=str, default="1,2", help="Comma-separated list.")
    parser.add_argument("--capacity-factors", type=str, default="1.0,1.25,1.5", help="Comma-separated list.")
    parser.add_argument("--seeds", type=str, default="3,13", help="Comma-separated list.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path (e.g., artifacts/moe_validation.json).",
    )
    args = parser.parse_args()

    cfg = MoeInferenceConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        ffn_size=args.ffn_size,
        num_layers=args.layers,
        num_moe_layers=args.moe_layers,
        num_experts=args.experts,
        top_k=args.top_k,
        moe_layer_frequency=args.moe_frequency,
        batch_size=args.batch_size,
        context_window=args.context_window,
        decode_tokens=args.decode_tokens,
        router_noise=args.router_noise,
        capacity_factor=None if args.capacity_factor == 0.0 else args.capacity_factor,
        dtype=torch.bfloat16,
    )

    k_vals = _parse_csv(args.k_values, int)
    cf_vals = _parse_csv(args.capacity_factors, float)
    seed_vals = _parse_csv(args.seeds, int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sweep = MoeValidationSweep(
        config=cfg,
        k_values=k_vals,
        capacity_factors=cf_vals,
        eval_seeds=seed_vals,
        device=device,
    )
    records = sweep.run()
    summary = _summarize(records)
    config_payload = asdict(cfg)
    for key, value in config_payload.items():
        if isinstance(value, torch.dtype):
            config_payload[key] = str(value)

    payload = {
        "config": config_payload,
        "device": str(device),
        "records": records,
        "summary": summary,
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
