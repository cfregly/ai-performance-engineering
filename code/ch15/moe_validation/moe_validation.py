"""MoE routing guardrails sweep tool (loss/throughput + overflow/Gini/entropy).

This is a chapter tool (not a comparable baseline/optimized benchmark).
Run via `python -m cli.aisp tools moe-validation -- --out artifacts/moe_validation.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
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
        self.entropy: List[float] = []

    def update(self, stats: Dict[str, torch.Tensor]) -> None:
        if not stats:
            return
        expert_indices = stats.get("expert_indices")
        if expert_indices is not None:
            flat = expert_indices.reshape(-1)
            valid = (flat >= 0) & (flat < self.num_experts)
            if valid.any():
                self.expert_counts += torch.bincount(
                    flat[valid],
                    minlength=self.num_experts,
                ).cpu()
            self.total_tokens += int(expert_indices.shape[0])

        overflow_mask = stats.get("overflow_mask")
        if overflow_mask is not None:
            self.overflow_tokens += int(overflow_mask.sum().item())

        entropy_val = stats.get("router_entropy")
        if entropy_val is not None:
            self.entropy.append(float(entropy_val))

    def summarize(self) -> Dict[str, float]:
        overflow_rate = self.overflow_tokens / self.total_tokens if self.total_tokens > 0 else 0.0
        gini = compute_gini(self.expert_counts)
        entropy = statistics.mean(self.entropy) if self.entropy else 0.0
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

    def setup(self) -> None:
        torch.manual_seed(42)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(42)
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats(self.device)
        self.model = SimpleMoEGPT(self.config, device=self.device).eval()

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
        total_tokens = cfg.context_window + cfg.decode_tokens
        kv_cache = allocate_kv_cache(
            cfg.batch_size,
            total_tokens,
            cfg.hidden_size,
            cfg.dtype_obj,
            self.device,
        )

        with torch.no_grad():
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

            seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            decode_losses: List[torch.Tensor] = []
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
                decode_losses.append(step_loss)
                for stats in decode_stats:
                    moe_logger.update(stats)
                seed_tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_s = max(time.perf_counter() - start, 1e-6)

        summary = moe_logger.summarize()
        loss_values = torch.stack((token_loss.detach(), *(loss.detach() for loss in decode_losses))).cpu().tolist()
        avg_decode_loss = (
            sum(loss_values[1:]) / max(len(decode_losses), 1) if decode_losses else 0.0
        )
        avg_loss = float(loss_values[0] + avg_decode_loss)
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
    best = max(records, key=lambda r: float(r.get("tokens_per_sec", 0.0)))
    overflow_mean = statistics.mean(float(r.get("overflow_rate", 0.0)) for r in records)
    gini_mean = statistics.mean(float(r.get("gini", 0.0)) for r in records)
    loss_std = statistics.pstdev(float(r.get("loss", 0.0)) for r in records) if len(records) > 1 else 0.0
    return {
        "best_tok_s": float(best.get("tokens_per_sec", 0.0)),
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

    payload = {
        "config": asdict(cfg),
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
