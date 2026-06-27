"""Perplexity evaluation helper for the GPT benchmark model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from ch16.gpt_large_benchmark import GPTConfig, GPTModel, transformer_engine_available


def load_tokens(path: Path) -> List[int]:
    text = path.read_text().strip().split()
    try:
        return [int(tok) for tok in text]
    except ValueError as exc:
        raise SystemExit("Token file must contain space-separated integers.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Perplexity evaluator")
    parser.add_argument("tokens", type=Path, help="Path to tokenized dataset (space-separated integers)")
    parser.add_argument("--seq-len", type=int, default=512, help="Context length for evaluation")
    parser.add_argument("--stride", type=int, default=256, help="Stride between evaluation windows")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--fp8-mode",
        choices=["none", "weight-only", "transformer-engine", "auto"],
        default="none",
        help="Optional FP8 conversion mode for the GPT model.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state-dict", type=Path, help="Optional checkpoint to load (torch.load)")
    parser.add_argument("--output-json", type=Path, help="Optional path to write metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = load_tokens(args.tokens)
    if len(tokens) < args.seq_len + 1:
        raise SystemExit("Token file too short for requested sequence length.")

    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    resolved_fp8_mode = args.fp8_mode
    if resolved_fp8_mode == "auto":
        resolved_fp8_mode = "transformer-engine" if transformer_engine_available() else "weight-only"

    config = GPTConfig(
        vocab_size=max(tokens) + 1,
        n_layers=8,
        n_heads=16,
        d_model=1024,
        d_ff=4096,
        max_seq_len=args.seq_len,
    )

    model = GPTModel(config, devices=[device], dtype=dtype, fp8_mode=resolved_fp8_mode).to(device)
    model.eval()

    if args.state_dict:
        state = torch.load(args.state_dict, map_location="cpu")
        model.load_state_dict(state, strict=False)

    token_ids_i64 = torch.tensor(tokens, device=device, dtype=torch.int64)
    token_ids_i32 = token_ids_i64.to(torch.int32)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(tokens) - args.seq_len - 1, args.stride):
            input_ids = token_ids_i32.narrow(0, start, args.seq_len).unsqueeze(0)
            target_ids = token_ids_i64.narrow(0, start + 1, args.seq_len).unsqueeze(0)
            logits = model(input_ids)
            # Match sizes: logits and targets should have same length
            min_len = min(logits.size(1), target_ids.size(1))
            loss = F.cross_entropy(
                logits[:, :min_len, :].reshape(-1, config.vocab_size),
                target_ids[:, :min_len].reshape(-1),
                reduction="sum",
            )
            total_loss += loss.detach()
            total_tokens += min_len

    avg_loss = float(total_loss / total_tokens)
    perplexity = math.exp(avg_loss)
    metrics = {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "tokens_evaluated": total_tokens,
        "sequence_length": args.seq_len,
        "stride": args.stride,
        "dtype": args.dtype,
        "fp8_mode_requested": args.fp8_mode,
        "fp8_mode_effective": resolved_fp8_mode,
    }

    print("=== Perplexity Evaluation ===")
    print(f"Tokens evaluated: {total_tokens}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.3f}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
