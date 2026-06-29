"""
Quick B200 inference micro-benchmark to toggle feature flags one by one.

Runs a prefill + decode loop for a fixed prompt length and reports tokens/sec.
Defaults keep all new features off; each mode enables one more flag so you can
see the incremental benefit on Blackwell (B200).
"""

import argparse
from typing import Callable, Dict, Tuple

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine, KVCache


def _time_cuda_region_seconds(fn: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    start.record(current_stream)
    fn()
    end.record(current_stream)
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def configure_mode(mode: str, config) -> Dict[str, bool]:
    """
    Set torch SDP backends and model config for a benchmark mode.
    Returns a dict describing which flags were enabled.
    """
    enabled = {}
    # Explicitly set global SDP kernel preferences for reproducibility.
    if mode == "baseline":
        # Math-only fallback, FA3 off.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = False
        config.use_flash3 = False
    elif mode == "flash_sdp":
        # Torch SDP (FlashAttention-2/torch flash) on, FA3 off.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = True
        config.use_flash3 = False
        enabled["use_flash_sdp"] = True
    elif mode == "flash3":
        # FlashAttention-3 varlen path on (TMA/TMEM on B200).
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = True
        config.use_flash3 = True
        enabled["use_flash3"] = True
    elif mode == "flash3_block":
        # FA3 plus block/paged KV cache layout (staging hint for TMA).
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = True
        config.use_flash3 = True
        config.kv_block_size = config.kv_block_size or 32
        config.kv_page_size = config.kv_page_size or 1024
        enabled["use_flash3"] = True
        enabled["kv_block_size"] = config.kv_block_size
    elif mode == "persistent":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = True
        config.use_flash3 = True
        config.enable_persistent_decode = True
        enabled["enable_persistent_decode"] = True
        enabled["use_flash3"] = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return enabled


@torch.inference_mode()
def bench_once(
    model,
    batch_size: int,
    prompt_len: int,
    decode_len: int,
    device: torch.device,
    engine: Engine | None = None,
) -> Tuple[float, float]:
    """
    Run one prefill + decode sweep and return (prefill_tok_s, decode_tok_s).
    """
    cfg = model.config
    head_dim = cfg.n_embd // cfg.n_head
    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=cfg.n_kv_head,
        seq_len=prompt_len + decode_len + 8,  # small headroom
        head_dim=head_dim,
        num_layers=cfg.n_layer,
        block_size=getattr(cfg, "kv_block_size", None),
        page_size=getattr(cfg, "kv_page_size", None),
    )
    prompt = torch.randint(0, cfg.vocab_size, (batch_size, prompt_len), device=device, dtype=torch.long)

    # Prefill
    prefill_seconds = _time_cuda_region_seconds(lambda: model(prompt, kv_cache=kv_cache))
    prefill_tok_s = (batch_size * prompt_len) / prefill_seconds

    # Decode steady-state (T=1)
    decode_tokens = torch.randint(0, cfg.vocab_size, (batch_size, decode_len), device=device, dtype=torch.long)
    decode_token_steps = tuple(
        decode_tokens[:, t:t + 1]
        for t in range(decode_len)
    )

    def _run_decode() -> None:
        for step_ids in decode_token_steps:
            if engine is None:
                _ = model(step_ids, kv_cache=kv_cache)
            else:
                _ = engine._execute_decode(step_ids, kv_cache)

    decode_seconds = _time_cuda_region_seconds(_run_decode)
    decode_tok_s = (batch_size * decode_len) / decode_seconds
    return prefill_tok_s, decode_tok_s


def run_benchmark(args):
    device = torch.device("cuda")
    model, tokenizer, _ = load_model("sft", device=device, phase="eval")
    model.to(dtype=args.dtype, device=device)

    modes = ["baseline", "flash_sdp", "flash3", "flash3_block", "persistent"]
    results = []
    for mode in modes:
        enabled = configure_mode(mode, model.config)
        engine = None
        if getattr(model.config, "enable_persistent_decode", False) or getattr(model.config, "use_cuda_graphs", False):
            engine = Engine(model, tokenizer, enable_batch_decode=False)
        # Warmup
        _ = bench_once(model, args.batch_size, args.prompt_len, args.decode_len, device, engine=engine)
        prefill_total = 0.0
        decode_total = 0.0
        for _ in range(args.iters):
            prefill_tok_s, decode_tok_s = bench_once(
                model,
                args.batch_size,
                args.prompt_len,
                args.decode_len,
                device,
                engine=engine,
            )
            prefill_total += prefill_tok_s
            decode_total += decode_tok_s
        results.append(
            dict(
                mode=mode,
                enabled=enabled,
                prefill_tok_s=prefill_total / args.iters,
                decode_tok_s=decode_total / args.iters,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=3, help="timed repetitions per mode")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    args.dtype = dtype

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    results = run_benchmark(args)
    print("=== B200 flag sweep ===")
    print(f"batch={args.batch_size}, prompt_len={args.prompt_len}, decode_len={args.decode_len}, dtype={args.dtype}")
    baseline = results[0]
    base_prefill, base_decode = baseline["prefill_tok_s"], baseline["decode_tok_s"]
    for res in results:
        flag_str = ", ".join([f"{k}={v}" for k, v in res["enabled"].items()]) or "none"
        prefill_gain = (res["prefill_tok_s"] / base_prefill - 1.0) * 100 if base_prefill else 0.0
        decode_gain = (res["decode_tok_s"] / base_decode - 1.0) * 100 if base_decode else 0.0
        print(
            f"{res['mode']:>12} | flags: {flag_str:<35} | prefill {res['prefill_tok_s']:.1f} tok/s ({prefill_gain:+.1f}%) | "
            f"decode {res['decode_tok_s']:.1f} tok/s ({decode_gain:+.1f}%)"
        )


if __name__ == "__main__":
    main()
