"""
Quick B200 inference micro-benchmark to toggle feature flags one by one.

Runs a prefill + decode loop for a fixed prompt length and reports tokens/sec.
Defaults keep all new features off. Each mode enables one more flag so you can
see the incremental benefit on Blackwell (B200). Every mode decodes through
Engine with torch.compile off, so compilation does not differ between rows.
"""

import argparse
import time
from typing import Any, Callable, Dict, Tuple

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine, KVCache

BENCHMARK_MODES = (
    "baseline",
    "flash_sdp",
    "flash_attn_varlen",
    "flash_attn_varlen_aligned_cache",
    "persistent",
)


def _time_cuda_region_seconds(fn: Callable[[], None]) -> float:
    # End-to-end region latency includes host submission and all CUDA streams.
    # Synchronizing after a current-stream end event does not repair that event
    # if another stream's work was never joined into its dependency chain.
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def configure_mode(mode: str, config) -> Dict[str, Any]:
    """
    Set torch SDP backends and model config for a benchmark mode.
    Returns a dict describing which flags were enabled.
    """
    enabled = {}
    config.enable_persistent_decode = False
    config.use_cuda_graphs = False
    config.kv_block_size = None
    config.kv_page_size = None
    # Explicitly set global SDP kernel preferences for reproducibility.
    if mode == "baseline":
        # Math-only SDPA control.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = False
        config.use_flash_attn_varlen = False
    elif mode == "flash_sdp":
        # Torch SDPA preferences on, external FlashAttention varlen off.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = True
        config.use_flash_attn_varlen = False
        enabled["use_flash_sdp"] = True
    elif mode == "flash_attn_varlen":
        # External FlashAttention 2 varlen callable from the pinned dependency.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = True
        config.use_flash_attn_varlen = True
        enabled["use_flash_attn_varlen"] = True
    elif mode == "flash_attn_varlen_aligned_cache":
        # The KV tensors remain contiguous. These values only round capacity.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = True
        config.use_flash_attn_varlen = True
        config.kv_block_size = config.kv_block_size or 32
        config.kv_page_size = config.kv_page_size or 1024
        enabled["use_flash_attn_varlen"] = True
        enabled["kv_block_size"] = config.kv_block_size
        enabled["kv_page_size"] = config.kv_page_size
        enabled["kv_cache_layout"] = "contiguous"
    elif mode == "persistent":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        config.use_flash_sdp = True
        config.use_flash_attn_varlen = True
        config.enable_persistent_decode = True
        enabled["enable_persistent_decode"] = True
        enabled["use_flash_attn_varlen"] = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return enabled


def build_sweep_engine(model, tokenizer) -> Engine:
    """Build the decode Engine for one mode with torch.compile held off."""
    return Engine(model, tokenizer, enable_batch_decode=False, allow_compile=False)


def require_isolated_decode(engine: Engine, config) -> Dict[str, Any]:
    """Fail unless decode ran uncompiled on the path the mode's flags request.

    Returns the Engine compile diagnostics for the result row.
    """
    expected_execution = "side_stream" if config.enable_persistent_decode else "eager"
    if engine.decode_execution_mode != expected_execution:
        raise RuntimeError(
            f"Requested {expected_execution} decode, observed {engine.decode_execution_mode}"
        )
    diagnostics = engine.get_compile_diagnostics()
    if diagnostics["status"] != "disabled":
        raise RuntimeError(
            "B200 flag sweep rows must run without torch.compile, "
            f"observed compile status {diagnostics['status']}"
        )
    return diagnostics


@torch.inference_mode()
def bench_once(
    model,
    batch_size: int,
    prompt_len: int,
    decode_len: int,
    device: torch.device,
    engine: Engine,
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
            _ = engine._execute_decode(step_ids, kv_cache)

    decode_seconds = _time_cuda_region_seconds(_run_decode)
    decode_tok_s = (batch_size * decode_len) / decode_seconds
    return prefill_tok_s, decode_tok_s


def run_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("B200 flag sweep requires a real CUDA device")
    device = torch.device("cuda")
    model, tokenizer, _ = load_model("sft", device=device, phase="eval")
    model.to(dtype=args.dtype, device=device)

    results = []
    for mode in BENCHMARK_MODES:
        enabled = configure_mode(mode, model.config)
        # Attention modules cache flags at construction; changing config alone
        # does not activate a different attention backend on a loaded model.
        for block in model.transformer.h:
            block.attn.use_flash_sdp = model.config.use_flash_sdp
            block.attn.use_flash_attn_varlen = model.config.use_flash_attn_varlen
            if model.config.use_flash_attn_varlen:
                block.attn._init_flash_attn_varlen()
                if (
                    not block.attn.use_flash_attn_varlen
                    or block.attn.flash_attn_varlen_fn is None
                    or block.attn.flash_attn_varlen_backend != "flash_attn_2_cuda"
                ):
                    raise RuntimeError(
                        f"{mode} requested the pinned FlashAttention 2 varlen backend, "
                        f"but it is unavailable: {block.attn.flash_attn_varlen_error}"
                    )
        if model.config.use_flash_attn_varlen:
            attention_backend = "flash_attn_2_cuda"
        elif mode == "baseline":
            attention_backend = "torch_math_sdpa"
        else:
            attention_backend = "torch_sdpa"
        engine = build_sweep_engine(model, tokenizer)
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
        compile_diagnostics = require_isolated_decode(engine, model.config)
        results.append(
            dict(
                mode=mode,
                enabled=enabled,
                attention_backend=attention_backend,
                decode_execution=engine.decode_execution_mode,
                compile_status=compile_diagnostics["status"],
                timing="synchronized_wall_seconds_including_host_submission",
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
            f"{res['mode']:>36} | backend: {res['attention_backend']} | "
            f"execution: {res['decode_execution']} | torch.compile: {res['compile_status']} | "
            f"flags: {flag_str:<35} | prefill {res['prefill_tok_s']:.1f} tok/s ({prefill_gain:+.1f}%) | "
            f"decode {res['decode_tok_s']:.1f} tok/s ({decode_gain:+.1f}%)"
        )


if __name__ == "__main__":
    main()
