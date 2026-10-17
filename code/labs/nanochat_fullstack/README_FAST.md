# Fast-Path Flags & Dynamic Batching

The optional B200 paths include the pinned FlashAttention 2 varlen callable,
contiguous KV cache capacity alignment, and side-stream buffer reuse. Each path
is flag-gated so benchmark rows can identify the backend and execution mode.

## New flags
- `use_padded_attention` (GPTConfig): enables key padding masks + per-row rotary positions. Default `False`.
- `enable_batch_decode` (Engine / `labs/nanochat_fullstack/scripts/chat_web.py`): turns on padded attention and the batched decode path. Default `False`.
- `--enable-dynamic-batching` (chat_web): queues requests and forms batches up to `--batch-size` or `--batch-timeout-ms`. Defaults off.
- `use_flash_attn_varlen` (GPTConfig): calls `flash_attn.flash_attn_interface.flash_attn_varlen_func` only when that interface identifies its extension as `flash_attn_2_cuda`. The ordinary model path disables this option when the dependency is unavailable. The B200 sweep fails instead of timing a fallback under this row. Default `True`.
- `kv_block_size` / `kv_page_size` (GPTConfig): optional alignment boundaries for the capacity and growth of one contiguous KV tensor. They do not create a page table or change the attention kernel layout. Defaults `None`.
- `enable_persistent_decode` / `use_cuda_graphs` (GPTConfig): side-stream decode with reusable buffers, plus optional CUDA Graph capture on a shared stream. Graphs recapture when KV cache grows. Defaults `False`.
- `use_te_weight_only` / `te_weight_dtype` (GPTConfig): Transformer Engine weight-only linears for q/k/v/proj/lm_head. If TE is missing, this flag fails fast. Default `False` (enable to measure gains).
- `use_clustered_attention_kernel` (GPTConfig): experimental hook to call a custom clustered attention kernel. Supply via `clustered_attention_impl="module:function"` or `stubs.register_clustered_attention_kernel(fn)`. Disabled by default; without an override it raises unless `allow_kernel_stub_fallback=True` is set to use the reference path.
- `use_persistent_decode_kernel` (GPTConfig): experimental hook to call a custom resident decode kernel. Supply via `persistent_decode_impl="module:function"` or `stubs.register_persistent_decode_kernel(fn)`. Both hooks fail fast without an override unless `allow_kernel_stub_fallback=True` is set.

## Usage
- Programmatic: instantiate `Engine(model, tokenizer, enable_batch_decode=True)` to use `generate_batched(prompts, max_tokens=..., temperature=..., top_k=...)` on padded multi-prompt batches. Per-request temperatures/top-k/max-tokens are accepted as scalars or lists.
- Web server: 
  ```bash
  PYTHONPATH=labs/nanochat_fullstack \
  python -m labs.nanochat_fullstack.scripts.chat_web \
    --enable-batch-decode \
    --enable-dynamic-batching \
    --batch-size 4 \
    --batch-timeout-ms 10 \
    --temperature 0.8 --top-k 50 --max-tokens 512
  ```
  With all flags off, chat_web runs the original single-request decode path.

### Custom kernel hooks
- `use_clustered_attention_kernel` / `use_persistent_decode_kernel` dispatch through `nanochat.kernels.stubs`.
- Supply a custom kernel via `GPTConfig.clustered_attention_impl` / `GPTConfig.persistent_decode_impl` (`module:function`) or register one in Python at startup.
- Set `GPTConfig.allow_kernel_stub_fallback=True` to route those flags to the reference Python/Torch implementations instead of failing fast. Intended for debugging, not benchmarking.

## Caveats
- `use_cuda_graphs` recaptures on KV-cache growth; set `max_tokens` with headroom to avoid recapture churn.
- `use_te_weight_only` requires `transformer_engine`; enable once installed to measure the gain.

## Benchmarks
- Target: prefill + decode tokens/sec with flags enabled one by one. Use the built-in sweep to get % gains vs baseline.
- How to run: `python -m labs.nanochat_fullstack.scripts.bench_b200_flags --batch-size 2 --prompt-len 256 --decode-len 64 --iters 2` (see `labs/nanochat_fullstack/scripts/bench_b200_flags.py`). It reports tokens/sec and % change vs baseline.
- Web server: `python -m labs.nanochat_fullstack.scripts.chat_web --enable-batch-decode --enable-dynamic-batching --batch-size 4 --batch-timeout-ms 10 ...` and measure end-to-end throughput; toggle `--enable-batch-decode` off to get the baseline.
- Quick local run (NVIDIA B200, `karpathy/nanochat-d32` step 650, bf16 autocast, 8 mixed-length prompts, `max_new_tokens=64`, `temperature=0.0`, `top_k=50`):
  - Single-path (`enable_batch_decode` off): 512 new tokens in 8.25s → ~62 tok/s.
  - Batched decode (`enable_batch_decode` on): 512 new tokens in 2.57s → ~199 tok/s (**~3.2x / +220% decode throughput** vs single-path).
  - Reuse-ids buffer: no meaningful uplift in this micro-run (~199 tok/s without reuse vs ~175 tok/s with reuse).
- Benchmark (real checkpoint, GPU): NVIDIA B200, `karpathy/nanochat-d32` step 650, 8 mixed-length prompts (Matrix summary, EV pros/cons, code/email, etc.), `max_new_tokens=64`, `temperature=0.0`, `top_k=50`, 5 runs:
  - Baseline (flags off, single-path): p50 6.55s, p90 6.55s, mean 6.61s; ~77.4 tok/s.
  - + `enable_batch_decode` (padded attention + batched decode): p50 2.61s, p90 2.61s, mean 2.63s; ~194.5 tok/s; **+151% throughput** and **~2.5x lower p50/p90 latency** vs baseline.
  - Dynamic batching in `chat_web` will use the batched path once batches form (`--enable-dynamic-batching --batch-size ... --batch-timeout-ms ...`); expect similar gains when your request mix allows batching.
  - Replace with your own numbers; record the % deltas per flag.

## Sanity checks
- Engine KV-cache resize test: `PYTHONPATH=labs/nanochat_fullstack python -m pytest tests/test_engine.py -q`.
- Manual check: default (unpadded) forward with loss and padded-attention forward with an attention mask both execute on small random inputs.
- Single-request inference remains the default path unless `--enable-batch-decode` is provided.
- Training path: unchanged. A quick micro-bench on the d32 checkpoint (B200, bf16 autocast, batch=1, seq=512, 5 train steps with backward) runs at ~3.4K tok/s. No training flags were added; focus here is inference.
- Training throughput snapshot (d32, B200, bf16 autocast, seq=512, 2 steps per run): flash SDP on, torch.compile off, scaling batch size pushes throughput; fp32 logits slow the fastest configs. Measured tok/s:
  - b1 flash + fp32 logits: ~2.0K tok/s; flash + bf16 logits: ~6.6K tok/s.
  - b2 flash + fp32 logits: ~8.3K tok/s; flash + bf16 logits: ~9.6K tok/s.
  - b4 flash + fp32 logits: ~14.5K tok/s; flash + bf16 logits: ~18.3K tok/s.
  - b8 flash + fp32 logits: ~22.2K tok/s; flash + bf16 logits: ~30.9K tok/s.
  - b12 flash + bf16 logits: ~18.2K tok/s (throughput flattened vs b8); b16 flash + bf16 logits: ~31.6K tok/s. Above b8, gains come from larger batch; flash still enabled.
  - Longer seq (1024) with flash + bf16 logits: b1 ~4.0K tok/s, b2 ~15.7K tok/s, b4 ~24.6K tok/s; no OOM at b4. Grad accumulation can be used to simulate larger batches if memory caps out.
  - Heavier configs (flash + bf16 logits): b32 x 512 ~22.3K tok/s; b8 x 1536 ~32.2K tok/s. Past b8, returns flatten; pushing seq length can help if memory allows.
  - torch.compile (as of this nightly) hurt training throughput badly here; keep it off unless profiled otherwise. Increase batch size (with grad accumulation if needed) and use flash SDP to maximize training tok/s; disabling fp32 logits boosts raw throughput further at the cost of logits precision.
- Training toggles micro-bench (random data, d32 checkpoint, B200, bf16 autocast, bs=1, seq=256, 3 timed steps):
  - Flash off / fp32 logits off: ~822 tok/s.
  - Flash off / fp32 logits on: ~937 tok/s.
  - Flash on / fp32 logits off: ~972 tok/s.
  - Flash on / fp32 logits on: ~889 tok/s.
  - torch.compile + flash on / fp32 logits off: ~437 tok/s (timed phase only; ~43s one-time compile overhead made it a net loss). On this nightly build, compile regressed performance—keep it off for training.

## B200 flag sweep (prefill + steady-state decode)
`python -m labs.nanochat_fullstack.scripts.bench_b200_flags --batch-size 2 --prompt-len 256 --decode-len 64 --iters 2`

- Device/checkpoint: NVIDIA B200, `chatsft_checkpoints/d32`, bf16.
- Rows are `baseline` (math SDPA), `flash_sdp` (PyTorch SDPA preferences), `flash_attn_varlen` (the pinned FA2 extension), `flash_attn_varlen_aligned_cache` (the same FA2 callable with contiguous capacity alignment), and `persistent` (the FA2 callable with side-stream buffer reuse).
- Output records the backend, execution mode, tokens/sec, and delta versus baseline. The aligned-cache row does not claim paged attention.

## Decode Optimization Results (B200)

Measured via `labs/decode_optimization/` (batch=8, prompt=256, decode=64, hidden=1024):

| Optimization | Speedup | Tokens/sec | What it does |
|--------------|---------|------------|--------------|
| **CUDA Graphs (full)** | **4.39x** | 704K | Prefill + decode captured in graphs |
| **CUDA Graphs (decode)** | **4.18x** | 690K | Decode-only graph capture |
| **torch.compile** | **3.25x** | 317K | Kernel fusion via Inductor/Triton |
| Pinned Memory | 1.43x | 170K | Async H2D with pinned host buffers |
| Streams | ~1.0x | 163K | Overlapped copy/compute (no gain here) |

The CUDA Graph measurements in this separate table do not establish a combined
gain with the external FlashAttention varlen path.

Run the microbenchmarks:
```bash
python -m cli.aisp bench run --targets labs/decode_optimization --profile none
```
