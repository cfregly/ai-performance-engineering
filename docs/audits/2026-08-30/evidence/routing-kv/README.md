# Router telemetry and KV transfer source batch

The CPU integration gate passed 17 tests with 9 explicit runtime skips. The new two-rank Gloo test verifies 36 full-tensor handoffs, including strided multi-batch prefixes, warm/active sources, and both directions. The parser checks use cumulative request-output payload fixtures; they do not execute vLLM.

W1-078 now orders buffer reuse and warmup across actual CUDA streams. Its changing-input full-output CUDA gate was not run. The original identical-input benchmark cannot demonstrate visible corruption from the race.

All three findings remain awaiting their actual CUDA/NCCL/vLLM runtime gates. See `receipt.json` for exact source hashes, failed and successful attempts, external API contracts, and limitations. No GPU workload was launched, no model was downloaded, and no measured speedup is claimed.
