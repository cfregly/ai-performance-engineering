#!/usr/bin/env python3
"""Baseline: NanoChat inference loop with conservative attention backend.

This benchmark is a harness-comparable pair with optimized_nanochat_inference.py.
It measures a fixed prefill + decode workload using the NanoChat GPT model.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

from labs.nanochat_fullstack.nanochat.engine import KVCache
from labs.nanochat_fullstack.nanochat.gpt import GPT, GPTConfig


class BaselineNanochatInferenceBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = False

    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 4
        self.prompt_len = 512
        self.decode_len = 64
        self.vocab_size = 10_000
        self.n_layer = 4
        self.n_head = 8
        self.n_kv_head = 8
        self.n_embd = 512

        self.model: Optional[GPT] = None
        self.kv_cache: Optional[KVCache] = None
        self.prompt: Optional[torch.Tensor] = None
        self.decode_tokens: Optional[torch.Tensor] = None
        self.decode_token_steps: tuple[torch.Tensor, ...] = ()
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

        self.register_workload_metadata(
            tokens_per_iteration=float(self.batch_size * (self.prompt_len + self.decode_len)),
            requests_per_iteration=1.0,
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: nanochat inference benchmark requires CUDA")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        cfg = GPTConfig(
            sequence_len=1024,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            n_embd=self.n_embd,
            use_flash_sdp=False,
            use_flash3=False,
            use_cta_clustering=False,
            kv_block_size=None,
            kv_page_size=None,
        )

        with torch.device("meta"):
            model = GPT(cfg)
        model.to_empty(device=self.device)
        model.init_weights()
        model = model.to(dtype=torch.bfloat16)
        model.eval()

        self.model = model
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())

        self.prompt = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.prompt_len),
            device=self.device,
            dtype=torch.long,
        )
        self.decode_tokens = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.decode_len),
            device=self.device,
            dtype=torch.long,
        )
        self.decode_token_steps = tuple(
            self.decode_tokens[:, t : t + 1]
            for t in range(self.decode_len)
        )

        head_dim = cfg.n_embd // cfg.n_head
        self.kv_cache = KVCache(
            batch_size=self.batch_size,
            num_heads=cfg.n_kv_head,
            seq_len=self.prompt_len + self.decode_len + 16,
            head_dim=head_dim,
            num_layers=cfg.n_layer,
            block_size=cfg.kv_block_size,
            page_size=cfg.kv_page_size,
        )
        self._verify_output_buffer = torch.empty(
            (self.batch_size, 1, self.vocab_size),
            device=self.device,
            dtype=torch.float32,
        )

    def benchmark_fn(self) -> None:
        if (
            self.model is None
            or self.kv_cache is None
            or self.prompt is None
            or self.decode_tokens is None
            or not self.decode_token_steps
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        self.kv_cache.reset()
        with torch.inference_mode():
            self.model(self.prompt, kv_cache=self.kv_cache)
            logits = None
            for step_ids in self.decode_token_steps:
                logits = self.model(step_ids, kv_cache=self.kv_cache)
            if logits is None:
                raise RuntimeError("decode loop did not execute")
            self.output = logits

    def capture_verification_payload(self) -> None:
        if (
            self.prompt is None
            or self.decode_tokens is None
            or self.output is None
            or self.model is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)

        self._set_verification_payload(
            inputs={"prompt": self.prompt, "decode_tokens": self.decode_tokens},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags={"fp16": False, "bf16": True, "fp8": False, "tf32": False},
            output_tolerance=(0.05, 0.2),
        )

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=self.batch_size,
            dtype="int64",
            prompt_len=self.prompt_len,
            decode_len=self.decode_len,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            n_embd=self.n_embd,
            precision_flags=PrecisionFlags(bf16=True, tf32=False),
        ).to_dict()

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None

    def teardown(self) -> None:
        self.model = None
        self.kv_cache = None
        self.prompt = None
        self.decode_tokens = None
        self.decode_token_steps = ()
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)


def get_benchmark() -> BaseBenchmark:
    return BaselineNanochatInferenceBenchmark()
