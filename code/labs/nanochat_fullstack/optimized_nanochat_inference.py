#!/usr/bin/env python3
"""Optimized: replay the complete fixed-shape NanoChat request in a CUDA graph."""

from __future__ import annotations

import time

import torch

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.nanochat_fullstack.nanochat.engine import KVCache
from labs.nanochat_fullstack.nanochat.gpt import GPT, GPTConfig


class OptimizedNanochatInferenceBenchmark(VerificationPayloadMixin, BaseBenchmark):
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

        self.model: GPT | None = None
        self._request_graph: torch.cuda.CUDAGraph | None = None
        self._graph_output: torch.Tensor | None = None
        self.capture_warmup_ms = 0.0
        self.capture_ms = 0.0
        self.kv_cache: KVCache | None = None
        self.prompt: torch.Tensor | None = None
        self.decode_tokens: torch.Tensor | None = None
        self.decode_token_steps: tuple[torch.Tensor, ...] = ()
        self.output: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None
        self._payload_parameter_count = 0

        self.register_workload_metadata(
            tokens_per_iteration=float(self.batch_size * (self.prompt_len + self.decode_len)),
            requests_per_iteration=1.0,
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: nanochat inference benchmark requires CUDA")

        cfg = GPTConfig(
            sequence_len=1024,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            n_embd=self.n_embd,
            # Match the baseline attention kernels and cache layout exactly.
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
        # Match the baseline's non-degenerate synthetic inference weights.
        model.apply(model._init_weights)
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

        # Materialize the lazy KV cache and all kernel workspaces on a side stream.
        # Setup is excluded from steady-state replay timing; retain its costs.
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for _ in range(3):
                self.kv_cache.reset()
                self._run_request()
        capture_stream.synchronize()
        self.capture_warmup_ms = (time.perf_counter() - started) * 1000.0

        # Capture unrolls the host cache positions for the entire request. Replay
        # overwrites the prompt and every decode position in order, so it neither
        # reads stale KV entries nor depends on resetting the Python position.
        self.kv_cache.reset()
        self._request_graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        with torch.inference_mode(), torch.cuda.graph(
            self._request_graph, stream=capture_stream
        ):
            self._graph_output = self._run_request()
        capture_stream.synchronize()
        self.capture_ms = (time.perf_counter() - started) * 1000.0
        if self.kv_cache.get_pos() != self.prompt_len + self.decode_len:
            raise RuntimeError("CUDA graph did not capture the complete request")

    def _run_request(self) -> torch.Tensor:
        if (
            self.model is None
            or self.kv_cache is None
            or self.prompt is None
            or self.decode_tokens is None
            or not self.decode_token_steps
        ):
            raise RuntimeError("Request tensors must be initialized before capture")

        self.model(self.prompt, kv_cache=self.kv_cache)
        logits = None
        for step_ids in self.decode_token_steps:
            logits = self.model(step_ids, kv_cache=self.kv_cache)
        if logits is None:
            raise RuntimeError("decode loop did not execute")
        return logits

    def benchmark_fn(self) -> None:
        if self._request_graph is None or self._graph_output is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self._request_graph.replay()
        self.output = self._graph_output

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

    def validate_result(self) -> str | None:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        if not bool(torch.isfinite(self.output).all()):
            return "benchmark_fn() produced non-finite logits"
        if not bool(torch.count_nonzero(self.output)):
            return "benchmark_fn() produced degenerate all-zero logits"
        return None

    def teardown(self) -> None:
        if self._request_graph is not None:
            torch.cuda.synchronize(self.device)
        self._request_graph = None
        self._graph_output = None
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

    def get_custom_metrics(self) -> dict[str, float]:
        return {
            "setup.graph_warmup_ms": self.capture_warmup_ms,
            "setup.graph_capture_ms": self.capture_ms,
        }


def get_benchmark() -> BaseBenchmark:
    return OptimizedNanochatInferenceBenchmark()
