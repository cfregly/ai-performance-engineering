"""Shared EOS early-exit decode-loop benchmark for Chapter 18."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


@dataclass
class EosEarlyExitConfig:
    batch_size: int = 64
    prompt_tokens: int = 32
    decode_tokens: int = 128
    force_eos_after_tokens: int = 16
    hidden_size: int = 256
    vocab_size: int = 4096
    eos_token_id: int = 2
    stop_on_all_done: bool = False
    iterations: int = 20
    warmup: int = 10
    label: str = "eos_early_exit"


class EosEarlyExitBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Toy decode loop that isolates tail work skipped by EOS early exit."""

    def __init__(self, cfg: EosEarlyExitConfig):
        super().__init__()
        if cfg.force_eos_after_tokens < 1:
            raise ValueError("force_eos_after_tokens must be >= 1")
        if cfg.force_eos_after_tokens > cfg.decode_tokens:
            raise ValueError("force_eos_after_tokens must be <= decode_tokens")
        self.cfg = cfg
        self.dtype = torch.float16
        self.embedding_weight: Optional[torch.Tensor] = None
        self.transition_weight: Optional[torch.Tensor] = None
        self.lm_head_weight: Optional[torch.Tensor] = None
        self._lm_head_weight_t: Optional[torch.Tensor] = None
        self.prompt_ids: Optional[torch.Tensor] = None
        self.state_buffer: Optional[torch.Tensor] = None
        self.next_state_buffer: Optional[torch.Tensor] = None
        self.token_hidden_buffer: Optional[torch.Tensor] = None
        self.combined_buffer: Optional[torch.Tensor] = None
        self.logits_buffer: Optional[torch.Tensor] = None
        self.max_values_buffer: Optional[torch.Tensor] = None
        self.next_token_buffer: Optional[torch.Tensor] = None
        self.done_mask_buffer: Optional[torch.Tensor] = None
        self.eos_compare_buffer: Optional[torch.Tensor] = None
        self.generated_tokens: Optional[torch.Tensor] = None
        self.config_tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._completion_checks = 0
        self._decoded_steps = 0
        self._full_decode_range = range(cfg.decode_tokens)
        self._early_exit_range = range(cfg.force_eos_after_tokens)
        self._decode_token_divisor = float(max(cfg.decode_tokens, 1))
        self._custom_metrics: Dict[str, float] = {}
        self.parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=float(cfg.batch_size),
            tokens_per_iteration=float(cfg.batch_size * (cfg.prompt_tokens + cfg.decode_tokens)),
        )

    def setup(self) -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(18018)
        cfg = self.cfg
        self.embedding_weight = torch.randn(
            cfg.vocab_size,
            cfg.hidden_size,
            generator=g,
            dtype=self.dtype,
            device="cpu",
        ).to(self.device)
        self.transition_weight = torch.randn(
            cfg.hidden_size,
            cfg.hidden_size,
            generator=g,
            dtype=self.dtype,
            device="cpu",
        ).to(self.device)
        self.transition_weight.mul_(1.0 / max(cfg.hidden_size, 1))
        self.lm_head_weight = torch.randn(
            cfg.vocab_size,
            cfg.hidden_size,
            generator=g,
            dtype=self.dtype,
            device="cpu",
        ).to(self.device)
        self._lm_head_weight_t = self.lm_head_weight.t()
        self.prompt_ids = torch.randint(
            low=3,
            high=cfg.vocab_size,
            size=(cfg.batch_size, cfg.prompt_tokens),
            generator=g,
            dtype=torch.long,
            device="cpu",
        ).to(self.device)
        self.state_buffer = torch.empty(
            cfg.batch_size,
            cfg.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.next_state_buffer = torch.empty_like(self.state_buffer)
        self.token_hidden_buffer = torch.empty_like(self.state_buffer)
        self.combined_buffer = torch.empty_like(self.state_buffer)
        self.logits_buffer = torch.empty(
            cfg.batch_size,
            cfg.vocab_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.max_values_buffer = torch.empty(cfg.batch_size, device=self.device, dtype=self.dtype)
        self.next_token_buffer = torch.empty(cfg.batch_size, device=self.device, dtype=torch.long)
        self.done_mask_buffer = torch.empty(cfg.batch_size, device=self.device, dtype=torch.bool)
        self.eos_compare_buffer = torch.empty(cfg.batch_size, device=self.device, dtype=torch.bool)
        self.generated_tokens = torch.empty(
            cfg.batch_size,
            cfg.decode_tokens,
            device=self.device,
            dtype=torch.long,
        )
        self.config_tensor = torch.tensor(
            [
                cfg.batch_size,
                cfg.prompt_tokens,
                cfg.decode_tokens,
                cfg.force_eos_after_tokens,
                cfg.hidden_size,
                cfg.vocab_size,
            ],
            device="cpu",
            dtype=torch.int64,
        )
        self.parameter_count = int(
            self.embedding_weight.numel()
            + self.transition_weight.numel()
            + self.lm_head_weight.numel()
        )
        self._refresh_static_metrics()

    def _refresh_static_metrics(self) -> None:
        cfg = self.cfg
        self._custom_metrics = {
            "eos_early_exit.max_decode_steps": float(cfg.decode_tokens),
            "eos_early_exit.force_eos_after_tokens": float(cfg.force_eos_after_tokens),
            "eos_early_exit.stop_on_all_done": float(cfg.stop_on_all_done),
        }

    def _prefill(self) -> None:
        if self.embedding_weight is None or self.prompt_ids is None or self.state_buffer is None:
            raise RuntimeError("EOS early-exit buffers are not initialized")
        first_tokens = self.prompt_ids[:, 0].contiguous()
        torch.index_select(self.embedding_weight, 0, first_tokens, out=self.state_buffer)
        self.next_token_buffer.copy_(self.prompt_ids[:, -1])

    def _decode_step(self) -> torch.Tensor:
        if (
            self.embedding_weight is None
            or self.transition_weight is None
            or self._lm_head_weight_t is None
            or self.token_hidden_buffer is None
            or self.combined_buffer is None
            or self.next_state_buffer is None
            or self.state_buffer is None
            or self.logits_buffer is None
            or self.max_values_buffer is None
            or self.next_token_buffer is None
        ):
            raise RuntimeError("EOS early-exit decode buffers are not initialized")
        torch.index_select(
            self.embedding_weight,
            0,
            self.next_token_buffer,
            out=self.token_hidden_buffer,
        )
        torch.add(self.state_buffer, self.token_hidden_buffer, out=self.combined_buffer)
        torch.mm(self.combined_buffer, self.transition_weight, out=self.next_state_buffer)
        self.state_buffer.copy_(self.next_state_buffer)
        torch.mm(self.state_buffer, self._lm_head_weight_t, out=self.logits_buffer)
        torch.max(
            self.logits_buffer,
            dim=-1,
            out=(self.max_values_buffer, self.next_token_buffer),
        )
        return self.next_token_buffer

    def benchmark_fn(self) -> None:
        if self.generated_tokens is None or self.done_mask_buffer is None or self.eos_compare_buffer is None:
            raise RuntimeError("EOS early-exit output buffers are not initialized")
        self._completion_checks = 0
        self._decoded_steps = 0
        self._prefill()
        self.done_mask_buffer.zero_()
        filled = 0
        decode_step_range = self._early_exit_range if self.cfg.stop_on_all_done else self._full_decode_range
        with torch.inference_mode(), self._nvtx_range(self.cfg.label):
            for step in decode_step_range:
                next_token = self._decode_step()
                if (step + 1) >= self.cfg.force_eos_after_tokens:
                    next_token.fill_(self.cfg.eos_token_id)
                self.generated_tokens[:, step].copy_(next_token)
                filled = step + 1
                torch.eq(next_token, self.cfg.eos_token_id, out=self.eos_compare_buffer)
                self.done_mask_buffer.logical_or_(self.eos_compare_buffer)
                if self.cfg.stop_on_all_done:
                    self._completion_checks += 1
                next_token.masked_fill_(self.done_mask_buffer, self.cfg.eos_token_id)
            if filled < self.cfg.decode_tokens:
                self.generated_tokens[:, filled:].fill_(self.cfg.eos_token_id)
        self._decoded_steps = filled
        self.output = self.generated_tokens
        metrics = self._custom_metrics
        metrics["eos_early_exit.decoded_steps"] = float(filled)
        metrics["eos_early_exit.skipped_decode_steps"] = float(self.cfg.decode_tokens - filled)
        metrics["eos_early_exit.completion_checks"] = float(self._completion_checks)
        metrics["eos_early_exit.effective_decode_fraction_pct"] = (
            100.0 * float(filled) / self._decode_token_divisor
        )

    def capture_verification_payload(self) -> None:
        if self.prompt_ids is None or self.output is None or self.config_tensor is None:
            raise RuntimeError("setup() and benchmark_fn() must run before verification capture")
        self._set_verification_payload(
            inputs={"prompt_ids": self.prompt_ids, "config": self.config_tensor},
            output=self.output,
            batch_size=self.cfg.batch_size,
            parameter_count=self.parameter_count,
            precision_flags=PrecisionFlags(fp16=True, tf32=torch.backends.cuda.matmul.allow_tf32),
            output_tolerance=(0.0, 0.0),
        )

    def get_input_signature(self) -> InputSignature:
        cfg = self.cfg
        return InputSignature(
            shapes={
                "prompt_ids": (cfg.batch_size, cfg.prompt_tokens),
                "config": (6,),
                "output": (cfg.batch_size, cfg.decode_tokens),
            },
            dtypes={
                "prompt_ids": str(torch.int64),
                "config": str(torch.int64),
                "output": str(torch.int64),
            },
            batch_size=cfg.batch_size,
            parameter_count=self.parameter_count or (
                (cfg.vocab_size * cfg.hidden_size * 2) + (cfg.hidden_size * cfg.hidden_size)
            ),
            precision_flags=PrecisionFlags(fp16=True, tf32=True),
        )

    def teardown(self) -> None:
        for attr in (
            "embedding_weight",
            "transition_weight",
            "lm_head_weight",
            "_lm_head_weight_t",
            "prompt_ids",
            "state_buffer",
            "next_state_buffer",
            "token_hidden_buffer",
            "combined_buffer",
            "logits_buffer",
            "max_values_buffer",
            "next_token_buffer",
            "done_mask_buffer",
            "eos_compare_buffer",
            "generated_tokens",
            "config_tensor",
            "output",
        ):
            setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.cfg.iterations,
            warmup=self.cfg.warmup,
            enable_memory_tracking=True,
        )

    def get_custom_metrics(self) -> Dict[str, float]:
        return self._custom_metrics
