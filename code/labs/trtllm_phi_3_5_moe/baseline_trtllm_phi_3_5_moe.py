"""Baseline Transformers generate for Phi-3.5-MoE."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.trtllm_phi_3_5_moe.trtllm_common import (
    build_prompt_tokens,
    disable_accelerate_transformer_engine,
    ensure_trtllm_assets,
    parse_trtllm_args,
    resolve_model_path,
    slice_generated_token_ids,
    verification_token_prefix_length,
)


class BaselineTrtLlmPhi35MoeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline HF Transformers generation for Phi-3.5-MoE."""

    def __init__(self) -> None:
        super().__init__()
        args = parse_trtllm_args()
        self.model_path = Path(args.model_path)
        self.prompt_len = args.prompt_len
        self.max_new_tokens = args.max_new_tokens
        self.batch_size = args.batch_size
        self.vocab_slice = args.vocab_slice
        self.model = None
        self.tokenizer = None
        self.input_ids: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.prompt_lengths: Optional[list[int]] = None
        self.pad_token_id: int = 0
        self._generated_output_ids: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        tokens = float(self.prompt_len + self.max_new_tokens)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA is required for the TRT-LLM Phi-3.5-MoE baseline")
        self.model_path = resolve_model_path(self.model_path)
        ensure_trtllm_assets(self.model_path, require_engine=False)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        disable_accelerate_transformer_engine()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("SKIPPED: transformers is required for the baseline TRT-LLM lab") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        input_ids, attention_mask = build_prompt_tokens(
            self.tokenizer,
            prompt_len=self.prompt_len,
            batch_size=self.batch_size,
        )
        self.prompt_lengths = [input_ids.size(1)] * self.batch_size
        self.input_ids = input_ids.to(self.device)
        self.attention_mask = attention_mask.to(self.device)
        self._verify_output_buffer = torch.empty(
            1,
            verification_token_prefix_length(self.max_new_tokens),
            dtype=torch.long,
            device="cpu",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.model is None or self.input_ids is None or self.attention_mask is None:
            raise RuntimeError("Benchmark not initialized")
        with self._nvtx_range("baseline_trtllm_phi_3_5_moe"):
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                self._generated_output_ids = outputs.sequences
                self.output = self._generated_output_ids
        if self._generated_output_ids is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if (
            self._generated_output_ids is None
            or self.input_ids is None
            or self.prompt_lengths is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        generated_prefix = slice_generated_token_ids(
            self._generated_output_ids[:1],
            prompt_lengths=[self.prompt_lengths[0]],
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.pad_token_id,
        )[:, : self._verify_output_buffer.shape[1]]
        self._verify_output_buffer.copy_(generated_prefix, non_blocking=False)
        # Keep signature fields backend-agnostic so baseline Transformers and optimized
        # TRT-LLM engine runs compare on equivalent workload semantics.
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self._verify_output_buffer,
            batch_size=int(self.batch_size),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        self.input_ids = None
        self.attention_mask = None
        self.prompt_lengths = None
        self._generated_output_ids = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            timing_method="wall_clock",
            full_device_sync=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=self.batch_size,
            dtype="int64",
            prompt_len=self.prompt_len,
            max_new_tokens=self.max_new_tokens,
            vocab_slice=self.vocab_slice,
            precision_flags=PrecisionFlags(fp16=True, tf32=False),
        ).to_dict()

    def validate_result(self) -> Optional[str]:
        if self._generated_output_ids is None:
            return "benchmark_fn() did not produce output"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineTrtLlmPhi35MoeBenchmark()
