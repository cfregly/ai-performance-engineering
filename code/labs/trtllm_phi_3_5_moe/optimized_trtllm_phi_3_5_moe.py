"""Optimized TensorRT-LLM generation for Phi-3.5-MoE."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gc
import torch

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.trtllm_phi_3_5_moe.trtllm_common import (
    build_prompt_tokens,
    disable_accelerate_transformer_engine,
    ensure_trtllm_assets,
    load_trtllm_runtime,
    parse_trtllm_args,
    resolve_model_path,
    slice_generated_token_ids,
    verification_token_prefix_length,
)

class OptimizedTrtLlmPhi35MoeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """TensorRT-LLM optimized generation for Phi-3.5-MoE."""

    def __init__(self) -> None:
        super().__init__()
        args = parse_trtllm_args()
        self.model_path = Path(args.model_path)
        self.engine_path = Path(args.engine_path) if args.engine_path is not None else None
        self.prompt_len = args.prompt_len
        self.max_new_tokens = args.max_new_tokens
        self.batch_size = args.batch_size
        self.vocab_slice = args.vocab_slice
        self.runner = None
        self.tokenizer = None
        self.sampling_config = None
        self.input_ids: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.prompt_lengths: Optional[list[int]] = None
        self._batch_inputs: Optional[list[torch.Tensor]] = None
        self.pad_token_id: int = 0
        self._generated_output_ids: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
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
            raise RuntimeError("SKIPPED: CUDA is required for the TRT-LLM Phi-3.5-MoE benchmark")
        self.model_path = resolve_model_path(self.model_path)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        disable_accelerate_transformer_engine()
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("SKIPPED: transformers is required for tokenizer support") from exc

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
        self._batch_inputs = [
            self.input_ids[i, :valid_len].contiguous()
            for i, valid_len in enumerate(self.prompt_lengths)
        ]
        self._verify_output_buffer = torch.empty(
            1,
            verification_token_prefix_length(self.max_new_tokens),
            dtype=torch.long,
            device="cpu",
        )
        ensure_trtllm_assets(
            self.model_path,
            engine_path=self.engine_path,
            require_engine=True,
        )
        runtime = load_trtllm_runtime()
        ModelRunner = runtime.ModelRunner
        SamplingConfig = runtime.SamplingConfig
        if self.engine_path is None:
            raise RuntimeError("SKIPPED: TensorRT-LLM engine_path is required")
        if self.engine_path.is_dir():
            if not hasattr(ModelRunner, "from_dir"):
                raise RuntimeError("SKIPPED: ModelRunner.from_dir is unavailable; provide an engine file path")
            self.runner = ModelRunner.from_dir(str(self.engine_path))
        else:
            self.runner = ModelRunner.from_engine(str(self.engine_path))
        self.sampling_config = SamplingConfig(
            end_id=int(self.tokenizer.eos_token_id),
            pad_id=int(self.tokenizer.pad_token_id),
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            return_dict=True,
            output_sequence_lengths=True,
            top_k=1,
            top_p=0.0,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.input_ids is None:
            raise RuntimeError("Benchmark not initialized")
        if self.runner is None or self.sampling_config is None:
            raise RuntimeError("TensorRT-LLM backend not initialized")
        with self._nvtx_range("optimized_trtllm_phi_3_5_moe"):
            if self.attention_mask is None:
                raise RuntimeError("Attention mask not initialized")
            if self._batch_inputs is None:
                raise RuntimeError("Prompt input batch not initialized")
            with torch.inference_mode():
                outputs = self.runner.generate(self._batch_inputs, sampling_config=self.sampling_config)
                output_ids = self._normalize_output_ids(outputs)
                self._generated_output_ids = output_ids
                self.output = self._generated_output_ids
        if self._generated_output_ids is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    @staticmethod
    def _normalize_output_ids(outputs) -> torch.Tensor:
        """Normalize TRT-LLM outputs into a token-id tensor for verification parity."""
        if isinstance(outputs, dict):
            if "output_ids" not in outputs:
                raise RuntimeError("TensorRT-LLM generate must return output_ids when return_dict=True")
            outputs = outputs["output_ids"]

        if not isinstance(outputs, torch.Tensor):
            raise RuntimeError(
                "Unsupported TRT-LLM output type "
                f"{type(outputs).__name__}; expected Tensor or dict containing output_ids."
            )
        if outputs.dim() not in (2, 3):
            raise RuntimeError(
                "TensorRT-LLM output_ids could not be normalized to [batch, seq] or [batch, beam, seq]; "
                f"got shape={tuple(outputs.shape)}"
            )
        return outputs

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
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        runner = self.runner
        if runner is not None:
            for method_name in ("shutdown", "close", "stop", "finalize", "destroy", "release"):
                method = getattr(runner, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                except Exception:
                    continue
        self.runner = None
        self.tokenizer = None
        self.sampling_config = None
        self.input_ids = None
        self.attention_mask = None
        self.prompt_lengths = None
        self._batch_inputs = None
        self._generated_output_ids = None
        self.output = None
        self._verify_output_buffer = None
        del runner
        gc.collect()
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        # TensorRT-LLM's Python runtime spawns helper processes even for world_size=1.
        # Normal timing runs still use the isolated subprocess path so the parent harness
        # can serialize results and reap any detached helpers after exit. Profiler wrappers
        # intentionally hard-exit without calling teardown because explicit TRT-LLM shutdown
        # under Nsight can segfault in cuMemFreeAsync / IExecutionContext deallocation.
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
    return OptimizedTrtLlmPhi35MoeBenchmark()
