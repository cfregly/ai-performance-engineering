#!/usr/bin/env python3
"""Baseline: vLLM v1 integration without optimization.

Demonstrates basic vLLM v1 usage without advanced features like:
- Bucketed CUDA graphs
- Optimized KV cache management
- Prefix caching
"""

import importlib
import importlib.metadata
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from core.utils.python_entrypoints import build_repo_python_env, install_local_module_override

repo_root = Path(__file__).resolve().parents[1]
hack_path = repo_root / "hack"

os.environ.update(build_repo_python_env(repo_root, base_env=os.environ, extra_pythonpath=[hack_path]))
install_local_module_override("numba", hack_path / "numba")
import numba  # noqa: F401

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.harness.serving_stack import (
    configure_serving_stack_cache_env,
    configure_serving_stack_runtime_env,
    get_serving_stack_pins,
    preload_serving_stack_shared_libs,
)
from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.logger import get_logger
from ch18.vllm_process_cleanup import shutdown_vllm_runtime

logger = get_logger(__name__)
_SERVING_STACK = get_serving_stack_pins()
_SERVING_STACK_LIB_DIRS = configure_serving_stack_runtime_env()
_SERVING_STACK_CACHE_DIRS = configure_serving_stack_cache_env()
_SERVING_STACK_PRELOADED_LIBS = preload_serving_stack_shared_libs()


def _is_vllm_abi_mismatch_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        ("undefined symbol" in text and ("vllm/_c.abi3.so" in text or "vllm._c" in text))
        or "c10_cuda_check_implementation" in text
    )


def _dist_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _assert_serving_stack_versions() -> None:
    if torch.__version__ != _SERVING_STACK.torch_version:
        raise RuntimeError(
            "FAIL FAST: Serving stack mismatch for torch "
            f"(expected {_SERVING_STACK.torch_version}, got {torch.__version__}). "
            f"Remediation: pin and reinstall {_SERVING_STACK.pinned_stack_str}."
        )
    vllm_version = _dist_version("vllm")
    if vllm_version != _SERVING_STACK.vllm_version:
        raise RuntimeError(
            "FAIL FAST: Serving stack mismatch for vllm "
            f"(expected {_SERVING_STACK.vllm_version}, got {vllm_version}). "
            f"Remediation: pin and reinstall {_SERVING_STACK.pinned_stack_str}."
        )
    flashinfer_version = _dist_version("flashinfer-python")
    if flashinfer_version != _SERVING_STACK.flashinfer_version:
        raise RuntimeError(
            "FAIL FAST: Serving stack mismatch for flashinfer-python "
            f"(expected {_SERVING_STACK.flashinfer_version}, got {flashinfer_version}). "
            f"Remediation: pin and reinstall {_SERVING_STACK.pinned_stack_str}."
        )


def _assert_vllm_runtime_ready(import_error: Optional[BaseException]) -> None:
    _assert_serving_stack_versions()
    if import_error is not None:
        if _is_vllm_abi_mismatch_error(import_error):
            raise RuntimeError(
                "FAIL FAST: vLLM ABI mismatch detected while importing vLLM. "
                f"Remediation: pin and reinstall {_SERVING_STACK.pinned_stack_str}. "
                "Then verify with: "
                "`python -c \"import importlib, importlib.metadata as md, torch, vllm; "
                "importlib.import_module('vllm._C'); "
                "print(torch.__version__, md.version('vllm'), vllm.__version__)\"`. "
                f"Original error: {import_error}"
            )
        raise RuntimeError(
            "FAIL FAST: vLLM import failed before benchmark setup. "
            f"Remediation: pin and reinstall {_SERVING_STACK.pinned_stack_str}. "
            f"Original error: {import_error}"
        )
    try:
        importlib.import_module("vllm._C")
    except Exception as exc:
        if _is_vllm_abi_mismatch_error(exc):
            raise RuntimeError(
                "FAIL FAST: vLLM ABI mismatch detected while loading vllm._C. "
                f"Remediation: pin and reinstall {_SERVING_STACK.pinned_stack_str}. "
                "Then verify with: "
                "`python -c \"import importlib, importlib.metadata as md, torch, vllm; "
                "importlib.import_module('vllm._C'); "
                "print(torch.__version__, md.version('vllm'), vllm.__version__)\"`. "
                f"Original error: {exc}"
            ) from exc
        raise RuntimeError(
            "FAIL FAST: vllm._C failed to import. "
            f"Remediation: pin and reinstall {_SERVING_STACK.pinned_stack_str}. "
            f"Original error: {exc}"
        ) from exc


def _fixed_kv_cache_memory_bytes() -> int:
    """Use a deterministic KV-cache budget to avoid unstable init profiling.

    vLLM's default startup path profiles available GPU memory and can fail when
    free memory changes mid-profile in shared/containerized environments.
    Supplying kv_cache_memory_bytes skips that fragile profiling path.
    """
    return 2 * 1024**3  # 2 GiB is sufficient for opt-125m @ max_model_len=512

# Check for vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import TokensPrompt
    VLLM_AVAILABLE = True
    _IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    LLM = SamplingParams = TokensPrompt = None  # type: ignore[assignment]
    VLLM_AVAILABLE = False
    _IMPORT_ERROR = exc
    logger.warning("vLLM import failed during module load: %s", exc)


class BaselineVLLMV1Integration:
    """Baseline vLLM v1 without optimizations."""
    
    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        max_tokens: int = 128,
        batch_size: int = 8,
        use_vllm: bool = True,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        if not use_vllm:
            raise RuntimeError(
                "FAIL FAST: BaselineVLLMV1Integration requires vLLM execution "
                "(use_vllm=False is unsupported)."
            )
        self.use_vllm = True
        self._token_id_preview: List[int] = []
        self._result_payload: Dict[str, Any] = {
            "mean_latency_ms": 0.0,
            "throughput_tokens_per_sec": 0.0,
            "total_tokens": 0,
            "token_ids": self._token_id_preview,
        }

    def _store_token_preview(self, token_ids: Sequence[int]) -> List[int]:
        preview_count = min(16, len(token_ids))
        preview = self._token_id_preview
        if len(preview) < preview_count:
            preview.extend([0] * (preview_count - len(preview)))
        elif len(preview) > preview_count:
            del preview[preview_count:]
        for index in range(preview_count):
            preview[index] = int(token_ids[index])
        return preview

    def _new_llm(self) -> "LLM":
        return LLM(
            model=self.model_name,
            enforce_eager=True,  # Disable CUDA graphs (baseline)
            enable_prefix_caching=False,  # Disable prefix caching
            enable_chunked_prefill=True,
            # Keep headroom for shared GPU environments; vLLM hard-gates init on this ratio.
            gpu_memory_utilization=0.15,
            kv_cache_memory_bytes=_fixed_kv_cache_memory_bytes(),
            dtype="bfloat16",
            tensor_parallel_size=1,  # Single GPU to avoid coordination issues
            max_model_len=512,  # Limit context length to reduce memory
        )
    
    def setup(self):
        """Initialize vLLM model."""
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        _assert_vllm_runtime_ready(_IMPORT_ERROR)
        import gc

        # Force cleanup before vLLM initialization to avoid resource conflicts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        last_engine_error: Optional[RuntimeError] = None
        for attempt in range(2):
            try:
                # Baseline: no CUDA graphs, no prefix caching.
                self.llm = self._new_llm()
                logger.info(f"Loaded model: {self.model_name}")
                logger.info("Baseline config: eager execution, no prefix caching")
                break
            except RuntimeError as e:
                error_msg = str(e)
                if "Engine core initialization failed" not in error_msg:
                    raise
                last_engine_error = e
                if attempt == 0:
                    logger.warning(
                        "Baseline eager vLLM init failed on first attempt; forcing cleanup and retrying once: %s",
                        error_msg,
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    continue
                raise RuntimeError(
                    "FAIL FAST: Baseline vLLM engine initialization failed in eager mode "
                    "after retry. Remediation: ensure the serving stack matches pinned versions "
                    "and rerun after clearing GPU state. "
                    f"Original error: {error_msg}"
                ) from e
        else:
            if last_engine_error is not None:
                raise RuntimeError(
                    "FAIL FAST: Baseline eager vLLM engine initialization failed unexpectedly."
                ) from last_engine_error
            raise RuntimeError("FAIL FAST: Baseline eager vLLM engine did not initialize")

        # Build pre-tokenized prompts so CPU tokenization doesn't dominate, and so
        # prefix caching (in the optimized variant) can skip meaningful prefill.
        tokenizer = self.llm.get_tokenizer()
        base_ids = tokenizer.encode("Once upon a time in a land far away, ", add_special_tokens=False)
        if not base_ids:
            raise RuntimeError("Tokenizer returned empty token IDs for baseline prefix")

        max_prompt_len = 512 - self.max_tokens
        suffix_ids: List[List[int]] = []
        for i in range(self.batch_size):
            ids = tokenizer.encode(f"there was a {i}.", add_special_tokens=False)
            if not ids:
                raise RuntimeError(f"Tokenizer returned empty token IDs for suffix {i}")
            suffix_ids.append(ids)
        max_suffix = max(len(ids) for ids in suffix_ids)
        if max_suffix >= max_prompt_len:
            raise RuntimeError("Suffix length exceeds max prompt length; reduce max_tokens or suffix text.")

        target_prefix_len = max_prompt_len - max_suffix
        repeats = (target_prefix_len + len(base_ids) - 1) // len(base_ids)
        prefix_ids = (base_ids * repeats)[:target_prefix_len]

        self.prompts = [
            TokensPrompt(prompt_token_ids=prefix_ids + suffix_ids[i])
            for i in range(self.batch_size)
        ]
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )
    
    def run(self) -> Dict[str, Any]:
        """Execute baseline vLLM inference."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Generate
        outputs = self.llm.generate(self.prompts, self.sampling_params, use_tqdm=False)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = end - start
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        first_ids = outputs[0].outputs[0].token_ids if outputs else []
        token_ids = self._store_token_preview(first_ids)
        throughput = total_tokens / elapsed
        mean_latency_ms = (elapsed / len(self.prompts)) * 1000
        
        logger.info("Throughput: %.2f tokens/sec", throughput)
        logger.info("Mean latency: %.2f ms", mean_latency_ms)

        metrics = self._result_payload
        metrics["mean_latency_ms"] = mean_latency_ms
        metrics["throughput_tokens_per_sec"] = throughput
        metrics["total_tokens"] = total_tokens
        metrics["token_ids"] = token_ids
        return metrics
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'llm'):
            try:
                shutdown_vllm_runtime(self.llm, logger=logger.warning)
            finally:
                del self.llm
        torch.cuda.empty_cache()


class BaselineVLLMV1IntegrationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness wrapper for the baseline vLLM integration."""
    allowed_benchmark_fn_antipatterns = ("sync",)

    def __init__(self):
        super().__init__()
        self.runner = BaselineVLLMV1Integration()
        # Profiling wrappers should teardown vLLM workers instead of hard-exiting.
        self.profile_require_teardown = True
        # vLLM kernels execute in worker subprocesses, so parent NVTX include
        # filters can hide all kernels from NCU capture.
        self.disable_ncu_nvtx_filter = True
        # Keep NCU replay scoped to kernel to avoid repeated full application
        # replays that can exceed benchmark timeout budgets.
        self.preferred_ncu_replay_mode = "kernel"
        self._metrics: Dict[str, Any] = {}
        self.output: Optional[torch.Tensor] = None
        self._last_token_ids: Optional[torch.Tensor] = None
        self._token_id_buffer: Optional[torch.Tensor] = None
        self._batch_size_tensor: Optional[torch.Tensor] = None
        self._max_tokens_tensor: Optional[torch.Tensor] = None
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.runner.batch_size),
            tokens_per_iteration=float(self.runner.batch_size * self.runner.max_tokens),
        )

    def setup(self) -> None:
        self.runner.setup()
        self._metrics = {}
        self.output = None
        self._last_token_ids = None
        self._token_id_buffer = None
        self._batch_size_tensor = torch.empty((), dtype=torch.int64)
        self._max_tokens_tensor = torch.empty((), dtype=torch.int64)
        self._batch_size_tensor.fill_(int(self.runner.batch_size))
        self._max_tokens_tensor.fill_(int(self.runner.max_tokens))

    def _materialize_token_ids(self, token_ids: Sequence[int]) -> torch.Tensor:
        num_ids = len(token_ids)
        if self._token_id_buffer is None or self._token_id_buffer.numel() < num_ids:
            self._token_id_buffer = torch.empty(num_ids, dtype=torch.int32)
        token_view = self._token_id_buffer.narrow(0, 0, num_ids)
        for index, token_id in enumerate(token_ids):
            token_view[index] = int(token_id)
        return token_view

    def benchmark_fn(self) -> None:
        self._metrics = self.runner.run()
        token_ids = self._metrics.get("token_ids")
        if token_ids is None:
            raise RuntimeError("Runner did not return token_ids for verification")
        self._last_token_ids = self._materialize_token_ids(token_ids)
        self.output = self._last_token_ids

    def capture_verification_payload(self) -> None:
        if self._last_token_ids is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._batch_size_tensor is None or self._max_tokens_tensor is None:
            raise RuntimeError("setup() must initialize verification metadata tensors")
        self._set_verification_payload(
            inputs={
                "batch_size": self._batch_size_tensor,
                "max_tokens": self._max_tokens_tensor,
            },
            output=self.output,
            batch_size=self.runner.batch_size,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": True, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.runner.cleanup()
        self._batch_size_tensor = None
        self._max_tokens_tensor = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=3,
            warmup=1,
            use_subprocess=True,
            setup_timeout_seconds=600,
            measurement_timeout_seconds=600,
            timing_method="wall_clock",
        )

    def get_workload_metadata(self) -> WorkloadMetadata | None:
        return WorkloadMetadata(
            requests_per_iteration=float(self.runner.batch_size),
            tokens_per_iteration=float(self.runner.batch_size * self.runner.max_tokens),
        )

    def get_input_signature(self) -> dict:
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        return simple_signature(
            batch_size=self.runner.batch_size,
            dtype="int64",
            max_tokens=self.runner.max_tokens,
            precision_flags=PrecisionFlags(bf16=True, tf32=tf32_enabled),
        ).to_dict()

    def get_custom_metrics(self) -> Dict[str, Any]:
        return self._metrics


def get_benchmark() -> BaseBenchmark:
    return BaselineVLLMV1IntegrationBenchmark()
