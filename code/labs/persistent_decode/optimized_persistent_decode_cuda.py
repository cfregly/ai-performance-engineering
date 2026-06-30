"""Persistent decode in CUDA via an out-of-line extension (no fallbacks)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import functools
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.extension_loader_template import load_cuda_extension_v2
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    build_decode_input_signature,
    get_decode_options,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


@functools.lru_cache(None)
def _load_extension() -> object:
    """Compile and return the CUDA extension once per process."""
    include_dirs = [
        # Stick to the repo-pinned CUTLASS to avoid mixing cute headers from TransformerEngine.
        REPO_ROOT / "third_party" / "cutlass" / "include",
        REPO_ROOT / "core" / "common" / "headers",
    ]
    return load_cuda_extension_v2(
        name="persistent_decode_ext",
        sources=[Path(__file__).with_name("persistent_decode_ext.cu")],
        extra_cuda_cflags=[
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DCUTE_ARCH_TCGEN05_TMEM_ENABLED",
            # sm_100 (B200) SASS does not load on sm_103 (GB300). Match the sibling
            # tma_extension.py / optimized_tma_prefill_decode.py arch set so the
            # persistent decode kernel has a Blackwell-Ultra (sm_103) image too.
            "-gencode=arch=compute_100,code=sm_100",
            "-gencode=arch=compute_103,code=sm_103",
            "-gencode=arch=compute_120,code=sm_120",
            "-gencode=arch=compute_121,code=sm_121",
        ] + [f"-I{p}" for p in include_dirs if p.exists()],
    )


class OptimizedPersistentDecodeCUDABenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Persistent decode using a cooperative CUDA kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.options = get_decode_options()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.batch_size = self.batch
        self.hidden_dim = self.head_dim
        self.blocks = 8
        self._ext: Optional[object] = None
        self.output: Optional[torch.Tensor] = None
        self._output_view: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        """Initialize the persistent decode CUDA extension and inputs."""
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for persistent decode benchmark")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Try to load the extension - this may fail if not pre-built
        try:
            self._ext = _load_extension()
        except Exception as exc:
            raise RuntimeError(
                f"SKIPPED: persistent_decode_ext failed to build ({type(exc).__name__}: {exc}). "
                "Build offline with: cd labs/persistent_decode && python -c 'from optimized_persistent_decode_cuda import _load_extension; _load_extension()'"
            ) from exc
        
        self.inputs = build_inputs(self.batch, self.seq_len, self.head_dim, self.device)
        self._output_view = self.inputs.out[:1, : min(8, self.inputs.out.shape[1])]
        self._verify_output_buffer = torch.empty_like(self._output_view, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        """Run the persistent decode kernel."""
        if self._ext is None or self.inputs is None or self._output_view is None:
            raise RuntimeError("SKIPPED: persistent_decode_ext not initialized")
        
        # Call the extension's forward pass
        with torch.inference_mode(), self._nvtx_range("persistent_decode_cuda"):
            self._ext.persistent_decode(
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.blocks,
            )
        # Capture a representative slice of the output
        self.output = self._output_view
        if self.inputs is None or self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "q": self.inputs.q.detach(),
                "k": self.inputs.k.detach(),
                "v": self.inputs.v.detach(),
            },
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={
                "fp16": self.inputs.q.dtype == torch.float16,
                "bf16": self.inputs.q.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.output = None
        self._output_view = None
        self._verify_output_buffer = None

    def get_config(self) -> BenchmarkConfig:
        # NOTE: CUDA binary runs external executable, warmup=5 ensures CUDA driver is initialized
        return BenchmarkConfig(
            iterations=5,
            warmup=5,  # Required to warm CUDA driver and JIT
            use_subprocess=True,
            measurement_timeout_seconds=600,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "persistent_decode_cu.batch_size": float(getattr(self, 'batch_size', 0)),
            "persistent_decode_cu.seq_len": float(getattr(self, 'seq_len', 0)),
            "persistent_decode_cu.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def get_input_signature(self) -> dict:
        return build_decode_input_signature(
            batch=self.batch,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            quantization=self.options.quantization,
        )

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None

def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeCUDABenchmark()
