"""cuDNN vs Flash SDP attention microbenchmark lab."""

from __future__ import annotations

import argparse
from typing import List, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.backends.cuda import SDPAParams, can_use_cudnn_attention

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

_CLI_BACKEND: Optional[str] = None
_BACKEND_CHOICES = ("auto", "cudnn", "flash", "math")


def _resolve_backend(cli_choice: Optional[str] = None) -> str:
    """Resolve backend selection from an explicit CLI choice or default to auto."""
    if cli_choice in _BACKEND_CHOICES:
        return cli_choice  # type: ignore[return-value]
    return "auto"


def _select_backend(requested: Optional[str]) -> str:
    """Choose a backend, falling back when cuDNN is unsupported."""
    backend = _resolve_backend(requested).lower()
    if backend != "cudnn":
        return backend
    if not torch.cuda.is_available():
        return "flash"
    try:
        q = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.float16)
        params = SDPAParams(q, q, q, None, 0.0, False, False)
        if not can_use_cudnn_attention(params, False):
            return "flash"
    except Exception:
        return "flash"
    return backend


def _sdpa_context(backend: str):
    """Return the configured SDPA context."""
    backend = backend.lower()
    if backend == "cudnn":
        preference = []
        for name in ("CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"):
            if hasattr(SDPBackend, name):
                preference.append(getattr(SDPBackend, name))
        return sdpa_kernel(preference)
    if backend == "flash":
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    if backend == "math":
        if hasattr(SDPBackend, "MATH"):
            return sdpa_kernel([getattr(SDPBackend, "MATH")])
        return nullcontext()

    # Auto: try cuDNN, then Flash, then efficient attention.
    preference = []
    for name in ("CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"):
        if hasattr(SDPBackend, name):
            preference.append(getattr(SDPBackend, name))
    return sdpa_kernel(preference)


def _ensure_backend_available(backend: str) -> None:
    """Fail fast by actually invoking the selected backend."""
    if not torch.cuda.is_available():
        raise RuntimeError("SDP benchmark requires a CUDA device.")
    q = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.float16)
    with torch.inference_mode(), _sdpa_context(backend):
        _ = F.scaled_dot_product_attention(q, q, q, is_causal=False)
    torch.cuda.synchronize()


class SDPAAttentionModule(nn.Module):
    """Minimal attention block with selectable SDP backend."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, backend: str = "auto"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.backend = backend
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.hidden_dim // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with _sdpa_context(self.backend):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, self.hidden_dim)
        return out


class FlashSDPLabBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs SDPA attention with selectable backend for cuDNN versus Flash comparisons."""

    def __init__(self, backend: Optional[str] = None):
        super().__init__()
        self.model: Optional[SDPAAttentionModule] = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.seq_len = 256
        self.batch = 8
        self.hidden = 512
        self.backend = _select_backend(backend or _CLI_BACKEND)
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        import gc

        # Clean up CUDA state to prevent RNG corruption from previous benchmarks.
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        try:
            if hasattr(torch.cuda, "graph_pool_trim"):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass

        _ensure_backend_available(self.backend)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.model = SDPAAttentionModule(hidden_dim=self.hidden, num_heads=8, backend=self.backend).to(
            self.device, dtype=torch.float16
        )
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        # Use CPU randn + to(device) to avoid CUDA RNG graph issues
        self.inputs = torch.randn(self.batch, self.seq_len, self.hidden, dtype=torch.float16).to(self.device)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model/inputs not initialized")
        nvtx_label = f"sdp_{self.backend}_baseline"
        with nvtx_range(nvtx_label, enable=self._enable_nvtx):
            with torch.inference_mode():
                self.output = self.model(self.inputs)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.inputs.detach()},
            output=self.output,
            batch_size=self.batch,
            parameter_count=self._payload_parameter_count,
            precision_flags={"fp16": True, "bf16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.output = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/inputs not initialized"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            enable_profiling=False,
            enable_nsys=False,
            enable_ncu=False,
            measurement_timeout_seconds=30,
            setup_timeout_seconds=30,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return {"backend": self.backend}

    # Harness hook: apply per-target CLI args (e.g., --backend cudnn)
    def apply_target_overrides(self, argv: List[str]) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--backend",
            choices=_BACKEND_CHOICES,
            default=None,
            help="Select SDP backend: auto (cuDNN->Flash), cudnn, flash, or math",
        )
        args, _ = parser.parse_known_args(argv)
        if args.backend:
            self.backend = _select_backend(args.backend)

def get_benchmark() -> BaseBenchmark:
    return FlashSDPLabBenchmark()


def _parse_cli_backend(argv: Optional[list[str]] = None) -> Optional[str]:
    parser = argparse.ArgumentParser(description="cuDNN vs Flash SDP lab")
    parser.add_argument(
        "--backend",
        choices=_BACKEND_CHOICES,
        default=None,
        help="Select SDP backend: auto (cuDNN->Flash), cudnn, flash, or math",
    )
    args, _ = parser.parse_known_args(argv)
    return args.backend
