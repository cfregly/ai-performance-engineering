"""Lightweight helpers for launching training demos via the benchmark harness."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.benchmark.verification import PrecisionFlags, InputSignature
from core.benchmark.verification_mixin import VerificationPayloadMixin


class TorchrunScriptBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Wrap a script-based training demo so the harness can launch it via torchrun."""

    def __init__(
        self,
        *,
        script_path: Path,
        base_args: Optional[List[str]] = None,
        target_label: Optional[str] = None,
        config_arg_map: Optional[Dict[str, str]] = None,
        multi_gpu_required: bool = True,
        default_nproc_per_node: Optional[int] = None,
        default_iterations: Optional[int] = None,
        measurement_timeout_seconds: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self._script_path = Path(script_path)
        self._base_args = list(base_args) if base_args else []
        self._config_arg_map = config_arg_map or {}
        self._multi_gpu_required = multi_gpu_required
        self._default_nproc_per_node = default_nproc_per_node
        self._default_iterations = default_iterations
        self._measurement_timeout_seconds = measurement_timeout_seconds
        self._env = dict(env) if env else {}
        self._target_label = target_label
        self.name = name or self._script_path.stem
        # Compliance: verification interface
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._batch_size = 4
        self._hidden_dim, self._meta_dim = self._resolve_signature_dims()
        self._model: Optional[nn.Linear] = None
        self._input: Optional[torch.Tensor] = None
        self._meta: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._parameter_count = 0

    def _signature_seed(self) -> int:
        identity = f"{self._target_label or self.name}|{self._script_path.name}"
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    def _resolve_signature_dims(self) -> Tuple[int, int]:
        seed = self._signature_seed()
        hidden = 128 + (seed % 128)
        meta = 32 + ((seed >> 8) % 64)
        return hidden, meta

    def _signature_world_size(self) -> int:
        if self._default_nproc_per_node is not None:
            return int(self._default_nproc_per_node)
        return 2 if self._multi_gpu_required else 1

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._model = nn.Linear(self._hidden_dim, self._hidden_dim, bias=False).to(self.device)
        self._input = torch.randn(
            self._batch_size,
            self._hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._meta = torch.randn(
            self._batch_size,
            self._meta_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._parameter_count = sum(p.numel() for p in self._model.parameters())

    def benchmark_fn(self) -> None:
        if self._model is None or self._input is None or self._meta is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        with torch.inference_mode():
            output = self._model(self._input)
            meta_scale = self._meta.mean(dim=-1, keepdim=True)
            output.add_(meta_scale)
            self._output = output

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input is None or self._meta is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        world_size = self._resolve_nproc_per_node() or 1
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        self._set_verification_payload(
            inputs={"input": self._input, "meta": self._meta},
            output=self._output,
            batch_size=self._batch_size,
            parameter_count=int(self._parameter_count),
            precision_flags=PrecisionFlags(tf32=tf32_enabled),
            output_tolerance=(0.1, 1.0),
            signature_overrides={"world_size": world_size},
        )

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.setup()
        try:
            self.benchmark_fn()
            self.capture_verification_payload()
            self._subprocess_verify_output = self.get_verify_output()
            self._subprocess_output_tolerance = self.get_output_tolerance()
            self._subprocess_input_signature = self.get_input_signature()
        finally:
            self.teardown()

    def teardown(self) -> None:
        self._model = None
        self._input = None
        self._meta = None
        self._output = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self._output is None:
            return "No output captured"
        return None

    def get_input_signature(self) -> InputSignature:
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        return InputSignature(
            shapes={
                "input": (self._batch_size, self._hidden_dim),
                "meta": (self._batch_size, self._meta_dim),
                "output": (self._batch_size, self._hidden_dim),
            },
            dtypes={
                "input": "torch.float32",
                "meta": "torch.float32",
                "output": "torch.float32",
            },
            batch_size=self._batch_size,
            parameter_count=int(self._hidden_dim * self._hidden_dim),
            precision_flags=PrecisionFlags(tf32=tf32_enabled),
            world_size=self._signature_world_size(),
        )

    def _resolve_nproc_per_node(self) -> Optional[int]:
        if self._default_nproc_per_node is None and not self._multi_gpu_required:
            return None
        if self._default_nproc_per_node is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for multi-GPU torchrun benchmarks")
            requested = torch.cuda.device_count()
        else:
            requested = int(self._default_nproc_per_node)
        if self._multi_gpu_required and requested < 2:
            raise RuntimeError("multi_gpu_required benchmarks need >=2 GPUs")
        if torch.cuda.is_available():
            available = torch.cuda.device_count()
            if requested > available:
                raise RuntimeError(f"nproc_per_node={requested} exceeds available GPUs ({available})")
        return requested

    def get_config(self) -> BenchmarkConfig:
        cfg = BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            multi_gpu_required=self._multi_gpu_required,
            nproc_per_node=self._resolve_nproc_per_node(),
        )
        if self._default_iterations is not None:
            cfg.iterations = int(self._default_iterations)
        if self._measurement_timeout_seconds is not None:
            cfg.measurement_timeout_seconds = int(self._measurement_timeout_seconds)
        cfg.target_label = self._target_label
        return cfg

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        return TorchrunLaunchSpec(
            script_path=self._script_path,
            script_args=list(self._base_args),
            env=dict(self._env),
            multi_gpu_required=self._multi_gpu_required,
            config_arg_map=self._config_arg_map,
            name=self.name,
        )
