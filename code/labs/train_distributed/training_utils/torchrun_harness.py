"""Training-script metadata; generic child-training verification is unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NoReturn, Optional

import torch

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)


CHILD_TRAINING_VERIFICATION_UNSUPPORTED = (
    "SKIPPED: TorchrunScriptBenchmark actual child-training verification is unsupported. "
    "The wrapper does not collect child model outputs, losses, gradients or optimizer state. "
    "Run the training script directly; harness comparison requires a child-produced "
    "verification contract and an independent reference."
)


class TorchrunScriptBenchmark(BaseBenchmark):
    """Keep script configuration discoverable without certifying unrelated work.

    Direct training entrypoints remain usable. Harness execution and verification
    fail until the actual child training exposes an independently checked result.
    A parent-side toy forward cannot establish that contract.
    """

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

    def _unsupported_child_verification(self) -> NoReturn:
        raise RuntimeError(CHILD_TRAINING_VERIFICATION_UNSUPPORTED)

    def setup(self) -> None:
        self._unsupported_child_verification()

    def benchmark_fn(self) -> None:
        self._unsupported_child_verification()

    def capture_verification_payload(self) -> None:
        self._unsupported_child_verification()

    def _prepare_verification_payload(self) -> NoReturn:
        # Retained for callers of the former helper; stale cached attributes must
        # never turn unobserved child training into accepted verification.
        self._unsupported_child_verification()

    def get_verify_inputs(self) -> NoReturn:
        self._unsupported_child_verification()

    def get_verify_output(self) -> NoReturn:
        self._unsupported_child_verification()

    def get_input_signature(self) -> NoReturn:
        self._unsupported_child_verification()

    def get_output_tolerance(self) -> NoReturn:
        self._unsupported_child_verification()

    def teardown(self) -> None:
        # No surrogate tensors or CUDA resources are allocated by this wrapper.
        return None

    def validate_result(self) -> str:
        return CHILD_TRAINING_VERIFICATION_UNSUPPORTED

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
        # The harness must not launch and time a generic training child while
        # presenting an unrelated parent-side tensor as correctness evidence.
        self._unsupported_child_verification()
