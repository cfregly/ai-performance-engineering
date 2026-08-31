"""Opt-in torchrun adapter for the four verified ZeRO factories only."""

from __future__ import annotations

import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, cast

import torch

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.harness.benchmark_harness import (
    BenchmarkConfig,
    TorchrunLaunchSpec,
    _lookup_target_extra_args,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark
from labs.train_distributed.zero2_child_protocol import (
    DEFAULT_PROFILE,
    LAUNCH_MONOTONIC_NS_ENV,
    LAUNCH_WALL_NS_ENV,
    MODE_ENV,
    POST_TIMING_PROFILE_KIND,
    PROFILE_KIND_ENV,
    RESULT_CALLBACK,
    RESULT_DIR_ENV,
    RUN_ID_ENV,
    VARIANT_ENV,
    validate_zero2_result_bundle,
)


class Zero2TorchrunBenchmark(TorchrunScriptBenchmark):
    """ZeRO-specific post-timing result transport.

    The generic training wrapper remains unsupported.  This subclass is used by
    exactly four ZeRO factories whose child script implements the v3 result
    contract and independently checked correctness profile.
    """

    def __init__(self, *, mode: str, variant: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if mode not in {"baseline", "optimized"}:
            raise ValueError(f"Invalid ZeRO mode {mode!r}")
        if variant not in {"single", "multigpu"}:
            raise ValueError(f"Invalid ZeRO variant {variant!r}")
        self._zero2_mode = mode
        self._zero2_variant = variant
        self._zero2_result_context: Optional[Dict[str, Any]] = None
        self._zero2_result_bundle: Optional[Dict[str, Any]] = None

    @staticmethod
    def _exact_world_size(config: BenchmarkConfig, nproc_per_node: int) -> int:
        nnodes = getattr(config, "nnodes", None) or 1
        try:
            nodes = int(nnodes)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "ZeRO child-result verification requires an exact integer nnodes value"
            ) from exc
        if nodes != 1:
            raise RuntimeError(
                "ZeRO child-result verification currently requires nnodes == 1; "
                "its result directory and monotonic freshness clock are host-local"
            )
        return nodes * int(nproc_per_node)

    def _reject_reserved_overrides(self, config: BenchmarkConfig) -> None:
        target_overrides = _lookup_target_extra_args(
            getattr(config, "target_extra_args", {}) or {},
            self._target_label,
        )
        if not target_overrides:
            return
        arguments = (
            shlex.split(target_overrides)
            if isinstance(target_overrides, str)
            else list(target_overrides)
        )
        reserved = {
            "--mode",
            "--variant",
            "--verification-only",
            "--verification-backend",
            "--compile",
        }
        for argument in arguments:
            flag = str(argument).split("=", 1)[0]
            reserved_control = next(
                (
                    control
                    for control in reserved
                    if control == flag
                    or (flag.startswith("--") and control.startswith(flag))
                ),
                None,
            )
            if reserved_control is not None:
                raise RuntimeError(
                    "ZeRO harness overrides may not set reserved control "
                    f"{reserved_control!r}"
                )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        if config is None:
            config = self.get_config()
        self._reject_reserved_overrides(config)
        nproc = getattr(config, "nproc_per_node", None)
        if nproc is None:
            nproc = self._resolve_nproc_per_node()
        nproc = 1 if nproc is None else int(nproc)
        if nproc <= 0:
            raise RuntimeError("ZeRO child-result verification requires nproc_per_node >= 1")
        world_size = self._exact_world_size(config, nproc)
        if self._zero2_variant == "single" and world_size != 1:
            raise RuntimeError("ZeRO single child verification requires world_size == 1")
        if self._zero2_variant == "multigpu" and world_size < 2:
            raise RuntimeError("SKIPPED: ZeRO multigpu child verification requires world_size >= 2")

        previous_context = self._zero2_result_context
        if previous_context is not None:
            previous_dir = Path(previous_context["result_dir"])
            if previous_dir.exists():
                previous_status = previous_context.get("retention", {}).get(
                    "status", "unknown"
                )
                raise RuntimeError(
                    "Refusing to replace an unconsumed ZeRO child-result context: "
                    f"status={previous_status}, path={previous_dir}"
                )

        result_dir = Path(tempfile.mkdtemp(prefix="aisp-zero2-child-result-"))
        if any(result_dir.iterdir()):
            raise RuntimeError("Fresh ZeRO child-result directory is unexpectedly non-empty")
        run_id = uuid.uuid4().hex
        profile_kind = POST_TIMING_PROFILE_KIND
        env = dict(self._env)
        env.update(
            {
                RESULT_DIR_ENV: str(result_dir),
                RUN_ID_ENV: run_id,
                MODE_ENV: self._zero2_mode,
                VARIANT_ENV: self._zero2_variant,
                PROFILE_KIND_ENV: profile_kind,
            }
        )
        self._zero2_result_context = {
            "result_dir": result_dir,
            "run_id": run_id,
            "mode": self._zero2_mode,
            "variant": self._zero2_variant,
            "world_size": world_size,
            "profile_kind": profile_kind,
            "retention": {
                "policy": "delete-after-success-retain-failure",
                "status": "pending-child-result",
                "path": str(result_dir),
            },
        }
        self._zero2_result_bundle = None
        for attribute in (
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
            "_zero2_verify_inputs",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return TorchrunLaunchSpec(
            script_path=self._script_path,
            script_args=list(self._base_args),
            env=env,
            multi_gpu_required=self._multi_gpu_required,
            config_arg_map=self._config_arg_map,
            name=self.name,
            result_callback=RESULT_CALLBACK,
        )

    def consume_zero2_child_results(
        self,
        *,
        launch_wall_ns: int,
        launch_monotonic_ns: int,
        finish_wall_ns: int,
        finish_monotonic_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        """Validate the fresh rank quorum and register post-timing outputs."""

        context = self._zero2_result_context
        if context is None:
            raise RuntimeError("ZeRO child-result callback has no prepared launch context")
        if returncode != 0:
            retained_path = str(context["result_dir"])
            context["retention"] = {
                "policy": "delete-after-success-retain-failure",
                "status": "retained-child-failure",
                "path": retained_path,
            }
            raise RuntimeError(
                "ZeRO child-result callback requires a clean child exit, "
                f"got {returncode}; artifacts retained at {retained_path}"
            )
        if context.get("profile_kind") != POST_TIMING_PROFILE_KIND:
            retained_path = str(context["result_dir"])
            context["retention"] = {
                "policy": "delete-after-success-retain-failure",
                "status": "retained-execution-kind-mismatch",
                "path": retained_path,
            }
            raise RuntimeError(
                "The performance harness accepts only post-timing ZeRO child results; "
                f"artifacts retained at {retained_path}"
            )
        try:
            bundle = validate_zero2_result_bundle(
                context["result_dir"],
                run_id=context["run_id"],
                mode=context["mode"],
                variant=context["variant"],
                world_size=context["world_size"],
                launch_wall_ns=int(launch_wall_ns),
                launch_monotonic_ns=int(launch_monotonic_ns),
                finish_wall_ns=int(finish_wall_ns),
                finish_monotonic_ns=int(finish_monotonic_ns),
                profile_kind=context["profile_kind"],
            )
        except Exception as exc:
            retained_path = str(context["result_dir"])
            context["retention"] = {
                "policy": "delete-after-success-retain-failure",
                "status": "retained-validation-failure",
                "path": retained_path,
            }
            raise RuntimeError(
                f"{exc}; failed ZeRO child artifacts retained at {retained_path}"
            ) from exc
        verify_inputs = bundle["verify_inputs"]
        verify_output = bundle["verify_output"]
        parameter_count = sum(
            tensor.numel()
            for name, tensor in verify_output.items()
            if name.startswith("parameter:")
        )
        signature = InputSignature(
            shapes={
                "x": tuple(verify_inputs["x"].shape),
                "y": tuple(verify_inputs["y"].shape),
                "rank_final_microbatch_losses": tuple(
                    verify_output["rank_final_microbatch_losses"].shape
                ),
            },
            dtypes={
                "x": str(verify_inputs["x"].dtype),
                "y": str(verify_inputs["y"].dtype),
                "rank_final_microbatch_losses": str(
                    verify_output["rank_final_microbatch_losses"].dtype
                ),
            },
            batch_size=DEFAULT_PROFILE.batch_size * context["world_size"],
            parameter_count=parameter_count,
            precision_flags=PrecisionFlags(tf32=False),
            world_size=context["world_size"],
        )
        self._zero2_result_bundle = bundle
        self._zero2_verify_inputs = verify_inputs
        self._subprocess_verify_output = verify_output
        self._subprocess_output_tolerance = bundle["output_tolerance"]
        self._subprocess_input_signature = signature
        retention = {
            "policy": "delete-after-success-retain-failure",
            "status": "cleaned-after-success",
            "path": str(context["result_dir"]),
            "diagnostics": "validated manifests and tensors remain in the in-memory result bundle",
        }
        try:
            shutil.rmtree(context["result_dir"])
        except OSError as exc:
            retention.update(
                status="retained-cleanup-failure",
                cleanup_error=str(exc),
            )
        context["retention"] = retention
        bundle["artifact_retention"] = retention

    def _require_result_bundle(self) -> Dict[str, Any]:
        if self._zero2_result_bundle is None:
            raise RuntimeError(
                "ZeRO verification payload is unavailable until the fresh child-result callback succeeds"
            )
        return self._zero2_result_bundle

    def get_verify_inputs(self) -> Dict[str, torch.Tensor]:
        self._require_result_bundle()
        return cast(Dict[str, torch.Tensor], self._zero2_verify_inputs)

    def get_verify_output(self) -> Dict[str, torch.Tensor]:
        self._require_result_bundle()
        return cast(Dict[str, torch.Tensor], self._subprocess_verify_output)

    def get_input_signature(self) -> InputSignature:
        self._require_result_bundle()
        return self._subprocess_input_signature

    def get_output_tolerance(self) -> tuple[float, float]:
        self._require_result_bundle()
        return cast(tuple[float, float], self._subprocess_output_tolerance)

    def validate_result(self) -> Optional[str]:
        if self._zero2_result_bundle is None:
            message = "ZeRO fresh child-result callback has not succeeded"
            context = self._zero2_result_context
            if context is not None and Path(context["result_dir"]).exists():
                message += f"; artifacts retained at {context['result_dir']}"
            return message
        return None

    def teardown(self) -> None:
        context = self._zero2_result_context
        if (
            self._zero2_result_bundle is None
            and context is not None
            and Path(context["result_dir"]).exists()
        ):
            print(
                f"[zero2] retained unsuccessful child artifacts at {context['result_dir']}",
                flush=True,
            )
        return None


__all__ = ["Zero2TorchrunBenchmark", "LAUNCH_WALL_NS_ENV", "LAUNCH_MONOTONIC_NS_ENV"]
