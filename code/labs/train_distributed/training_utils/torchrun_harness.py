"""Fail-closed torchrun metadata with an explicit child-result opt-in."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, NoReturn, cast

import torch

from core.benchmark.verification import InputSignature
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from labs.train_distributed.training_utils.child_result import (
    CONTRACT_ENV,
    ITERATIONS_ENV,
    RESULT_CALLBACK,
    RESULT_DIR_ENV,
    RUN_ID_ENV,
    WORLD_SIZE_ENV,
    TorchrunChildResultContract,
    validate_training_child_result_bundle,
)

CHILD_TRAINING_VERIFICATION_UNSUPPORTED = (
    "SKIPPED: TorchrunScriptBenchmark actual child-training verification is unsupported. "
    "The wrapper does not collect child model outputs, losses, gradients or optimizer state. "
    "Run the training script directly; harness comparison requires an explicit "
    "child-produced verification contract and an independent reference."
)

CHILD_RESULT_NOT_READY = (
    "Torchrun child-result verification is unavailable until the opted-in child "
    "publishes and the parent validates a fresh complete result bundle"
)


class TorchrunScriptBenchmark(BaseBenchmark):
    """Keep script configuration discoverable without certifying unrelated work.

    Direct training entrypoints remain usable.  Generic instances fail before
    launch.  A workload may opt in only by naming a concrete child-result contract
    and making its child script publish the required real tensors.  A parent-side
    toy forward cannot establish that contract.
    """

    def __init__(
        self,
        *,
        script_path: Path,
        base_args: list[str] | None = None,
        target_label: str | None = None,
        config_arg_map: dict[str, str] | None = None,
        multi_gpu_required: bool = True,
        default_nproc_per_node: int | None = None,
        default_iterations: int | None = None,
        measurement_timeout_seconds: int | None = None,
        env: dict[str, str] | None = None,
        name: str | None = None,
        child_result_contract: TorchrunChildResultContract | None = None,
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
        if child_result_contract is not None:
            child_result_contract.validate()
        self._child_result_contract = child_result_contract
        self._child_result_context: dict[str, Any] | None = None
        self._child_result_bundle: dict[str, Any] | None = None

    def _unsupported_child_verification(self) -> NoReturn:
        raise RuntimeError(CHILD_TRAINING_VERIFICATION_UNSUPPORTED)

    def _require_torchrun_launcher(self) -> NoReturn:
        if self._child_result_contract is None:
            self._unsupported_child_verification()
        raise RuntimeError(
            "Opted-in training child verification must execute through launch_via=torchrun"
        )

    def setup(self) -> None:
        self._require_torchrun_launcher()

    def benchmark_fn(self) -> None:
        self._require_torchrun_launcher()

    def capture_verification_payload(self) -> None:
        self._require_torchrun_launcher()

    def _prepare_verification_payload(self) -> NoReturn:
        # Retained for callers of the former helper; stale cached attributes must
        # never turn unobserved child training into accepted verification.
        self._require_torchrun_launcher()

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        if self._child_result_contract is None:
            self._unsupported_child_verification()
        if self._child_result_bundle is None:
            raise RuntimeError(CHILD_RESULT_NOT_READY)
        return cast(dict[str, torch.Tensor], self._subprocess_verify_inputs)

    def get_verify_output(self) -> dict[str, torch.Tensor]:
        if self._child_result_contract is None:
            self._unsupported_child_verification()
        if self._child_result_bundle is None:
            raise RuntimeError(CHILD_RESULT_NOT_READY)
        return cast(dict[str, torch.Tensor], self._subprocess_verify_output)

    def get_input_signature(self) -> InputSignature:
        if self._child_result_contract is None:
            self._unsupported_child_verification()
        if self._child_result_bundle is None:
            raise RuntimeError(CHILD_RESULT_NOT_READY)
        return cast(InputSignature, self._subprocess_input_signature)

    def get_output_tolerance(self) -> tuple[float, float]:
        if self._child_result_contract is None:
            self._unsupported_child_verification()
        if self._child_result_bundle is None:
            raise RuntimeError(CHILD_RESULT_NOT_READY)
        return cast(tuple[float, float], self._subprocess_output_tolerance)

    def teardown(self) -> None:
        context = self._child_result_context
        if (
            self._child_result_bundle is None
            and context is not None
            and Path(context["result_dir"]).exists()
        ):
            print(
                f"[training-child-result] retained unsuccessful child artifacts at "
                f"{context['result_dir']}",
                flush=True,
            )
        return None

    def validate_result(self) -> str | None:
        if self._child_result_contract is None:
            return CHILD_TRAINING_VERIFICATION_UNSUPPORTED
        if self._child_result_bundle is None:
            context = self._child_result_context
            message = CHILD_RESULT_NOT_READY
            if context is not None and Path(context["result_dir"]).exists():
                message += f"; artifacts retained at {context['result_dir']}"
            return message
        return None

    def _resolve_nproc_per_node(self) -> int | None:
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
                raise RuntimeError(
                    f"nproc_per_node={requested} exceeds available GPUs ({available})"
                )
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

    def get_torchrun_spec(self, config: BenchmarkConfig | None = None) -> TorchrunLaunchSpec:
        contract = self._child_result_contract
        if contract is None:
            # The harness must not launch and time a generic training child while
            # presenting an unrelated parent-side tensor as correctness evidence.
            self._unsupported_child_verification()
        if config is None:
            config = self.get_config()

        nnodes = getattr(config, "nnodes", None) or 1
        try:
            nodes = int(nnodes)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Training child-result verification requires an exact integer nnodes value"
            ) from exc
        if nodes != 1:
            raise RuntimeError(
                "Training child-result verification currently requires nnodes == 1; "
                "its result directory and monotonic freshness clock are host-local"
            )
        nproc = getattr(config, "nproc_per_node", None)
        if nproc is None:
            nproc = self._resolve_nproc_per_node()
        nproc = 1 if nproc is None else int(nproc)
        if nproc <= 0:
            raise RuntimeError("Training child-result verification requires nproc_per_node >= 1")
        if self._multi_gpu_required and nproc < 2:
            raise RuntimeError(
                "SKIPPED: Training child-result multi-GPU verification requires >=2 ranks"
            )
        if nproc > 1 and (
            contract.collective_type is None or contract.collective_algorithm is None
        ):
            raise RuntimeError(
                "Multi-rank training child-result contracts must declare collective type and algorithm"
            )
        iterations = getattr(config, "iterations", None)
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise RuntimeError(
                "Training child-result verification requires a positive iteration count"
            )

        previous = self._child_result_context
        if previous is not None and Path(previous["result_dir"]).exists():
            raise RuntimeError(
                "Refusing to replace an unconsumed training child-result context: "
                f"status={previous['retention']['status']}, path={previous['result_dir']}"
            )
        result_dir = Path(tempfile.mkdtemp(prefix="aisp-training-child-result-"))
        run_id = uuid.uuid4().hex
        env = dict(self._env)
        env.update(
            {
                RESULT_DIR_ENV: str(result_dir),
                RUN_ID_ENV: run_id,
                CONTRACT_ENV: json.dumps(contract.to_dict(), sort_keys=True, separators=(",", ":")),
                WORLD_SIZE_ENV: str(nproc),
                ITERATIONS_ENV: str(iterations),
            }
        )
        self._child_result_context = {
            "result_dir": result_dir,
            "run_id": run_id,
            "contract": contract,
            "world_size": nproc,
            "requested_iterations": iterations,
            "retention": {
                "policy": "delete-after-success-retain-failure",
                "status": "pending-child-result",
                "path": str(result_dir),
            },
        }
        self._child_result_bundle = None
        for attribute in (
            "_subprocess_verify_inputs",
            "_subprocess_verify_output",
            "_subprocess_output_tolerance",
            "_subprocess_input_signature",
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
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=iterations,
        )

    def consume_training_child_results(
        self,
        *,
        spec: TorchrunLaunchSpec,
        config: BenchmarkConfig,
        launch_wall_ns: int,
        launch_monotonic_ns: int,
        finish_wall_ns: int,
        finish_monotonic_ns: int,
        returncode: int,
        **_: Any,
    ) -> None:
        """Validate a fresh complete child quorum and expose its real tensors."""

        context = self._child_result_context
        if context is None:
            raise RuntimeError("Training child-result callback has no prepared launch context")
        result_dir = Path(context["result_dir"])
        if returncode != 0:
            context["retention"]["status"] = "retained-child-failure"
            raise RuntimeError(
                "Training child-result callback requires a clean child exit; "
                f"artifacts retained at {result_dir}"
            )
        try:
            bundle = validate_training_child_result_bundle(
                result_dir,
                contract=context["contract"],
                run_id=context["run_id"],
                world_size=context["world_size"],
                requested_iterations=context["requested_iterations"],
                launch_wall_ns=int(launch_wall_ns),
                launch_monotonic_ns=int(launch_monotonic_ns),
                finish_wall_ns=int(finish_wall_ns),
                finish_monotonic_ns=int(finish_monotonic_ns),
            )
            requested_seed = getattr(config, "seed", None)
            if requested_seed is not None and bundle["torch_seed"] != int(requested_seed):
                raise RuntimeError(
                    "Training child-result seed differs from the harness request: "
                    f"child={bundle['torch_seed']}, requested={int(requested_seed)}"
                )
        except Exception as exc:
            context["retention"]["status"] = "retained-validation-failure"
            raise RuntimeError(
                f"{exc}; failed training child artifacts retained at {result_dir}"
            ) from exc
        self._child_result_bundle = bundle
        self._subprocess_verify_inputs = bundle["verify_inputs"]
        self._subprocess_verify_output = bundle["verify_output"]
        self._subprocess_output_tolerance = bundle["output_tolerance"]
        self._subprocess_input_signature = bundle["input_signature"]
        # ``--steps`` is an upper bound for dataset-backed training scripts.
        # Bind the rank-0 aggregate timing to the work the child actually
        # completed rather than pretending an exhausted loader ran the maximum.
        spec.timing_iterations_per_sample = bundle["completed_iterations"]
        context["retention"]["status"] = "cleaned-after-success"
        try:
            shutil.rmtree(result_dir)
        except OSError as exc:
            context["retention"].update(
                status="retained-cleanup-failure",
                cleanup_error=str(exc),
            )
