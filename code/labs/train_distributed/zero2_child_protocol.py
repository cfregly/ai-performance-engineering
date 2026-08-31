"""Fresh child-result contract for the four ZeRO training benchmarks.

The performance workload remains in :mod:`zero2_common`.  When the ZeRO
torchrun adapter explicitly requests evidence, each child executes this small
correctness profile *after* the measured workload and atomically publishes its
own parameters, gradients, optimizer state, losses and inputs.  The parent
validates the complete rank quorum against an independently constructed dense
AdamW reference before exposing a post-timing verification payload.

The explicit ``verification-only`` route is intended for bounded CPU/Gloo
validation.  It never reports performance timing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


SCHEMA_VERSION = "aisp.zero2.child-result.v3"
RESULT_CALLBACK = "consume_zero2_child_results"
RESULT_DIR_ENV = "AISP_ZERO2_RESULT_DIR"
RUN_ID_ENV = "AISP_ZERO2_RESULT_RUN_ID"
MODE_ENV = "AISP_ZERO2_RESULT_MODE"
VARIANT_ENV = "AISP_ZERO2_RESULT_VARIANT"
PROFILE_KIND_ENV = "AISP_ZERO2_PROFILE_KIND"
LAUNCH_WALL_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"
LAUNCH_MONOTONIC_NS_ENV = "AISP_TORCHRUN_RESULT_LAUNCH_MONOTONIC_NS"
POST_TIMING_PROFILE_KIND = "post-timing-correctness"
VERIFICATION_ONLY_PROFILE_KIND = "verification-only"
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Zero2ProfileConfig:
    """Bounded correctness workload, deliberately separate from performance."""

    hidden_size: int = 8
    batch_size: int = 3
    updates: int = 3
    grad_accum: int = 2
    learning_rate: float = 1.0e-3
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    weight_decay: float = 0.05
    model_seed: int = 42017
    input_seed: int = 73001
    process_torch_seed: int = 42

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "batch_size": self.batch_size,
            "updates": self.updates,
            "grad_accum": self.grad_accum,
            "learning_rate": self.learning_rate,
            "betas": list(self.betas),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "model_seed": self.model_seed,
            "input_seed": self.input_seed,
            "process_torch_seed": self.process_torch_seed,
        }


DEFAULT_PROFILE = Zero2ProfileConfig()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _rng_snapshot(device: torch.device) -> Dict[str, Any]:
    cpu_state = torch.get_rng_state()
    snapshot: Dict[str, Any] = {
        "torch_initial_seed": int(torch.initial_seed()),
        "cpu_state_sha256": _tensor_sha256(cpu_state),
    }
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
        snapshot.update(
            {
                "cuda_initial_seed": int(torch.cuda.initial_seed()),
                "cuda_state_sha256": _tensor_sha256(cuda_state),
            }
        )
    return snapshot


@contextmanager
def _independent_reference_math(device: torch.device):
    """Use exact FP32 matmul for the profile, then restore backend policy."""

    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32: Optional[bool] = None
    if device.type == "cuda":
        previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _profile_inputs(
    config: Zero2ProfileConfig,
    *,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(config.input_seed + rank)
    shape = (config.updates, config.grad_accum, config.batch_size, config.hidden_size)
    x = torch.randn(shape, generator=generator, dtype=torch.float32)
    y = torch.randn(shape, generator=generator, dtype=torch.float32)
    return x, y


def _optimizer_state(
    optimizer: Any,
    named_parameters: Mapping[str, torch.nn.Parameter],
) -> Dict[str, Dict[str, torch.Tensor]]:
    local_optimizer = getattr(optimizer, "optim", optimizer)
    name_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    captured: Dict[str, Dict[str, torch.Tensor]] = {}
    for parameter, state in local_optimizer.state.items():
        name = name_by_id.get(id(parameter))
        if name is None:
            raise RuntimeError("Local AdamW state contains an unknown parameter")
        required = {"step", "exp_avg", "exp_avg_sq"}
        missing = required - set(state)
        if missing:
            raise RuntimeError(f"Local AdamW state for {name} is missing {sorted(missing)}")
        captured[name] = {
            key: torch.as_tensor(state[key]).detach().cpu().clone()
            for key in sorted(required)
        }
    return captured


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists() or destination.exists():
        raise RuntimeError(f"Refusing to overwrite ZeRO result artifact: {destination}")
    with temporary.open("xb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists() or destination.exists():
        raise RuntimeError(f"Refusing to overwrite ZeRO result manifest: {destination}")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _required_result_environment() -> Dict[str, str]:
    names = (
        RESULT_DIR_ENV,
        RUN_ID_ENV,
        MODE_ENV,
        VARIANT_ENV,
        PROFILE_KIND_ENV,
        LAUNCH_WALL_NS_ENV,
        LAUNCH_MONOTONIC_NS_ENV,
    )
    values: Dict[str, str] = {}
    for name in names:
        value = os.environ.get(name)
        if value is None or not value:
            raise RuntimeError(f"ZeRO child-result protocol requires {name}")
        values[name] = value
    return values


def result_protocol_requested() -> bool:
    """Return whether the parent explicitly requested a ZeRO result bundle."""

    return bool(os.environ.get(RESULT_DIR_ENV))


def required_profile_kind() -> str:
    """Return the parent-requested execution kind, rejecting unknown values."""

    value = os.environ.get(PROFILE_KIND_ENV)
    if value not in {POST_TIMING_PROFILE_KIND, VERIFICATION_ONLY_PROFILE_KIND}:
        raise RuntimeError(
            f"ZeRO child-result protocol requires a supported {PROFILE_KIND_ENV}; "
            f"got {value!r}"
        )
    return value


def _validate_execution_kind(
    profile_kind: str,
    *,
    backend: str,
    device_type: str,
) -> None:
    """Bind evidence labels to the child execution path that produced them."""

    if profile_kind == POST_TIMING_PROFILE_KIND:
        if (backend, device_type) != ("nccl", "cuda"):
            raise RuntimeError(
                "Post-timing ZeRO correctness requires the NCCL/CUDA performance child; "
                f"got backend={backend!r}, device_type={device_type!r}"
            )
        return
    if profile_kind == VERIFICATION_ONLY_PROFILE_KIND:
        if (backend, device_type) not in {("gloo", "cpu"), ("nccl", "cuda")}:
            raise RuntimeError(
                "Verification-only ZeRO evidence requires Gloo/CPU or NCCL/CUDA; "
                f"got backend={backend!r}, device_type={device_type!r}"
            )
        return
    raise RuntimeError(f"Unsupported ZeRO profile kind {profile_kind!r}")


def run_zero2_result_profile(
    *,
    optimized: bool,
    variant: str,
    device: torch.device,
    profile: Zero2ProfileConfig = DEFAULT_PROFILE,
) -> None:
    """Run a fresh production-component profile and publish one rank result.

    This function requires an initialized process group.  It uses the same model,
    DDP construction, communication hook, sharded optimizer and training-step
    implementation as the performance workload.  Its fixed small inputs and
    reference-friendly FP32 execution are intentionally outside the timed region.
    """

    from labs.train_distributed.zero2_common import (
        build_model,
        build_training_components,
        get_zero2_communication_evidence,
        training_step,
    )

    if not dist.is_initialized():
        raise RuntimeError("ZeRO child-result profile requires an initialized process group")
    env = _required_result_environment()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mode = "optimized" if optimized else "baseline"
    if env[MODE_ENV] != mode:
        raise RuntimeError(f"ZeRO mode mismatch: parent={env[MODE_ENV]!r}, child={mode!r}")
    if env[VARIANT_ENV] != variant:
        raise RuntimeError(
            f"ZeRO variant mismatch: parent={env[VARIANT_ENV]!r}, child={variant!r}"
        )
    if variant not in {"single", "multigpu"}:
        raise RuntimeError(f"Invalid ZeRO variant {variant!r}")
    if variant == "single" and world_size != 1:
        raise RuntimeError("ZeRO single verification requires world_size == 1")
    if variant == "multigpu" and world_size < 2:
        raise RuntimeError("ZeRO multigpu verification requires world_size >= 2")
    if int(torch.initial_seed()) != profile.process_torch_seed:
        raise RuntimeError(
            "ZeRO verification child started with an unexpected torch seed: "
            f"expected {profile.process_torch_seed}, got {int(torch.initial_seed())}"
        )

    result_dir = Path(env[RESULT_DIR_ENV]).resolve()
    if not result_dir.is_dir():
        raise RuntimeError(f"ZeRO result directory does not exist: {result_dir}")
    launch_wall_ns = int(env[LAUNCH_WALL_NS_ENV])
    launch_monotonic_ns = int(env[LAUNCH_MONOTONIC_NS_ENV])
    backend = str(dist.get_backend())
    profile_kind = required_profile_kind()
    _validate_execution_kind(
        profile_kind,
        backend=backend,
        device_type=device.type,
    )
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    if started_wall_ns < launch_wall_ns or started_monotonic_ns < launch_monotonic_ns:
        raise RuntimeError("ZeRO child-result clock predates the parent launch window")

    rng_before = _rng_snapshot(device)
    cuda_devices: list[int] = []
    if device.type == "cuda":
        if device.index is None:
            raise RuntimeError("CUDA verification device must have an explicit index")
        cuda_devices = [device.index]

    with (
        torch.random.fork_rng(devices=cuda_devices, enabled=True),
        _independent_reference_math(device),
    ):
        # build_model initializes its Linear modules on CPU before moving them to
        # the requested device. Seed only the CPU default generator so this
        # profile does not reset every visible accelerator RNG.
        torch.default_generator.manual_seed(profile.model_seed)
        model = build_model(profile.hidden_size, device)
        named_parameters = dict(model.named_parameters())
        initial_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in named_parameters.items()
        }
        ddp, optimizer = build_training_components(
            model,
            profile.learning_rate,
            optimized=optimized,
            device_ids=[device.index] if device.type == "cuda" else None,
        )

        input_x, input_y = _profile_inputs(profile, rank=rank)
        x = torch.empty(profile.batch_size, profile.hidden_size, device=device)
        y = torch.empty_like(x)
        final_gradients: Dict[str, torch.Tensor] = {}
        final_microbatch_losses: list[torch.Tensor] = []

        def capture_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
            nonlocal final_gradients
            parameter_list = list(parameters)
            name_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
            captured: Dict[str, torch.Tensor] = {}
            for parameter in parameter_list:
                if parameter.grad is None:
                    raise RuntimeError("ZeRO correctness profile observed a missing gradient")
                name = name_by_id[id(parameter)]
                captured[name] = parameter.grad.detach().cpu().clone()
            final_gradients = captured

        for update in range(profile.updates):
            microbatches = [
                (
                    input_x[update, micro].to(device=device),
                    input_y[update, micro].to(device=device),
                )
                for micro in range(profile.grad_accum)
            ]
            final_microbatch_loss = training_step(
                ddp,
                optimizer,
                x,
                y,
                None,
                profile.grad_accum,
                fixed_microbatches=microbatches,
                autocast_enabled=False,
                post_clip_callback=capture_gradients,
            )
            final_microbatch_losses.append(
                final_microbatch_loss.detach().cpu().clone()
            )

        final_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in named_parameters.items()
        }
        local_optimizer_state = _optimizer_state(optimizer, named_parameters)
        communication = get_zero2_communication_evidence(ddp)
        optimizer_group = getattr(optimizer, "optim", optimizer).param_groups[0]
        optimizer_config = {
            "lr": float(optimizer_group["lr"]),
            "betas": [float(value) for value in optimizer_group["betas"]],
            "eps": float(optimizer_group["eps"]),
            "weight_decay": float(optimizer_group["weight_decay"]),
            "amsgrad": bool(optimizer_group.get("amsgrad", False)),
            "maximize": bool(optimizer_group.get("maximize", False)),
            "fused": bool(optimizer_group.get("fused", False)),
        }

    rng_after = _rng_snapshot(device)
    if rng_after != rng_before:
        raise RuntimeError("ZeRO result profile mutated the wrapper-managed RNG state")
    dist.barrier()
    finished_monotonic_ns = time.monotonic_ns()
    finished_wall_ns = time.time_ns()

    config = profile.as_dict()
    config.update(
        {
            "mode": mode,
            "variant": variant,
            "world_size": world_size,
            "backend": backend,
            "device_type": device.type,
            "profile_kind": profile_kind,
            "optimizer": optimizer_config,
        }
    )
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": env[RUN_ID_ENV],
        "rank": rank,
        "world_size": world_size,
        "config": config,
        "freshness": {
            "launch_wall_ns": launch_wall_ns,
            "launch_monotonic_ns": launch_monotonic_ns,
            "started_wall_ns": started_wall_ns,
            "started_monotonic_ns": started_monotonic_ns,
            "finished_wall_ns": finished_wall_ns,
            "finished_monotonic_ns": finished_monotonic_ns,
        },
        "rng_before": rng_before,
        "rng_after": rng_after,
        "communication": communication,
        "inputs": {"x": input_x, "y": input_y},
        "initial_parameters": initial_parameters,
        "final_parameters": final_parameters,
        "final_gradients": final_gradients,
        "local_optimizer_state": local_optimizer_state,
        "final_microbatch_losses": torch.stack(final_microbatch_losses),
    }
    payload_path = result_dir / f"rank-{rank}.pt"
    _atomic_torch_save(payload, payload_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": env[RUN_ID_ENV],
        "rank": rank,
        "world_size": world_size,
        "config": config,
        "freshness": payload["freshness"],
        "rng_before": rng_before,
        "rng_after": rng_after,
        "communication": communication,
        "payload_file": payload_path.name,
        "payload_size": payload_path.stat().st_size,
        "payload_sha256": _sha256_file(payload_path),
    }
    _atomic_json(manifest, result_dir / f"rank-{rank}.json")


def _require_dict(value: Any, label: str) -> MutableMapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a dict")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _iter_tensors(value: Any, prefix: str = "payload") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_tensors(child, f"{prefix}[{index}]")


def _independent_forward(
    inputs: torch.Tensor,
    parameters: Mapping[str, torch.nn.Parameter],
) -> torch.Tensor:
    """Dense seven-linear reference without importing the production builder."""

    output = inputs
    module_indices = sorted(
        int(name.split(".", 1)[0])
        for name in parameters
        if name.endswith(".weight")
    )
    if module_indices != [0, 2, 4, 6, 8, 10, 12]:
        raise RuntimeError(f"Unexpected ZeRO model parameter layout: {module_indices}")
    for offset, module_index in enumerate(module_indices):
        output = F.linear(
            output,
            parameters[f"{module_index}.weight"],
            parameters[f"{module_index}.bias"],
        )
        if offset + 1 != len(module_indices):
            output = F.gelu(output)
    return output


def _independent_reference(
    payloads: Sequence[Mapping[str, Any]],
    profile: Zero2ProfileConfig,
) -> Dict[str, Any]:
    first = payloads[0]
    initial = _require_dict(first["initial_parameters"], "initial_parameters")
    parameters = {
        name: torch.nn.Parameter(tensor.detach().cpu().clone())
        for name, tensor in initial.items()
    }
    optimizer_config = _require_dict(first["config"], "config")["optimizer"]
    optimizer = torch.optim.AdamW(
        list(parameters.values()),
        lr=profile.learning_rate,
        betas=profile.betas,
        eps=profile.eps,
        weight_decay=profile.weight_decay,
        amsgrad=False,
        maximize=False,
        fused=bool(optimizer_config["fused"]),
    )
    final_gradients: Dict[str, torch.Tensor] = {}
    rank_final_microbatch_losses: list[list[torch.Tensor]] = [
        [] for _ in payloads
    ]
    for update in range(profile.updates):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(profile.grad_accum):
            local_losses: list[torch.Tensor] = []
            for rank, payload in enumerate(payloads):
                inputs = payload["inputs"]
                x = inputs["x"][update, micro]
                y = inputs["y"][update, micro]
                local_loss = F.mse_loss(_independent_forward(x, parameters), y) / profile.grad_accum
                local_losses.append(local_loss)
                if micro + 1 == profile.grad_accum:
                    rank_final_microbatch_losses[rank].append(
                        local_loss.detach().clone()
                    )
            (sum(local_losses) / len(local_losses)).backward()
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0)
        final_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in parameters.items()
        }
        optimizer.step()
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, parameter in parameters.items():
        state[name] = {
            key: torch.as_tensor(optimizer.state[parameter][key]).detach().clone()
            for key in ("step", "exp_avg", "exp_avg_sq")
        }
    return {
        "final_parameters": {name: value.detach().clone() for name, value in parameters.items()},
        "final_gradients": final_gradients,
        "optimizer_state": state,
        "final_microbatch_losses": [
            torch.stack(values) for values in rank_final_microbatch_losses
        ],
    }


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-6,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise RuntimeError(f"ZeRO child-result mismatch for {label}: {exc}") from exc


def validate_zero2_result_bundle(
    result_dir: Path,
    *,
    run_id: str,
    mode: str,
    variant: str,
    world_size: int,
    launch_wall_ns: int,
    launch_monotonic_ns: int,
    finish_wall_ns: int,
    finish_monotonic_ns: int,
    profile_kind: str,
    profile: Zero2ProfileConfig = DEFAULT_PROFILE,
) -> Dict[str, Any]:
    """Validate and independently reproduce a complete child result quorum."""

    if mode not in {"baseline", "optimized"}:
        raise RuntimeError(f"Invalid ZeRO mode {mode!r}")
    if profile_kind not in {POST_TIMING_PROFILE_KIND, VERIFICATION_ONLY_PROFILE_KIND}:
        raise RuntimeError(f"Unsupported ZeRO profile kind {profile_kind!r}")
    if variant == "single" and world_size != 1:
        raise RuntimeError("ZeRO single result requires world_size == 1")
    if variant == "multigpu" and world_size < 2:
        raise RuntimeError("ZeRO multigpu result requires world_size >= 2")
    if variant not in {"single", "multigpu"}:
        raise RuntimeError(f"Invalid ZeRO variant {variant!r}")
    result_dir = Path(result_dir).resolve()
    if not result_dir.is_dir():
        raise RuntimeError(f"Missing ZeRO child-result directory: {result_dir}")
    expected_names = {
        f"rank-{rank}.{suffix}"
        for rank in range(world_size)
        for suffix in ("json", "pt")
    }
    actual_names = {path.name for path in result_dir.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError(
            "ZeRO child-result quorum mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )

    payloads: list[Mapping[str, Any]] = []
    manifests: list[Mapping[str, Any]] = []
    expected_payload_keys = {
        "schema_version",
        "run_id",
        "rank",
        "world_size",
        "config",
        "freshness",
        "rng_before",
        "rng_after",
        "communication",
        "inputs",
        "initial_parameters",
        "final_parameters",
        "final_gradients",
        "local_optimizer_state",
        "final_microbatch_losses",
    }
    expected_manifest_keys = {
        "schema_version",
        "run_id",
        "rank",
        "world_size",
        "config",
        "freshness",
        "rng_before",
        "rng_after",
        "communication",
        "payload_file",
        "payload_size",
        "payload_sha256",
    }
    base_config = profile.as_dict()
    base_config.update(
        {
            "mode": mode,
            "variant": variant,
            "world_size": world_size,
            "profile_kind": profile_kind,
        }
    )
    runtime_config_reference: Optional[Dict[str, Any]] = None
    optimized_communication_counts: Optional[tuple[int, int, int]] = None
    for rank in range(world_size):
        manifest_path = result_dir / f"rank-{rank}.json"
        payload_path = result_dir / f"rank-{rank}.pt"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid ZeRO rank-{rank} manifest: {exc}") from exc
        _require_exact_keys(manifest, expected_manifest_keys, f"rank-{rank} manifest")
        if manifest["payload_file"] != payload_path.name:
            raise RuntimeError(f"rank-{rank} manifest points to the wrong payload")
        payload_size = payload_path.stat().st_size
        if payload_size > MAX_PAYLOAD_BYTES:
            raise RuntimeError(
                f"rank-{rank} payload exceeds the {MAX_PAYLOAD_BYTES}-byte safety limit"
            )
        if int(manifest["payload_size"]) != payload_size:
            raise RuntimeError(f"rank-{rank} payload size mismatch")
        if manifest["payload_sha256"] != _sha256_file(payload_path):
            raise RuntimeError(f"rank-{rank} payload checksum mismatch")
        try:
            payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Invalid ZeRO rank-{rank} tensor payload: {exc}") from exc
        payload = _require_dict(payload, f"rank-{rank} payload")
        _require_exact_keys(payload, expected_payload_keys, f"rank-{rank} payload")
        for source_name, source in (("manifest", manifest), ("payload", payload)):
            if source["schema_version"] != SCHEMA_VERSION:
                raise RuntimeError(f"rank-{rank} {source_name} schema mismatch")
            if source["run_id"] != run_id:
                raise RuntimeError(f"rank-{rank} {source_name} run_id mismatch")
            if int(source["rank"]) != rank or int(source["world_size"]) != world_size:
                raise RuntimeError(f"rank-{rank} {source_name} rank quorum mismatch")
        if manifest["config"] != payload["config"]:
            raise RuntimeError(f"rank-{rank} manifest/payload config mismatch")
        if manifest["freshness"] != payload["freshness"]:
            raise RuntimeError(f"rank-{rank} manifest/payload freshness mismatch")
        if manifest["rng_before"] != payload["rng_before"] or manifest["rng_after"] != payload["rng_after"]:
            raise RuntimeError(f"rank-{rank} manifest/payload RNG mismatch")
        if manifest["communication"] != payload["communication"]:
            raise RuntimeError(f"rank-{rank} manifest/payload communication mismatch")

        config = _require_dict(payload["config"], f"rank-{rank} config")
        _require_exact_keys(
            config,
            set(base_config) | {"backend", "device_type", "optimizer"},
            f"rank-{rank} config",
        )
        for key, expected in base_config.items():
            if config.get(key) != expected:
                raise RuntimeError(
                    f"rank-{rank} config mismatch for {key}: {config.get(key)!r} != {expected!r}"
                )
        if config.get("backend") not in {"gloo", "nccl"}:
            raise RuntimeError(f"rank-{rank} unsupported verification backend {config.get('backend')!r}")
        if config.get("device_type") not in {"cpu", "cuda"}:
            raise RuntimeError(f"rank-{rank} unsupported verification device {config.get('device_type')!r}")
        _validate_execution_kind(
            profile_kind,
            backend=str(config["backend"]),
            device_type=str(config["device_type"]),
        )
        optimizer_config = _require_dict(config.get("optimizer"), f"rank-{rank} optimizer config")
        expected_optimizer = {
            "lr": profile.learning_rate,
            "betas": list(profile.betas),
            "eps": profile.eps,
            "weight_decay": profile.weight_decay,
            "amsgrad": False,
            "maximize": False,
        }
        for key, expected in expected_optimizer.items():
            if optimizer_config.get(key) != expected:
                raise RuntimeError(f"rank-{rank} optimizer config mismatch for {key}")
        _require_exact_keys(
            optimizer_config,
            set(expected_optimizer) | {"fused"},
            f"rank-{rank} optimizer config",
        )
        if not isinstance(optimizer_config.get("fused"), bool):
            raise RuntimeError(f"rank-{rank} optimizer fused field must be bool")
        runtime_config = {
            "backend": config["backend"],
            "device_type": config["device_type"],
            "optimizer": dict(optimizer_config),
        }
        if runtime_config_reference is None:
            runtime_config_reference = runtime_config
        elif runtime_config != runtime_config_reference:
            raise RuntimeError(f"rank-{rank} runtime config differs across the quorum")

        freshness = _require_dict(payload["freshness"], f"rank-{rank} freshness")
        _require_exact_keys(
            freshness,
            {
                "launch_wall_ns",
                "launch_monotonic_ns",
                "started_wall_ns",
                "started_monotonic_ns",
                "finished_wall_ns",
                "finished_monotonic_ns",
            },
            f"rank-{rank} freshness",
        )
        if int(freshness["launch_wall_ns"]) != launch_wall_ns:
            raise RuntimeError(f"rank-{rank} wall-clock launch token mismatch")
        if int(freshness["launch_monotonic_ns"]) != launch_monotonic_ns:
            raise RuntimeError(f"rank-{rank} monotonic launch token mismatch")
        started_wall = int(freshness["started_wall_ns"])
        finished_wall = int(freshness["finished_wall_ns"])
        started_monotonic = int(freshness["started_monotonic_ns"])
        finished_monotonic = int(freshness["finished_monotonic_ns"])
        if not launch_wall_ns <= started_wall <= finished_wall <= finish_wall_ns:
            raise RuntimeError(f"rank-{rank} wall-clock freshness is outside the parent launch window")
        if not launch_monotonic_ns <= started_monotonic <= finished_monotonic <= finish_monotonic_ns:
            raise RuntimeError(f"rank-{rank} monotonic freshness is outside the parent launch window")
        for artifact in (manifest_path, payload_path):
            mtime_ns = artifact.stat().st_mtime_ns
            if not launch_wall_ns <= mtime_ns <= finish_wall_ns:
                raise RuntimeError(f"rank-{rank} artifact mtime is outside the parent launch window")

        rng_before = _require_dict(payload["rng_before"], f"rank-{rank} rng_before")
        rng_after = _require_dict(payload["rng_after"], f"rank-{rank} rng_after")
        expected_rng_keys = {"torch_initial_seed", "cpu_state_sha256"}
        if config["device_type"] == "cuda":
            expected_rng_keys |= {"cuda_initial_seed", "cuda_state_sha256"}
        _require_exact_keys(rng_before, expected_rng_keys, f"rank-{rank} rng_before")
        _require_exact_keys(rng_after, expected_rng_keys, f"rank-{rank} rng_after")
        if rng_before != rng_after:
            raise RuntimeError(f"rank-{rank} verification profile mutated RNG state")
        if int(rng_before.get("torch_initial_seed", -1)) != profile.process_torch_seed:
            raise RuntimeError(f"rank-{rank} wrapper seed mismatch")
        communication = _require_dict(
            payload["communication"], f"rank-{rank} communication"
        )
        _require_exact_keys(
            communication,
            {
                "mechanism",
                "hook_invocations",
                "reduce_scatter_completions",
                "all_gather_completions",
            },
            f"rank-{rank} communication",
        )
        expected_mechanism = (
            "reduce-scatter-all-gather" if mode == "optimized" else "ddp-all-reduce"
        )
        if communication["mechanism"] != expected_mechanism:
            raise RuntimeError(
                f"rank-{rank} communication mechanism mismatch: "
                f"{communication['mechanism']!r} != {expected_mechanism!r}"
            )
        hook_invocations = int(communication["hook_invocations"])
        reduce_scatter_completions = int(communication["reduce_scatter_completions"])
        all_gather_completions = int(communication["all_gather_completions"])
        if mode == "optimized" and hook_invocations <= 0:
            raise RuntimeError(f"rank-{rank} RS/AG hook was not invoked")
        if mode == "optimized":
            if world_size > 1 and (
                reduce_scatter_completions != hook_invocations
                or all_gather_completions != hook_invocations
            ):
                raise RuntimeError(
                    f"rank-{rank} completed RS/AG counts do not match hook invocations: "
                    f"hooks={hook_invocations}, reduce_scatter={reduce_scatter_completions}, "
                    f"all_gather={all_gather_completions}"
                )
            if world_size == 1 and (
                reduce_scatter_completions != 0 or all_gather_completions != 0
            ):
                raise RuntimeError(
                    f"rank-{rank} single-rank profile reported multirank collective completions"
                )
            communication_counts = (
                hook_invocations,
                reduce_scatter_completions,
                all_gather_completions,
            )
            if optimized_communication_counts is None:
                optimized_communication_counts = communication_counts
            elif communication_counts != optimized_communication_counts:
                raise RuntimeError("RS/AG communication counts differ across ranks")
        if mode == "baseline" and any(
            count != 0
            for count in (
                hook_invocations,
                reduce_scatter_completions,
                all_gather_completions,
            )
        ):
            raise RuntimeError(f"rank-{rank} baseline reported custom collective activity")
        for tensor_name, tensor in _iter_tensors(payload, f"rank-{rank}"):
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"Non-finite tensor in {tensor_name}")
        payloads.append(payload)
        manifests.append(manifest)

    parameter_names = set(_require_dict(payloads[0]["initial_parameters"], "initial_parameters"))
    if not parameter_names:
        raise RuntimeError("ZeRO child-result has no model parameters")
    reference_initial = payloads[0]["initial_parameters"]
    for rank, payload in enumerate(payloads):
        inputs = _require_dict(payload["inputs"], f"rank-{rank} inputs")
        _require_exact_keys(inputs, {"x", "y"}, f"rank-{rank} inputs")
        for state_name in ("initial_parameters", "final_parameters", "final_gradients"):
            state = _require_dict(payload[state_name], f"rank-{rank} {state_name}")
            if set(state) != parameter_names:
                raise RuntimeError(f"rank-{rank} {state_name} parameter coverage mismatch")
        for name in parameter_names:
            _assert_close(
                payload["initial_parameters"][name],
                reference_initial[name],
                label=f"rank-{rank} initial parameter {name}",
                rtol=0.0,
                atol=0.0,
            )
        expected_x, expected_y = _profile_inputs(profile, rank=rank)
        _assert_close(payload["inputs"]["x"], expected_x, label=f"rank-{rank} input x", rtol=0.0, atol=0.0)
        _assert_close(payload["inputs"]["y"], expected_y, label=f"rank-{rank} input y", rtol=0.0, atol=0.0)
        if tuple(payload["final_microbatch_losses"].shape) != (profile.updates,):
            raise RuntimeError(f"rank-{rank} final-microbatch loss update count mismatch")

    reference = _independent_reference(payloads, profile)
    total_delta = 0.0
    total_grad = 0.0
    for rank, payload in enumerate(payloads):
        for name in sorted(parameter_names):
            final_parameter = payload["final_parameters"][name]
            final_gradient = payload["final_gradients"][name]
            _assert_close(
                final_parameter,
                reference["final_parameters"][name],
                label=f"rank-{rank} final parameter {name}",
            )
            _assert_close(
                final_gradient,
                reference["final_gradients"][name],
                label=f"rank-{rank} final gradient {name}",
            )
            total_delta += float((final_parameter - payload["initial_parameters"][name]).abs().sum())
            total_grad += float(final_gradient.abs().sum())
        _assert_close(
            payload["final_microbatch_losses"],
            reference["final_microbatch_losses"][rank],
            label=f"rank-{rank} final scaled microbatch losses",
        )
    if not math.isfinite(total_delta) or total_delta <= 0.0:
        raise RuntimeError("ZeRO child-result detected an optimizer no-op")
    if not math.isfinite(total_grad) or total_grad <= 0.0:
        raise RuntimeError("ZeRO child-result detected zero final gradients")

    ownership: list[set[str]] = []
    for rank, payload in enumerate(payloads):
        local_state = _require_dict(payload["local_optimizer_state"], f"rank-{rank} local optimizer state")
        local_names = set(local_state)
        if mode == "baseline" and not local_names:
            raise RuntimeError(f"rank-{rank} has no local AdamW state")
        if not local_names <= parameter_names:
            raise RuntimeError(f"rank-{rank} local AdamW state has unknown parameters")
        ownership.append(local_names)
        for name in sorted(local_names):
            state = _require_dict(local_state[name], f"rank-{rank} optimizer state {name}")
            _require_exact_keys(state, {"step", "exp_avg", "exp_avg_sq"}, f"rank-{rank} optimizer state {name}")
            if int(torch.as_tensor(state["step"]).item()) != profile.updates:
                raise RuntimeError(f"rank-{rank} optimizer update count mismatch for {name}")
            for key in ("step", "exp_avg", "exp_avg_sq"):
                _assert_close(
                    torch.as_tensor(state[key]),
                    reference["optimizer_state"][name][key],
                    label=f"rank-{rank} optimizer {name}.{key}",
                )
            if float(torch.as_tensor(state["exp_avg"]).abs().sum()) <= 0.0:
                raise RuntimeError(f"rank-{rank} optimizer state is a no-op for {name}")
    if mode == "baseline":
        if any(names != parameter_names for names in ownership):
            raise RuntimeError("Baseline AdamW must retain full state on every rank")
    else:
        union = set().union(*ownership)
        if union != parameter_names:
            raise RuntimeError("Sharded AdamW state does not cover every parameter")
        expected_nonempty_owners = min(world_size, len(parameter_names))
        actual_nonempty_owners = sum(bool(names) for names in ownership)
        if actual_nonempty_owners != expected_nonempty_owners:
            raise RuntimeError(
                "Sharded AdamW state has degenerate owner coverage: "
                f"expected {expected_nonempty_owners} nonempty ranks, "
                f"got {actual_nonempty_owners}"
            )
        for left in range(len(ownership)):
            for right in range(left + 1, len(ownership)):
                if ownership[left] & ownership[right]:
                    raise RuntimeError("Sharded AdamW state ownership overlaps between ranks")

    rank0 = payloads[0]
    verify_output: Dict[str, torch.Tensor] = {}
    for name in sorted(parameter_names):
        verify_output[f"parameter:{name}"] = rank0["final_parameters"][name].detach().clone()
        verify_output[f"gradient:{name}"] = rank0["final_gradients"][name].detach().clone()
    verify_output["rank_final_microbatch_losses"] = torch.stack(
        [payload["final_microbatch_losses"] for payload in payloads]
    )
    verify_inputs = {
        "x": torch.stack([payload["inputs"]["x"] for payload in payloads]),
        "y": torch.stack([payload["inputs"]["y"] for payload in payloads]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "variant": variant,
        "world_size": world_size,
        "profile_kind": profile_kind,
        "verify_output": verify_output,
        "verify_inputs": verify_inputs,
        "output_tolerance": (1.0e-5, 1.0e-6),
        "parameter_names": sorted(parameter_names),
        "manifests": manifests,
    }
