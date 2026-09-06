"""Fail-fast CUDA device identity and current-context contract.

This contract binds a configured logical CUDA device to an expected full-device
UUID and compute capability when those expectations are supplied.  It also
detects current-device drift at harness phase boundaries.

The boundary checks do not prove that kernels never ran on another device.  A
benchmark can switch devices and switch back between two checks, so undeclared
device execution remains a trace-backed Nsys concern scoped to the measured
NVTX range.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from core.profiling.gpu_telemetry import normalize_gpu_uuid, resolve_nvml_device_handle

ComputeCapabilityInput = str | Sequence[int]


class DeviceIdentityError(RuntimeError):
    """Raised when configured and observed CUDA device identity diverge."""


def normalize_compute_capability(value: ComputeCapabilityInput) -> str:
    """Return a canonical ``major.minor`` compute-capability string.

    Accepted forms are ``(10, 0)``, ``"10.0"``, ``"sm_100"``, and
    ``"compute_100"`` (with an optional architecture suffix such as ``a``).
    Ambiguous integer-only values are rejected.
    """
    major: int
    minor: int
    if isinstance(value, str):
        normalized = value.strip().lower()
        dotted = re.fullmatch(r"(\d+)\.(\d+)", normalized)
        if dotted:
            major, minor = int(dotted.group(1)), int(dotted.group(2))
        else:
            compact = re.fullmatch(r"(?:sm_|compute_)(\d+)[a-z]?", normalized)
            if compact is None:
                raise ValueError(
                    "Compute capability must use 'major.minor', 'sm_NN', "
                    f"'compute_NN', or a two-integer sequence; got {value!r}"
                )
            digits = compact.group(1)
            if len(digits) < 2:
                raise ValueError(f"Invalid compact compute capability: {value!r}")
            major, minor = int(digits[:-1]), int(digits[-1])
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, bytes | bytearray)
        and len(value) == 2
        and all(isinstance(part, int) and not isinstance(part, bool) for part in value)
    ):
        major, minor = int(value[0]), int(value[1])
    else:
        raise TypeError(
            "Compute capability must be a string or a two-integer sequence; "
            f"got {type(value).__name__}"
        )

    if major < 1 or minor < 0 or minor > 9:
        raise ValueError(f"Invalid compute capability: {value!r}")
    return f"{major}.{minor}"


@dataclass(frozen=True)
class DeviceIdentityExpectation:
    """Caller-declared identity for one configured CUDA device."""

    expected_uuid: str
    expected_compute_capability: str

    def __post_init__(self) -> None:
        normalized_uuid = normalize_gpu_uuid(self.expected_uuid)
        if normalized_uuid is None:
            raise ValueError("expected_device_uuid must be a non-empty GPU UUID")
        object.__setattr__(self, "expected_uuid", normalized_uuid)
        object.__setattr__(
            self,
            "expected_compute_capability",
            normalize_compute_capability(self.expected_compute_capability),
        )


@dataclass(frozen=True)
class DeviceIdentityObservation:
    """CUDA and NVML identity observed for a configured logical device."""

    logical_index: int
    current_logical_index: int
    cuda_uuid: str | None
    nvml_uuid: str
    compute_capability: str


def build_device_identity_expectation(
    expected_uuid: object | None,
    expected_compute_capability: ComputeCapabilityInput | None,
) -> DeviceIdentityExpectation | None:
    """Build an all-or-nothing expected-device declaration."""
    if expected_uuid is None and expected_compute_capability is None:
        return None
    if expected_uuid is None or expected_compute_capability is None:
        raise ValueError(
            "expected_device_uuid and expected_compute_capability must be configured together"
        )
    return DeviceIdentityExpectation(
        expected_uuid=str(expected_uuid),
        expected_compute_capability=normalize_compute_capability(expected_compute_capability),
    )


def _configured_logical_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"CUDA device identity requires a CUDA device, got {device}")
    if not torch.cuda.is_available():
        raise DeviceIdentityError(f"CUDA device {device} was configured but CUDA is unavailable")
    logical_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
    device_count = int(torch.cuda.device_count())
    if logical_index < 0 or logical_index >= device_count:
        raise DeviceIdentityError(
            f"Configured CUDA logical index {logical_index} is outside visible device count {device_count}"
        )
    return logical_index


def observe_cuda_device_identity(device: torch.device) -> DeviceIdentityObservation:
    """Read one logical CUDA device through both CUDA and NVML."""
    logical_index = _configured_logical_index(device)
    current_logical_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(logical_index)
    cuda_uuid = normalize_gpu_uuid(getattr(properties, "uuid", None))
    compute_capability = normalize_compute_capability(
        torch.cuda.get_device_capability(logical_index)
    )

    try:
        import pynvml
    except ImportError as exc:
        raise DeviceIdentityError(
            "CUDA device identity validation requires pynvml (nvidia-ml-py)"
        ) from exc

    nvml_initialized = False
    try:
        pynvml.nvmlInit()
        nvml_initialized = True
        handle = resolve_nvml_device_handle(
            pynvml,
            logical_index,
            cuda_device_uuid=cuda_uuid,
        )
        nvml_uuid = normalize_gpu_uuid(pynvml.nvmlDeviceGetUUID(handle))
        if nvml_uuid is None:
            raise DeviceIdentityError(
                f"NVML returned an empty UUID for configured CUDA logical index {logical_index}"
            )
    except DeviceIdentityError:
        raise
    except Exception as exc:
        raise DeviceIdentityError(
            f"Failed to resolve CUDA logical index {logical_index} through NVML: {exc}"
        ) from exc
    finally:
        if nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:
                raise DeviceIdentityError(f"Failed to shut down NVML after identity query: {exc}") from exc

    return DeviceIdentityObservation(
        logical_index=logical_index,
        current_logical_index=current_logical_index,
        cuda_uuid=cuda_uuid,
        nvml_uuid=nvml_uuid,
        compute_capability=compute_capability,
    )


def validate_device_identity(
    expectation: DeviceIdentityExpectation | None,
    observation: DeviceIdentityObservation,
    boundary: str,
) -> None:
    """Reject current-context, CUDA/NVML, or declared-identity mismatches."""
    boundary_name = str(boundary).strip()
    if not boundary_name:
        raise ValueError("Device identity boundary must be non-empty")
    if observation.current_logical_index != observation.logical_index:
        raise DeviceIdentityError(
            f"CUDA current-device drift at {boundary_name}: configured logical index "
            f"{observation.logical_index}, current logical index {observation.current_logical_index}"
        )
    if observation.cuda_uuid is not None and observation.cuda_uuid != observation.nvml_uuid:
        raise DeviceIdentityError(
            f"CUDA/NVML device identity mismatch at {boundary_name}: "
            f"cuda={observation.cuda_uuid!r}, nvml={observation.nvml_uuid!r}"
        )
    if expectation is None:
        return
    if observation.nvml_uuid != expectation.expected_uuid:
        raise DeviceIdentityError(
            f"Expected GPU UUID mismatch at {boundary_name}: "
            f"expected={expectation.expected_uuid!r}, observed={observation.nvml_uuid!r}"
        )
    if observation.cuda_uuid is not None and observation.cuda_uuid != expectation.expected_uuid:
        raise DeviceIdentityError(
            f"Expected CUDA UUID mismatch at {boundary_name}: "
            f"expected={expectation.expected_uuid!r}, observed={observation.cuda_uuid!r}"
        )
    if observation.compute_capability != expectation.expected_compute_capability:
        raise DeviceIdentityError(
            f"Expected compute capability mismatch at {boundary_name}: "
            f"expected={expectation.expected_compute_capability!r}, "
            f"observed={observation.compute_capability!r}"
        )


def validate_device_identity_stable(
    initial: DeviceIdentityObservation,
    observation: DeviceIdentityObservation,
    boundary: str,
) -> None:
    """Reject identity changes relative to the first harness boundary."""
    drift = []
    if observation.logical_index != initial.logical_index:
        drift.append(f"logical_index {initial.logical_index}->{observation.logical_index}")
    if observation.cuda_uuid != initial.cuda_uuid:
        drift.append(f"cuda_uuid {initial.cuda_uuid!r}->{observation.cuda_uuid!r}")
    if observation.nvml_uuid != initial.nvml_uuid:
        drift.append(f"nvml_uuid {initial.nvml_uuid!r}->{observation.nvml_uuid!r}")
    if observation.compute_capability != initial.compute_capability:
        drift.append(
            "compute_capability "
            f"{initial.compute_capability!r}->{observation.compute_capability!r}"
        )
    if drift:
        raise DeviceIdentityError(
            f"CUDA device identity drift at {boundary}: {'; '.join(drift)}"
        )


class DeviceIdentityContract:
    """Track CUDA identity across the phase boundaries of one harness run."""

    def __init__(
        self,
        device: torch.device,
        expectation: DeviceIdentityExpectation | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(f"DeviceIdentityContract requires a CUDA device, got {device}")
        self.device = device
        self.expectation = expectation
        self._logical_index: int | None = None
        self._entry_logical_index: int | None = None
        self._initial_observation: DeviceIdentityObservation | None = None

    def establish_configured_device(self) -> int:
        """Select the configured device and remember the caller's entry context."""
        if self._entry_logical_index is not None:
            raise RuntimeError("Configured CUDA device was already established for this contract")
        entry_index = int(torch.cuda.current_device())
        logical_index = _configured_logical_index(self.device)
        self._entry_logical_index = entry_index
        self._logical_index = logical_index
        if entry_index != logical_index:
            torch.cuda.set_device(logical_index)
        return entry_index

    def check(self, boundary: str) -> DeviceIdentityObservation:
        """Validate the current boundary and reject identity drift."""
        if self._logical_index is None:
            raise RuntimeError("establish_configured_device() must run before device identity checks")
        observation = observe_cuda_device_identity(torch.device("cuda", self._logical_index))
        validate_device_identity(self.expectation, observation, boundary)
        initial = self._initial_observation
        if initial is None:
            self._initial_observation = observation
            return observation
        validate_device_identity_stable(initial, observation, boundary)
        return observation

    def restore_entry_device(self) -> None:
        """Restore the caller's CUDA current device after the run."""
        if self._entry_logical_index is None:
            return
        entry_index = self._entry_logical_index
        if not torch.cuda.is_available():
            raise DeviceIdentityError(
                f"CUDA became unavailable before restoring entry logical index {entry_index}"
            )
        if int(torch.cuda.current_device()) != entry_index:
            torch.cuda.set_device(entry_index)
        self._entry_logical_index = None
        self._logical_index = None
        self._initial_observation = None
